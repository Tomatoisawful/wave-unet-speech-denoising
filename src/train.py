"""
train.py

Wave-U-Net 及其 LSTM/自注意力瓶颈消融模型的训练流程。

主要流程：
  1. 从预处理的训练/验证 `.pt` 片段构建 DataLoader；
  2. 创建模型、损失函数和 AdamW 优化器；
  3. 配置带预热的余弦学习率调度；
  4. 每轮训练全部批次，验证并记录 SI-SDR，保存最佳权重，必要时早停；
  5. 可选地记录到 Weights & Biases。

Run:
    python src/train.py
"""

import argparse
import os
import math

import torch
import torch.optim as optim

import config
import utils
from preprocess import PreprocessedDataset
from wave_model import WaveUNetDenoiser
from losses     import DenoisingLoss
from torch.utils.data import DataLoader


try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def lr_lambda(epoch: int) -> float:
    """
    返回指定训练轮次的学习率乘数。

    调度规则：预热阶段由 0 线性增加至 1，之后由 1 余弦衰减至 MIN_LR/LR。

    预热可避免随机初始化模型在训练初期产生过大的梯度更新。
    """
    if epoch < config.WARMUP_EPOCHS:
        return (epoch + 1) / config.WARMUP_EPOCHS

    # 余弦衰减阶段
    progress = (epoch - config.WARMUP_EPOCHS) / max(config.MAX_EPOCHS - config.WARMUP_EPOCHS, 1)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_frac = config.MIN_LR / config.LEARNING_RATE
    return min_frac + (1.0 - min_frac) * cosine


def train_one_epoch(
    model:      WaveUNetDenoiser,
    loader,
    criterion:  DenoisingLoss,
    optimiser:  optim.Optimizer,
    device:     torch.device,
    epoch:      int,
) -> dict[str, float]:
    """
    遍历训练加载器中的全部批次，计算损失、反向传播并更新模型权重。

    返回用于日志记录的平均损失分量字典。
    """
    model.train()
    totals    = {}
    n_batches = 0

    for batch in loader:
        noisy = batch["noisy"].to(device, non_blocking=True)   # (B, 1, samples)
        clean = batch["clean"].to(device, non_blocking=True)   # (B, 1, samples)

        estimated_wav = model(noisy).squeeze(1)
        clean_wav = clean.squeeze(1)

        # 计算损失
        loss, components = criterion(estimated_wav, clean_wav)

        # 反向传播
        optimiser.zero_grad()
        loss.backward()
        # 梯度裁剪，降低训练过程中梯度异常增大的风险
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimiser.step()

        # 累加指标用于日志记录
        for k, v in components.items():
            totals[k] = totals.get(k, 0.0) + v
        n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}


@torch.no_grad()
def validate(
    model:     WaveUNetDenoiser,
    loader,
    criterion: DenoisingLoss,
    device:    torch.device,
) -> dict[str, float]:
    """
    在不计算梯度的情况下运行验证集，并返回平均损失分量。
    """
    model.eval()
    totals    = {}
    n_batches = 0

    for batch in loader:
        noisy = batch["noisy"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)

        estimated_wav = model(noisy).squeeze(1)
        clean_wav = clean.squeeze(1)

        _, components = criterion(estimated_wav, clean_wav)

        for k, v in components.items():
            totals[k] = totals.get(k, 0.0) + v
        n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}


def save_checkpoint(model: WaveUNetDenoiser, epoch: int, val_si_sdr: float, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":       epoch,
        "model_state": model.state_dict(),
        "val_si_sdr":  val_si_sdr,
        "model_config": model.model_config,
    }, path)
    print(f"Checkpoint saved -> {path}")


def load_checkpoint(model: WaveUNetDenoiser, path: str) -> tuple[int, float]:
    """
    将权重载入模型，并返回 (epoch, val_si_sdr)。
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return ckpt["epoch"], ckpt["val_si_sdr"]


def train(
    lstm_direction: str = "bidirectional",
    checkpoint_name: str | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    use_attention: bool = True,
) -> None:
    utils.ensure_dirs()

    if lstm_direction not in {"bidirectional", "unidirectional", "none"}:
        raise ValueError(f"不支持的 LSTM 方向：{lstm_direction}")
    use_lstm = lstm_direction != "none"
    lstm_bidirectional = lstm_direction == "bidirectional"
    if use_lstm and not use_attention:
        raise ValueError("已决定不运行 LSTM-only 消融；--no-attention 只能与 --lstm-direction none 配合")
    batch_size = config.BATCH_SIZE if batch_size is None else batch_size
    num_workers = config.NUM_WORKERS if num_workers is None else num_workers
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size 必须大于 0，num_workers 不能小于 0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Using device: {device}")

    # 数据
    _seg_root    = os.path.join(config.PROCESSED_DIR, "segments")
    train_ds     = PreprocessedDataset(os.path.join(_seg_root, "train"), augment=True,  max_files=config.MAX_TRAIN_FILES)
    val_ds       = PreprocessedDataset(os.path.join(_seg_root, "val"),   augment=False, max_files=config.MAX_VAL_FILES)
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_options.update(persistent_workers=True, prefetch_factor=2)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_options)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_options)

    # 模型
    model = WaveUNetDenoiser(
        use_lstm=use_lstm,
        use_attention=use_attention,
        lstm_bidirectional=lstm_bidirectional,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    if not use_lstm and not use_attention:
        bottleneck_name = "none (pure Wave-U-Net)"
    else:
        bottleneck_name = {
            "bidirectional": "BiLSTM + self-attention",
            "unidirectional": "UniLSTM + self-attention",
            "none": "self-attention only (no LSTM)",
        }[lstm_direction]
    print(f"[train] Bottleneck: {bottleneck_name}")
    print(f"[train] Batch size: {batch_size}; DataLoader workers: {num_workers}")
    print(f"[train] Model parameters: {total_params:,}")

    # 优化器与学习率调度
    optimiser = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.LambdaLR(optimiser, lr_lambda=lr_lambda)
    criterion = DenoisingLoss()

    # WANDB 实验跟踪
    if WANDB_AVAILABLE and config.WANDB_PROJECT:
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY, config={
            "lr": config.LEARNING_RATE, "batch_size": batch_size,
            "epochs": config.MAX_EPOCHS, "lstm_hidden": config.LSTM_HIDDEN,
            "lstm_direction": lstm_direction, "use_lstm": use_lstm,
            "use_attention": use_attention, "num_workers": num_workers,
        })

    # 训练循环
    best_val_si_sdr  = float("-inf")
    patience_counter = 0
    if checkpoint_name is None:
        if not use_lstm and not use_attention:
            checkpoint_name = "best_wave_unet_only.pt"
        else:
            checkpoint_name = {
                "bidirectional": "best_wave_model.pt",
                "unidirectional": "best_wave_unilstm_attention.pt",
                "none": "best_wave_attention_only.pt",
            }[lstm_direction]
    best_ckpt_path = os.path.join(config.CHECKPOINT_DIR, checkpoint_name)

    for epoch in range(config.MAX_EPOCHS):
        current_lr = optimiser.param_groups[0]["lr"]
        print(f"\nEpoch {epoch + 1}/{config.MAX_EPOCHS}  (lr={current_lr:.2e})")

        # 训练
        train_metrics = train_one_epoch(model, train_loader, criterion, optimiser, device, epoch)
        # 验证
        val_metrics   = validate(model, val_loader, criterion, device)

        scheduler.step()

        # 损失中 SI-SDR 以负值保存，此处恢复为正指标值
        val_si_sdr = -val_metrics["loss_si_sdr"]

        print(
            f"  train loss={train_metrics['loss_total']:.4f} "
            f"| val loss={val_metrics['loss_total']:.4f} "
            f"| val SI-SDR={val_si_sdr:.2f} dB"
        )

        # WANDB 日志
        if WANDB_AVAILABLE and config.WANDB_PROJECT:
            wandb.log({"epoch": epoch + 1, "lr": current_lr,
                       **{f"train/{k}": v for k, v in train_metrics.items()},
                       **{f"val/{k}":   v for k, v in val_metrics.items()}})

        # 保存最佳权重
        if val_si_sdr > best_val_si_sdr:
            best_val_si_sdr = val_si_sdr
            patience_counter = 0
            save_checkpoint(model, epoch + 1, val_si_sdr, best_ckpt_path)
        else:
            patience_counter += 1
            print(f"无提升（{patience_counter}/{config.EARLY_STOP_PATIENCE}）")

        # 早停
        if patience_counter >= config.EARLY_STOP_PATIENCE:
            print(f"\n[train] 在第 {epoch + 1} 轮触发早停")
            break

    print(f"\n[train] 完成。最佳验证集 SI-SDR：{best_val_si_sdr:.2f} dB")
    if WANDB_AVAILABLE and config.WANDB_PROJECT:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 Wave-U-Net 语音去噪模型")
    parser.add_argument(
        "--lstm-direction",
        choices=("bidirectional", "unidirectional", "none"),
        default="bidirectional",
        help="瓶颈 LSTM 的方向；none 表示移除 LSTM，仅保留自注意力",
    )
    parser.add_argument(
        "--checkpoint-name",
        default=None,
        help="最佳权重文件名；不指定时根据瓶颈类型自动命名",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="批大小；不指定时使用 config.py 中的 BATCH_SIZE",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader 子进程数；不指定时使用 config.py 中的 NUM_WORKERS",
    )
    parser.add_argument(
        "--attention",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否使用自注意力；纯 Wave-U-Net 使用 --lstm-direction none --no-attention",
    )
    arguments = parser.parse_args()
    train(
        arguments.lstm_direction,
        arguments.checkpoint_name,
        arguments.batch_size,
        arguments.num_workers,
        arguments.attention,
    )
