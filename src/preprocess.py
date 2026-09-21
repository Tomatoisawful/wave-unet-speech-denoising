"""
preprocess.py

一次性预处理脚本。
请在训练前运行一次。

功能：DNS 模式扫描纯净语音和噪声目录，合成带噪/纯净文件对；Edinburgh 模式匹配
数据库中的带噪/纯净 WAV，并创建说话人不重叠的训练/验证划分。两种模式均会将训练数据
切为重叠率 50% 的 2 秒片段，并把 (noisy_wave, clean_wave) 张量对保存到
data/processed_wave/segments/。

执行后，train.py 会直接读取 .pt 文件；训练阶段无需加载音频，因此每轮更快。

Run:
    python src/preprocess.py
    python src/preprocess.py --clean_dir path/to/clean --noise_dir path/to/noise
    python src/preprocess.py --limit 500   # process only 500 pairs (for quick tests)
"""

import argparse
import os
import glob
import random

import torch

import config
import utils
from dataset import segment_waveform


def mix_at_snr(
    clean: torch.Tensor,
    noise: torch.Tensor,
    snr_db: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    以指定的 SNR（dB）混合纯净语音和噪声。

    SNR = 10 * log10(P_speech / P_noise)
        即缩放噪声，使功率比符合目标 SNR。

    参数：
        clean:  一维 float32 张量（语音信号）
        noise:  一维 float32 张量（噪声信号，会裁剪或循环以匹配长度）
        snr_db: 目标信噪比，单位 dB

    Returns:
        noisy：纯净语音加缩放噪声，长度与 clean 相同
    """
    # 循环或裁剪噪声，使其与纯净语音长度一致
    if noise.shape[0] < clean.shape[0]:
        repeats = (clean.shape[0] // noise.shape[0]) + 1
        noise   = noise.repeat(repeats)
    noise = noise[:clean.shape[0]]

    # 随机偏移，避免每次均从噪声文件的同一位置开始
    offset = random.randint(0, max(0, noise.shape[0] - clean.shape[0]))
    noise  = noise[offset:offset + clean.shape[0]]

    # 计算每个信号的功率
    clean_power = (clean ** 2).mean().clamp(min=eps)
    noise_power = (noise ** 2).mean().clamp(min=eps)

    # 缩放噪声以达到目标 SNR
    target_noise_power = clean_power / (10 ** (snr_db / 10))
    noise_scale        = (target_noise_power / noise_power).sqrt()

    return clean + noise_scale * noise


def apply_impulse_response(
    waveform: torch.Tensor,
    ir: torch.Tensor,
) -> torch.Tensor:
    """
    将波形与脉冲响应卷积，以模拟房间声学。

    这会让合成训练数据更真实，使模型学习对真实房间录制的语音去噪，而非仅对无响室
    纯净语音叠加噪声的信号去噪。

    为加快计算，卷积在频域中完成（基于 FFT）；输出会裁剪至原始波形长度。

    参数：
        waveform: 一维 float32 张量（纯净语音）
        ir:       一维 float32 张量（脉冲响应）

    Returns:
        混响波形，长度与输入相同
    """
    n     = waveform.shape[0] + ir.shape[0] - 1
    # 取不小于长度的最小 2 的幂，以提升 FFT 效率
    nfft  = 1 << (n - 1).bit_length()

    W = torch.fft.rfft(waveform, n=nfft)
    H = torch.fft.rfft(ir,       n=nfft)
    Y = torch.fft.irfft(W * H,   n=nfft)

    # 裁剪回原始长度并归一化，避免削波
    Y = Y[:waveform.shape[0]]
    peak = Y.abs().max()
    if peak > 1.0:
        Y = Y / peak

    return Y


def collect_wavs(directory: str) -> list[str]:
    """递归查找目录下的全部 WAV 文件。"""
    pattern = os.path.join(directory, "**", "*.wav")
    return sorted(glob.glob(pattern, recursive=True))


def collect_pairs(noisy_dir: str, clean_dir: str) -> list[tuple[str, str, str]]:
    """按文件名匹配带噪/纯净 WAV 文件对。"""
    clean_by_name = {
        os.path.basename(path): path for path in collect_wavs(clean_dir)
    }
    pairs = []
    for noisy_path in collect_wavs(noisy_dir):
        name = os.path.basename(noisy_path)
        clean_path = clean_by_name.get(name)
        if clean_path is None:
            continue
        stem = os.path.splitext(name)[0]
        speaker_id = stem.split("_")[0]
        pairs.append((noisy_path, clean_path, speaker_id))
    return pairs


def split_pairs_by_speaker(
    pairs: list[tuple[str, str, str]],
    train_ratio: float = config.TRAIN_RATIO,
    seed: int = 42,
) -> tuple[list, list]:
    """按说话人划分配对文件，避免说话人泄漏到验证集。"""
    by_speaker = {}
    for pair in pairs:
        by_speaker.setdefault(pair[2], []).append(pair)
    speakers = sorted(by_speaker)
    random.seed(seed)
    random.shuffle(speakers)
    n_train = max(1, int(len(speakers) * train_ratio))
    train_speakers = set(speakers[:n_train])
    train = [p for p in pairs if p[2] in train_speakers]
    val = [p for p in pairs if p[2] not in train_speakers]
    return train, val


def preprocess_paired_files(
    pairs: list[tuple[str, str, str]],
    output_dir: str,
    limit: int = 0,
) -> None:
    """预处理已有的带噪/纯净配对文件，不再合成噪声。"""
    os.makedirs(output_dir, exist_ok=True)
    if limit:
        pairs = pairs[:limit]

    total_segments = 0
    skipped = 0
    print(f"[preprocess] Paired files: {len(pairs)}")
    print(f"[preprocess] Output dir : {output_dir}")

    for file_idx, (noisy_path, clean_path, _speaker_id) in enumerate(pairs):
        try:
            noisy_wav = utils.load_audio(noisy_path)
            clean_wav = utils.load_audio(clean_path)
            min_len = min(noisy_wav.shape[0], clean_wav.shape[0])
            noisy_wav = noisy_wav[:min_len]
            clean_wav = clean_wav[:min_len]
            noisy_segs = segment_waveform(noisy_wav)
            clean_segs = segment_waveform(clean_wav)
            n_segs = min(len(noisy_segs), len(clean_segs))

            for seg_idx in range(n_segs):
                noisy_seg = noisy_segs[seg_idx]
                clean_seg = clean_segs[seg_idx]
                noise = noisy_seg - clean_seg
                snr_db = float(10 * torch.log10(
                    (clean_seg.square().mean() + 1e-8) /
                    (noise.square().mean() + 1e-8)
                ))
                stem = os.path.splitext(os.path.basename(noisy_path))[0]
                filename = f"{stem}_seg{seg_idx:03d}.pt"
                torch.save({
                    "noisy_wave": noisy_seg.unsqueeze(0),
                    "clean_wave": clean_seg.unsqueeze(0),
                    "snr_db": snr_db,
                    "clean_path": clean_path,
                }, os.path.join(output_dir, filename))
                total_segments += 1
        except Exception as exc:
            print(f"  [skip] {noisy_path}: {exc}")
            skipped += 1

        if (file_idx + 1) % 100 == 0 or file_idx + 1 == len(pairs):
            print(f"  [{file_idx + 1}/{len(pairs)}] {total_segments} segments saved")

    print(f"[preprocess] Done: {total_segments} segments, {skipped} skipped")


def preprocess_edinburgh(
    clean_train_dir: str,
    noisy_train_dir: str,
    output_dir: str,
    limit: int = 0,
) -> None:
    """仅预处理 Edinburgh 配对录音中的训练与验证数据。

    完整测试语音直接从原始 WAV 文件评估，不会转换为 2 秒 ``.pt`` 测试片段。
    """
    train_pairs = collect_pairs(noisy_train_dir, clean_train_dir)
    if not train_pairs:
        raise RuntimeError(
            f"No Edinburgh training pairs found in {noisy_train_dir} / {clean_train_dir}"
        )

    train_pairs, val_pairs = split_pairs_by_speaker(train_pairs)
    print(f"[edinburgh] train={len(train_pairs)}, val={len(val_pairs)} paired files")
    segment_root = os.path.join(output_dir, "segments")
    preprocess_paired_files(train_pairs, os.path.join(segment_root, "train"), limit)
    preprocess_paired_files(val_pairs, os.path.join(segment_root, "val"), limit)


def preprocess(
    clean_dir:  str,
    noise_dir:  str,
    output_dir: str,
    ir_dir:     str,
    snr_min:    float = -5.0,
    snr_max:    float = 30.0,
    limit:      int   = 0,
    seed:       int   = 42,
) -> None:
    """
    合成带噪/纯净文件对、进行分段并保存为 .pt 文件。

    参数：
        clean_dir:  包含纯净语音 WAV 的目录
        noise_dir:  包含噪声 WAV 的目录
        output_dir: .pt 片段文件的保存位置
        snr_min:    混合时的最小 SNR（dB）
        snr_max:    混合时的最大 SNR（dB）
        limit:      若设置，则至多处理该数量的纯净文件（快速测试）
        seed:       用于复现的随机种子
    """
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    clean_files = collect_wavs(clean_dir)
    noise_files = collect_wavs(noise_dir)

    if not clean_files:
        raise RuntimeError(f"No WAV files found in clean_dir: {clean_dir}")
    if not noise_files:
        raise RuntimeError(f"No WAV files found in noise_dir: {noise_dir}")

    ir_files = collect_wavs(ir_dir) if ir_dir and os.path.isdir(ir_dir) else []

    if limit:
        clean_files = clean_files[:limit]

    print(f"[preprocess] Clean files : {len(clean_files)}")
    print(f"[preprocess] Noise files : {len(noise_files)}")
    print(f"[preprocess] IR files    : {len(ir_files)} ({'enabled' if ir_files else 'disabled'})")
    print(f"[preprocess] Output dir  : {output_dir}")
    print(f"[preprocess] SNR range   : [{snr_min}, {snr_max}] dB")

    total_segments = 0
    skipped        = 0

    for file_idx, clean_path in enumerate(clean_files):
        # 加载纯净语音
        try:
            clean_wav = utils.load_audio(clean_path)
        except Exception as e:
            print(f"  [skip] {clean_path}: {e}")
            skipped += 1
            continue

        # 随机选择噪声文件和 SNR
        noise_path = random.choice(noise_files)
        snr_db     = random.uniform(snr_min, snr_max)

        try:
            noise_wav = utils.load_audio(noise_path)
        except Exception as e:
            print(f"  [skip] noise {noise_path}: {e}")
            skipped += 1
            continue

        # 若有可用数据，则应用脉冲响应（房间声学模拟）
        if ir_files:
            ir_path = random.choice(ir_files)
            try:
                ir_wav    = utils.load_audio(ir_path)
                clean_wav = apply_impulse_response(clean_wav, ir_wav)
            except Exception:
                pass

        # 合成带噪波形
        noisy_wav = mix_at_snr(clean_wav, noise_wav, snr_db)

        # 对两个波形进行分段
        clean_segs = segment_waveform(clean_wav)
        noisy_segs = segment_waveform(noisy_wav)

        n_segs = min(len(clean_segs), len(noisy_segs))

        for seg_idx in range(n_segs):
            clean_seg = clean_segs[seg_idx]
            noisy_seg = noisy_segs[seg_idx]

            # 构造输出文件名，格式：fileIDX_segIDX_snrVALUE.pt
            stem     = os.path.splitext(os.path.basename(clean_path))[0]
            filename = f"{stem}_seg{seg_idx:03d}_snr{snr_db:.1f}.pt"
            out_path = os.path.join(output_dir, filename)

            # 保存为张量字典
            torch.save({
                "noisy_wave":  noisy_seg.unsqueeze(0),
                "clean_wave":  clean_seg.unsqueeze(0),
                "snr_db":      snr_db,       # float - useful for stratified eval
                "clean_path":  clean_path,   # str  - for debugging
            }, out_path)

            total_segments += 1

        # 输出进度
        if (file_idx + 1) % 100 == 0 or (file_idx + 1) == len(clean_files):
            print(f"  [{file_idx + 1}/{len(clean_files)}] {total_segments} segments saved")

    print(f"\n[preprocess] Done.")
    print(f"  Total segments : {total_segments}")
    print(f"  Skipped files  : {skipped}")
    print(f"  Output dir     : {output_dir}")


class PreprocessedDataset(torch.utils.data.Dataset):
    """
    比 DenoisingDataset 更快的替代方案：读取预先保存的 .pt 文件，
    而非实时加载和处理 WAV。

    运行一次 preprocess.py 后，在 train.py 中如下使用：
        from preprocess import PreprocessedDataset
        ds = PreprocessedDataset("data/processed/train")

    每个样本存储配对的带噪和纯净波形片段。
    """

    def __init__(self, pt_dir: str, augment: bool = False, max_files = None):
        files = sorted(glob.glob(os.path.join(pt_dir, "*.pt")))

        if not files:
            raise RuntimeError(f"No .pt files found in {pt_dir}. Run preprocess.py first.")

        if max_files is not None and max_files < len(files):
            import random
            random.seed(42)
            files = random.sample(files, max_files)

        self.files   = files
        self.augment = augment

        total = len(glob.glob(os.path.join(pt_dir, "*.pt")))
        if max_files is not None and max_files < total:
            print(f"[PreprocessedDataset] {len(self.files)}/{total} segments in {pt_dir} (limited)")
        else:
            print(f"[PreprocessedDataset] {len(self.files)} segments in {pt_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        data = torch.load(self.files[idx], map_location="cpu", weights_only=True)

        if "noisy_wave" not in data or "clean_wave" not in data:
            raise RuntimeError(
                f"Legacy spectrogram cache found: {self.files[idx]}. "
                "Run preprocess.py again to create Wave-U-Net waveform caches."
            )

        noisy = data["noisy_wave"].float()
        clean = data["clean_wave"].float()

        if self.augment:
            # 在保持纯净目标不变的前提下，改变原始噪声强度。
            shift_db = random.uniform(-config.SNR_AUGMENT_DB, config.SNR_AUGMENT_DB)
            noise_component = noisy - clean
            noisy = clean + noise_component * (10 ** (shift_db / 20))

        return {
            "noisy": noisy,
            "clean": clean,
            "snr_db": torch.tensor(float(data.get("snr_db", 0.0))),
        }


def split_preprocessed(
    pt_dir:      str,
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio:   float = config.VAL_RATIO,
    seed:        int   = 42,
) -> None:
    """
    预处理后调用本函数，将 .pt 文件移动到 train/val/test 子目录，
    以便 PreprocessedDataset 分别加载每个划分。

    该划分在文件级随机进行，因合成后不一定保留说话人信息，故不是按说话人分层。
    若需要严格的说话人分层划分，请改用 dataset.py 中的 build_dataloaders。

    参数：
        pt_dir: 包含全部 .pt 文件的目录（preprocess() 的输出）
    """
    files = sorted(glob.glob(os.path.join(pt_dir, "*.pt")))
    random.seed(seed)
    random.shuffle(files)

    n       = len(files)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    splits = {
        "train": files[:n_train],
        "val":   files[n_train:n_train + n_val],
        "test":  files[n_train + n_val:],
    }

    for split_name, split_files in splits.items():
        split_dir = os.path.join(pt_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        for src in split_files:
            dst = os.path.join(split_dir, os.path.basename(src))
            os.rename(src, dst)
        print(f"  {split_name}: {len(split_files)} segments -> {split_dir}")

    print("[split] Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="预处理语音去噪数据集")
    parser.add_argument("--dataset", choices=["edinburgh", "dns"],
                        default=config.DATASET,
                        help="要预处理的数据集格式")

    parser.add_argument("--clean_dir",  default=config.CLEAN_DIR,
                        help="包含纯净语音 WAV 的目录")
    parser.add_argument("--noise_dir",  default=config.NOISE_DIR,
                        help="包含噪声 WAV 的目录")
    parser.add_argument("--ir_dir",     default=None,
                        help="包含脉冲响应 WAV 的目录（可选）")
    parser.add_argument("--output_dir", default=config.PROCESSED_DIR,
                        help="保存 .pt 文件的位置")
    parser.add_argument("--snr_min",    type=float, default=-5.0,
                        help="混音时的最小 SNR（dB）")
    parser.add_argument("--snr_max",    type=float, default=30.0,
                        help="混音时的最大 SNR（dB）")
    parser.add_argument("--limit",      type=int,   default=None,
                        help="仅处理指定数量的纯净文件（用于快速测试）")
    parser.add_argument("--split",      action="store_true",
                        help="预处理后划分为 train/val/test 子目录")

    args = parser.parse_args()

    if args.dataset == "edinburgh":
        preprocess_edinburgh(
            clean_train_dir=config.EDINBURGH_CLEAN_TRAIN_DIR,
            noisy_train_dir=config.EDINBURGH_NOISY_TRAIN_DIR,
            output_dir=args.output_dir,
            limit=args.limit or 0,
        )
    else:
        preprocess(
            clean_dir=args.clean_dir,
            noise_dir=args.noise_dir,
            output_dir=args.output_dir,
            ir_dir=args.ir_dir,
            snr_min=args.snr_min,
            snr_max=args.snr_max,
            limit=args.limit,
        )
        if args.split:
            print("\n[preprocess] Splitting into train/val/test...")
            split_preprocessed(args.output_dir)
