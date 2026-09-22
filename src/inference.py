"""
inference.py

使用训练好的模型为单个 WAV 文件去噪。

Usage:
    python src/inference.py --input path/to/noisy.wav --output path/to/clean.wav
    python src/inference.py --input path/to/noisy.wav   # saves to outputs/ folder

脚本加载最优权重，执行完整推理流程，并将去噪 WAV 写入磁盘。
"""

import argparse
import os

import torch
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import librosa.display

import config
import utils
from wave_model import WaveUNetDenoiser
from train    import load_checkpoint
from evaluate import denoise_waveform


FULL_UTTERANCE_OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, "full_utterances")


def plot_spectrograms(
    noisy_wav:   torch.Tensor,
    clean_wav:   torch.Tensor,
    output_path: str,
    sr:          int = config.SAMPLE_RATE,
) -> None:
    """
    将带噪与去噪语音的并列频谱对比保存为 PNG。

    使用 Mel 标度的对数幅度频谱，更符合人耳感知特性。

    Args:
        noisy_wav:   原始带噪信号（一维张量）
        clean_wav:   去噪信号（一维张量）
        output_path: 输出 WAV 路径，PNG 会保存到同一位置
    """
    noisy_np = noisy_wav.numpy()
    clean_np = clean_wav[:noisy_wav.shape[0]].numpy()   # align lengths

    # Mel 频谱参数
    n_fft   = config.N_FFT
    hop     = config.HOP_LENGTH
    n_mels  = 128

    mel_noisy = librosa.feature.melspectrogram(
        y=noisy_np, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels
    )
    mel_clean = librosa.feature.melspectrogram(
        y=clean_np, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels
    )

    # 转换为 dB 标度
    mel_noisy_db = librosa.power_to_db(mel_noisy, ref=np.max)
    mel_clean_db = librosa.power_to_db(mel_clean, ref=np.max)

    # 两个子图使用相同色标，便于比较
    vmin = min(mel_noisy_db.min(), mel_clean_db.min())
    vmax = max(mel_noisy_db.max(), mel_clean_db.max())

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)

    for ax, mel_db, title in zip(
        axes,
        [mel_noisy_db, mel_clean_db],
        ["输入（带噪语音）", "输出（去噪语音）"],
    ):
        img = librosa.display.specshow(
            mel_db,
            sr=sr,
            hop_length=hop,
            x_axis="time",
            y_axis="mel",
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            cmap="magma",
        )
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("时间（秒）")
        ax.set_ylabel("频率（Mel）")

    fig.colorbar(img, ax=axes, format="%+2.0f dB", label="幅度（dB）")
    fig.suptitle(
        "去噪前后频谱对比",
        fontsize=14,
        fontweight="bold",
    )

    png_path = os.path.splitext(output_path)[0] + "_spectrogram.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[inference] Spectrogram saved  -> {png_path}")


def postprocess(wav: torch.Tensor, reference: torch.Tensor, sr: int = config.SAMPLE_RATE) -> torch.Tensor:
    """
    推理后处理：
      1. 使用 80–7500 Hz 带通滤波突出语音频段；
      2. 进行峰值归一化，使输出响度与输入匹配。
    """
    wav_np = wav.numpy().astype(np.float64)
    b, a   = butter(2, 80   / (sr / 2), btype='high')
    wav_np = filtfilt(b, a, wav_np)
    b, a   = butter(2, 7500 / (sr / 2), btype='low')
    wav_np = filtfilt(b, a, wav_np)
    result = torch.from_numpy(wav_np.astype(np.float32))

    input_peak  = reference.abs().max().clamp(min=1e-8)
    output_peak = result.abs().max().clamp(min=1e-8)
    return result * (input_peak / output_peak)


def run_inference(
    input_path: str,
    output_path: str,
    checkpoint_path: str,
    apply_postprocess: bool = True,
) -> None:
    """
    加载模型和音频，执行去噪，并保存结果及频谱对比图。

    Args:
        input_path:      带噪输入 WAV 的路径
        output_path:     去噪 WAV 的保存路径
        checkpoint_path: 模型权重路径
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[inference] Device: {device}")

    # 加载模型
    model = WaveUNetDenoiser().to(device)
    epoch, _ = load_checkpoint(model, checkpoint_path)
    model.eval()
    print(f"[inference] Loaded model from epoch {epoch}")

    # 加载音频
    noisy_wav = utils.load_audio(input_path)
    duration  = noisy_wav.shape[0] / config.SAMPLE_RATE
    print(f"[inference] Input: {input_path}  ({duration:.2f}s)")

    # 去噪
    with torch.no_grad():
        clean_wav = denoise_waveform(model, noisy_wav, device)

    if apply_postprocess:
        clean_wav = postprocess(clean_wav, noisy_wav)

    # 保存结果
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    utils.save_audio(clean_wav, output_path)
    print(f"[inference] Saved denoised audio  -> {output_path}")

    # 保存频谱对比图
    plot_spectrograms(noisy_wav, clean_wav, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为单个 WAV 或一批随机文件去噪")
    parser.add_argument("--input",  required=False, help="带噪输入 WAV 的路径")
    parser.add_argument(
        "--input-dir", required=False,
        help="完整语音批量输入目录；指定后按文件名顺序处理目录内 WAV",
    )
    parser.add_argument(
        "--output", required=False,
        help="去噪输出 WAV 的路径（默认：outputs/<输入文件名>）",
    )
    parser.add_argument(
        "--output-dir", default=FULL_UTTERANCE_OUTPUT_DIR,
        help="批量模式 WAV 与频谱图的目录（默认：outputs/full_utterances/）",
    )
    parser.add_argument(
        "--postprocess", action=argparse.BooleanOptionalAction, default=True,
        help="应用带通滤波和响度匹配（默认启用）。",
    )
    parser.add_argument(
        "--plots", action=argparse.BooleanOptionalAction, default=True,
        help="批量模式保存每条语音的频谱对比图（默认启用）。",
    )
    parser.add_argument(
        "--save-noisy", action=argparse.BooleanOptionalAction, default=True,
        help="批量模式将输入带噪 WAV 复制到输出目录（默认启用）。",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="批量模式跳过已经存在的去噪 WAV，用于中断后续跑。",
    )
    parser.add_argument(
        "--checkpoint", required=False,
        default=os.path.join(config.CHECKPOINT_DIR, "best_wave_model.pt"),
        help="模型权重路径",
    )
    parser.add_argument(
        "--batch", type=int, default=0,
        help="批量模式最多处理 N 条完整 WAV；设为 0 且指定 --input-dir 时处理全部",
    )
    args = parser.parse_args()

    if args.input_dir or args.batch > 0:
        # 批量模式：处理指定目录，或从默认测试集随机抽取 N 条带噪 WAV
        import glob, random
        wav_dir = args.input_dir or (
            config.EDINBURGH_NOISY_TEST_DIR
            if config.DATASET.lower() == "edinburgh"
            else config.DEV_NOISY_DIR
        )
        wav_files = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
        if not wav_files:
            raise RuntimeError(f"No WAV files found in {wav_dir}")

        if args.input_dir:
            chosen = wav_files[:args.batch] if args.batch > 0 else wav_files
        else:
            random.seed(None)   # 每次运行均使用新的随机种子
            chosen = random.sample(wav_files, min(args.batch, len(wav_files)))
        print(f"[inference] 批量模式：正在处理 {len(chosen)} 条完整带噪测试语音")

        # 仅加载一次模型，供全部文件复用
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = WaveUNetDenoiser().to(device)
        epoch, _ = load_checkpoint(model, args.checkpoint)
        model.eval()
        print(f"[inference] Loaded model from epoch {epoch}")

        os.makedirs(args.output_dir, exist_ok=True)

        for i, wav_path in enumerate(chosen, 1):
            stem      = os.path.splitext(os.path.basename(wav_path))[0]
            out_wav   = os.path.join(args.output_dir, f"{stem}_denoised.wav")
            noisy_out = os.path.join(args.output_dir, f"{stem}_noisy.wav")

            if args.resume and os.path.exists(out_wav):
                if i % 100 == 0 or i == len(chosen):
                    print(f"  [{i}/{len(chosen)}] 已跳过现有结果")
                continue

            noisy_wav = utils.load_audio(wav_path)

            with torch.no_grad():
                clean_wav = denoise_waveform(model, noisy_wav, device)
            if args.postprocess:
                clean_wav = postprocess(clean_wav, noisy_wav)

            utils.save_audio(clean_wav, out_wav)
            if args.save_noisy:
                utils.save_audio(noisy_wav, noisy_out)
            if args.plots:
                plot_spectrograms(noisy_wav, clean_wav, out_wav)
            print(f"  [{i}/{len(chosen)}] {stem}")

        print(f"[inference] Done. Results in {args.output_dir}")

    else:
        # 单文件模式
        if not args.input:
            parser.error("请提供 --input <wav> 或 --batch <N>")

        if not args.output:
            filename    = os.path.basename(args.input)
            args.output = os.path.join(args.output_dir, filename)

        run_inference(args.input, args.output, args.checkpoint, args.postprocess)
