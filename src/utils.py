"""
utils.py

全项目共用的辅助函数：
  - WAV 文件读写
  - STFT / iSTFT 封装
  - 掩码应用
  - 目录创建
"""

import os
import math
import torch
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

import config


def load_audio(path: str) -> torch.Tensor:
    """
    读取 WAV 文件并返回一维 float32 张量（单声道、16 kHz）。

    多声道文件会混合为单声道；若采样率不同于 config.SAMPLE_RATE 则重采样。

    Returns:
        waveform: 形状为 (num_samples,)
    """
    data, sr = sf.read(path, dtype="float32", always_2d=True)  # (num_samples, num_channels)

    # 如有需要，混合为单声道
    if data.shape[1] > 1:
        data = data.mean(axis=1, keepdims=True)

    data = data[:, 0]  # (num_samples,)

    # 如有需要，进行重采样
    if sr != config.SAMPLE_RATE:
        gcd = math.gcd(sr, config.SAMPLE_RATE)
        up, down = config.SAMPLE_RATE // gcd, sr // gcd
        data = resample_poly(data, up, down).astype(np.float32)

    return torch.from_numpy(data)   # (num_samples,)


def save_audio(waveform: torch.Tensor, path: str) -> None:
    """
    将一维 float32 张量保存为 16 kHz 单声道 WAV 文件。

    Args:
        waveform: 形状为 (num_samples,)
        path:     目标文件路径（父目录必须存在）
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, waveform.numpy(), config.SAMPLE_RATE)


def compute_stft(waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    计算波形的短时傅里叶变换（STFT）。

    分别返回幅度和相位，使模型仅处理幅度，而相位用于后续重建。

    Args:
        waveform: shape (num_samples,)

    Returns:
        magnitude: shape (n_freq_bins, n_frames)  - non-negative real values
        phase:     shape (n_freq_bins, n_frames)  - values in [-pi, pi]
    """
    window = torch.hann_window(config.WIN_LENGTH, device=waveform.device)

    # stft 返回形状为 (n_freq_bins, n_frames) 的复数张量
    stft_complex = torch.stft(
        waveform,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        win_length=config.WIN_LENGTH,
        window=window,
        return_complex=True,
    )

    magnitude = stft_complex.abs()           # |z|
    phase     = torch.angle(stft_complex)    # angle of z in radians

    return magnitude, phase


def reconstruct_waveform(magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """
    通过逆 STFT（iSTFT）由幅度和相位重建波形。

    使用正向 STFT 保存的原始相位，避免相位失真伪影。

    Args:
        magnitude: shape (n_freq_bins, n_frames)
        phase:     shape (n_freq_bins, n_frames)

    Returns:
        waveform: shape (num_samples,)
    """
    # 重建复频谱：z = |z| × e^(i×angle(z))
    stft_complex = magnitude * torch.exp(1j * phase)

    window = torch.hann_window(config.WIN_LENGTH, device=magnitude.device)

    waveform = torch.istft(
        stft_complex,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        win_length=config.WIN_LENGTH,
        window=window,
    )

    return waveform


def apply_mask(noisy_magnitude: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    将模型预测的软掩码应用于带噪幅度频谱。

    The mask M ∈ [0, 1] acts as a per-bin gain:
        clean_estimate = M * noisy_magnitude

    接近 1 的值表示保留（可能是语音），接近 0 的值表示抑制（可能是噪声）。

    Args:
        noisy_magnitude: shape (batch, 1, n_freq_bins, n_frames)
        mask:            shape (batch, 1, n_freq_bins, n_frames), values in [0, 1]

    Returns:
        clean_magnitude: shape (batch, 1, n_freq_bins, n_frames)
    """
    return mask * noisy_magnitude


def log_compress(magnitude: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    应用 log(1 + magnitude) 压缩以减小动态范围。

    否则高响度成分会主导输入，安静的语音细节几乎无法被网络感知。

    Args:
        magnitude: any shape, non-negative
        eps:       small constant to avoid log(0)

    Returns:
        compressed magnitude, same shape
    """
    return torch.log1p(magnitude + eps)


def normalise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    沿全部维度对张量进行 Z-score 标准化。

    减去均值并除以标准差，可使输入近似零均值、单位方差，从而帮助基于梯度的优化更快收敛。

    Returns:
        x_norm: normalised tensor
        mean:   scalar mean (saved so you can undo normalisation later)
        std:    scalar std
    """
    mean = x.mean()
    std  = x.std().clamp(min=1e-8)   # 避免静音片段发生除零
    return (x - mean) / std, mean, std


def normalise_with_stats(
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """
    使用外部提供的均值和标准差对张量标准化。

    Args:
        x:    tensor to normalise (any shape)
        mean: scalar mean from the reference signal (noisy)
        std:  scalar std from the reference signal (noisy)

    Returns:
        x_norm: normalised tensor, same shape as x
    """
    std_clamped = std.clamp(min=1e-8)
    return (x - mean) / std_clamped


def denormalise(x_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """撤销 Z-score 标准化。"""
    return x_norm * std + mean


def ensure_dirs() -> None:
    """创建尚不存在的全部项目输出目录。"""
    for d in [config.PROCESSED_DIR, config.CHECKPOINT_DIR, config.OUTPUT_DIR]:
        os.makedirs(d, exist_ok=True)
