"""客观评估指标与完整波形去噪辅助函数。
"""

import math

import numpy as np
import torch
from pesq import pesq
from pystoi import stoi
from scipy.signal import resample_poly

import config
from dataset import segment_waveform
from wave_model import WaveUNetDenoiser


def _resample(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    """按整数多相比例重采样一维信号。"""
    if source_sr == target_sr:
        return audio.astype(np.float32, copy=False)
    divisor = math.gcd(source_sr, target_sr)
    return resample_poly(audio, target_sr // divisor, source_sr // divisor).astype(np.float32)


def compute_pesq_nb(ref: np.ndarray, deg: np.ndarray, sr: int = 16000) -> float:
    """在 PESQ-NB 要求的 8 kHz 采样率下计算窄带 PESQ。"""
    try:
        ref_8k = _resample(ref, sr, 8000)
        deg_8k = _resample(deg, sr, 8000)
        length = min(len(ref_8k), len(deg_8k))
        return float(pesq(8000, ref_8k[:length], deg_8k[:length], "nb"))
    except Exception:
        return float("nan")


def compute_pesq_wb(ref: np.ndarray, deg: np.ndarray, sr: int = 16000) -> float:
    """在 PESQ-WB 要求的 16 kHz 采样率下计算宽带 PESQ。"""
    try:
        ref_16k = _resample(ref, sr, 16000)
        deg_16k = _resample(deg, sr, 16000)
        length = min(len(ref_16k), len(deg_16k))
        return float(pesq(16000, ref_16k[:length], deg_16k[:length], "wb"))
    except Exception:
        return float("nan")


def compute_pesq(ref: np.ndarray, deg: np.ndarray, sr: int = 16000) -> float:
    """宽带 PESQ 的向后兼容别名。"""
    return compute_pesq_wb(ref, deg, sr)


def compute_stoi(ref: np.ndarray, deg: np.ndarray, sr: int = 16000) -> float:
    """STOI 可懂度指标，取值范围 [0, 1]，越高越好。"""
    try:
        return float(stoi(ref, deg, sr, extended=False))
    except Exception:
        return float("nan")


def compute_si_sdr(ref: np.ndarray, pred: np.ndarray, eps: float = 1e-8) -> float:
    """以 dB 表示的尺度不变信号失真比（SI-SDR）。"""
    ref = ref - ref.mean()
    pred = pred - pred.mean()
    scale = np.dot(pred, ref) / (np.dot(ref, ref) + eps)
    target = scale * ref
    error = pred - target
    return float(10 * np.log10((np.square(target).sum() + eps) /
                               (np.square(error).sum() + eps)))


def compute_snr(signal: np.ndarray, noise: np.ndarray, eps: float = 1e-8) -> float:
    """全局信噪比（dB）；其中 ``noise`` 为信号估计误差。"""
    return float(10 * np.log10((np.square(signal).sum() + eps) /
                               (np.square(noise).sum() + eps)))


def compute_ssnr(
    signal: np.ndarray,
    noise: np.ndarray,
    sr: int = 16000,
    frame_ms: float = 20.0,
    hop_ms: float = 10.0,
    min_db: float = -10.0,
    max_db: float = 35.0,
    eps: float = 1e-8,
) -> float:
    """对有效 20 ms 帧取平均的分段信噪比。

    帧级数值裁剪至常用的 [-10, 35] dB 范围。纯净语音能量近似为零的帧会
    被排除，避免填充和静音主导结果。
    """
    length = min(len(signal), len(noise))
    frame_len = max(1, int(sr * frame_ms / 1000.0))
    hop_len = max(1, int(sr * hop_ms / 1000.0))
    if length < frame_len:
        signal = np.pad(signal[:length], (0, frame_len - length))
        noise = np.pad(noise[:length], (0, frame_len - length))
        length = frame_len

    values = []
    for start in range(0, length - frame_len + 1, hop_len):
        clean_frame = signal[start:start + frame_len]
        noise_frame = noise[start:start + frame_len]
        clean_energy = float(np.square(clean_frame).sum())
        if clean_energy <= eps:
            continue
        noise_energy = float(np.square(noise_frame).sum())
        frame_snr = 10 * np.log10((clean_energy + eps) / (noise_energy + eps))
        values.append(np.clip(frame_snr, min_db, max_db))
    return float(np.mean(values)) if values else float("nan")


@torch.no_grad()
def denoise_waveform(
    model: WaveUNetDenoiser,
    noisy_wav: torch.Tensor,
    device: torch.device,
    batch_size: int = 8,
) -> torch.Tensor:
    """通过分段重叠相加，为任意长度的单声道波形去噪。"""
    original_len = noisy_wav.numel()
    segments = segment_waveform(noisy_wav.float().cpu())
    clean_segments = []

    for start in range(0, len(segments), batch_size):
        batch = torch.stack(segments[start:start + batch_size]).unsqueeze(1).to(device)
        predicted = model(batch).squeeze(1).cpu()
        clean_segments.extend(predicted.unbind(0))

    return _overlap_add(clean_segments, config.SEGMENT_SAMPLES, config.OVERLAP)[:original_len]


def _overlap_add(segments: list[torch.Tensor], seg_len: int, overlap: float) -> torch.Tensor:
    """对重叠的模型输出取平均，重建完整波形。"""
    if not segments:
        return torch.empty(0)
    step = int(seg_len * (1 - overlap))
    total = step * (len(segments) - 1) + seg_len
    output = torch.zeros(total)
    counts = torch.zeros(total)
    for index, segment in enumerate(segments):
        start = index * step
        end = start + segment.numel()
        output[start:end] += segment
        counts[start:end] += 1
    return output / counts.clamp_min(1)


def snr_group(noisy: np.ndarray, clean: np.ndarray) -> str:
    """根据输入 SNR 对样本分组，用于按难度分层汇报。"""
    value = compute_snr(clean, noisy - clean)
    if value < 5:
        return "low"
    if value < 15:
        return "mid"
    return "high"
