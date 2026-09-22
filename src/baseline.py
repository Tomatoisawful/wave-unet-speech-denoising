"""Edinburgh 测试集上的完整语音传统基线方法。
"""

import argparse
import csv
import glob
import os

import numpy as np
import torch

import config
import utils
from inference import postprocess
from evaluate import (
    compute_pesq_nb, compute_pesq_wb, compute_si_sdr, compute_snr,
    compute_ssnr, compute_stoi, snr_group,
)

METRICS = ("snr_in", "snr_out", "ssnr_in", "ssnr_out", "pesq_nb", "pesq_wb", "stoi", "si_sdr")
METHODS = ("noisy", "spectral_subtraction", "wiener")
DEFAULT_REPORT = os.path.join("logs", "full_utterances", "baseline_full_utterances.log")


@torch.no_grad()
def spectral_subtraction(noisy: torch.Tensor, noise_frames: int = 10) -> torch.Tensor:
    magnitude, phase = utils.compute_stft(noisy)
    power = magnitude.square()
    noise = power[:, :noise_frames].mean(dim=1, keepdim=True)
    enhanced = torch.clamp(power - noise, min=0.001 * power)
    return utils.reconstruct_waveform(torch.sqrt(enhanced), phase)


@torch.no_grad()
def wiener_filter(noisy: torch.Tensor, noise_frames: int = 10, eps: float = 1e-8) -> torch.Tensor:
    magnitude, phase = utils.compute_stft(noisy)
    power = magnitude.square()
    noise = power[:, :noise_frames].mean(dim=1, keepdim=True) + eps
    signal = torch.clamp(power - noise, min=0.0)
    gain = signal / (signal + noise)
    return utils.reconstruct_waveform(gain * magnitude, phase)


def _scores(clean: np.ndarray, noisy: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    length = min(len(clean), len(noisy), len(pred))
    clean, noisy, pred = clean[:length], noisy[:length], pred[:length]
    return {
        "snr_in": compute_snr(clean, noisy - clean),
        "snr_out": compute_snr(clean, pred - clean),
        "ssnr_in": compute_ssnr(clean, noisy - clean, config.SAMPLE_RATE),
        "ssnr_out": compute_ssnr(clean, pred - clean, config.SAMPLE_RATE),
        "pesq_nb": compute_pesq_nb(clean, pred, config.SAMPLE_RATE),
        "pesq_wb": compute_pesq_wb(clean, pred, config.SAMPLE_RATE),
        "stoi": compute_stoi(clean, pred, config.SAMPLE_RATE),
        "si_sdr": compute_si_sdr(clean, pred),
    }


def _match_length(waveform: torch.Tensor, length: int) -> torch.Tensor:
    """使每种基线输出都与原始语音长度对齐。"""
    if waveform.numel() >= length:
        return waveform[:length]
    return torch.nn.functional.pad(waveform, (0, length - waveform.numel()))


def _summary(rows: list[dict[str, float | str]], method: str) -> list[str]:
    lines = [f"\nMethod: {method}", f"{'Group':<7} {'N':>5} {'SNR-in':>8} {'SNR-out':>8} {'SSNR-in':>8} {'SSNR-out':>9} {'PESQ-NB':>8} {'PESQ-WB':>8} {'STOI':>7} {'SI-SDR':>8}"]
    for group in ("low", "mid", "high", "all"):
        subset = rows if group == "all" else [row for row in rows if row["group"] == group]
        if not subset:
            continue
        mean = lambda metric: float(np.nanmean([float(row[metric]) for row in subset]))
        lines.append(f"{group:<7} {len(subset):>5} {mean('snr_in'):>8.2f} {mean('snr_out'):>8.2f} {mean('ssnr_in'):>8.2f} {mean('ssnr_out'):>9.2f} {mean('pesq_nb'):>8.3f} {mean('pesq_wb'):>8.3f} {mean('stoi'):>7.3f} {mean('si_sdr'):>8.2f}")
    return lines


def main(report_path: str, max_files: int | None = None) -> None:
    noisy_paths = sorted(glob.glob(os.path.join(config.EDINBURGH_NOISY_TEST_DIR, "*.wav")))
    if max_files is not None:
        noisy_paths = noisy_paths[:max_files]
    if not noisy_paths:
        raise RuntimeError(f"No noisy WAV files in {config.EDINBURGH_NOISY_TEST_DIR}")
    rows_by_method = {method: [] for method in METHODS}

    for index, noisy_path in enumerate(noisy_paths, 1):
        stem = os.path.splitext(os.path.basename(noisy_path))[0]
        clean_path = os.path.join(config.EDINBURGH_CLEAN_TEST_DIR, f"{stem}.wav")
        if not os.path.exists(clean_path):
            raise RuntimeError(f"Missing clean WAV for {stem}")
        noisy, clean = utils.load_audio(noisy_path), utils.load_audio(clean_path)
        length = min(noisy.numel(), clean.numel())
        noisy, clean = noisy[:length], clean[:length]
        noisy_np, clean_np = noisy.numpy(), clean.numpy()
        group = snr_group(noisy_np, clean_np)
        predictions = {
            "noisy": noisy,
            "spectral_subtraction": postprocess(spectral_subtraction(noisy), noisy),
            "wiener": postprocess(wiener_filter(noisy), noisy),
        }
        for method, pred in predictions.items():
            pred = _match_length(pred, length)
            rows_by_method[method].append({"method": method, "file": stem, "group": group, **_scores(clean_np, noisy_np, pred.numpy())})
        if index % 50 == 0 or index == len(noisy_paths):
            print(f"[baseline-full] 已处理 {index}/{len(noisy_paths)}")

    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    csv_path = os.path.splitext(report_path)[0] + ".csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("method", "file", "group", *METRICS))
        writer.writeheader()
        for method in METHODS:
            writer.writerows(rows_by_method[method])

    lines = ["=" * 105, "完整语音基线评估（Edinburgh 测试 WAV）", "=" * 105]
    for method in METHODS:
        lines.extend(_summary(rows_by_method[method], method))
    lines.extend(("=" * 105, f"逐文件 CSV：{csv_path}", "谱减法和维纳滤波输出使用与 Wave-U-Net 相同的推理后处理。"))
    report = "\n".join(lines) + "\n"
    print(report, end="")
    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估完整语音的传统基线方法")
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--max-files", type=int, default=None, help="可选的冒烟测试文件数量上限")
    args = parser.parse_args()
    main(args.report, args.max_files)
