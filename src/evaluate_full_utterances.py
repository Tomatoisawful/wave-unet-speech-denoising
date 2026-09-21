"""使用 Edinburgh 纯净 WAV 对已保存的完整语音去噪结果进行评估。"""

import argparse
import csv
import glob
import os

import numpy as np

import config
import utils
from evaluate import (
    compute_pesq_nb,
    compute_pesq_wb,
    compute_si_sdr,
    compute_snr,
    compute_ssnr,
    compute_stoi,
    snr_group,
)


METRICS = (
    "snr_in", "snr_out", "ssnr_in", "ssnr_out",
    "pesq_nb", "pesq_wb", "stoi", "si_sdr",
)
DEFAULT_OUTPUTS_DIR = os.path.join(config.OUTPUT_DIR, "full_utterances")
DEFAULT_REPORT_PATH = os.path.join("logs", "full_utterances", "evaluation_full_utterances.log")


def _score(clean: np.ndarray, noisy: np.ndarray, pred: np.ndarray) -> dict[str, float]:
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


def main(outputs_dir: str, report_path: str) -> None:
    noisy_paths = sorted(glob.glob(os.path.join(config.EDINBURGH_NOISY_TEST_DIR, "*.wav")))
    if not noisy_paths:
        raise RuntimeError(f"No noisy WAV files in {config.EDINBURGH_NOISY_TEST_DIR}")

    grouped = {name: {metric: [] for metric in METRICS} for name in ("low", "mid", "high", "all")}
    rows: list[dict[str, float | str]] = []

    for index, noisy_path in enumerate(noisy_paths, 1):
        stem = os.path.splitext(os.path.basename(noisy_path))[0]
        clean_path = os.path.join(config.EDINBURGH_CLEAN_TEST_DIR, f"{stem}.wav")
        pred_path = os.path.join(outputs_dir, f"{stem}_denoised.wav")
        if not os.path.exists(clean_path) or not os.path.exists(pred_path):
            raise RuntimeError(f"Missing clean or denoised pair for {stem}")

        noisy = utils.load_audio(noisy_path).numpy()
        clean = utils.load_audio(clean_path).numpy()
        pred = utils.load_audio(pred_path).numpy()
        scores = _score(clean, noisy, pred)
        group = snr_group(noisy[:min(len(noisy), len(clean))], clean[:min(len(noisy), len(clean))])
        row = {"file": stem, "group": group, **scores}
        rows.append(row)
        for name in (group, "all"):
            for metric, value in scores.items():
                grouped[name][metric].append(value)
        if index % 50 == 0 or index == len(noisy_paths):
            print(f"[full-eval] 已处理 {index}/{len(noisy_paths)}")

    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    csv_path = os.path.splitext(report_path)[0] + ".csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("file", "group", *METRICS))
        writer.writeheader()
        writer.writerows(rows)

    lines = ["=" * 105, "已保存完整语音去噪结果的评估", "=" * 105]
    lines.append(f"{'Group':<7} {'N':>5} {'SNR-in':>8} {'SNR-out':>8} {'SSNR-in':>8} {'SSNR-out':>9} {'PESQ-NB':>8} {'PESQ-WB':>8} {'STOI':>7} {'SI-SDR':>8}")
    lines.append("=" * 105)
    for name in ("low", "mid", "high", "all"):
        values = grouped[name]
        n = len(values["snr_in"])
        if not n:
            continue
        mean = lambda metric: float(np.nanmean(values[metric]))
        lines.append(
            f"{name:<7} {n:>5} {mean('snr_in'):>8.2f} {mean('snr_out'):>8.2f} "
            f"{mean('ssnr_in'):>8.2f} {mean('ssnr_out'):>9.2f} "
            f"{mean('pesq_nb'):>8.3f} {mean('pesq_wb'):>8.3f} "
            f"{mean('stoi'):>7.3f} {mean('si_sdr'):>8.2f}"
        )
    lines.append("=" * 105)
    lines.extend((f"逐文件 CSV：{csv_path}", "评估使用已保存的推理输出，其中包含推理后处理。"))
    report = "\n".join(lines) + "\n"
    print(report, end="")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估 824 条完整语音的去噪结果")
    parser.add_argument("--outputs-dir", default=DEFAULT_OUTPUTS_DIR)
    parser.add_argument("--report", default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()
    main(args.outputs_dir, args.report)
