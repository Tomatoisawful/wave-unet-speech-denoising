"""生成 VCTK + DEMAND 完整语音跨数据集测试集。

脚本会排除 Edinburgh 训练集和测试集中已经出现的说话人，仅选择 VCTK
的 mic1 录音，并按指定 SNR 与 DEMAND 噪声混合。输出包含完整的纯净、
带噪 WAV 和逐文件元数据，不生成 2 秒测试片段。
"""

import argparse
import csv
import glob
import os
import random

import soundfile as sf
import torch

import config
import utils


def speaker_id(path: str) -> str:
    """从 pXXX_YYY 格式的文件名中取得说话人编号。"""
    return os.path.basename(path).split("_")[0]


def edinburgh_speakers() -> set[str]:
    """收集 Edinburgh 训练和测试数据中出现过的全部说话人。"""
    speakers: set[str] = set()
    for directory in (
        config.EDINBURGH_CLEAN_TRAIN_DIR,
        config.EDINBURGH_CLEAN_TEST_DIR,
    ):
        for path in glob.glob(os.path.join(directory, "*.wav")):
            speakers.add(speaker_id(path))
    return speakers


def collect_vctk_clean(vctk_root: str) -> list[str]:
    """查找 VCTK 的 mic1 FLAC，并排除 Edinburgh 已使用的说话人。"""
    used_speakers = edinburgh_speakers()
    paths = glob.glob(
        os.path.join(vctk_root, "wav48_silence_trimmed", "p*", "*_mic1.flac")
    )
    return sorted(path for path in paths if speaker_id(path) not in used_speakers)


def collect_demand_noise(demand_root: str) -> list[str]:
    """递归收集已解压 DEMAND 场景中的 WAV 噪声录音。"""
    return sorted(glob.glob(os.path.join(demand_root, "**", "*.wav"), recursive=True))


def crop_or_repeat(noise: torch.Tensor, length: int, rng: random.Random) -> torch.Tensor:
    """随机裁剪噪声；不足目标长度时循环拼接。"""
    if noise.numel() < length:
        repeats = (length + noise.numel() - 1) // noise.numel()
        noise = noise.repeat(repeats)
    max_start = noise.numel() - length
    start = rng.randint(0, max_start) if max_start > 0 else 0
    return noise[start:start + length]


def load_noise_segment(path: str, length: int, rng: random.Random) -> torch.Tensor:
    """只读取所需长度的噪声，避免反复载入整段数分钟录音。"""
    info = sf.info(path)
    if info.samplerate != config.SAMPLE_RATE or info.frames < length:
        return crop_or_repeat(utils.load_audio(path), length, rng)

    max_start = info.frames - length
    start = rng.randint(0, max_start) if max_start > 0 else 0
    data, _ = sf.read(
        path,
        start=start,
        frames=length,
        dtype="float32",
        always_2d=True,
    )
    if data.shape[1] > 1:
        data = data.mean(axis=1, keepdims=True)
    return torch.from_numpy(data[:, 0].copy())


def mix_at_snr(clean: torch.Tensor, noise: torch.Tensor, snr_db: float) -> tuple[torch.Tensor, torch.Tensor]:
    """按目标 SNR 混合，并对纯净参考和混合语音做相同峰值缩放。"""
    eps = 1e-8
    clean_rms = clean.square().mean().clamp_min(eps).sqrt()
    noise_rms = noise.square().mean().clamp_min(eps).sqrt()
    target_noise_rms = clean_rms / (10 ** (snr_db / 20.0))
    scaled_noise = noise * (target_noise_rms / noise_rms)
    noisy = clean + scaled_noise

    peak = torch.maximum(clean.abs().max(), noisy.abs().max()).clamp_min(eps)
    scale = min(1.0, 0.99 / float(peak))
    return clean * scale, noisy * scale


def main(vctk_root: str, demand_root: str, output_root: str, count: int, seed: int) -> None:
    """生成指定数量的完整语音配对测试文件。"""
    clean_candidates = collect_vctk_clean(vctk_root)
    noise_paths = collect_demand_noise(demand_root)
    if len(clean_candidates) < count:
        raise RuntimeError(
            f"可用 VCTK mic1 音频只有 {len(clean_candidates)} 条，少于要求的 {count} 条。"
            "请确认 VCTK 已完整解压。"
        )
    if not noise_paths:
        raise RuntimeError(f"在 {demand_root} 中没有找到 DEMAND WAV 文件。")

    rng = random.Random(seed)
    rng.shuffle(clean_candidates)
    selected = clean_candidates[:count]
    snr_levels = (-5.0, 0.0, 5.0, 10.0, 15.0)

    clean_out = os.path.join(output_root, "clean")
    noisy_out = os.path.join(output_root, "noisy")
    os.makedirs(clean_out, exist_ok=True)
    os.makedirs(noisy_out, exist_ok=True)
    rows = []

    for index, clean_path in enumerate(selected, 1):
        clean = utils.load_audio(clean_path)
        noise_path = noise_paths[(index - 1) % len(noise_paths)]
        noise = load_noise_segment(noise_path, clean.numel(), rng)
        snr_db = snr_levels[(index - 1) % len(snr_levels)]
        clean_ref, noisy = mix_at_snr(clean, noise, snr_db)

        stem = os.path.basename(clean_path).replace("_mic1.flac", "")
        filename = f"{stem}_snr{snr_db:+.0f}.wav"
        utils.save_audio(clean_ref, os.path.join(clean_out, filename))
        utils.save_audio(noisy, os.path.join(noisy_out, filename))
        rows.append({
            "file": filename,
            "speaker": speaker_id(clean_path),
            "snr_db": snr_db,
            "noise_file": os.path.relpath(noise_path, demand_root),
            "vctk_file": os.path.relpath(clean_path, vctk_root),
        })
        if index % 50 == 0 or index == count:
            print(f"[VCTK-DEMAND] 已生成 {index}/{count}")

    metadata_path = os.path.join(output_root, "metadata.csv")
    with open(metadata_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"[VCTK-DEMAND] 完成：{output_root}")
    print(f"[VCTK-DEMAND] 纯净语音：{clean_out}")
    print(f"[VCTK-DEMAND] 带噪语音：{noisy_out}")
    print(f"[VCTK-DEMAND] 元数据：{metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成 VCTK + DEMAND 完整语音测试集")
    parser.add_argument(
        "--vctk-root",
        default=os.path.join(config.DATA_ROOT, "vctk", "extracted"),
        help="VCTK 解压目录",
    )
    parser.add_argument(
        "--demand-root",
        default=os.path.join(config.DATA_ROOT, "demand"),
        help="DEMAND 解压目录",
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join(config.DATA_ROOT, "vctk_demand_test"),
        help="配对测试集输出目录",
    )
    parser.add_argument("--count", type=int, default=824, help="完整测试语音数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()
    main(args.vctk_root, args.demand_root, args.output_root, args.count, args.seed)
