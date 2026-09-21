"""
dataset.py

为训练循环提供 (noisy_spectrogram, clean_spectrogram) 配对数据的 PyTorch 数据集。

每对音频的处理流程：
  1. 加载带噪与纯净 WAV；
  2. 分为 2 秒片段，重叠率为 50%；
  3. 执行 STFT 并取得幅度；
  4. 对幅度做对数压缩；
  5. Z-score 标准化（带噪语音的均值/标准差同时用于两个信号）；
  6. 返回形状为 (1, n_freq_bins, n_frames) 的张量对，供旧频谱模型按单通道图像处理。

该数据集还支持：短片段零填充、跳过实际内容不足 25% 的尾部片段、
训练时随机 ±3 dB 的 SNR 增强，以及按说话人划分训练/验证/测试集。
"""

import os
import glob
import random
from collections import defaultdict

import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader

import config
import utils

MIN_CONTENT_RATIO = 0.25  # 丢弃实际音频内容不足 25% 的尾部片段


def segment_waveform(waveform: torch.Tensor) -> list[torch.Tensor]:
    """
    将一维波形分为长度为 SEGMENT_SAMPLES 的重叠片段。

    重叠率为 50%，步长为 SEGMENT_SAMPLES // 2。最后一个片段若不足长度则补零，
    但必须至少包含 MIN_CONTENT_RATIO 的真实采样点；近乎为空的尾部片段会被丢弃。
    短于 SEGMENT_SAMPLES 的波形会补零后作为单个片段返回。

    Returns:
        张量列表，每个张量形状为 (SEGMENT_SAMPLES,)。
    """
    seg_len = config.SEGMENT_SAMPLES
    step    = int(seg_len * (1 - config.OVERLAP))   # 50% overlap -> step = seg_len // 2
    total   = waveform.shape[0]

    if total < seg_len:
        # 填充整个波形，并将其作为一个片段返回
        pad = torch.zeros(seg_len - total)
        return [torch.cat([waveform, pad])]

    segments = []
    start = 0
    while start < total:
        end        = start + seg_len
        real_len   = min(end, total) - start   # 本片段中的真实音频采样点数

        if real_len < seg_len * MIN_CONTENT_RATIO:
            break

        chunk = waveform[start:end]
        if chunk.shape[0] < seg_len:
            pad   = torch.zeros(seg_len - chunk.shape[0])
            chunk = torch.cat([chunk, pad])

        segments.append(chunk)
        start += step

    return segments


def _count_segments_from_info(path: str) -> int:
    """
    仅根据时长元数据估计文件会产生的片段数，不解码音频。

    使用只读取文件头的 soundfile.info()；无论文件时长如何，I/O 均为 O(1)。
    """
    try:
        info     = sf.info(path)
        # 如有需要，重采样帧数（逻辑与 utils.load_audio 相同）
        n_frames = info.frames
        if info.samplerate != config.SAMPLE_RATE:
            import math
            n_frames = int(n_frames * config.SAMPLE_RATE / info.samplerate)

        seg_len = config.SEGMENT_SAMPLES
        step    = int(seg_len * (1 - config.OVERLAP))

        if n_frames < seg_len:
            return 1

        count = 0
        start = 0
        while start < n_frames:
            real_len = min(start + seg_len, n_frames) - start
            if real_len < seg_len * MIN_CONTENT_RATIO:
                break
            count += 1
            start += step

        return max(count, 1)   # 至少返回一个片段

    except Exception:
        return 1


def waveform_to_input(
    waveform: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    将波形片段转换为可供旧频谱 U-Net 使用的频谱张量。

    Returns:
        spec_norm:  (1, n_freq_bins, n_frames)，经对数压缩和标准化
        phase:      (n_freq_bins, n_frames)，用于 iSTFT 的原始相位
        mean, std:  重建时撤销标准化所需的标量
    """
    magnitude, phase = utils.compute_stft(waveform)
    compressed       = utils.log_compress(magnitude)
    spec_norm, mean, std = utils.normalise(compressed)
    # 添加通道维度，使其形如单通道图像：(1, F, T)
    return spec_norm.unsqueeze(0), phase, mean, std


def normalise_with_stats(
    waveform: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    将波形转换为对数幅度频谱，并使用外部提供的均值和标准差（来自配对带噪信号）进行标准化。

    Returns:
        spec_norm:  (1, n_freq_bins, n_frames)
        phase:      (n_freq_bins, n_frames)
    """
    magnitude, phase = utils.compute_stft(waveform)
    compressed       = utils.log_compress(magnitude)
    spec_norm        = utils.normalise_with_stats(compressed, mean, std)
    return spec_norm.unsqueeze(0), phase



class DenoisingDataset(Dataset):
    """
    DNS-Challenge 数据集的 (noisy_spectrogram, clean_spectrogram) 片段对。

    每个样本是一个字典：
        {
          "noisy":       (1, n_freq_bins, n_frames)  float32 tensor
          "clean":       (1, n_freq_bins, n_frames)  float32 tensor - same normalisation as noisy
          "noisy_phase": (n_freq_bins, n_frames)     for inference reconstruction
          "clean_phase": (n_freq_bins, n_frames)
          "mean":        scalar float  (from noisy, shared by clean)
          "std":         scalar float  (from noisy, shared by clean)
          "speaker_id":  str
        }

    参数：
        pairs:       (noisy_path, clean_path, speaker_id) 元组列表
        augment:     为 True 时应用随机 SNR 偏移（仅训练使用）
    """

    def __init__(self, pairs: list[tuple[str, str, str]], augment: bool = False):
        self.pairs   = pairs
        self.augment = augment
        self._items = self._build_index()


    def _build_index(self) -> list[tuple[tuple, int]]:
        """
        遍历所有文件对，构建扁平的 (pair, seg_idx) 列表，使 __getitem__ 可 O(1) 访问。
        """
        items = []
        for pair in self.pairs:
            noisy_path, clean_path, speaker_id = pair
            try:
                n_segs = _count_segments_from_info(noisy_path)
                for seg_idx in range(n_segs):
                    items.append((pair, seg_idx))
            except Exception as e:
                print(f"[dataset] Skipping {noisy_path}: {e}")
        return items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        (noisy_path, clean_path, speaker_id), seg_idx = self._items[idx]

        # 加载完整波形
        noisy_wav = utils.load_audio(noisy_path)
        clean_wav = utils.load_audio(clean_path)

        # 裁剪至共同长度（DNS 开发/测试集的纯净文件与带噪文件长度可能略有差异）
        min_len   = min(noisy_wav.shape[0], clean_wav.shape[0])
        noisy_wav = noisy_wav[:min_len]
        clean_wav = clean_wav[:min_len]

        # 应用 SNR 增强：将噪声分量缩放 ±SNR_AUGMENT_DB
        if self.augment:
            shift_db    = random.uniform(-config.SNR_AUGMENT_DB, config.SNR_AUGMENT_DB)
            noise       = noisy_wav - clean_wav
            noise_scale = 10 ** (shift_db / 20)
            noisy_wav   = clean_wav + noise * noise_scale

        # 提取所需片段
        noisy_segs = segment_waveform(noisy_wav)
        clean_segs = segment_waveform(clean_wav)

        # 防止索引越界
        seg_idx = min(seg_idx, len(noisy_segs) - 1, len(clean_segs) - 1)

        noisy_seg = noisy_segs[seg_idx]
        clean_seg = clean_segs[seg_idx]

        # 先计算带噪频谱的均值/标准差，再以相同统计量标准化纯净频谱
        noisy_spec, noisy_phase, mean, std = waveform_to_input(noisy_seg)
        clean_spec, clean_phase            = normalise_with_stats(clean_seg, mean, std)

        return {
            "noisy":       noisy_spec,
            "clean":       clean_spec,
            "noisy_phase": noisy_phase,
            "clean_phase": clean_phase,
            "mean":        mean,
            "std":         std,
            "speaker_id":  speaker_id,
        }


def collect_pairs(noisy_dir: str, clean_dir: str) -> list[tuple[str, str, str]]:
    """
    按文件名主干，将 noisy_dir 中每个带噪 WAV 与 clean_dir 中对应纯净 WAV 匹配。

    DNS 文件名遵循以下模式：
        noisy_dir/fileid_<id>_snr<X>_tl<Y>_fileid_<Z>.wav
        clean_dir/fileid_<id>.wav

    参数：
        noisy_dir: 带噪 WAV 文件目录
        clean_dir: 对应纯净 WAV 文件目录

    Returns:
        (noisy_path, clean_path, speaker_id) 元组列表
    """
    noisy_paths = sorted(glob.glob(os.path.join(noisy_dir, "*.wav")))
    pairs = []

    for noisy_path in noisy_paths:
        stem  = os.path.splitext(os.path.basename(noisy_path))[0]
        # DNS 开发测试集的带噪和纯净文件使用相同文件名
        clean_path = os.path.join(clean_dir, os.path.basename(noisy_path))

        if not os.path.exists(clean_path):
            fileid = stem.split("_snr")[0]
            clean_path = os.path.join(clean_dir, fileid + ".wav")

        if not os.path.exists(clean_path):
            continue

        speaker_id = stem.split("_")[0]
        pairs.append((noisy_path, clean_path, speaker_id))

    return pairs


def split_by_speaker(
    pairs: list[tuple[str, str, str]],
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio:   float = config.VAL_RATIO,
    seed:        int   = 42,
) -> tuple[list, list, list]:
    """
    按 speaker_id 分组并打乱说话人，再将完整说话人分到训练/验证/测试集，
    确保同一说话人不会出现在两个划分中。

    Returns:
        train_pairs、val_pairs、test_pairs
    """
    by_speaker: dict[str, list] = defaultdict(list)
    for pair in pairs:
        by_speaker[pair[2]].append(pair)

    speakers = list(by_speaker.keys())
    random.seed(seed)
    random.shuffle(speakers)

    n          = len(speakers)
    n_train    = int(n * train_ratio)
    n_val      = int(n * val_ratio)

    train_spks = speakers[:n_train]
    val_spks   = speakers[n_train:n_train + n_val]
    test_spks  = speakers[n_train + n_val:]

    train_pairs = [p for s in train_spks for p in by_speaker[s]]
    val_pairs   = [p for s in val_spks   for p in by_speaker[s]]
    test_pairs  = [p for s in test_spks  for p in by_speaker[s]]

    print(f"[split] {len(train_spks)} train / {len(val_spks)} val / {len(test_spks)} test speakers")
    print(f"[split] {len(train_pairs)} / {len(val_pairs)} / {len(test_pairs)} file pairs")

    return train_pairs, val_pairs, test_pairs


def build_dataloaders(
    noisy_dir:   str = config.DEV_NOISY_DIR,
    clean_dir:   str = config.DEV_CLEAN_DIR,
    batch_size:  int = config.BATCH_SIZE,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    完整流程：扫描目录 → 按说话人划分 → 创建 DataLoader。

    Returns:
        train_loader、val_loader、test_loader
    """
    pairs = collect_pairs(noisy_dir, clean_dir)
    if not pairs:
        raise RuntimeError(f"No matching pairs found in {noisy_dir} / {clean_dir}")

    train_pairs, val_pairs, test_pairs = split_by_speaker(pairs)

    train_ds = DenoisingDataset(train_pairs, augment=True)
    val_ds   = DenoisingDataset(val_pairs,   augment=False)
    test_ds  = DenoisingDataset(test_pairs,  augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
