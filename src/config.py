"""
config.py
语音去噪项目的集中配置文件。
所有超参数和路径均在此定义；进行实验时通常只需修改本文件。
"""

import os

#----------------------------------------------------------------------------------
# 路径
#----------------------------------------------------------------------------------

# 数据集选择。当前实验使用 "edinburgh"。
DATASET = "edinburgh"

# 已下载数据的根目录
DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")

# 原始 WAV 文件路径
CLEAN_DIR  = os.path.join(DATA_ROOT, "raw/clean_fullband/mnt/dnsv5/clean")    # 纯净语音 WAV
NOISE_DIR  = os.path.join(DATA_ROOT, "raw/noise_fullband")                    # 背景噪声 WAV

# DNS 开发测试集
DEV_NOISY_DIR = os.path.join(DATA_ROOT, "processed", "noisy_testclips")
DEV_CLEAN_DIR = os.path.join(DATA_ROOT, "processed", "clean_testclips")

# Edinburgh Noisy Speech Database（Valentini-Botinhao）
# 解压后的预期目录结构：
# data/edinburgh/
#   clean_trainset_28spk_wav/
#   noisy_trainset_28spk_wav/
#   clean_testset_wav/
#   noisy_testset_wav/
EDINBURGH_ROOT = os.path.join(DATA_ROOT, "edinburgh")
EDINBURGH_CLEAN_TRAIN_DIR = os.path.join(EDINBURGH_ROOT, "clean_trainset_28spk_wav")
EDINBURGH_NOISY_TRAIN_DIR = os.path.join(EDINBURGH_ROOT, "noisy_trainset_28spk_wav")
EDINBURGH_CLEAN_TEST_DIR  = os.path.join(EDINBURGH_ROOT, "clean_testset_wav")
EDINBURGH_NOISY_TEST_DIR  = os.path.join(EDINBURGH_ROOT, "noisy_testset_wav")

# Wave-U-Net 的波形缓存独立存储
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed_wave")

# 训练得到的模型权重保存位置
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")

# 推理时去噪 WAV 的输出位置
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


#----------------------------------------------------------------------------------
# 音频设置
#----------------------------------------------------------------------------------

SAMPLE_RATE    = 16000   # 将所有输入音频重采样至 16 kHz
N_FFT          = 512     # FFT 窗长，决定频率分辨率
HOP_LENGTH     = 128     # 相邻 STFT 帧的跳步长度（N_FFT 的 25%）
WIN_LENGTH     = 512     # 分析窗长度（与 N_FFT 相同）

# STFT 后频率轴包含 N_FFT//2 + 1 = 257 个频点
N_FREQ_BINS = N_FFT // 2 + 1


#----------------------------------------------------------------------------------
# 分段设置（用于训练）
#----------------------------------------------------------------------------------

SEGMENT_DURATION = 2.0                                   # 每个训练片段的时长（秒）
SEGMENT_SAMPLES  = int(SAMPLE_RATE * SEGMENT_DURATION)   # 32000 个采样点
OVERLAP          = 0.5                                   # 50% 重叠，即每隔 1 秒产生一个新片段


#----------------------------------------------------------------------------------
# 数据集划分
#----------------------------------------------------------------------------------

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

# 按 speaker_id 划分，确保同一说话人不会同时出现在多个划分中

# 每个划分最多加载的片段数；设为 None 则使用完整数据集
MAX_TRAIN_FILES = None
MAX_VAL_FILES   = None


#----------------------------------------------------------------------------------
# 模型结构
#----------------------------------------------------------------------------------

ENCODER_CHANNELS = [32, 64, 128, 256, 512]    # 为旧频谱模型保留的兼容配置
WAVE_ENCODER_CHANNELS = [32, 64, 128, 256, 512]
LSTM_HIDDEN      = 256                        # 双向各 256 维，共 512 个瓶颈通道
LSTM_LAYERS      = 2                          # 堆叠的 BiLSTM 层数
ATTENTION_HEADS  = 8
ATTENTION_DROPOUT = 0.1


#----------------------------------------------------------------------------------
# 训练设置
#----------------------------------------------------------------------------------

BATCH_SIZE    = 8       # 波形模型占用显存较多
MAX_EPOCHS    = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4

WARMUP_EPOCHS   = 3
MIN_LR          = 1e-5  # 避免学习率衰减过低

# 早停
EARLY_STOP_PATIENCE = 15

# 数据增强
SNR_AUGMENT_DB = 3.0


#----------------------------------------------------------------------------------
# 损失函数权重
#----------------------------------------------------------------------------------

STFT_LOSS_WEIGHT = 0.1   # L = -SI_SDR + 0.1 × STFT_loss

# 多分辨率 STFT 损失：(n_fft, hop_length, win_length) 元组列表
STFT_RESOLUTIONS = [
    (256,  64,  256),
    (512,  128, 512),
    (1024, 256, 1024),
]


WANDB_PROJECT = None
WANDB_ENTITY  = None
