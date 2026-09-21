# Wave-U-Net 语音去噪实验

本项目实现端到端波形语音去噪模型：**Wave-U-Net + BiLSTM + 多头自注意力**。模型以带噪语音波形为输入，直接输出去噪波形；训练使用 Edinburgh Noisy Speech Database（VoiceBank-DEMAND），最终测试以 **824 条完整原始测试语音** 为单位进行，而非对测试语音做片段级指标统计。

## 1. 算法结构

```text
带噪单声道波形（16 kHz）
        │
        ▼
5 层 Wave-U-Net 编码器：一维卷积 + 下采样
通道数：1 → 32 → 64 → 128 → 256 → 512
        │
        ▼
瓶颈层：2 层 BiLSTM（双向各 256 维）
        │
        ▼
8 头多头自注意力 + 前馈网络
        │
        ▼
5 层 Wave-U-Net 解码器：线性插值上采样 + 跳跃连接
        │
        ▼
一维卷积预测残差波形
        │
        ▼
去噪结果 = 带噪输入 + 预测残差
```

Wave-U-Net 的跳跃连接保留不同时间尺度的局部细节；BiLSTM 建模双向长程时序关系；自注意力进一步学习语音帧之间的全局依赖。模型参数量约为 **17.9 M**。

### 损失函数

训练目标为：

```text
L = -SI-SDR + 0.1 × 多分辨率 STFT 损失
```

- `SI-SDR`：约束整体时域波形保真度；训练时最小化其相反数。
- 多分辨率 STFT 损失：在 `(256, 64, 256)`、`(512, 128, 512)`、`(1024, 256, 1024)` 三种时频分辨率下约束频谱差异。

## 2. 数据集与目录

训练和正式评估使用 Edinburgh Noisy Speech Database。解压后的目录应为：

```text
data/
└── edinburgh/
    ├── clean_trainset_28spk_wav/
    ├── noisy_trainset_28spk_wav/
    ├── clean_testset_wav/
    └── noisy_testset_wav/
```

项目按说话人将 11,572 条训练配对文件划分为训练集和验证集，避免同一说话人同时出现于两者。训练阶段会将训练/验证数据转换为 2 秒、50% 重叠的波形缓存；**这仅用于训练，不用于正式测试指标**。

正式测试直接使用 `clean_testset_wav` 和 `noisy_testset_wav` 中的 824 条完整语音。

## 3. 环境安装（Windows + Miniconda）

本项目推荐使用现有的 `mamba` Conda 环境。以下命令每行单独执行：

```powershell
conda activate mamba
cd "D:\人工智能算法综合课程设计\audio-denoising-main"

# 先安装与 CUDA 版本匹配的 PyTorch；示例为 CUDA 12.4
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

检查 GPU 是否可用：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

若 PowerShell 中的 `python` 未指向 mamba 环境，可固定使用：

```powershell
$mambaPy = "D:\miniconda3\envs\mamba\python.exe"
& $mambaPy -c "import torch; print(torch.cuda.is_available())"
```

## 4. 训练流程

### 4.1 预处理训练与验证数据

```powershell
$mambaPy = "D:\miniconda3\envs\mamba\python.exe"
& $mambaPy .\src\preprocess.py --dataset edinburgh
```

缓存保存到：

```text
data/processed_wave/segments/train/
data/processed_wave/segments/val/
```

不会生成测试集的 2 秒缓存。

### 4.2 配置训练参数

在 `src/config.py` 中主要修改：

```python
BATCH_SIZE = 8          # 根据显存逐步增大；显存不足则减小
MAX_EPOCHS = 100
MAX_TRAIN_FILES = None  # None 表示使用全部训练片段
MAX_VAL_FILES = None    # None 表示使用全部验证片段
```

### 4.3 开始训练

前台训练：

```powershell
& $mambaPy -u .\src\train.py
```

训练完成后最佳权重保存为：

```text
checkpoints/best_wave_model.pt
```

训练以验证集 SI-SDR 作为最佳权重选择依据；若连续 15 轮未提升则早停。

## 5. 完整语音推理与正式评估

### 5.1 对 824 条完整测试语音生成去噪结果

```powershell
& $mambaPy -u .\src\inference.py --batch 824 --checkpoint .\checkpoints\best_wave_model.pt
```

输出目录：

```text
outputs/full_utterances/
├── *_denoised.wav
├── *_noisy.wav
└── *_denoised_spectrogram.png
```

任意长度语音在推理内部会采用 2 秒窗口和重叠相加重建，以控制显存占用；最终保存和评估的对象始终是完整 WAV。默认后处理已启用，包括 80–7500 Hz 带通滤波和与输入匹配的峰值响度归一化。

### 5.2 计算客观指标

```powershell
& $mambaPy -u .\src\evaluate_full_utterances.py
```

结果保存为：

```text
logs/full_utterances/evaluation_full_utterances.log
logs/full_utterances/evaluation_full_utterances.csv
```

评估指标：

- SNR：全局信噪比。
- SSNR：分段信噪比，对有效 20 ms 语音帧取平均。
- PESQ-NB：8 kHz 窄带感知语音质量。
- PESQ-WB：16 kHz 宽带感知语音质量。
- STOI：短时客观可懂度，范围为 0–1，越高越好。
- SI-SDR：尺度不变信号失真比，单位 dB，越高越好。

## 6. 传统基线对比

项目提供三种完整语音基线：原始带噪语音、谱减法和维纳滤波。谱减法与维纳滤波使用与模型相同的后处理，确保系统级比较公平。

```powershell
& $mambaPy -u .\src\baseline.py --report .\logs\full_utterances\baselines_with_postprocess.log
```

对应 CSV 会写入同一目录。

## 7. 当前完整测试结果

已训练的 `best_wave_model.pt` 在 Edinburgh 824 条完整测试语音上的平均指标如下：

| 方法 | SNR-out | SSNR-out | PESQ-NB | PESQ-WB | STOI | SI-SDR |
|---|---:|---:|---:|---:|---:|---:|
| 原始带噪语音 | 8.45 | 1.52 | 2.945 | 1.967 | 0.921 | 8.45 |
| 谱减法 | 15.49 | 6.48 | 3.105 | 2.308 | 0.921 | 16.00 |
| 维纳滤波 | 15.60 | 6.68 | 3.131 | 2.328 | 0.920 | 16.11 |
| **Wave-U-Net + BiLSTM + Attention** | **16.94** | **8.49** | **3.381** | **2.542** | **0.937** | **17.70** |

完整结果图位于：

```text
result/38ba233466f457f33deeb32cafd799f3.png
result/a2372aacf60e0628915c69c3b507433b.png
result/full_utterance_example/
```

## 8. 单条外部音频去噪

```powershell
& $mambaPy -u .\src\inference.py `
  --input .\example_noisy.wav `
  --output .\outputs\example_denoised.wav `
  --checkpoint .\checkpoints\best_wave_model.pt
```

命令会保存去噪 WAV 和对应频谱对比图。单条外部音频没有纯净参考时，只能试听和观察频谱，无法计算 PESQ、STOI、SNR 等有参考指标。

## 9. VCTK 跨数据集测试

`data/vctk/` 中的 VCTK 是**纯净语音**数据集，本身不包含带噪配对语音，不能直接进行去噪指标评估。若需跨数据集测试：

1. 选取 VCTK 纯净 WAV；
2. 选取独立噪声集，例如 DEMAND 或 NoiseX-92；
3. 按给定 SNR 混合生成带噪 VCTK；
4. 用当前模型对完整带噪 WAV 推理；
5. 用原始 VCTK WAV 作为参考，计算同一套指标。

该流程不需要重新训练，反映模型的跨数据集泛化能力；若目标是提升 VCTK 上的效果，则应将 VCTK 参与训练或微调。

## 10. 主要文件

| 文件 | 作用 |
|---|---|
| `src/wave_model.py` | Wave-U-Net、BiLSTM 与自注意力模型定义 |
| `src/losses.py` | SI-SDR 与多分辨率 STFT 组合损失 |
| `src/preprocess.py` | Edinburgh 训练/验证数据预处理 |
| `src/train.py` | 训练、验证、早停与权重保存 |
| `src/inference.py` | 单条或完整测试集去噪、后处理与频谱图生成 |
| `src/evaluate_full_utterances.py` | 824 条完整测试语音的指标评估 |
| `src/baseline.py` | 带噪、谱减和维纳滤波基线对比 |
| `src/config.py` | 数据路径与超参数配置 |

## 11. 常见问题

**`ModuleNotFoundError: No module named 'torch'`**

说明当前 `python` 不在 mamba 环境中。使用 `$mambaPy` 指向 `D:\miniconda3\envs\mamba\python.exe` 后重新运行。

**`No .pt files found ... segments/train`**

尚未完成训练/验证数据预处理。执行第 4.1 节命令。

**显存不足（CUDA out of memory）**

在 `src/config.py` 中调小 `BATCH_SIZE`，例如从 64 改为 32 或 16。完整语音推理由内部窗口化和重叠相加完成，不需要把整条语音一次送入显存。

