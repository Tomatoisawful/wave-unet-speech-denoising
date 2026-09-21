"""
losses.py

训练时使用的损失函数。

组合损失：
    L = -SI_SDR + 0.1 * MultiResolutionSTFTLoss

SI-SDR（尺度不变信号失真比）：
    衡量预测波形与纯净波形的相似度，不受绝对响度影响；SI-SDR 越高越好，
    因此训练时最小化 -SI-SDR。

多分辨率 STFT 损失：
    在多个时频分辨率下比较频谱，使模型同时保留快速瞬态细节和缓慢的语音结构。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


def si_sdr_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    尺度不变信号失真比损失。

    直接作用于原始波形，而非频谱。

    Formula:
        s_target = (<s_hat, s> / ||s||^2) * s          (projection of pred onto target)
        e_noise  = s_hat - s_target                    (distortion component)
        SI-SDR   = 10 * log10(||s_target||^2 / ||e_noise||^2)
        loss     = -mean(SI-SDR)                       (minimise the negative)

    Args:
        pred:   (batch, samples)，重建波形
        target: (batch, samples)，纯净参考波形

    Returns:
        标量损失张量
    """
    # 两个信号均去均值（SI-SDR 定义在零均值信号上）
    pred   = pred   - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    # 计算 pred 在 target 上的投影
    dot        = (pred * target).sum(dim=-1, keepdim=True)
    target_pow = (target * target).sum(dim=-1, keepdim=True).clamp(min=eps)
    s_target   = (dot / target_pow) * target

    # 失真为投影信号之外的全部成分
    e_noise = pred - s_target

    # 以 dB 表示的 SI-SDR
    si_sdr = 10 * torch.log10(
        (s_target ** 2).sum(dim=-1).clamp(min=eps) /
        (e_noise  ** 2).sum(dim=-1).clamp(min=eps)
    )

    # 返回负均值（优化目标为最小化，故损失越低代表 SI-SDR 越高）
    return -si_sdr.mean()


def _stft_loss_single(
    pred:       torch.Tensor,
    target:     torch.Tensor,
    n_fft:      int,
    hop_length: int,
    win_length: int,
) -> torch.Tensor:
    """
    计算单个分辨率下的 STFT 损失。

    Two components:
        spectral_convergence = ||M_target - M_pred||_F / ||M_target||_F
        log_magnitude        = ||log(M_target + eps) - log(M_pred + eps)||_1 / N

    两个分量都会惩罚幅度频谱差异：谱收敛项关注高能量区域，
    对数幅度项更关注安静细节。

    Args:
        pred / target: (batch, samples) - waveforms

    Returns:
        scalar loss tensor
    """
    eps    = 1e-7
    window = torch.hann_window(win_length, device=pred.device)

    def magnitude(x):
        stft = torch.stft(
            x, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            window=window, return_complex=True,
        )
        return stft.abs()

    M_pred   = magnitude(pred)
    M_target = magnitude(target)

    # 谱收敛损失（Frobenius 范数比）
    sc_loss = torch.norm(M_target - M_pred, p="fro") / torch.norm(M_target, p="fro").clamp(min=eps)

    # 对数幅度损失（L1）
    log_pred   = torch.log(M_pred   + eps)
    log_target = torch.log(M_target + eps)
    lm_loss    = F.l1_loss(log_pred, log_target)

    return sc_loss + lm_loss


def multi_resolution_stft_loss(
    pred:        torch.Tensor,
    target:      torch.Tensor,
    resolutions: list[tuple[int, int, int]] = config.STFT_RESOLUTIONS,
) -> torch.Tensor:
    """
    对多个 (n_fft, hop_length, win_length) 设置下的 STFT 损失取平均。

    多分辨率迫使模型同时保证较高时间分辨率（小 n_fft，捕获快速瞬态）
    和较高频率分辨率（大 n_fft，捕获音调结构）。

    Args:
        pred / target: (batch, samples)
        resolutions:   list of (n_fft, hop, win) tuples

    Returns:
        scalar loss tensor
    """
    total = torch.tensor(0.0, device=pred.device)
    for n_fft, hop, win in resolutions:
        total = total + _stft_loss_single(pred, target, n_fft, hop, win)
    return total / len(resolutions)


class DenoisingLoss(nn.Module):
    """
    组合损失：L = -SI_SDR + w × MultiResSTFTLoss。

    STFT 损失提供符合感知的频谱约束，SI-SDR 保证整体波形保真；
    权重（默认 0.1）使两项的数值范围相近。

    pred 和 target 均须为形状 (batch, samples) 的原始波形。
    """

    def __init__(self, stft_weight: float = config.STFT_LOSS_WEIGHT):
        super().__init__()
        self.stft_weight = stft_weight

    def forward(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Returns:
            total_loss: 用于反向传播的标量张量
            components: 记录各损失项数值的字典
        """
        loss_si_sdr  = si_sdr_loss(pred, target)
        loss_stft    = multi_resolution_stft_loss(pred, target)
        total        = loss_si_sdr + self.stft_weight * loss_stft

        components = {
            "loss_si_sdr": loss_si_sdr.item(),
            "loss_stft":   loss_stft.item(),
            "loss_total":  total.item(),
        }
        return total, components
