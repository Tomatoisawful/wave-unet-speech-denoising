"""支持完整模型及瓶颈模块消融的 Wave-U-Net 语音增强模型。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class EncoderBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv1d(out_channels, out_channels, 15, padding=7, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.PReLU(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 15, padding=7, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv1d(out_channels, out_channels, 15, padding=7, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.PReLU(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock1D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 5, padding=2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.PReLU(out_channels),
        )
        self.fuse = ConvBlock1D(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        return self.fuse(torch.cat([self.up(x), skip], dim=1))


class LSTMAttentionBottleneck(nn.Module):
    """可切换双向或单向 LSTM 的注意力瓶颈。"""

    def __init__(
        self,
        channels: int,
        hidden: int,
        layers: int,
        heads: int,
        dropout: float,
        bidirectional: bool = True,
    ):
        super().__init__()
        directions = 2 if bidirectional else 1
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            channels, hidden, num_layers=layers, batch_first=True,
            bidirectional=bidirectional, dropout=dropout if layers > 1 else 0.0,
        )
        lstm_output = directions * hidden
        self.output_projection = (
            nn.Identity() if lstm_output == channels
            else nn.Linear(lstm_output, channels)
        )
        self.norm1 = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels, heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(channels * 2, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.transpose(1, 2)
        seq, _ = self.lstm(seq)
        seq = self.output_projection(seq)
        seq = self.norm1(seq)
        attn, _ = self.attention(seq, seq, seq, need_weights=False)
        seq = self.norm2(seq + attn)
        seq = seq + self.ffn(seq)
        return seq.transpose(1, 2)


class AttentionBottleneck(nn.Module):
    """不含循环网络的纯自注意力瓶颈，用于消融实验。"""

    def __init__(self, channels: int, heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels, heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(channels * 2, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.transpose(1, 2)
        seq = self.norm1(seq)
        attn, _ = self.attention(seq, seq, seq, need_weights=False)
        seq = self.norm2(seq + attn)
        seq = seq + self.ffn(seq)
        return seq.transpose(1, 2)


class WaveUNetDenoiser(nn.Module):
    """直接将带噪波形映射为去噪波形的 Wave-U-Net 去噪器。"""

    def __init__(
        self,
        encoder_channels: list[int] = config.WAVE_ENCODER_CHANNELS,
        lstm_hidden: int | None = None,
        lstm_layers: int = config.LSTM_LAYERS,
        use_lstm: bool = config.USE_LSTM,
        lstm_bidirectional: bool = config.LSTM_BIDIRECTIONAL,
        use_attention: bool = config.USE_ATTENTION,
        attention_heads: int = config.ATTENTION_HEADS,
        attention_dropout: float = config.ATTENTION_DROPOUT,
    ):
        super().__init__()
        channels = [1] + encoder_channels
        self.encoders = nn.ModuleList([
            EncoderBlock1D(channels[i], channels[i + 1])
            for i in range(len(encoder_channels))
        ])
        if lstm_hidden is None:
            # 双向与单向实验均使用 256 维隐藏状态。单向输出通过线性层投影
            # 到 512 维，使注意力层和解码器保持不变。
            lstm_hidden = config.LSTM_HIDDEN
        self.use_lstm = use_lstm
        self.use_attention = use_attention
        self.lstm_bidirectional = lstm_bidirectional
        self.model_config = {
            "use_lstm": use_lstm,
            "use_attention": use_attention,
            "lstm_bidirectional": lstm_bidirectional,
        }
        if use_lstm and not use_attention:
            raise ValueError("当前实验方案不包含 LSTM-only 结构；请启用注意力或同时移除 LSTM")
        if use_lstm:
            self.bottleneck = LSTMAttentionBottleneck(
                encoder_channels[-1], lstm_hidden, lstm_layers,
                attention_heads, attention_dropout, lstm_bidirectional,
            )
        elif use_attention:
            self.bottleneck = AttentionBottleneck(
                encoder_channels[-1], attention_heads, attention_dropout,
            )
        else:
            # 纯 Wave-U-Net：编码器输出直接送入解码器。
            self.bottleneck = nn.Identity()

        # 每次下采样前保存编码器跳跃连接，其通道数为
        # [1, 32, 64, 128, 256]，解码时按相反顺序使用。
        skip_channels = list(reversed([1] + encoder_channels[:-1]))
        decoder_outputs = list(reversed(encoder_channels[:-1])) + [16]
        self.decoders = nn.ModuleList()
        in_channels = encoder_channels[-1]
        for skip_ch, out_ch in zip(skip_channels, decoder_outputs):
            self.decoders.append(DecoderBlock1D(in_channels, skip_ch, out_ch))
            in_channels = out_ch
        self.output = nn.Conv1d(in_channels, 1, kernel_size=1)
        # 网络预测残差。初始化时保持接近恒等映射，避免向带噪输入叠加
        # 幅度很大的随机波形。
        nn.init.normal_(self.output.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.output.bias)

    def forward(self, noisy_wave: torch.Tensor) -> torch.Tensor:
        if noisy_wave.ndim != 3 or noisy_wave.shape[1] != 1:
            raise ValueError(f"期望输入形状为 (B, 1, samples)，实际为 {tuple(noisy_wave.shape)}")
        original = noisy_wave
        skips = []
        x = noisy_wave
        for encoder in self.encoders:
            skips.append(x)
            x = encoder(x)
        x = self.bottleneck(x)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)
        residual = self.output(x)
        if residual.shape[-1] != original.shape[-1]:
            residual = F.interpolate(residual, size=original.shape[-1], mode="linear", align_corners=False)
        return original + residual


# 保留旧导入名称，兼容已有调用代码。
UNetDenoiser = WaveUNetDenoiser
BiLSTMAttentionBottleneck = LSTMAttentionBottleneck


if __name__ == "__main__":
    model = WaveUNetDenoiser()
    dummy = torch.randn(2, 1, config.SEGMENT_SAMPLES)
    output = model(dummy)
    print(f"Input shape : {dummy.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters  : {sum(p.numel() for p in model.parameters()):,}")
