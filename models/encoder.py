import torch
import torch.nn as nn
from models.module import Conv2d, StyleAttentionBlock

_ENCODER_CHANNEL_DEFAULT = 256


class Encoder(nn.Module):
    def __init__(self, hp, in_channels=1, out_channels=_ENCODER_CHANNEL_DEFAULT):
        super().__init__()
        self.hp = hp
        self.module = nn.ModuleList()

    def forward(self, x):
        for block in self.module:
            x = block(x)
        return x


class ContentVanillaEncoder(Encoder):
    def __init__(self, hp, in_channels, out_channels):
        super().__init__(hp, in_channels, out_channels)
        self.depth = hp.encoder.content.depth
        assert out_channels // (2 ** self.depth) >= in_channels * 2, "Output channel should be increased"

        self.module = nn.ModuleList()
        self.module.append(
            Conv2d(in_channels, out_channels // (2 ** self.depth),
                   kernel_size=7, padding=3, padding_mode='reflect', bias=False)
        )

        for layer_idx in range(1, self.depth + 1):  # downsample
            self.module.append(
                Conv2d(out_channels // (2 ** (self.depth - layer_idx + 1)),
                       out_channels // (2 ** (self.depth - layer_idx)),
                       kernel_size=3, stride=2, padding=1, bias=False)
            )


class StyleVanillaEncoder(Encoder):
    def __init__(self, hp, in_channels, out_channels):
        super().__init__(hp, in_channels, out_channels)
        self.depth = hp.encoder.style.depth
        assert out_channels // (2 ** self.depth) >= in_channels * 2, "Output channel should be increased"

        encoder_module = []
        encoder_module.append(
            Conv2d(in_channels, out_channels // (2 ** self.depth),
                   kernel_size=7, padding=3, padding_mode='reflect', bias=False)
        )

        for layer_idx in range(1, self.depth + 1):  # downsample
            encoder_module.append(
                Conv2d(out_channels // (2 ** (self.depth - layer_idx + 1)),
                       out_channels // (2 ** (self.depth - layer_idx)),
                       kernel_size=3, stride=2, padding=1, bias=False)
            )
        self.add_module("encoder_module", nn.Sequential(*encoder_module))
        self.add_module("attention_module", StyleAttentionBlock(out_channels))

    def forward(self, x):
        B, K, H, W = x.size()
        out = self.encoder_module(x.view(-1, 1, H, W))
        out = self.attention_module(out, B, K)
        return out
