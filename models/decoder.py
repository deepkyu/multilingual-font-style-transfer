import torch
import torch.nn as nn
from models.module import ResidualBlocks

_DECODER_CHANNEL_DEFAULT = 512


class Decoder(nn.Module):
    def __init__(self, hp, in_channels=_DECODER_CHANNEL_DEFAULT, out_channels=1):
        super().__init__()
        self.module = nn.ModuleList()

    def forward(self, x):
        for block in self.module:
            x = block(x)
        return x


class VanillaDecoder(Decoder):
    def __init__(self, hp, in_channels, out_channels):
        super().__init__(hp, in_channels, out_channels)
        self.depth = hp.decoder.depth
        self.blocks = hp.decoder.residual_blocks

        self.module = nn.ModuleList()
        if self.blocks > 0:
            self.module.append(ResidualBlocks(in_channels, n_blocks=self.blocks))

        for layer_idx in range(1, self.depth + 1):  # add upsampling layers
            self.module.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels // (2 ** (layer_idx - 1)),
                                   in_channels // (2 ** layer_idx),
                                   kernel_size=3, stride=2,
                                   padding=1, output_padding=1,
                                   bias=False),
                nn.BatchNorm2d(in_channels // (2 ** layer_idx)),
                nn.ReLU(True)
            ))

        final = nn.Sequential(
            nn.Conv2d(in_channels // (2 ** self.depth), out_channels, kernel_size=7, padding=3, padding_mode='reflect'),
            nn.Tanh()
        )

        self.module.append(final)
