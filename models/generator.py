import torch
import torch.nn as nn
from . import encoder, decoder

class Generator(nn.Module):
    def __init__(self, hp, in_channels=1):
        super().__init__()
        self.hp = hp
        _ngf = 64
        hidden_dim = _ngf * 4
        self.content_encoder = getattr(encoder, self.hp.encoder.content.type)(self.hp, in_channels, hidden_dim)
        self.style_encoder = getattr(encoder, self.hp.encoder.style.type)(self.hp, in_channels, hidden_dim)
        self.decoder = getattr(decoder, self.hp.decoder.type)(self.hp, hidden_dim * 2, in_channels)
                
    def forward(self, images):
        content_images, style_images = images
        content_feature = self.content_encoder(content_images)
        style_images = style_images * 2 - 1  # pixel value range -1 to 1
        style_feature = self.style_encoder(style_images)  # K-shot as batch
        _, _, H, W = content_feature.size()
        out = self.decoder(torch.cat([content_feature, style_feature.expand(-1, -1, H, W)], dim=1))
        return out