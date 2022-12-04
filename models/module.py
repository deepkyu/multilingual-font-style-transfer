import torch
import torch.nn as nn


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, padding_mode='zeros', bias=True, residual=False):
        super(Conv2d, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                      padding, padding_mode=padding_mode, bias=bias),
            nn.BatchNorm2d(out_channels)
        )
        self.residual = residual
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        out = self.act(out)
        return out


class ResnetBlock(nn.Module):
    def __init__(self, channel, padding_mode, norm_layer=nn.BatchNorm2d, bias=False):
        super().__init__()
        if padding_mode not in ['reflect', 'zero']:
            raise NotImplementedError(f"{padding_mode} is not supported!")

        self.block = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, padding_mode=padding_mode, bias=bias),
            norm_layer(channel)
        )
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.block(x)
        out = out + x
        out = self.act(out)
        return out


class ResidualBlocks(nn.Module):
    def __init__(self, channel, n_blocks=6):
        super().__init__()
        model = []
        for i in range(n_blocks):  # add ResNet blocks
            model += [ResnetBlock(channel, padding_mode='reflect')]

        self.module = nn.Sequential(*model)

    def forward(self, x):
        return self.module(x)


class SelfAttentionBlock(nn.Module):

    def __init__(self, in_dim):
        super().__init__()
        self.feature_dim = in_dim // 8
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.feature_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.feature_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        _query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)  # B x C x (H'*W')
        _key = self.key_conv(x).view(B, -1, H * W)  # B x C x (H'*W')
        attn_matrix = torch.bmm(_query, _key)
        attention = self.softmax(attn_matrix)  # B x (H'*W') x (H'*W')
        _value = self.value_conv(x).view(B, -1, H * W)  # B X C X (H * W)

        out = torch.bmm(_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        out = self.gamma * out + x
        return out


class ContextAwareAttentionBlock(nn.Module):

    def __init__(self, in_channels, hidden_dim=128):
        super().__init__()
        self.self_attn = SelfAttentionBlock(in_channels)
        self.fc = nn.Linear(in_channels, hidden_dim)
        self.context_vector = nn.Linear(hidden_dim, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, style_features):
        B, C, H, W = style_features.size()
        h = self.self_attn(style_features)
        h = h.permute(0, 2, 3, 1).reshape(-1, C)
        h = torch.tanh(self.fc(h))  # (B*H*W) x self.hidden_dim
        h = self.context_vector(h)  # (B*H*W) x 1
        attention_score = self.softmax(h.view(B, H * W)).view(B, 1, H, W)  # B x 1 x H x W
        return torch.sum(style_features * attention_score, dim=[2, 3])  # B x C


class LayerAttentionBlock(nn.Module):
    """from FTransGAN
    """

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.width_feat = 4
        self.height_feat = 4
        self.fc = nn.Linear(self.in_channels * self.width_feat * self.height_feat, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, style_features, style_features_1, style_features_2, style_features_3, B, K):
        style_features = torch.mean(style_features.view(B, K, self.in_channels, self.height_feat, self.width_feat), dim=1)
        style_features = style_features.view(B, -1)
        weight = self.softmax(self.fc(style_features))

        style_features_1 = torch.mean(style_features_1.view(B, K, self.in_channels), dim=1)
        style_features_2 = torch.mean(style_features_2.view(B, K, self.in_channels), dim=1)
        style_features_3 = torch.mean(style_features_3.view(B, K, self.in_channels), dim=1)

        style_features = (style_features_1 * weight.narrow(1, 0, 1) +
                          style_features_2 * weight.narrow(1, 1, 1) +
                          style_features_3 * weight.narrow(1, 2, 1))
        style_features = style_features.view(B, self.in_channels, 1, 1)
        return style_features


class StyleAttentionBlock(nn.Module):
    """from FTransGAN
    """

    def __init__(self, in_channels):
        super().__init__()
        self.num_local_attention = 3
        for module_idx in range(1, self.num_local_attention + 1):
            self.add_module(f"local_attention_{module_idx}",
                            ContextAwareAttentionBlock(in_channels))

        for module_idx in range(1, self.num_local_attention):
            self.add_module(f"downsample_{module_idx}",
                            Conv2d(in_channels, in_channels,
                                   kernel_size=3, stride=2, padding=1, bias=False))

        self.add_module(f"layer_attention", LayerAttentionBlock(in_channels))

    def forward(self, x, B, K):
        feature_1 = self.local_attention_1(x)

        x = self.downsample_1(x)
        feature_2 = self.local_attention_2(x)

        x = self.downsample_2(x)
        feature_3 = self.local_attention_3(x)

        out = self.layer_attention(x, feature_1, feature_2, feature_3, B, K)

        return out
