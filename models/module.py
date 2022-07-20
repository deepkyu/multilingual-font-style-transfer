import torch
import torch.nn as nn

class Conv2d(nn.Module):
    def __init__(self, in_dim, out_dim, ks, stride, padding, residual=False):
        super(Conv2d, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, ks, stride, padding),
            nn.BatchNorm2d(out_dim)
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
    def __init__(self, channel, padding_type, norm_layer):
        super().__init__()
        block = [nn.ReflectionPad2d(1)] if padding_type == 'reflect' else []
        p = 1 if padding_type == 'zero' else 0
        
        if padding_type not in ['reflect', 'zero']:
            raise NotImplementedError(f"{padding_type} is not supported!")

        block += [
            nn.Conv2d(channel, channel, kernel_size=3, padding=p, bias=False),
            norm_layer(channel),
        ]
        
        self.block = nn.Sequential(*block)
        self.act = nn.ReLU()

    def forward(self, x):
        out = x + self.block(x)
        out = self.act(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, channel, n_blocks=6):
        super().__init__()
        model = []
        for i in range(n_blocks):  # add ResNet blocks
            model += [ResnetBlock(channel, padding_type='reflect', norm_layer=nn.BatchNorm2d)]
        
        self.module = nn.Sequential(*model)
        
    def forward(self, x):
        return self.module(x)
    

class FTransGANSelfAttentionBlock(nn.Module):
    """ Self attention Layer from FTransGAN
    """

    def __init__(self, in_dim):
        super().__init__()
        self.attention_feature_dim = in_dim // 8
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.attention_feature_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.attention_feature_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out


class FTransGANLocalAttentionBlock(nn.Module):
    """from FTransGAN
    """
    def __init__(self, in_channels):
        super().__init__()
        self.hidden_dim = 100
        self.self_atten = FTransGANSelfAttentionBlock(in_channels)
        self.attention = nn.Linear(in_channels, self.hidden_dim)
        self.context_vec = nn.Linear(self.hidden_dim, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, style_features):
        B, C, H, W = style_features.shape
        h = self.self_atten(style_features)
        h = h.permute(0, 2, 3, 1).reshape(-1, C)
        h = torch.tanh(self.attention(h))  # (B*H*W, self.hidden_dim)
        h = self.context_vec(h)  # (B*H*W, 1)
        attention_map = self.softmax(h.view(B, H * W)).view(B, 1, H, W)  # (B, 1, H, W)
        style_features = torch.sum(style_features * attention_map, dim=[2, 3])
        return style_features
    
class FTransGANLayerAttentionBlock(nn.Module):
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
    
class FTransGANAttentionBlock(nn.Module):
    """from FTransGAN
    """
    def __init__(self, in_channels):
        super().__init__()
        self.num_local_attention = 3
        for module_idx in range(1, self.num_local_attention + 1):
            self.add_module(f"local_attention_{module_idx}",
                            FTransGANLocalAttentionBlock(in_channels))
            
        for module_idx in range(1, self.num_local_attention):
            self.add_module(f"downsample_{module_idx}",
                            nn.Sequential(
                nn.Conv2d(in_channels, in_channels,
                          kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU()
            ))
            
        self.add_module(f"layer_attention", FTransGANLayerAttentionBlock(in_channels))
        
    def forward(self, x, B, K):
        feature_1 = self.local_attention_1(x)

        x = self.downsample_1(x)
        feature_2 = self.local_attention_2(x)

        x = self.downsample_2(x)
        feature_3 = self.local_attention_3(x)

        out = self.layer_attention(x, feature_1, feature_2, feature_3, B, K)

        return out