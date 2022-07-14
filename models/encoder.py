import torch
import torch.nn as nn

class ContentEncoder(nn.Module):
    def __init__(self, hp, input_channel, output_channel):
        super().__init__()
        self.hp = hp
        self.module = nn.ModuleList()
            
    def forward(self, x):
        for block in self.module:
            x = block(x)
        return x
    

class ContentVanillaEncoder(ContentEncoder):
    """Following the basic architecture in https://github.com/ligoudaner377/font_translator_gan
    
    """
    def __init__(self, hp, input_channel, output_channel):
        super().__init__(hp, input_channel, output_channel)
        self.depth = self.hp.depth
        assert output_channel // (2 ** self.depth) >= input_channel * 2, "Output channel should be increased"
        
        self.module = nn.ModuleList()
        self.module.append(nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channel, output_channel // (2 ** self.depth), kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(output_channel // (2 ** self.depth)),
            nn.ReLU()
        ))
        
        for layer_idx in range(1, self.depth + 1):  # downsample
            self.module.append(nn.Sequential(
                nn.Conv2d(output_channel // (2 ** (self.depth - layer_idx + 1)),
                          output_channel // (2 ** (self.depth - layer_idx)),
                          kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(output_channel // (2 ** (self.depth - layer_idx))),
                nn.ReLU()
            ))