import torch.nn as nn
import torch.nn.functional as F

class EncBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bn=True, pool=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity()
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.should_bn = bn
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        if self.should_bn:
            x = self.bn(x)
        x = self.relu(x)
        return x
    
class DecBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, upsample=False):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.upsample = nn.Upsample(scale_factor=2) if upsample else nn.Identity()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        return x

class mnist_cnn_encoder(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.enc_blocks = nn.ModuleList([
            EncBlock(1, 32, 3, 1, 1, pool=True),
            EncBlock(32, 64, 3, 1, 1, pool=True),
            EncBlock(64, 128, 3, 1, 0),
            EncBlock(128, 256, 3, 1, 0),
            EncBlock(256, num_features, 3, 1, 0, bn=False),
        ])
    
    def forward(self, x, stop_at=None):
        for i, block in enumerate(self.enc_blocks):
            if i == stop_at:
                break
            x = block(x)
        if stop_at is not None:
            return x
        else:
            return x.flatten(1)

class mnist_cnn_decoder(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.dec_blocks = nn.ModuleList([
            DecBlock(num_features, 256, 3, 1, 0),
            DecBlock(256, 128, 3, 1, 0),
            DecBlock(128, 64, 3, 1, 0),
            DecBlock(64, 32, 3, 1, 1, upsample=True),
            DecBlock(32, 1, 3, 1, 1, upsample=True),
        ])

    def forward(self, z, stop_at=None):
        z = z.view(-1, 256, 1, 1)
        for i, block in enumerate(self.dec_blocks):
            if stop_at is not None and i == len(self.dec_blocks) - stop_at:
                break
            z = block(z)
            if i < len(self.dec_blocks) - 1:
                z = F.relu(z)
            else:
                z = F.sigmoid(z)
        return z
    