from functools import partial
from typing import Any, Callable, List, Optional, Type, Union
import torch.nn as nn
from torch import Tensor
from typing import Tuple
from Utils.nn.parts import TransformerEncoderBottleneck

# Adapted from https://github.com/locuslab/convmixer/blob/main/convmixer.py
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

# defaults for 128x128 images
class ConvMixer(nn.Module):
    def __init__(self, dim, depth, kernel_size=5, patch_size=4):
        super().__init__()
        self.net = nn.Sequential(
        nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        )

    def forward(self, x, stop_at=None):
        if stop_at == 0:
            return x
        else: 
            return self.net(x)