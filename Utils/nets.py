from collections import OrderedDict
from functools import partial
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.vision_transformer import ConvStemConfig, EncoderBlock, VisionTransformer

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
        self.convt = nn.ConvTranspose2d(in_channels, in_channels, kernel_size, stride, padding)
        self.upsample = nn.Upsample(scale_factor=2) if upsample else nn.Identity()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
    
    def forward(self, x):
        x = self.convt(x)
        x = self.upsample(x)
        x = self.conv(x)
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
    
    def forward(self, x):
        for block in self.enc_blocks:
            x = block(x)
        return x.flatten(1)

class mnist_cnn_decoder(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(num_features, 256, 3, 1),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, 3, 3),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, 3, 3),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, 1, 1),
        )

    def forward(self, z):
        z = z.view(-1, self.num_features, 1, 1)
        return self.decoder(z)


class ViTEncoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        num_registers: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default

        # initialise registers
        self.num_registers = num_registers
        self.registers = nn.Parameter(torch.empty(1, num_registers, hidden_dim).normal_(std=0.02)) # copied pos_embedding init. (not optimised)

        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor, stop_at:Optional[int]=None):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        # add registers to input, without positional embedding
        x = torch.cat([input, self.registers.expand(input.size(0), -1, -1)], dim=1)
        # output = self.ln(self.layers(self.dropout(input)))
        x = self.dropout(x)
        for i, layer in enumerate(self.layers):
            if i == stop_at:
                break
            x = layer(x)
        output = self.ln(x)

        # return output, excluding the registers
        if self.num_registers > 0:
            output = output[:, :-self.num_registers, :]
        return output

class mnist_vit(VisionTransformer):
    def __init__(self, num_features):
        super().__init__(
            image_size=28,
            patch_size=7,
            num_layers=6,
            num_heads=4,
            hidden_dim=num_features,
            mlp_dim=num_features*4,
            dropout=0.1,
            attention_dropout=0.1,
            num_classes=0,
            representation_size=None,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            conv_stem_configs=None,
        )

        self.num_registers = 4
        self.seq_length = (28 // self.patch_size) ** 2 + 1

        self.encoder = ViTEncoder(
            seq_length=self.seq_length,
            num_layers=6,
            num_heads=4,
            hidden_dim=num_features,
            num_registers=4,
            mlp_dim=num_features*4,
            dropout=0.1,
            attention_dropout=0.1,
            norm_layer = partial(nn.LayerNorm, eps=1e-6),
        )

        self.conv_proj = nn.Conv2d(1, num_features, kernel_size=7, stride=7)
        self.heads = nn.Identity()
    
    def forward(self, x: torch.Tensor, stop_at:Optional[int]=None):
        x = self._process_input(x)
        n = x.shape[0]

        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x, stop_at)

        if stop_at is None:
            return x[:, 0]
        else:
            return x[:, 1:]

class Decoder224(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(num_features, 256, 5, 1),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, 4, 4),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 128, 4, 4),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, 3, 3),
            nn.ReLU(),

            nn.Conv2d(64, 64, 7, 1, 0),
            nn.ReLU(),

            nn.Conv2d(64, 64, 7, 1, 0),
            nn.ReLU(),

            nn.Conv2d(64, 1, 5, 1, 0),
        )

    def forward(self, z):
        z = z.view(-1, self.num_features, 1, 1)
        return self.decoder(z)