from collections import OrderedDict
from functools import partial
from typing import Callable, List, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.vision_transformer import ConvStemConfig, EncoderBlock, VisionTransformer
from Utils.nn.parts import EncBlock, DecBlock, ConvResidualBlock, SelfAttentionBlock, TransformerEncoderBottleneck, TransformerDecoderBottleneck

class Encoder28(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.enc_blocks = nn.ModuleList([
            EncBlock(1, 32, 3, 1, 1, pool=True),
            EncBlock(32, 64, 3, 1, 1, pool=True),
            EncBlock(64, 128, 3, 1, 0),
            EncBlock(128, 256, 3, 1, 0),
            EncBlock(256, num_features, 3, 1, 0, bn=False),
        ])
    
    def forward(self, x, stop_at=-1):

        for i, block in enumerate(self.enc_blocks):
            if i == stop_at:
                break
            x = block(x)
        
        if x.shape[2] == 1:
            x = x.flatten(1)
        return x

class Decoder1(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features

        self.decoder = nn.Sequential(
            nn.Linear(num_features, num_features*2),
            nn.ReLU(),
            nn.Linear(num_features*2, num_features*2),
            nn.ReLU(),
            nn.Linear(num_features*2, num_features*2),
            nn.ReLU(),
            nn.Linear(num_features*2, num_features*2),
            nn.ReLU(),
            nn.Linear(num_features*2, num_features),
        )

    def forward(self, x):
        return self.decoder(x)

class Decoder5(nn.Module):
    def __init__(self, num_features, out_features=128):
        super().__init__()
        self.num_features = num_features

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(num_features, max(256, out_features), 2, 1),
            nn.ReLU(),

            nn.ConvTranspose2d(max(256, out_features), max(128, out_features), 3, 1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(max(128, out_features), max(64, out_features), 2, 1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(max(64, out_features), max(32, out_features), 3, 1, 1),

            nn.ReLU(),
            nn.Conv2d(max(32, out_features), out_features, 3, 1, 1),
        )

    def forward(self, z):
        z = z.view(-1, self.num_features, 1, 1)
        return self.decoder(z)

class Decoder28(nn.Module):
    def __init__(self, num_features, out_features=1):
        super().__init__()
        self.num_features = num_features

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(num_features, max(256, out_features), 3, 1),
            nn.ReLU(),

            nn.ConvTranspose2d(max(256, out_features), max(128, out_features), 3, 3),
            nn.ReLU(),
            
            nn.ConvTranspose2d(max(128, out_features), max(64, out_features), 3, 3),
            nn.ReLU(),
            
            nn.ConvTranspose2d(max(64, out_features), max(32, out_features), 2, 1),

            nn.ReLU(),
            nn.Conv2d(max(32, out_features), out_features, 3, 1, 1),
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

class Encoder128(nn.Module):
    def __init__(self, in_features, num_features):
        super().__init__()
        self.in_features = in_features
        self.num_features = num_features

        self.encoder = nn.ModuleList([

            # (-1, in_features, 128, 128) -> (-1, 64, 128, 128)
            nn.Sequential(
                nn.Conv2d(in_features, 32, 3, 1, 1),
                ConvResidualBlock(32, 32),
            ),

            # (-1, 64, 128, 128) -> (-1, 128, 64, 64)
            nn.Sequential(
                # nn.Conv2d(64, 64, 2, 2, 0),
                nn.MaxPool2d(2, 2),
                ConvResidualBlock(32, 64),
            ),

            # (-1, 128, 64, 64) -> (-1, 256, 32, 32)
            nn.Sequential(
                # nn.Conv2d(128, 128, 2, 2, 0),
                nn.MaxPool2d(2, 2),
                ConvResidualBlock(64, 128),
            ),

            # (-1, 256, 32, 32) -> (-1, 512, 16, 16)
            nn.Sequential(
                # nn.Conv2d(256, 256, 2, 2, 0),
                nn.MaxPool2d(2, 2),
                ConvResidualBlock(128, 256),
            ),
        
            # (-1, 512, 16, 16) -> (-1, 512, 8, 8)
            nn.Sequential(
                # nn.Conv2d(512, 512, 2, 2, 0),
                nn.MaxPool2d(2, 2),
                ConvResidualBlock(256, num_features),
            ),

            TransformerEncoderBottleneck(num_features, (8,8), 4, 4, 4, 1024, 0.1, 0.1),
        ])
    
    def forward(self, x, stop_at=-1):
        for i, block in enumerate(self.encoder):
            if i == stop_at:
                break
            x = block(x)
        return x


class Decoder128(nn.Module):
    def __init__(self, out_channels, num_features):
        super().__init__()
        self.num_features = num_features
        self.upsample = nn.Upsample(scale_factor=2)
        self.decoder = nn.ModuleList([

            # TransformerDecoderBottleneck(num_features, (8,8), 1, 4, 4, 1024, 0.0, 0.0),
            nn.ConvTranspose2d(num_features, 512, 8, 1),

            # nn.ConvTranspose2d(512, 512, 2, 2),
            ConvResidualBlock(num_features, 512),

            # nn.ConvTranspose2d(512, 512, 2, 2),
            ConvResidualBlock(512, 256),

            # nn.ConvTranspose2d(256, 256, 2, 2),
            ConvResidualBlock(256, 128),

            # nn.ConvTranspose2d(128, 128, 2, 2),
            ConvResidualBlock(128, 128),

            nn.Sequential(
                nn.BatchNorm2d(128),
                nn.SiLU(),
                nn.Conv2d(128, out_channels, 3, 1, 1),
            ),
        ])

    def forward(self, z, stop_at=0):
        z = z.view(-1, self.num_features, 1, 1)
        i = len(self.decoder)
        for layer in self.decoder:
            if i < len(self.decoder) and i > 1:
                z = self.upsample(z)
            z = layer(z)
            i -= 1
        return z


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

class GRU(nn.Module):
    def __init__(self, num_features, num_layers):
        super().__init__()
        self.gru = nn.GRU(num_features, num_features, num_layers, batch_first=True)

    def forward(self, x):
        x = x.squeeze(2).permute(0, 2, 1)
        return self.gru(x)[0][:, -1]

class VoxEncoder(nn.Module):
    def __init__(self, num_features, num_layers=12):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, num_features, (128, 8), 8),
            GRU(num_features, num_layers),
        )

    def forward(self, x):
        return self.encoder(x)

class VoxDecoder(nn.Module):
    def __init__(self, out_channels, num_features):
        super().__init__()
        self.num_features = num_features
        self.upsample = nn.Upsample(scale_factor=2)
        self.decoder = nn.ModuleList([

            # TransformerDecoderBottleneck(num_features, (8,8), 1, 4, 4, 1024, 0.0, 0.0),
            nn.ConvTranspose2d(num_features, 512, (8,16), 1),
            # nn.ConvTranspose2d(512, num_features, 8, 1),

            # nn.ConvTranspose2d(512, 512, 2, 2),
            ConvResidualBlock(512, 512),

            # nn.ConvTranspose2d(512, 512, 2, 2),
            ConvResidualBlock(512, 256),

            # nn.ConvTranspose2d(256, 256, 2, 2),
            ConvResidualBlock(256, 128),

            # nn.ConvTranspose2d(128, 128, 2, 2),
            ConvResidualBlock(128, 128),

            nn.Sequential(
                nn.BatchNorm2d(128),
                nn.SiLU(),
                nn.Conv2d(128, out_channels, 3, 1, 1),
            ),
        ])

    def forward(self, z, stop_at=0):
        z = z.view(-1, self.num_features, 1, 1)
        i = len(self.decoder)
        for layer in self.decoder:
            if i < len(self.decoder) and i > 1:
                z = self.upsample(z)
            z = layer(z)
            i -= 1
        return z