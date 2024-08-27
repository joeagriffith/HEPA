import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import EncoderBlock

from Utils.masking import random_masking
from Utils.nn.transformer import Transformer

from typing import Callable
from functools import partial
from collections import OrderedDict
import math


class Encoder(Transformer):

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        input_shape: tuple[int, int],
        patch_size: int,
        num_layers: int,
        num_heads: int,
        num_registers: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__(
            in_dim=in_dim,
            embed_dim=embed_dim,
            input_shape=input_shape,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_registers=num_registers,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            norm_layer=norm_layer,
            cls_token=False,
        )
        self.mask_token = nn.Parameter(torch.empty(1, 1, embed_dim).normal_(std=0.02))

    def forward(self, x: torch.Tensor, mask_ratio=0.75):
        torch._assert(x.dim() == 4, f"Expected (batch_size, seq_length, hidden_dim) got {x.shape}")


        x = self._patchify(x)
        B, N, C = x.shape

        # add pos_embeddings
        x = x + self.pos_embed.repeat(B, 1, 1)

        # mask
        x, mask, ids_restore = random_masking(x, mask_ratio)

        # add registers to input
        x = torch.cat([x, self.registers.expand(x.size(0), -1, -1)], dim=1)

        # encode input tokens
        x = self.ln(self.layers(self.dropout(x)))

        # remove registers
        if self.num_registers > 0:
            x = x[:, :-self.num_registers, :]
        
        # reintroduce mask tokens
        x = torch.cat([x, self.mask_token.expand(B, N-x.size(1), -1)], dim=1)
        # reorder
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, C))

        return x, mask


class MNISTDecoder(Transformer):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        out_dim: int,
        embed_dim: int,
        input_shape: tuple[int, int],
        patch_size: int,
        num_layers: int,
        num_heads: int,
        num_registers: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__(
            in_dim=out_dim,
            embed_dim=embed_dim,
            input_shape=input_shape,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_registers=num_registers,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            norm_layer=norm_layer,
            cls_token=False,
        )
        self.embed_patches = None
        self.out_proj = nn.ConvTranspose2d(embed_dim, out_dim, kernel_size=7, stride=7, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        torch._assert(x.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {x.shape}")
        B, N, C = x.shape

        # add positional embeddings
        x = x + self.pos_embed.repeat(B, 1, 1)
        
        # add registers to input
        x = torch.cat([x, self.registers.expand(x.size(0), -1, -1)], dim=1)

        # encode input tokens
        x = self.ln(self.layers(self.dropout(x)))

        # remove registers
        if self.num_registers > 0:
            x = x[:, :-self.num_registers, :]
        
        if mask is not None:
            x = x * mask.unsqueeze(-1)

        # reshape to image and project into pixel-space
        H = int(math.sqrt(N))
        W = H
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = self.out_proj(x)

        return x

class MAE(nn.Module):
    def __init__(self, in_features, input_size=(28, 28), patch_size=7):
        super().__init__()
        self.in_features = in_features
        self.input_size = input_size
        self.patch_size = patch_size

        assert isinstance(input_size, tuple) and len(input_size) == 2
        assert input_size[0] == input_size[1], "non-square input size not supported"
        
        small = input_size[0] <= 32
        self.num_features = 256  if small else 512
        num_layers = 4 if small else 6
        num_heads = 4 if small else 8

        self.encoder = Encoder(
            in_dim=in_features,
            embed_dim=self.num_features,
            input_shape=input_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_registers=4,
            mlp_dim=self.num_features*4,
            dropout=0.1,
            attention_dropout=0.1,
            norm_layer=nn.LayerNorm,
        )

        self.decoder = MNISTDecoder(
            out_dim=in_features,
            embed_dim=self.num_features,
            input_shape=input_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_registers=4,
            mlp_dim=self.num_features*4,
            dropout=0.1,
            attention_dropout=0.1,
            norm_layer=nn.LayerNorm,
        )

    def forward(self, x):
        z, _ = self.encoder(x, mask_ratio=0.0)
        return z.mean(1)
    
    def reconstruct(self, x, mask_ratio, mask_output=False):
        z, mask = self.encoder(x, mask_ratio=mask_ratio)
        output_mask = mask if mask_output else None
        pred = self.decoder(z, output_mask)
        return pred

    def copy(self):
        model = MAE(self.in_features, self.input_size, self.patch_size).to(next(self.parameters()).device)
        model.load_state_dict(self.state_dict())
        return model

    def loss(self, img1, **_):

        with torch.autocast(device_type=img1.device.type, dtype=torch.bfloat16):
            preds = self.reconstruct(img1, 0.75, mask_output=True)
            loss = F.mse_loss(preds, img1)
        return loss