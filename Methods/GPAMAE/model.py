import torch
import torch.nn as nn
from torchvision.models.vision_transformer import EncoderBlock

from Utils.nn.nets import mnist_cnn_encoder, mnist_cnn_decoder
from Utils.functional import create_sine_cosine_embeddings
from Utils.masking import random_masking

from typing import Callable
from functools import partial
from collections import OrderedDict
import math


class MNISTEncoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        in_features: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        num_registers: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        learnable_pos_embeddings: bool = True,
    ):
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default

        self.embed_patches = nn.Conv2d(in_features, hidden_dim, kernel_size=7, stride=7, bias=False)
        
        if learnable_pos_embeddings:
            self.pos_embedding = nn.Parameter(torch.randn(16, hidden_dim) * 0.02)
        else:
            self.pos_embedding = create_sine_cosine_embeddings(4, 4, hidden_dim)

        # initialise registers
        self.num_registers = num_registers
        self.registers = nn.Parameter(torch.empty(1, num_registers, hidden_dim).normal_(std=0.02)) # copied pos_embedding init. (not optimised)

        self.mask_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

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


    def _patchify(self, x):
        # x: (B, C, H, W) -> (B, N, C)
        patches = self.embed_patches(x)
        patches = patches.flatten(2).transpose(1, 2)
        return patches

    def forward(self, x: torch.Tensor, mask_ratio=0.75):
        torch._assert(x.dim() == 4, f"Expected (batch_size, seq_length, hidden_dim) got {x.shape}")

        # tokenize input
        x = self._patchify(x)
        B, N, C = x.shape

        # add pos_embeddings
        x = x + self.pos_embedding

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


class MNISTDecoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        out_features: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        num_registers: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        learnable_pos_embeddings: bool = True
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default

        if learnable_pos_embeddings:
            self.pos_embedding = nn.Parameter(torch.randn(16, hidden_dim) * 0.02)
        else:
            self.pos_embedding = create_sine_cosine_embeddings(4, 4, hidden_dim)

        # initialise registers
        self.num_registers = num_registers
        self.registers = nn.Parameter(torch.empty(1, num_registers, hidden_dim).normal_(std=0.02)) # copied pos_embedding init. (not optimised)

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

        self.out_proj = nn.ConvTranspose2d(hidden_dim, out_features, kernel_size=7, stride=7, bias=False)

    
    def forward(self, x: torch.Tensor, a: torch.Tensor, mask: torch.Tensor=None):
        torch._assert(x.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {x.shape}")
        B, N, C = x.shape

        # add positional embeddings
        x = x + self.pos_embedding
        
        # add registers and action token to input
        x = torch.cat([x, a.unsqueeze(1), self.registers.expand(x.size(0), -1, -1)], dim=1)

        # encode input tokens
        x = self.ln(self.layers(self.dropout(x)))

        # remove registers and action token
        if self.num_registers > 0:
            x = x[:, :-(self.num_registers+1), :]

        if mask is not None:
            x = x * mask.unsqueeze(-1)
        
        # reshape to image and project into pixel-space
        H = int(math.sqrt(N))
        W = H
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = self.out_proj(x)

        return x

class GPAMAE(nn.Module):
    def __init__(self, in_features, num_actions, learnable_pos_embeddings=True):
        super().__init__()
        self.in_features = in_features
        self.learnable_pos_embeddings = learnable_pos_embeddings
        self.backbone = 'custom'
        self.num_features = 256
        self.num_actions = num_actions

        self.action_encoder = nn.Sequential(
            nn.Linear(num_actions, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_features),
        )

        self.encoder = MNISTEncoder(
            in_features,
            num_layers=6,
            num_heads=8,
            hidden_dim=self.num_features,
            mlp_dim=self.num_features*4,
            dropout=0.1,
            attention_dropout=0.1,
            num_registers=4,
            norm_layer=nn.LayerNorm,
            learnable_pos_embeddings=learnable_pos_embeddings
        )

        self.decoder = MNISTDecoder(
            in_features,
            num_layers=6,
            num_heads=8,
            hidden_dim=self.num_features,
            mlp_dim=self.num_features*4,
            dropout=0.1,
            attention_dropout=0.1,
            num_registers=4,
            norm_layer=nn.LayerNorm,
            learnable_pos_embeddings=learnable_pos_embeddings
        )


    def forward(self, x):
        z, _ = self.encoder(x, mask_ratio=0.0)
        return z.mean(1)
    

    def predict(self, x, a=None, mask_ratio=0.75, mask_output=False):
        if a is None:
            a = torch.zeros(x.shape[0], self.num_actions, device=x.device)

        z, mask = self.encoder(x, mask_ratio=mask_ratio)
        output_mask = mask if mask_output else None
        a = self.action_encoder(a)
        z = self.decoder(z, a, output_mask)
        return z


    def copy(self):
        model = GPAMAE(self.in_features, self.learnable_pos_embeddings).to(next(self.parameters()).device)
        model.load_state_dict(self.state_dict())
        return model