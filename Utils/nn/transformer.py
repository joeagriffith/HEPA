import torch
import torch.nn as nn
from torchvision.models.vision_transformer import MLPBlock, EncoderBlock
from torchvision.ops import MLP
from flash_attn import flash_attn_qkvpacked_func

from typing import Callable
from functools import partial
from collections import OrderedDict
from flash_attn.modules.mha import MHA

from Utils.pos_embed import get_2d_sincos_pos_embed, interpolate_pos_embedding

class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        # self.self_attention = MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.self_attention = MHA(hidden_dim, num_heads, dropout=attention_dropout, use_flash_attn=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        # x, _ = self.self_attention(x, need_weights=False)
        x = self.self_attention(x)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y

class Transformer(nn.Module):

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
        cls_token: bool = True,
    ):
        assert embed_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default

        self.embed_patches = nn.Conv2d(in_dim, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        
        patched_shape = (input_shape[0] // patch_size, input_shape[1] // patch_size)
        num_patches = patched_shape[0] * patched_shape[1]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)
        pos_embedding = get_2d_sincos_pos_embed(embed_dim, patched_shape[0], cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embedding).float().unsqueeze(0))

        # initialise registers
        self.num_registers = num_registers
        self.registers = nn.Parameter(torch.empty(1, num_registers, embed_dim).normal_(std=0.02)) # copied pos_embedding init. (not optimised)
        if cls_token:
            self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim).normal_(std=0.02))
        else:
            self.cls_token = None

        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                embed_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(embed_dim)

    def _patchify(self, x):
        # x: (B, C, H, W) -> (B, N, C)
        patches = self.embed_patches(x)
        patches = patches.flatten(2).transpose(1, 2)
        return patches

    def forward(self, x: torch.Tensor):
        torch._assert(x.dim() == 4, f"Expected (batch_size, seq_length, hidden_dim) got {x.shape}")

        x = self._patchify(x)
        B, N, D = x.shape

        # add pos_embeddings
        pos_embed = interpolate_pos_embedding(x, self.pos_embed)
        x = x + pos_embed

        # add registers to input
        x = torch.cat([x, self.registers.expand(x.size(0), -1, -1)], dim=1)

        # add cls_token to input
        if self.cls_token is not None:
            x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1)

        # encode input tokens
        x = self.ln(self.layers(self.dropout(x)))

        # remove registers
        if self.num_registers > 0:
            x = x[:, :-self.num_registers, :]
        
        if self.cls_token is not None:
            return x[:, 0, :]
        else:
            return x