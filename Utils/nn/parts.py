import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import EncoderBlock

import math
from functools import partial
from typing import Callable
from collections import OrderedDict

from Utils.pos_embed import get_2d_sincos_pos_embed

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


class ConvResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.norm_1 = nn.BatchNorm2d(in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm_2 = nn.BatchNorm2d(out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, Channel, Height, Width)

        residual = x

        x = self.norm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)

        x = self.norm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residual)


class SelfAttention(nn.Module):

    def __init__(self, n_heads: int, d_embed: int, in_proj_bias: bool = True, out_proj_bias: bool = True):
        super().__init__()

        assert d_embed % n_heads == 0, f"d_embed {d_embed} must be divisible by n_heads {n_heads}"

        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

    def forward(self, x: torch.Tensor, causal_mask: bool = False) -> torch.Tensor:
        # x: (Batch_size, Seq_Len, Dim)

        input_shape = x.shape
        batch_size, seq_len, d_embed = input_shape        
        interim_shape = (batch_size, seq_len, self.n_heads, self.d_head)

        # (Batch_size, Seq_Len, Dim) -> (Batch_size, Seq_Len, 3*Dim) -> 3 x (Batch_size, Seq_Len, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1) 

        # (Batch_size, Seq_Len, Dim) -> (Batch_size, Seq_Len, n_heads, d_head) -> (Batch_size, n_heads, Seq_Len, d_head)
        q = q.view(*interim_shape).permute(0, 2, 1, 3).contiguous()
        k = k.view(*interim_shape).permute(0, 2, 1, 3).contiguous()
        v = v.view(*interim_shape).permute(0, 2, 1, 3).contiguous()

        # (Batch_size, n_heads, Seq_Len, d_head) -> (Batch_size, n_heads, Seq_Len, Seq_Len)
        # scores = q @ k.transpose(-2, -1)

        # if causal_mask:
            # mask = torch.ones_like(scores, dtype=torch.bool).triu(1)
            # scores.masked_fill_(mask, float('-inf'))
        
        # scores /= math.sqrt(self.d_head)

        # scores = scores.softmax(dim=-1)

        # Batch_size, n_heads, Seq_Len, Seq_Len) @ (Batch_size, n_heads, Seq_Len, d_head) -> (Batch_size, n_heads, Seq_Len, d_head)
        # output = scores @ v

        output = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)


        # (Batch_size, n_heads, Seq_Len, d_head) -> (Batch_size, Seq_Len, n_heads, d_head) -> (Batch_size, Seq_Len, Dim)
        output = output.permute(0, 2, 1, 3).contiguous().view(*input_shape)

        output = self.out_proj(output)

        # (Batch_size, Seq_Len, Dim)
        return output


class SelfAttentionBlock(nn.Module):

    def __init__(self, in_channels: int):
        super().__init__()
        self.layernorm = nn.LayerNorm(in_channels)
        self.attention = SelfAttention(1, in_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, Channel, Height, Width)
        residual = x

        x = self.layernorm(x)

        n, c, h, w = x.shape
        # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height*Width, Channel)
        x = x.view(n, c, h*w).permute(0, 2, 1).contiguous()

        # (Batch_Size, Height*Width, Channel) -> (Batch_Size, Height*Width, Channel)
        x = self.attention(x)

        # (Batch_Size, Height*Width, Channel) -> (Batch_Size, Channel, Height, Width)
        x = x.permute(0, 2, 1).contiguous().view(n, c, h, w)

        return x + residual

class TransformerEncoderBottleneck(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        in_shape: tuple[int, int],
        num_layers: int,
        num_heads: int,
        num_registers: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        assert embed_dim % num_heads == 0, f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"
        assert in_shape[0] == in_shape[1], "in_shape must be square"

        super().__init__()

        self.embed_dim = embed_dim
        self.in_shape = in_shape
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_registers = num_registers
        self.mlp_dim = mlp_dim
        self.norm_layer = norm_layer

        num_tokens = in_shape[0] * in_shape[1]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim), requires_grad=False)
        pos_embedding = get_2d_sincos_pos_embed(embed_dim, in_shape[0], cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embedding).float().unsqueeze(0))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.registers = nn.Parameter(torch.randn(1, num_registers, embed_dim) * 0.02)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape

        # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height*Width, Channel)
        x = x.view(n, c, h*w).permute(0, 2, 1).contiguous()
        x = x + self.pos_embed

        x = torch.cat([self.cls_token.expand(n, -1, -1), x, self.registers.expand(n, -1, -1)], dim=1)

        x = self.dropout(x)
        x = self.layers(x)
        x = self.ln(x)

        return x[:, 0, :]


class TransformerDecoderBottleneck(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        out_shape: tuple[int, int],
        num_layers: int,
        num_heads: int,
        num_registers: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        assert embed_dim % num_heads == 0, f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"
        assert out_shape[0] == out_shape[1], "in_shape must be square"

        super().__init__()

        self.embed_dim = embed_dim
        self.out_shape = out_shape
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_registers = num_registers
        self.mlp_dim = mlp_dim
        self.norm_layer = norm_layer

        self.num_tokens = out_shape[0] * out_shape[1]
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim), requires_grad=False)
        pos_embedding = get_2d_sincos_pos_embed(embed_dim, out_shape[0], cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embedding).float().unsqueeze(0))
        self.cls_token = nn.Parameter(torch.randn(1, embed_dim) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.registers = nn.Parameter(torch.randn(1, num_registers, embed_dim) * 0.02)

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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        n, c = z.shape
        z = z + self.cls_token

        x = self.mask_token.expand(n, self.num_tokens, -1)
        x = x + self.pos_embed

        x = torch.cat([x, z.unsqueeze(1), self.registers.expand(n, -1, -1)], dim=1)

        x = self.dropout(x)
        x = self.layers(x)
        x = self.ln(x)

        x = x[:, :self.num_tokens, :].view(n, c, self.out_shape[0], self.out_shape[1])
        return x

