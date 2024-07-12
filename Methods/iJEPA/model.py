import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import EncoderBlock

from Utils.nets import mnist_cnn_encoder, mnist_cnn_decoder
from Utils.functional import create_sine_cosine_embeddings, repeat_interleave_batch
from Utils.masking import random_masking, MaskGenerator, apply_masks

from typing import Callable
from functools import partial
from collections import OrderedDict
import math


class MNISTEncoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        patched_shape: tuple[int, int],
        num_layers: int,
        num_heads: int,
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
            num_patches = patched_shape[0] * patched_shape[1]
            self.pos_embedding = nn.Parameter(torch.randn(num_patches, hidden_dim) * 0.02)
        else:
            self.pos_embedding = create_sine_cosine_embeddings(patched_shape[0], patched_shape[1], hidden_dim)

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

    def forward(self, x: torch.Tensor, enc_masks=None):
        torch._assert(x.dim() == 4, f"Expected (batch_size, seq_length, hidden_dim) got {x.shape}")

        x = self._patchify(x)
        B, N, D = x.shape

        # add pos_embeddings
        x = x + self.pos_embedding

        x = apply_masks(x, enc_masks)

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


class MNISTPredictor(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        embed_dim: int,
        pred_embed_dim: int,
        patched_shape: tuple[int, int],
        num_layers: int,
        num_heads: int,
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
        self.predictor_embed = nn.Linear(embed_dim, pred_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.randn(1, 1, pred_embed_dim))

        if learnable_pos_embeddings:
            num_patches = patched_shape[0] * patched_shape[1]
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, pred_embed_dim) * 0.02)
        else:
            self.pos_embedding = create_sine_cosine_embeddings(patched_shape[0], patched_shape[1], pred_embed_dim).unsqueeze(0)
        
        # initialise registers
        self.num_registers = num_registers
        self.registers = nn.Parameter(torch.empty(1, num_registers, pred_embed_dim).normal_(std=0.02)) # copied pos_embedding init. (not optimised)

        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                pred_embed_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(pred_embed_dim)
        self.predictor_proj = nn.Linear(pred_embed_dim, embed_dim, bias=True)


    def forward(self, context: torch.Tensor, context_masks: torch.Tensor, target_masks: torch.Tensor):
        torch._assert(context.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {context.shape}")

        B = len(context) // len(context_masks)

        x = self.predictor_embed(context)

        context_pos_embed = self.pos_embedding.repeat(B, 1, 1)
        context_pos_embed = apply_masks(context_pos_embed, context_masks)
        x += context_pos_embed

        _, N_ctxt, D = context.shape


        # concat mask tokens to context
        target_pos_embed = self.pos_embedding.repeat(B, 1, 1)
        target_pos_embed = apply_masks(target_pos_embed, target_masks)
        target_pos_embed = repeat_interleave_batch(target_pos_embed, B, repeat=len(context_masks))

        target_tokens = self.mask_token.repeat(target_pos_embed.size(0), target_pos_embed.size(1), 1)
        target_tokens += target_pos_embed

        x = x.repeat(len(context_masks), 1, 1)
        x = torch.cat([x, target_tokens], dim=1)
        

        # add registers to input
        x = torch.cat([x, self.registers.expand(x.size(0), -1, -1)], dim=1)

        # fwd propagate
        x = self.ln(self.layers(self.dropout(x)))

        # remove registers
        if self.num_registers > 0:
            x = x[:, :-self.num_registers, :]
        
        # remove context
        x = x[:, N_ctxt:, :]

        # project to encoder embedding space
        x = self.predictor_proj(x)

        return x

class iJEPA(nn.Module):
    def __init__(self, in_features, input_size=(224,224), patch_size=16, learnable_pos_embeddings=True):
        super().__init__()
        self.in_features = in_features
        self.learnable_pos_embeddings = learnable_pos_embeddings
        self.backbone = 'custom'
        self.num_features = 256
        self.mask_generator = MaskGenerator(input_size=input_size, patch_size=patch_size, npred=2)
        patched_shape = (input_size[0]//patch_size, input_size[1]//patch_size)

        self.encoder = MNISTEncoder(
            in_features,
            hidden_dim=self.num_features,
            patched_shape=patched_shape,
            num_layers=6,
            num_heads=8,
            mlp_dim=self.num_features*4,
            dropout=0.1,
            attention_dropout=0.1,
            num_registers=4,
            norm_layer=nn.LayerNorm,
            learnable_pos_embeddings=learnable_pos_embeddings
        )

        self.predictor = MNISTPredictor(
            embed_dim=self.num_features,
            pred_embed_dim=self.num_features,
            patched_shape=patched_shape,
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
        z, _ = self.encoder(x)
        return z.mean(1)
    
    def predict(self, x, mask_ratio, mask_output=False):
        z, mask = self.encoder(x, mask_ratio=mask_ratio)
        output_mask = mask if mask_output else None
        pred = self.decoder(z, output_mask)
        return pred

    def copy(self):
        model = iJEPA(self.in_features, self.learnable_pos_embeddings).to(next(self.parameters()).device)
        model.load_state_dict(self.state_dict())
        return model

    def train_step(self, img1, img2, actions, teacher, epoch):
        assert img2 is None, 'img2 should be None for VAE.train_step()'
        assert actions is None, 'actions should be None for VAE.train_step()'

        enc_masks, pred_masks = self.mask_generator.sample_masks(img1.shape[0])

        def forward_target():
            with torch.no_grad():
                h = teacher(img1)
                h = F.layer_norm(h, (h.size(-1),))
                B = len(h)
                h = apply_masks(h, pred_masks)
                h = repeat_interleave_batch(h, B, repeat=len(enc_masks))
                return h
        
        def forward_context():
            z = self(img1, enc_masks)
            z = self.predict(z, enc_masks, pred_masks)
            return z

        with torch.autocast(device_type=img1.device.type, dtype=torch.bfloat16):
            target = forward_target()
            pred = forward_context()
            loss = F.smooth_l1_loss(pred, target)

        return loss