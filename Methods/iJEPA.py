import torch
import torch.nn as nn
import torch.nn.functional as F

from Utils.functional import repeat_interleave_batch
from Utils.masking import MaskGenerator, apply_masks
from Utils.nn.transformer import Transformer

from typing import Callable, List
from functools import partial


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

    def forward(self, x: torch.Tensor, enc_masks=None):
        torch._assert(x.dim() == 4, f"Expected (batch_size, seq_length, hidden_dim) got {x.shape}")

        x = self._patchify(x)
        B, N, D = x.shape

        # add pos_embeddings
        x = x + self.pos_embed.repeat(B, 1, 1)

        if enc_masks is not None:
            x = apply_masks(x, enc_masks)

        # add registers to input
        x = torch.cat([x, self.registers.expand(x.size(0), -1, -1)], dim=1)

        # encode input tokens
        x = self.ln(self.layers(self.dropout(x)))

        if self.num_registers > 0:
            x = x[:, :-self.num_registers, :] # remove registers
        
        return x
        


class Predictor(Transformer):
    """Transformer Model Encoder for sequence to sequence translation."""

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
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.embed_patches = None
        self.in_proj = nn.Linear(in_dim, embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.out_proj = nn.Linear(embed_dim, in_dim, bias=True)

    def forward(self, context: torch.Tensor, context_masks: List[torch.Tensor], target_masks: List[torch.Tensor]):
        B = len(context) // len(context_masks)
        assert len(context.shape) == 3, f"Expected (batch_size, seq_length, hidden_dim) got {context.shape}"

        # build context tokens
        context = self.in_proj(context)
        context_pos_embed = self.pos_embed.repeat(B, 1, 1)
        context_pos_embed = apply_masks(context_pos_embed, context_masks)
        context += context_pos_embed
        context = context.repeat(len(target_masks), 1, 1)
        _, N_ctxt, D = context.shape

        # build target tokens
        target_pos_embed = self.pos_embed.repeat(B, 1, 1)
        target_pos_embed = apply_masks(target_pos_embed, target_masks)
        target_pos_embed = repeat_interleave_batch(target_pos_embed, B, repeat=len(context_masks))
        targets = self.mask_token.repeat(target_pos_embed.size(0), target_pos_embed.size(1), 1)
        targets += target_pos_embed

        # build registers
        registers = self.registers.expand(targets.size(0), -1, -1)

        # build input tokens
        x = torch.cat([context, registers, targets], dim=1)

        # fwd propagate
        x = self.ln(self.layers(self.dropout(x)))

        # remove context and registers
        x = x[:, N_ctxt + self.num_registers:, :]
        
        # project to encoder embedding space
        x = self.out_proj(x)

        return x

class iJEPA(nn.Module):
    def __init__(self, in_features, input_size=(28,28), patch_size=7, min_keep=1):
        super().__init__()
        self.in_features = in_features
        self.input_size = input_size
        self.patch_size = patch_size

        assert isinstance(input_size, tuple)
        assert input_size[0] == input_size[1], "non-square input size not supported."
        self.mask_generator = MaskGenerator(input_size=input_size[0]//patch_size, npred=1, min_keep=min_keep, device='cpu')

        small = input_size[0] <= 32

        self.num_features = 256 if small else 512
        num_layers = 4 if small else 6
        num_heads = 4 if small else 8

        self.encoder = Encoder(
            in_features,
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

        self.predictor = Predictor(
            in_dim=self.num_features,
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
    
    def to(self, device):
        self.encoder.to(device)
        if self.predictor is not None:
            self.mask_generator.to(device)
            self.predictor.to(device)
        return self

    def copy(self):
        model = iJEPA(self.in_features, self.input_size, self.patch_size)
        model.load_state_dict(self.state_dict())
        # remove predictor as not used by teacher
        model.predictor = None
        model.mask_generator = None
        model = model.to(next(self.parameters()).device)
        return model

    def forward(self, x, reduction='mean'):
        z = self.encoder(x)
        assert reduction in ['mean', 'none']
        if reduction == 'mean':
            return z.mean(1)
        else:
            return z

    def loss(self, img1, teacher, **_):
        
        context_masks, target_masks = self.mask_generator.sample_masks(img1.shape[0])

        def forward_target():
            with torch.no_grad():
                h = teacher(img1, reduction='none')
                B = len(h)
                h = apply_masks(h, target_masks)
                h = repeat_interleave_batch(h, B, repeat=len(context_masks))
                return h
        
        def forward_context():
            z = self.encoder(img1, context_masks)
            z = self.predictor(z, context_masks, target_masks)
            return z

        with torch.autocast(device_type=img1.device.type, dtype=torch.bfloat16):
            target = forward_target()
            pred = forward_context()
            loss = F.smooth_l1_loss(pred, target)

        return loss