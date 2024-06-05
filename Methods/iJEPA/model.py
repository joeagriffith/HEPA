import torch
import torch.nn as nn
from torchvision.models.vision_transformer import VisionTransformer

class iJEPA(nn.Module):
    def __init__(self, in_features, backbone='alexnet'):
        super().__init__()
        self.in_features = in_features
        self.backbone = backbone

        if backbone == 'vit':
            self.vit = VisionTransformer(
                image_size=96,
                patch_size=16,
                num_layers=6,
                num_heads=6,
                hidden_dim=256,
                mlp_dim=512,
            )
            self.vit.conv_proj = nn.Conv2d(in_features, 256, kernel_size=(7, 7), stride=(7, 7), padding=(0, 0))
            self.vit.heads = nn.Linear(256, 256)
            self.num_features = 256
            self.num_patches = 16
        else:
            raise ValueError(f'backbone must be one of ["vit"], got {backbone}')

        self.predict = nn.Sequential(
            # MLP
            nn.Linear(self.num_features, 1024, bias=False),
            nn.GELU(),
            nn.Linear(1024, 1024, bias=False),
            nn.GELU(),
            nn.Linear(1024, 1024, bias=False),

            # LayerNorm
            nn.LayerNorm(1024, elementwise_affine=False),

            # Weight Normalised Linear
            nn.utils.weight_norm(nn.Linear(1024, self.num_features, bias=False)),
        )
    
    def encode(self, x, mask=None):
        tokens = self.vit.conv_proj(x)
        tokens = tokens.reshape(-1, tokens.shape[1], tokens.shape[2] * tokens.shape[3]).permute(0, 2, 1)

        # Apply Mask (if provided)
        # Mask is a boolean tensor of shape (batch_size, num_patches)
        if mask is not None:
            tokens = tokens * mask[:, :, None]

        batch_class_token = self.vit.class_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat([batch_class_token, tokens], dim=1)

        return self.vit.encoder(tokens)

    
    def forward(self, x):
        return self.encoder(x)
    
    def copy(self):
        model = iJEPA(self.in_features, backbone=self.backbone).to(next(self.parameters()).device)
        model.load_state_dict(self.state_dict())
        return model
