import torch
import torch.nn as nn
from torchvision.models import resnet18, alexnet
from torchvision.models.vision_transformer import VisionTransformer

class DINO(nn.Module):
    def __init__(self, in_features, backbone='alexnet'):
        super().__init__()
        self.in_features = in_features
        self.backbone = backbone

        if backbone == 'resnet18':
            self.encoder = resnet18()
            self.encoder.conv1 = nn.Conv2d(in_features, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.encoder.maxpool = nn.Identity()
            self.encoder.fc = nn.Flatten()
            self.num_features = 512
        elif backbone == 'alexnet':
            self.encoder = alexnet()
            self.encoder.features[0] = nn.Conv2d(in_features, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.encoder.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.encoder.classifier = nn.Flatten()
            self.num_features = 256
        elif backbone == 'vit':
            self.encoder = VisionTransformer(
                image_size=28,
                patch_size=7,
                num_layers=6,
                num_heads=6,
                hidden_dim=256,
                mlp_dim=512,
            )
            self.encoder.conv_proj = nn.Conv2d(in_features, 256, kernel_size=(7, 7), stride=(7, 7), padding=(0, 0))
            self.encoder.heads = nn.Identity()
            self.num_features = 256
        elif backbone == 'mnist_cnn':
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),

                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                nn.Conv2d(64, 128, kernel_size=3, stride=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),

                nn.Conv2d(128, 256, kernel_size=3, stride=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),

                nn.Conv2d(256, 256, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            self.num_features = 256
        else:
            raise ValueError(f'backbone must be one of ["resnet18", "alexnet", "vit", "mnist_cnn"], got {backbone}')


        self.project = nn.Sequential(
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

    def forward(self, x):
        return self.encoder(x)
    
    def copy(self):
        model = DINO(self.in_features, backbone=self.backbone).to(next(self.parameters()).device)
        model.load_state_dict(self.state_dict())
        return model
