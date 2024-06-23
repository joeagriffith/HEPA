import torch
import torch.nn as nn
from torchvision.models import resnet18, alexnet
from rvit import RegisteredVisionTransformer
from Utils.nets import mnist_cnn_encoder

class BYOL(nn.Module):
    def __init__(self, in_features, backbone='mnist_cnn'):
        super().__init__()
        self.in_features = in_features
        self.backbone = backbone

        # MNIST ONLY
        if backbone == 'vit':
            self.encoder = RegisteredVisionTransformer(
                image_size=28,
                patch_size=7,
                num_layers=6,
                num_heads=4,
                hidden_dim=256,
                num_registers=4,
                mlp_dim=1024,
            )
            self.encoder.conv_proj = nn.Conv2d(1, 256, kernel_size=7, stride=7)
            self.encoder.heads = nn.Identity()
            self.num_features = 256
        elif backbone == 'resnet18':
            self.encoder = resnet18()
            self.encoder.conv1 = nn.Conv2d(in_features, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.encoder.maxpool = nn.Identity()
            self.encoder.fc = nn.Linear(512, 256)
            self.num_features = 256
        elif backbone == 'alexnet':
            self.encoder = alexnet()
            self.encoder.features[0] = nn.Conv2d(in_features, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.encoder.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.encoder.classifier = nn.Flatten()
            self.num_features = 256
        elif backbone == 'mnist_cnn':
            self.num_features = 256
            self.encoder = mnist_cnn_encoder(self.num_features)

        self.project = nn.Sequential(
            nn.Linear(self.num_features, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256, bias=False),
            # nn.Linear(self.num_features, 1024, bias=False),
            # nn.ReLU(),
            # nn.Linear(1024, 512, bias=False),
            # nn.ReLU(),
            # nn.Linear(512, self.num_features, bias=False)
        )

        self.predict = nn.Sequential(
            nn.Linear(256, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256, bias=False),
            # nn.Linear(self.num_features, 1024, bias=False),
            # nn.ReLU(),
            # nn.Linear(1024, 512, bias=False),
            # nn.ReLU(),
            # nn.Linear(512, self.num_features, bias=False)
        )

    def forward(self, x):
        return self.encoder(x)
    
    def copy(self):
        model = BYOL(self.in_features, backbone=self.backbone).to(next(self.parameters()).device)
        model.load_state_dict(self.state_dict())
        return model