import torch
import torch.nn as nn
from torchvision.models import resnet18, alexnet

class SimSiam(nn.Module):
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

            # Initialise scale parameters as 0 in last BN layer of every residual block
            for n, m in self.encoder.named_modules():
                # if name contains 'bn2', set scale parameter to 0
                if 'bn2' in n:
                    nn.init.constant_(m.weight, 0)
        
        self.project = nn.Sequential(
            nn.Linear(self.num_features, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 2048, bias=False),
            nn.BatchNorm1d(2048),
        )

        self.predict = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 2048, bias=False),
        )

    def forward(self, x):
        return self.encoder(x)
    
    def copy(self):
        model = SimSiam(self.in_features, backbone=self.backbone).to(next(self.parameters()).device)
        model.load_state_dict(self.state_dict())
        return model
