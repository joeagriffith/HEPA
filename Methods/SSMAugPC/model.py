import torch
import torch.nn as nn

from torchvision.models import resnet18, alexnet
from rvit import RegisteredVisionTransformer

class SSMAugPC(nn.Module):
    def __init__(self, in_features, num_actions, backbone='resnet18'):
        super().__init__()
        self.in_features = in_features
        self.num_actions = num_actions
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
            # self.encoder.fc = nn.Flatten()
            self.encoder.fc = nn.Linear(512, 256)
            self.num_features = 256

        elif backbone == 'alexnet':
            self.encoder = alexnet()
            self.encoder.features[0] = nn.Conv2d(in_features, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.encoder.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.encoder.classifier = nn.Flatten()
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

        # NO BATCHNORM
        self.A = nn.Sequential(
            nn.Linear(self.num_features, 1024, bias=False),
            nn.ReLU(),
            nn.Linear(1024, 512, bias=False),
            nn.ReLU(),
            nn.Linear(512, self.num_features, bias=False)
        )
    
        self.B = nn.Sequential(
            nn.Linear(num_actions, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_features),
        )

        #for Mnist (-1, 1, 28, 28)
        # No BN, makes it worse
        self.C = nn.Sequential(
            nn.Unflatten(1, (self.num_features, 1, 1)),

            nn.ConvTranspose2d(self.num_features, 512, 3, 1),
            nn.ReLU(),

            nn.ConvTranspose2d(512, 256, 3, 3),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, 3, 3),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, 2, 1),
            nn.ReLU(),

            nn.Conv2d(64, 1, 3, 1, 1),
            # nn.Sigmoid(),
        )


    def forward(self, x):
        z = self.encoder(x)
        return z
    
    def predict(self, x, a=None):
        if a is None:
            a = torch.zeros(x.shape[0], self.num_actions, device=x.device)
        
        z = self.encoder(x)
        d_z_pred = self.A(z) + self.B(a)
        z_pred = z.detach() + d_z_pred
        pred = self.C(z_pred)
        return pred