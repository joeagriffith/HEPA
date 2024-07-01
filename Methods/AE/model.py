import torch
import torch.nn as nn

from torchvision.models import resnet18, alexnet
from rvit import RegisteredVisionTransformer
from Utils.nets import mnist_cnn_encoder, mnist_cnn_decoder

class AE(nn.Module):
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
            self.encoder.fc = nn.Flatten() # Actually performs better without this line
            self.num_features = 512

        elif backbone == 'alexnet':
            self.encoder = alexnet()
            self.encoder.features[0] = nn.Conv2d(in_features, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.encoder.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.encoder.classifier = nn.Flatten()
            self.num_features = 256

        elif backbone == 'mnist_cnn':
            self.num_features = 256
            self.encoder = mnist_cnn_encoder(self.num_features)
        
        elif backbone == 'none':
            self.num_features = 784
            self.encoder = nn.Flatten()

        self.pre_decode = nn.Sequential(
            nn.Linear(self.num_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_features)
        )

        #for Mnist (-1, 1, 28, 28)
        # No BN, makes it worse
        self.decoder = nn.Sequential(
            self.pre_decode,
            mnist_cnn_decoder(self.num_features),
        )
    
    def forward(self, x):
        z = self.encoder(x)
        return z
    
    def reconstruct(self, x):
        z = self.encoder(x)
        pred = self.decoder(z)
        return pred