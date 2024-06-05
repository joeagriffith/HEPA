import torch
import torch.nn as nn

from torchvision.models import resnet18, alexnet
from rvit import RegisteredVisionTransformer
from Utils.nets import mnist_cnn_encoder, mnist_cnn_decoder


class VAE(nn.Module):
    def __init__(self, in_features, z_dim, backbone='mnist_cnn'):
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
            self.h_dim = 256

        elif backbone == 'resnet18':
            self.encoder = resnet18()
            self.encoder.conv1 = nn.Conv2d(in_features, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.encoder.maxpool = nn.Identity()
            self.encoder.fc = nn.Flatten() # Actually performs better without this line
            self.h_dim = 512

        elif backbone == 'alexnet':
            self.encoder = alexnet()
            self.encoder.features[0] = nn.Conv2d(in_features, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.encoder.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.encoder.classifier = nn.Flatten()
            self.h_dim = 256

        elif backbone == 'mnist_cnn':
            self.h_dim = 256
            self.encoder = mnist_cnn_encoder(self.h_dim)
        
        self.num_features = z_dim
        
        self.mu = nn.Linear(self.h_dim, z_dim)
        self.logVar = nn.Linear(self.h_dim, z_dim)
        self.z2h = nn.Linear(z_dim, self.h_dim)

        #for Mnist (-1, 1, 28, 28)
        self.decoder = mnist_cnn_decoder(self.h_dim)

    def forward(self, x):
        h = self.encoder(x)
        return self.mu(h)
    
    def reparameterise(self, mu, logVar):
        std = torch.exp(0.5 * logVar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def reconstruct(self, x):
        h = self.encoder(x)
        mu, logVar = self.mu(h), self.logVar(h)
        z = self.reparameterise(mu, logVar)
        h = self.z2h(z)
        x_hat = self.decoder(h) 
        return x_hat, mu, logVar