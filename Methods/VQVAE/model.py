import torch
import torch.nn as nn

from torchvision.models import resnet18, alexnet
from Methods.VQVAE.vqvae import VQVAE

class VAE(nn.Module):
    def __init__(self, in_features, backbone='resnet18'):
        super().__init__()
        self.in_features = in_features
        self.backbone = backbone

        if backbone == 'resnet18':
            encoder = resnet18()
            encoder.conv1 = nn.Conv2d(in_features, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            encoder.maxpool = nn.Identity()
            encoder.fc = nn.Unflatten(1, (512, 1, 1))
            self.num_features = 512

        elif backbone == 'alexnet':
            encoder = alexnet()
            encoder.features[0] = nn.Conv2d(in_features, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            encoder.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            encoder.classifier = nn.Unflatten(1, (256, 1, 1))
            self.num_features = 256

        # self.vqvae = VQVAE(
        #     in_channels=1,
        #     num_hiddens=self.num_features,
        #     num_downsampling_layers=4,
        #     num_residual_layers=2,
        #     num_residual_hiddens=64,
        #     embedding_dim=256,
        #     num_embeddings=512,
        #     use_ema=True,
        #     decay=0.99,
        #     epsilon=1e-5,
        # )

        self.vqvae = VQVAE(
            in_channels=1,
            num_hiddens=self.num_features,
            num_downsampling_layers=5,
            num_residual_layers=2,
            num_residual_hiddens=32,
            embedding_dim=256,
            num_embeddings=512,
            use_ema=True,
            decay=0.99,
            epsilon=1e-5,
        )

        # self.vqvae.encoder = encoder
    
        # #for Mnist (-1, 1, 28, 28)
        # self.vqvae.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(self.num_features, 512, 3, 1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(512, 256, 3, 3),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(256, 128, 3, 3),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(128, 1, 2, 1),
        # )

    def forward(self, x):
        # pad x to 32x32
        x = nn.functional.pad(x, (2, 2, 2, 2), mode='constant', value=0)

        (z, _, _, _) = self.vqvae.quantize(x)
        return z.flatten(1)
    
    def reconstruct(self, x):
        return self.vqvae(x)