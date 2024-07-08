import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, alexnet
from rvit import RegisteredVisionTransformer
from Utils.nets import mnist_cnn_encoder, mnist_cnn_decoder

def vae_loss(recon_x, x, mu, logVar, beta=1.0):
    reconstruction_loss = F.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')
    mse = F.mse_loss(F.sigmoid(recon_x), x) * x.shape[0]
    kl_loss = -0.5 * torch.sum(1 + logVar - mu.pow(2) - logVar.exp())
    return reconstruction_loss + beta * kl_loss, mse

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
        self.z2h = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.h_dim)
        )
        # self.z2h = nn.Linear(z_dim, self.h_dim)

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

    def train_step(self, img1, img2, actions, teacher, epoch):
        assert img2 is None, 'img2 should be None for VAE.train_step()'
        assert teacher is None, 'teacher should be None for VAE.train_step()'
        assert actions is None, 'actions should be None for VAE.train_step()'
        with torch.autocast(device_type=img1.device.type, dtype=torch.bfloat16):
            x_hat, mu, logVar = self.reconstruct(img1)
            loss = vae_loss(x_hat, img1, mu, logVar)
        return loss