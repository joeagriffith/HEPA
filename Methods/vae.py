import torch
import torch.nn as nn
import torch.nn.functional as F

from Utils.nn.nets import Encoder28, Decoder28, Decoder128, Decoder224
from Utils.nn.resnet_encoder import resnet18


def vae_loss(recon_x, x, mu, logVar, beta=1.0):
    reconstruction_loss = F.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')
    mse = F.mse_loss(F.sigmoid(recon_x), x) * x.shape[0]
    kl_loss = -0.5 * torch.sum(1 + logVar - mu.pow(2) - logVar.exp())
    return reconstruction_loss + beta * kl_loss, mse

class VAE(nn.Module):
    def __init__(self, in_features, z_dim, resolution=28):
        super().__init__()
        self.in_features = in_features
        self.num_features = z_dim

        if resolution == 28:
            self.h_dim = 256
            self.encoder = Encoder28(self.h_dim)
            self.decoder = Decoder28(self.num_features)

        elif resolution in [128, 224]:
            self.h_dim = 512
            self.encoder = resnet18((in_features, resolution, resolution))
            if resolution == 128:
                self.decoder = Decoder128(self.num_features)
            elif resolution == 224:
                self.decoder = Decoder224(self.num_features)
        
        self.mu = nn.Linear(self.h_dim, z_dim)
        self.logVar = nn.Linear(self.h_dim, z_dim)
        self.z2h = nn.Sequential(
            nn.Linear(self.num_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.h_dim)
        )

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

    def loss(self, img1, **_):

        with torch.autocast(device_type=img1.device.type, dtype=torch.bfloat16):
            x_hat, mu, logVar = self.reconstruct(img1)
            loss = vae_loss(x_hat, img1, mu, logVar)[0]
        return loss