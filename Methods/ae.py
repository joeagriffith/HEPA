import torch
import torch.nn as nn
import torch.nn.functional as F

from Utils.nn.nets import Encoder28, Decoder28, Decoder128, Decoder224
from Utils.nn.resnet_encoder import resnet18

class AE(nn.Module):
    def __init__(self, in_features, resolution=28):
        super().__init__()
        self.in_features = in_features

        if resolution == 28:
            self.num_features = 256
            self.encoder = Encoder28(self.num_features)

        elif resolution in [128, 224]:
            self.encoder = resnet18((in_features, resolution, resolution))
            self.num_features = 512
                
        else:
            raise ValueError(f'Resolution {resolution} not supported')
            
        self.pre_decode = nn.Sequential(
            nn.Linear(self.num_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_features)
        )

        #for Mnist (-1, 1, 28, 28)
        # No BN, makes it worse
        dec_nets = [self.pre_decode]
        if resolution == 28:
            dec_nets.append(Decoder28(self.num_features))
        elif resolution == 128:
            dec_nets.append(Decoder128(in_features, self.num_features))
        elif resolution == 224:
            dec_nets.append(Decoder224(self.num_features))
        self.decoder = nn.Sequential(*dec_nets)
    
    def forward(self, x):
        z = self.encoder(x)
        return z
    
    def reconstruct(self, x):
        z = self.encoder(x)
        pred = self.decoder(z)
        return pred

    def loss(self, img1, **_):

        with torch.autocast(device_type=img1.device.type, dtype=torch.bfloat16):
            preds = self.reconstruct(img1)
            loss = F.mse_loss(preds, img1)
        return loss