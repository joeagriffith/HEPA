import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import alexnet
from Utils.nn.nets import mnist_cnn_encoder, mnist_cnn_decoder, Decoder128, Decoder224
from Utils.nn.parts import TransformerEncoderBottleneck
from Utils.nn.resnet_encoder import resnet18

class iGPA(nn.Module):
    def __init__(self, in_features, num_actions, stop_at=0, backbone='mnist_cnn', resolution=28):
        super().__init__()
        self.in_features = in_features
        self.num_actions = num_actions
        self.stop_at = stop_at # where to perform prediction, 0 = observation space, -1 = latent space
        self.backbone = backbone
        self.resolution = resolution

        # MNIST ONLY
        # if backbone == 'vit':
        #     self.encoder = RegisteredVisionTransformer(
        #         image_size=resolution,
        #         patch_size=7,
        #         num_layers=6,
        #         num_heads=4,
        #         hidden_dim=256,
        #         num_registers=4,
        #         mlp_dim=1024,
        #     )
        #     self.encoder.conv_proj = nn.Conv2d(1, 256, kernel_size=7, stride=7)
        #     self.encoder.heads = nn.Identity()
        #     self.num_features = 256

        if backbone == 'resnet18':
            self.encoder = resnet18((in_features, resolution, resolution))
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
    
        self.action_encoder = nn.Sequential(
            nn.Linear(num_actions, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # NO BATCHNORM
        self.transition = nn.Sequential(
            nn.Linear(self.num_features + 128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_features)
        )

        #for Mnist (-1, 1, 28, 28)
        if resolution == 28:
            self.decoder = mnist_cnn_decoder(self.num_features)
        elif resolution == 128:
            self.decoder = Decoder128(in_features, self.num_features)
        elif resolution == 224:
            self.decoder = Decoder224(self.num_features)

    def forward(self, x, stop_at=-1):
        if stop_at == 0:
            return x
        elif stop_at == -1:
            return self.encoder(x)
        else:
            raise NotImplementedError(f'stop_at={stop_at} not implemented')
    
    def predict(self, x, a=None, stop_at=None):
        if a is None:
            a = torch.zeros(x.shape[0], self.num_actions, device=x.device)
        else:
            a = torch.zeros_like(a)
        
        z = self.encoder(x)
        a = self.action_encoder(a)
        z_pred = self.transition(torch.cat([z, a], dim=1))
        if stop_at == -1:
            pred = z_pred
        elif stop_at == 0:
            pred = self.decoder(z_pred)
        else:
            raise NotImplementedError(f'stop_at={stop_at} not implemented')
        return pred
    
    def copy(self):
        model = iGPA(self.in_features, self.num_actions, self.stop_at, self.backbone, self.resolution).to(next(self.parameters()).device)
        model.load_state_dict(self.state_dict())
        return model

    def train_step(self, img1, img2, actions, teacher, epoch):
        with torch.autocast(device_type=img1.device.type, dtype=torch.bfloat16):
            with torch.no_grad():
                targets = teacher(img2, stop_at=self.stop_at)
            preds = self.predict(img1, actions, stop_at=self.stop_at)
            loss = F.mse_loss(preds, targets)
        return loss