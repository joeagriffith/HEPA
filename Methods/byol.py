import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from Utils.nn.resnet_encoder import resnet18
from Utils.nn.nets import Encoder28

class BYOL(nn.Module):
    def __init__(self, in_features, resolution=28):
        super().__init__()
        self.in_features = in_features
        self.resolution = resolution

        if resolution == 28:
            self.num_features = 256
            self.encoder = Encoder28(self.num_features)

        elif resolution in [128, 224]:
            self.encoder = resnet18((in_features, resolution, resolution))
            self.num_features = 512

        self.project = nn.Sequential(
            nn.Linear(self.num_features, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, self.num_features, bias=False),
        )

        self.predict = nn.Sequential(
            nn.Linear(self.num_features, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, self.num_features, bias=False),
        )

    def forward(self, x):
        return self.encoder(x)
    
    def copy(self):
        model = BYOL(self.in_features, resolution=self.resolution).to(next(self.parameters()).device)
        model.load_state_dict(self.state_dict())
        return model
    
    def transform(self, images):
        B, C, H, W = images.shape

        new_H = round(H*0.70)
        t = transforms.Compose([
            transforms.RandomCrop(new_H),
            transforms.Resize(H, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.75, 1.25), shear=25),
        ])
        return t(images)

    def loss(self, img1, teacher, **_):

        aug1 = self.transform(img1)
        aug2 = self.transform(img1)

        with torch.autocast(device_type=img1.device.type, dtype=torch.bfloat16):
            with torch.no_grad():
                y1_t, y2_t = teacher(aug1), teacher(aug2)
                z1_t, z2_t = teacher.project(y1_t), teacher.project(y2_t)
                z1_t, z2_t = F.normalize(z1_t, dim=-1), F.normalize(z2_t, dim=-1)

            y1_o, y2_o = self(aug1), self(aug2)
            z1_o, z2_o = self.project(y1_o), self.project(y2_o)
            p1_o, p2_o = self.predict(z1_o), self.predict(z2_o)
            p1_o, p2_o = F.normalize(p1_o, dim=-1), F.normalize(p2_o, dim=-1)

            loss = 0.5 * (F.mse_loss(p1_o, z2_t) + F.mse_loss(p2_o, z1_t))

        return loss