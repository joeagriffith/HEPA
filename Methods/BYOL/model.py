import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import alexnet
from Utils.nn.nets import mnist_cnn_encoder
from Utils.nn.resnet_encoder import resnet18

class BYOL(nn.Module):
    def __init__(self, in_features, backbone='mnist_cnn', resolution=28):
        super().__init__()
        self.in_features = in_features
        self.backbone = backbone
        self.resolution = resolution

        # # MNIST ONLY
        # if backbone == 'vit':
        #     self.encoder = RegisteredVisionTransformer(
        #         image_size=28,
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

        self.project = nn.Sequential(
            nn.Linear(self.num_features, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, self.num_features, bias=False),
            # nn.Linear(self.num_features, 1024, bias=False),
            # nn.ReLU(),
            # nn.Linear(1024, 512, bias=False),
            # nn.ReLU(),
            # nn.Linear(512, self.num_features, bias=False)
        )

        self.predict = nn.Sequential(
            nn.Linear(self.num_features, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, self.num_features, bias=False),
            # nn.Linear(self.num_features, 1024, bias=False),
            # nn.ReLU(),
            # nn.Linear(1024, 512, bias=False),
            # nn.ReLU(),
            # nn.Linear(512, self.num_features, bias=False)
        )

    def forward(self, x):
        return self.encoder(x)
    
    def copy(self):
        model = BYOL(self.in_features, backbone=self.backbone, resolution=self.resolution).to(next(self.parameters()).device)
        model.load_state_dict(self.state_dict())
        return model

    def train_step(self, img1, img2, actions, teacher, epoch):
        assert actions is None, 'actions should be None for AE.train_step()'
        with torch.autocast(device_type=img1.device.type, dtype=torch.bfloat16):
            with torch.no_grad():
                y1_t, y2_t = teacher(img1), teacher(img2)
                z1_t, z2_t = teacher.project(y1_t), teacher.project(y2_t)
                z1_t, z2_t = F.normalize(z1_t, dim=-1), F.normalize(z2_t, dim=-1)

            y1_o, y2_o = self(img1), self(img2)
            z1_o, z2_o = self.project(y1_o), self.project(y2_o)
            p1_o, p2_o = self.predict(z1_o), self.predict(z2_o)
            p1_o, p2_o = F.normalize(p1_o, dim=-1), F.normalize(p2_o, dim=-1)

            loss = 0.5 * (F.mse_loss(p1_o, z2_t, reduction='none').sum(-1).mean() + F.mse_loss(p2_o, z1_t, reduction='none').sum(-1).mean())

        return loss


