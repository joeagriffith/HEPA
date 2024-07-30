import torch
import torch.nn as nn

from torchvision.models import resnet18, alexnet
from Utils.nn.nets import mnist_cnn_encoder, mnist_cnn_decoder

class Supervised(nn.Module):
    def __init__(self, in_features, backbone='mnist_cnn', resolution=28):
        super().__init__()
        self.in_features = in_features
        self.backbone = backbone

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
            self.encoder = resnet18()
            self.encoder.conv1 = nn.Conv2d(in_features, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.encoder.maxpool = nn.Identity()
            self.encoder.fc = nn.Linear(512, 256)
            self.num_features = 256

        elif backbone == 'alexnet':
            self.encoder = alexnet()
            self.encoder.features[0] = nn.Conv2d(in_features, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.encoder.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.encoder.classifier = nn.Flatten()
            self.num_features = 256

        elif backbone == 'mnist_cnn':
            self.num_features = 256
            self.encoder = mnist_cnn_encoder(self.num_features)
        
        self.classifier = nn.Linear(self.num_features, 10, bias=False)
        self.decoder = mnist_cnn_decoder(self.num_features)

    def forward(self, x):
        return self.encoder(x)
    
    def predict(self, x):
        z = self.encoder(x)
        pred = self.classifier(z)
        return pred