import torch
import torch.nn as nn

from torchvision.models import resnet18, alexnet
from rvit import RegisteredVisionTransformer

class LAugPC2(nn.Module):
    def __init__(self, in_features, num_actions):
        super().__init__()
        self.in_features = in_features
        self.num_actions = num_actions
        self.backbone = 'vit'

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

        self.action_encoder = nn.Sequential(
            nn.Linear(num_actions, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.transition = nn.Sequential(
            nn.Linear(self.num_features + 128, 4096, bias=False),
            nn.ReLU(),
            nn.Linear(4096, 2048, bias=False),
            nn.ReLU(),
            nn.Linear(2048, self.num_features, bias=False)
        )

        self.predictor = RegisteredVisionTransformer(
            image_size=28,
            patch_size=7,
            num_layers=6,
            num_heads=4,
            hidden_dim=256,
            num_registers=4,
            mlp_dim=1024,
        )

    def vit_forward(self, x, vit, ln=True, stop_layer=None):

        # add positional encodings
        x = x + vit.encoder.pos_embedding

        # add registers, without positional embedding
        x = torch.cat([x, vit.encoder.registers.expand(x.shape[0], -1, -1)], dim=1)
        
        # forward through dropout
        x = vit.encoder.dropout(x)
        # forward partway through encoder blocks
        for i, layer in enumerate(vit.encoder.layers):
            if i == stop_layer:
                break
            x = layer(x)
        
        # forward through norm
        if ln:
            x = vit.encoder.ln(x)

        # remove registers
        if vit.encoder.num_registers > 0:
            x = x[:, :-vit.encoder.num_registers, :]
        
        return x

    def encode(self, x, ln=True, cls=True, stop_layer=None):
        if stop_layer is not None:
            assert stop_layer <= len(self.encoder.encoder.layers), 'Requested more layers than available'
        b_size = x.shape[0]

        # tokenise
        x = self.encoder.conv_proj(x)
        x = x.reshape(b_size, self.num_features, -1)
        x = x.permute(0, 2, 1)

        # add class token
        cls_token = self.encoder.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # forward through encoder
        x = self.vit_forward(x, self.encoder, ln=ln, stop_layer=stop_layer)

        if not cls:
            x = x[:,1:]

        return x

    def predict(self, x, a=None, ln=True, cls=True):
        if a is None:
            a = torch.zeros(x.shape[0], self.num_actions, device=x.device)
        
        z = self.encode(x)
        a = self.action_encoder(a)
        cls_pred = self.transition(torch.cat([z[:,0], a], dim=1))
        z[:,0] = cls_pred

        z = self.vit_forward(z, self.predictor, ln=ln)

        if not cls:
            z = z[:,1:]

        return z

    def forward(self, x):
        z = self.encoder(x)
        return z
    
    def copy(self):
        model = LAugPC2(self.in_features, self.num_actions)
        model.load_state_dict(self.state_dict())
        return model