import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2.functional as F_v2
from Utils.nn.resnet_encoder import resnet18
from Utils.nn.nets import Encoder28

class BYOPL(nn.Module):
    def __init__(self, in_features, num_actions, resolution=28, p=0.25, consider_actions=True):
        super().__init__()
        self.in_features = in_features
        self.num_actions = num_actions
        self.resolution = resolution
        self.p = p
        self.consider_actions = consider_actions

        if resolution == 28:
            self.num_features = 256
            self.encoder = Encoder28(self.num_features)
            self.project = nn.Sequential(
                nn.Linear(self.num_features, 1024, bias=False),
                # nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(1024, self.num_features, bias=False),
            )

        elif resolution in [128, 224]:
            self.encoder = resnet18((in_features, resolution, resolution))
            self.num_features = 512
            self.project = nn.Identity()

        self.action_encoder = nn.Sequential(
            nn.Linear(num_actions, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        

        self.predict = nn.Sequential(
            nn.Linear(self.num_features + 128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_features)
        )

    def forward(self, x):
        return self.encoder(x)
    
    def copy(self):
        model = BYOPL(self.in_features, self.num_actions, resolution=self.resolution, p=self.p, consider_actions=self.consider_actions).to(next(self.parameters()).device)
        model.load_state_dict(self.state_dict())
        return model

    def interact(self, images, groups=8):
        # Sample Action
        if images.shape[0] < 8:
            groups = images.shape[0]
        # assert images.shape[0] % groups == 0, f'images.shape[0]={images.shape[0]} must be divisible by groups={groups}'
        n_per = images.shape[0] // groups
        images_aug = torch.zeros_like(images)
        actions = torch.empty((images.shape[0], 5), device=images.device)
        for i in range(groups):
            act_p = torch.rand(5) # whether to apply each augmentation
            angle = torch.rand(1).item() * 360 - 180 if act_p[0] < self.p else 0
            translate_x = torch.randint(-8, 9, (1,)).item() if act_p[1] < self.p else 0
            translate_y = torch.randint(-8, 9, (1,)).item() if act_p[2] < self.p else 0
            scale = torch.rand(1).item() * 0.5 + 0.75 if act_p[3] < self.p else 1.0
            shear = torch.rand(1).item() * 50 - 25 if act_p[4] < self.p else 0
            lo, hi = i*n_per, min(images.shape[0], (i+1)*n_per)
            images_aug[lo:hi] = F_v2.affine(images[lo:hi], angle=angle, translate=(translate_x, translate_y), scale=scale, shear=shear)
            actions[lo:hi] = torch.tensor([angle/180, translate_x/8, translate_y/8, (scale-1.0)/0.25, shear/25], dtype=torch.float32, device=images.device).unsqueeze(0).repeat(hi-lo, 1)

        return images_aug, actions

    def loss(self, img1, img2, actions, teacher, **_):
        if img2 is None:
            img2, actions = self.interact(img1)
        
        if not self.consider_actions:
            actions *= 0.0

        with torch.autocast(device_type=img1.device.type, dtype=torch.bfloat16):
            with torch.no_grad():
                z_target = teacher(img2)
                z_target = teacher.project(z_target)
                z_target = F.normalize(z_target, dim=-1)

            z_x = self(img1)
            z_a = self.action_encoder(actions)
            z = torch.cat([z_x, z_a], dim=-1)
            z_pred = self.predict(z)
            z_pred = self.project(z_pred)
            z_pred = F.normalize(z_pred, dim=-1)

            loss = F.mse_loss(z_pred, z_target)

        return loss