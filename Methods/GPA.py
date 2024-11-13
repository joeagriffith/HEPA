import torch
import torch.nn as nn
import torch.nn.functional as F

from Utils.nn.nets import Encoder28, Decoder1, Decoder5, Decoder28, Decoder128, Decoder224, AudioEncoder
import torchvision.transforms.v2.functional as F_v2
from Utils.nn.resnet_encoder import resnet18, resnet34
from Utils.nn.conv_mixer import ConvMixer

class GPA(nn.Module):
    def __init__(self, in_features, num_actions, stop_at=0, resolution=28, p=0.25, consider_actions=True):
        super().__init__()
        self.in_features = in_features
        self.num_actions = num_actions
        self.stop_at = stop_at # where to perform prediction, 0 = observation space, -1 = latent space
        self.resolution = resolution
        self.p = p
        self.consider_actions = consider_actions

        if resolution == 28:
            self.num_features = 256
            self.encoder = Encoder28(self.num_features)
            if stop_at == 0:
                self.decoder = Decoder28(self.num_features, out_features=1)
            elif stop_at == 3:
                self.decoder = Decoder5(self.num_features, out_features=128)
            elif stop_at == -1:
                self.decoder = Decoder1(self.num_features)
            else:
                raise NotImplementedError(f'stop_at={stop_at} not implemented for iGPA')

        elif resolution in [128, 224]:
            self.encoder = resnet18((in_features, resolution, resolution))
            # self.encoder = resnet34((in_features, resolution, resolution))
            # self.encoder = ConvMixer(dim=512, depth=12)
            self.num_features = 512
            if resolution == 128:
                self.decoder = Decoder128(in_features, self.num_features)
            elif resolution == 224:
                self.decoder = Decoder224(self.num_features)
        
        elif resolution == 1: # VoxCeleb1
            self.encoder = AudioEncoder(self.num_features)
        
        else:
            raise NotImplementedError(f'resolution={resolution} not implemented for GPA')

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

    # def interact(self, images, groups=8):
    #     N, C, H, W = images.shape
    #     crop_size = 14
    #     num_per_group = N // groups
    #     img1 = torch.zeros((N, C, crop_size, crop_size), device=images.device)
    #     img2 = torch.zeros((N, C, crop_size, crop_size), device=images.device)
    #     actions = torch.zeros((N, 2), device=images.device)
    #     for g in range(groups):
    #         start_x = torch.randint(0, W-crop_size, (2,))
    #         start_y = torch.randint(0, H-crop_size, (2,))
    #         img1[g*num_per_group:(g+1)*num_per_group] = images[g*num_per_group:(g+1)*num_per_group, :, start_y[0]:start_y[0]+crop_size, start_x[0]:start_x[0]+crop_size]
    #         img2[g*num_per_group:(g+1)*num_per_group] = images[g*num_per_group:(g+1)*num_per_group, :, start_y[1]:start_y[1]+crop_size, start_x[1]:start_x[1]+crop_size]
    #         # actions[g*num_per_group:(g+1)*num_per_group] = torch.tensor([start_x[0]/14, start_y[0]/14, start_x[1]/14, start_y[1]/14], device=images.device).unsqueeze(0).repeat(num_per_group, 1)
    #         actions[g*num_per_group:(g+1)*num_per_group] = torch.tensor([(start_x[1] - start_x[0])/14, (start_y[1] - start_y[0])/14], device=images.device).unsqueeze(0).repeat(num_per_group, 1)
    #     img1 = F_v2.resize(img1, (28, 28))
    #     img2 = F_v2.resize(img2, (28, 28))

    #     return img1, img2, actions

    def forward(self, x, stop_at=-1):
        return self.encoder(x, stop_at)
    
    def predict(self, x, a=None):
        if a is None:
            a = torch.zeros(x.shape[0], self.num_actions, device=x.device)
        
        z_x = self.encoder(x)
        z_a = self.action_encoder(a)
        z_pred = self.transition(torch.cat([z_x, z_a], dim=1))
        pred = self.decoder(z_pred)

        return pred
    
    def copy(self):
        model = GPA(self.in_features, self.num_actions, self.stop_at, self.resolution).to(next(self.parameters()).device)
        model.load_state_dict(self.state_dict())
        return model
    
    def loss(self, img1, img2, actions, teacher, **_):
        if img2 is None:
            img2, actions = self.interact(img1)
            # img1, img2, actions = self.interact(img1)
        
        if not self.consider_actions:
            actions *= 0.0

        with torch.autocast(device_type=img1.device.type, dtype=torch.bfloat16):
            with torch.no_grad():
                if self.stop_at == 0:
                    targets = img2
                else:
                    targets = teacher(img2, stop_at=self.stop_at)
            preds = self.predict(img1, actions)
            loss = F.mse_loss(preds, targets)
        return loss