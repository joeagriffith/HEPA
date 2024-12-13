import torch
import torch.nn as nn
import torch.nn.functional as F

from Utils.nn.nets import Encoder28, Decoder1, Decoder5, Decoder28, Decoder128, Decoder224, VoxEncoder, VoxDecoder
import torchvision.transforms.v2.functional as F_v2
from Utils.nn.resnet_encoder import resnet18, resnet34
from Utils.nn.conv_mixer import ConvMixer

class PVAE(nn.Module):
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
            self.num_features = 256
            self.encoder = VoxEncoder(self.num_features, num_layers=4)
            self.decoder = VoxDecoder(1, self.num_features)
        
        else:
            raise NotImplementedError(f'resolution={resolution} not implemented for GPA')

        # self.action_encoder = nn.Sequential(
        #     nn.Linear(num_actions, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        # )

        # NO BATCHNORM
        self.transition = nn.Sequential(
            # nn.Linear(self.num_features + 128, 1024),
            nn.Linear(self.num_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_features)
        )

        self.mu = nn.Linear(self.num_features, self.num_features)
        self.logVar = nn.Linear(self.num_features, self.num_features)

    # For acting on MNIST images
    def transform_images(self, images):
        # Sample action
        action = torch.rand(5, dtype=images.dtype, device=images.device) * 2 - 1
        mask = torch.rand(5, dtype=images.dtype, device=images.device) < self.p
        action = action * mask

        # Calculate affine parameters
        angle = action[0].item() * 180
        translate_x, translate_y = action[1].item() * 8, action[2].item() * 8
        scale = action[3].item() * 0.25 + 1.0
        shear = action[4].item() * 25

        # Apply affine transformation
        images_aug = F_v2.affine(images, angle=angle, translate=(translate_x, translate_y), scale=scale, shear=shear)
        actions = action.unsqueeze(0).repeat(images.shape[0], 1)
        return images_aug, actions

    # For acting on VoxCeleb1 spectrograms
    def transform_spectrogram(self, images):
        _, _, original_height, original_width = images.size()

        width_factor = torch.randn(1).item() * 0.1 + 1.0
        while width_factor < 0.75 or width_factor > 1.25:
            width_factor = torch.randn(1).item() * 0.1 + 1.0
        height_factor = torch.randn(1).item() * 0.1 + 1.0
        while height_factor < 0.75 or height_factor > 1.25:
            height_factor = torch.randn(1).item() * 0.1 + 1.0
        shift = torch.randn((2,)) * 8
        while shift.abs().max() > 20:
            shift = torch.randn((2,)) * 8
        shift_x = int(shift[0].item())
        shift_y = int(shift[1].item())

        # Calculate new dimensions
        new_width = int(original_width * width_factor)
        new_height = int(original_height * height_factor)
        
        # Resize the image
        images_aug = F.interpolate(images, size=(new_height, new_width), mode='bilinear', align_corners=False)
        
        # Calculate padding if needed
        pad_width = max(0, original_width - new_width)
        pad_height = max(0, original_height - new_height)
        
        # Pad the image to the original size
        if pad_width > 0 or pad_height > 0:
            images_aug = F.pad(images_aug, 
                                (pad_width // 2, pad_width - pad_width // 2, 
                                pad_height // 2, pad_height - pad_height // 2))
        
        # Shift the image
        if shift_x != 0 or shift_y != 0:
            images_aug = F.pad(images_aug, 
                                (shift_x, -shift_x, 
                                shift_y, -shift_y))
        
        # Center crop to original size if necessary
        if width_factor > 1.0 or height_factor > 1.0:
            start_x = (images_aug.size(3) - original_width) // 2
            start_y = (images_aug.size(2) - original_height) // 2
            images_aug = images_aug[:, :, start_y:start_y + original_height, start_x:start_x + original_width]
        
        actions = torch.tensor([(width_factor-1.0)/0.25, (height_factor-1.0)/0.25, shift_x/8, shift_y/8], dtype=torch.bfloat16, device=images.device).unsqueeze(0).repeat(images.shape[0], 1)
        return images_aug, actions
    
    # # Fake interact that does nothing
    # def interact(self, images, groups=8):
    #     return images, torch.zeros(images.shape[0], 4, device=images.device)

    def interact(self, images, groups=8):
        """
        Interact with the images by applying either image or spectrogram transformations.
        
        Parameters:
        images (torch.Tensor): The input image tensor.
        groups (int): The number of groups to split the images into.
        
        Returns:
        torch.Tensor: The augmented images tensor.
        torch.Tensor: The actions tensor.

        """
        N, _, original_height, original_width = images.size()
        if N < groups:
            groups = N
        n_per = N // groups

        images_aug_arr = []
        actions_arr = []

        lo, hi = 0, n_per + N % groups
        while lo < N:
            if original_width <= 32:
                images_aug, actions = self.transform_images(images[lo:hi])
            else:
                images_aug, actions = self.transform_spectrogram(images[lo:hi])
            
            images_aug_arr.append(images_aug)
            actions_arr.append(actions)

            lo = hi
            hi = min(N, lo + n_per)
        
        return torch.cat(images_aug_arr, dim=0), torch.cat(actions_arr, dim=0)

    def forward(self, x, stop_at=-1):
        h = self.encoder(x)
        return self.mu(h)

    def reparameterise(self, mu, logVar):
        std = torch.exp(0.5 * logVar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def predict(self, x, a=None):
        if a is None:
            a = torch.zeros(x.shape[0], self.num_actions, device=x.device)
        
        h = self.encoder(x)
        mu, logVar = self.mu(h), self.logVar(h)
        z_x = self.reparameterise(mu, logVar)
        z_a = self.action_encoder(a)
        h = self.transition(torch.cat([z_x, z_a], dim=1))
        pred = self.decoder(h)

        return pred, mu, logVar
    
    def copy(self):
        model = PVAE(self.in_features, self.num_actions, self.stop_at, self.resolution).to(next(self.parameters()).device)
        model.load_state_dict(self.state_dict())
        return model

    def loss(self, img1, img2, actions, teacher, **_):
        if img2 is None:
            img2, actions = self.interact(img1)
        
        if not self.consider_actions:
            actions *= 0.0

        with torch.autocast(device_type=img1.device.type, dtype=torch.bfloat16):
            with torch.no_grad():
                if self.stop_at == 0:
                    targets = img2
                else:
                    targets = teacher(img2, stop_at=self.stop_at)
            preds, mu, logvar = self.predict(img1, actions)
            pred_loss = F.binary_cross_entropy_with_logits(preds, targets, reduction='sum') / targets.shape[0]
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]
            loss = pred_loss + self.beta * kl_loss
        return loss