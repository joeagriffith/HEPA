import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F
import os

class VoxCeleb1Dataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None, device='cpu'):
        self.images = images.to(device)
        self.transformed_images = None
        self.labels = labels.to(device)
        self.transform = transform
        self.device = device
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if self.transformed_images is None:
            return self.images[idx], self.labels[idx]
        else:
            return self.transformed_images[idx], self.labels[idx]

    def split_set(self, val_ratio): 
        split_idx = int(len(self.images) - (len(self.images) * val_ratio))
        split_label = self.labels[split_idx]
        while self.labels[split_idx] == split_label:
            split_idx += 1
        
        val_images = self.images[split_idx:]
        val_labels = self.labels[split_idx:]
        val_set = VoxCeleb1Dataset(val_images, val_labels, self.transform, self.device)

        train_images = self.images[:split_idx]
        train_labels = self.labels[:split_idx]
        train_set = VoxCeleb1Dataset(train_images, train_labels, self.transform, self.device)

        return train_set, val_set

    
    #  Transforms the data in batches so as not to overload memory
    def apply_transform(self, device=torch.device('cuda'), batch_size=500):
        if self.transform is not None:
            self.transformed_images = torch.empty_like(self.images)
            if device is None:
                device = self.device
            
            low = 0
            high = batch_size
            while low < len(self.images):
                if high > len(self.images):
                    high = len(self.images)
                self.transformed_images[low:high] = self.transform(self.images[low:high].to(device)).to(self.device)
                low += batch_size
                high += batch_size

class VoxCeleb1TripletDataset(VoxCeleb1Dataset):
    def __init__(self, images, labels, transform=None, device='cpu'):
        super().__init__(images, labels, transform, device)
        # Dictionary to store the indices of the images
        self.indices = {label: torch.where(self.labels == label)[0] for label in torch.unique(self.labels)}
    
    def __getitem__(self, idx):
        anchor_image, anchor_label = self.images[idx], self.labels[idx]

        pos_idx = self.indices[anchor_label][torch.randint(0, len(self.indices[anchor_label]), (1,))]
        i = 0
        while pos_idx == idx:
            i += 1
            pos_idx = self.indices[anchor_label][torch.randint(0, len(self.indices[anchor_label]), (1,))]
            if i > 100:
                raise ValueError('Could not find positive sample')
        positive_image = self.images[pos_idx]
        
        found, i = False, 0
        while not found:
            i += 1
            neg_idx = torch.randint(0, len(self.images), (1,))
            negative_image, negative_label = self.images[neg_idx], self.labels[neg_idx]
            found = negative_label != anchor_label
            if i > 100:
                raise ValueError('Could not find negative sample')
        
        return (anchor_image, positive_image, negative_image), anchor_label
        
        

def VoxCeleb1(
        root, 
        split, 
        n=None, 
        transform=None, 
        device='cpu', 
        use_tqdm=True, 
        resolution=1, 
        dataset_dtype='bfloat16', 
        rank=1, 
        world_size=1, 
        shard_type='mel',
        seed=42
    ):
    # Load data
    assert split in ['train', 'val', 'test']
    assert world_size == 1, 'ddp not implemented for VoxCeleb1'
    assert resolution == 1, 'non-default values for resolution not supported for VoxCeleb1'
    assert dataset_dtype == 'float32', 'dataset_dtype must be float32 for VoxCeleb1'
    assert shard_type in ['mel', 'wav'], 'shard_type must be either mel or wav'

    images, labels = [], []
    shard_dir = os.path.join(root, 'VoxCeleb1', f'{shard_type}_shards')
    for i in range(len(os.listdir(shard_dir))):
        shard_path = os.path.join(shard_dir, f'{i}.pt')
        (shard_images, shard_labels) = torch.load(shard_path)
        images.append(shard_images)
        labels.append(shard_labels)
    
    images = torch.cat(images).to(device)
    labels = torch.cat(labels).to(device)
    
    dataset = VoxCeleb1Dataset(images, labels, transform, device)

    return dataset

def VoxCeleb1Triplet(
        root, 
        split, 
        n=None, 
        transform=None, 
        device='cpu', 
        use_tqdm=True, 
        resolution=1, 
        dataset_dtype='bfloat16', 
        rank=1, 
        world_size=1, 
        shard_type='mel',
        seed=42
    ):
    # Load data
    assert split in ['train', 'val', 'test']
    assert world_size == 1, 'ddp not implemented for VoxCeleb1'
    assert resolution == 1, 'non-default values for resolution not supported for VoxCeleb1'
    assert dataset_dtype == 'float32', 'dataset_dtype must be float32 for VoxCeleb1'
    assert shard_type in ['mel', 'wav'], 'shard_type must be either mel or wav'

    images, labels = [], []
    shard_dir = os.path.join(root, 'VoxCeleb1', f'{shard_type}_shards')
    for i in range(len(os.listdir(shard_dir))):
        shard_path = os.path.join(shard_dir, f'{i}.pt')
        (shard_images, shard_labels) = torch.load(shard_path)
        images.append(shard_images)
        labels.append(shard_labels)
    
    images = torch.cat(images).to(device)
    labels = torch.cat(labels).to(device)
    
    dataset = VoxCeleb1TripletDataset(images, labels, transform, device)

    return dataset