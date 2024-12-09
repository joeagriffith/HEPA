import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F
import os
import time
import random

class VoxCeleb1(torch.utils.data.Dataset):
    def __init__(self, 
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
        self.root = root
        self.split = split
        self.n = n
        self.transform = transform
        self.device = device
        self.use_tqdm = use_tqdm
        self.resolution = resolution
        self.dataset_dtype = dataset_dtype
        self.rank = rank
        self.world_size = world_size
        self.shard_type = shard_type
        self.seed = seed

        assert split in ['train', 'val', 'test']
        assert world_size == 1, 'ddp not implemented for VoxCeleb1'
        assert resolution == 1, 'non-default values for resolution not supported for VoxCeleb1'
        assert dataset_dtype == 'bfloat16', 'dataset_dtype must be bfloat16 for VoxCeleb1'
        assert shard_type in ['mel', 'wav'], 'shard_type must be either mel or wav'

        if root is not None:
            self.images = torch.empty(332918, 1, 128, 256, dtype=torch.bfloat16, device=device)
            self.labels = torch.empty(332918, dtype=torch.long, device=device)
            # lo, hi = 0, 64000
            lo, hi = 0, 0
            shard_dir = os.path.join(root, 'VoxCeleb1', f'{shard_type}_shards')
            for i in range(len(os.listdir(shard_dir))):
                shard_path = os.path.join(shard_dir, f'{i}.pt')
                (shard_images, shard_labels) = torch.load(shard_path)
                hi += len(shard_images)
                self.images[lo:hi] = shard_images.to(device)
                self.labels[lo:hi] = shard_labels.to(device)
                lo += 64000
                # hi = min(hi + 64000, len(self.images))
        else:
            self.images = None
            self.labels = None
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if self.transform is None:
            return self.images[idx], self.labels[idx]
        else:
            return self.transform(self.images[idx]), self.labels[idx]
            # return self.transformed_images[idx], self.labels[idx]

    def split_set(self, train_ratio): 
        split_idx = int(len(self.images) * train_ratio)
        split_label = self.labels[split_idx]
        while self.labels[split_idx] == split_label:
            split_idx += 1
        
        val_set = VoxCeleb1(
            root=None, 
            split='val', 
            n=self.n, 
            transform=self.transform, 
            device=self.device, 
            use_tqdm=self.use_tqdm, 
            resolution=self.resolution, 
            dataset_dtype=self.dataset_dtype, 
            rank=self.rank, 
            world_size=self.world_size, 
            shard_type=self.shard_type,
            seed=self.seed
        )
        val_set.images = self.images[split_idx:]
        val_set.labels = self.labels[split_idx:]

        self.images = self.images[:split_idx]
        self.labels = self.labels[:split_idx]

        return self, val_set

    
    # #  Transforms the data in batches so as not to overload memory
    # def apply_transform(self, device=torch.device('cuda'), batch_size=500):
    #     if self.transform is not None:
    #         self.transformed_images = torch.empty_like(self.images)
    #         if device is None:
    #             device = self.device
            
    #         low = 0
    #         high = batch_size
    #         while low < len(self.images):
    #             if high > len(self.images):
    #                 high = len(self.images)
    #             self.transformed_images[low:high] = self.transform(self.images[low:high].to(device)).to(self.device)
    #             low += batch_size
    #             high += batch_size


# class VoxCeleb1TripletDataset(VoxCeleb1Dataset):
#     def __init__(self, images, labels, transform=None, device='cpu'):
#         super().__init__(images, labels, transform, device)
#         # Dictionary to store the indices of the images
#         self.indices = {label: torch.where(self.labels == label)[0] for label in torch.unique(self.labels)}
    
#     def __getitem__(self, idx):
#         anchor_image, anchor_label = self.images[idx], self.labels[idx]

#         pos_idx = self.indices[anchor_label][torch.randint(0, len(self.indices[anchor_label]), (1,))]
#         i = 0
#         while pos_idx == idx:
#             i += 1
#             pos_idx = self.indices[anchor_label][torch.randint(0, len(self.indices[anchor_label]), (1,))]
#             if i > 100:
#                 raise ValueError('Could not find positive sample')
#         positive_image = self.images[pos_idx]
        
#         found, i = False, 0
#         while not found:
#             i += 1
#             neg_idx = torch.randint(0, len(self.images), (1,))
#             negative_image, negative_label = self.images[neg_idx], self.labels[neg_idx]
#             found = negative_label != anchor_label
#             if i > 100:
#                 raise ValueError('Could not find negative sample')
        
#         return (anchor_image, positive_image, negative_image), anchor_label


class VoxCeleb1TripletDataset():
    def __init__(self, dataset, subset_ratio=None, device='cpu'):
        self.dataset = dataset
        self.indices = {label.item(): torch.where(self.dataset.labels == label)[0] for label in torch.unique(self.dataset.labels)}
        self.lens = {label.item(): len(self.indices[label.item()]) for label in torch.unique(self.dataset.labels)}
        self.subset_ratio = subset_ratio
        self.device = device
    def __len__(self):
        if self.subset_ratio is not None:
            return int(len(self.dataset) * self.subset_ratio)
        else:
            return len(self.dataset)
    
    def __getitem__(self, idx):
        anchor_image, anchor_label = self.dataset[idx]

        pos_indices = self.indices[anchor_label.item()]
        pos_idx = pos_indices[torch.randint(0, self.lens[anchor_label.item()], (1,))].item()
        i = 0
        while pos_idx == idx:
            i += 1
            pos_idx = self.indices[anchor_label.item()][torch.randint(0, self.lens[anchor_label.item()], (1,))].item()
            if i > 100:
                raise ValueError('Could not find positive sample')
        positive_image = self.dataset[pos_idx][0]

        found, i = False, 0
        while not found:
            i += 1
            neg_idx = torch.randint(0, len(self.dataset), (1,)).item()
            negative_image, negative_label = self.dataset[neg_idx]
            found = negative_label != anchor_label
            if i > 100:
                raise ValueError('Could not find negative sample')
        
        return (anchor_image, positive_image, negative_image), anchor_label
        
# def VoxCeleb1(
#         root, 
#         split, 
#         n=None, 
#         transform=None, 
#         device='cpu', 
#         use_tqdm=True, 
#         resolution=1, 
#         dataset_dtype='bfloat16', 
#         rank=1, 
#         world_size=1, 
#         shard_type='mel',
#         seed=42
#     ):
#     # Load data
#     assert split in ['train', 'val', 'test']
#     assert world_size == 1, 'ddp not implemented for VoxCeleb1'
#     assert resolution == 1, 'non-default values for resolution not supported for VoxCeleb1'
#     assert dataset_dtype == 'bfloat16', 'dataset_dtype must be bfloat16 for VoxCeleb1'
#     assert shard_type in ['mel', 'wav'], 'shard_type must be either mel or wav'

#     images, labels = [], []
#     shard_dir = os.path.join(root, 'VoxCeleb1', f'{shard_type}_shards')
#     for i in range(len(os.listdir(shard_dir))):
#         shard_path = os.path.join(shard_dir, f'{i}.pt')
#         (shard_images, shard_labels) = torch.load(shard_path)
#         images.append(shard_images)
#         labels.append(shard_labels)
    
#     print(f'Done building list', flush=True)
#     time.sleep(5)
#     print(f'Concatenating', flush=True)
#     images = torch.cat(images).to(device)
#     labels = torch.cat(labels).to(device)
#     print(f'Done concatenating', flush=True)
#     time.sleep(5)    

#     print(f'Building dataset', flush=True)
#     dataset = VoxCeleb1Dataset(images, labels, transform, device)
#     print(f'Done building dataset', flush=True)
#     time.sleep(5)

#     return dataset


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
    assert dataset_dtype == 'bfloat16', 'dataset_dtype must be bfloat16 for VoxCeleb1'
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