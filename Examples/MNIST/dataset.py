import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from Utils.dataset import PreloadedDataset
from tqdm import tqdm
import torch.nn.functional as F


def MNIST(cfg, split, n=None, transform=None):
    # Load data
    assert split in ['train', 'val', 'test']
    device = cfg['device']
    root = cfg['root']
    train = split in ['train', 'val']
    dataset = datasets.MNIST(root=root, train=train, transform=transforms.ToTensor(), download=True)
    if transform is None:
        transform = transforms.ToTensor()

    if split == 'train':
        # Build train dataset
        dataset = torch.utils.data.Subset(dataset, range(0, len(dataset) - 10000))
        dataset = PreloadedDataset.from_dataset(dataset, transform, device, cfg['local'])
        if n is not None:
            train_indices = []
            for i in range(10):
                idxs = torch.where(dataset.targets == i)[0][:n]
                train_indices.append(idxs)
            train_indices = torch.cat(train_indices)
            dataset.images = dataset.images[train_indices]
            dataset.transformed_images = dataset.transformed_images[train_indices]
            dataset.targets = dataset.targets[train_indices]
    
    elif split == 'val':
        # Build val dataset
        if n is not None:
            raise NotImplementedError('n not implemented for val_set')
        dataset = torch.utils.data.Subset(dataset, range(len(dataset) - 10000, len(dataset)))
        dataset = PreloadedDataset.from_dataset(dataset, transforms.ToTensor(), device, cfg['local'])
    
    elif split == 'test':
        if n is not None:
            raise NotImplementedError('n not implemented for test_set')
        dataset = PreloadedDataset.from_dataset(dataset, transform, device, cfg['local'])

    return dataset