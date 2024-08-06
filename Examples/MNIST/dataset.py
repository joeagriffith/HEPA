import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from Utils.dataset import PreloadedDataset
from tqdm import tqdm
import torch.nn.functional as F


def MNIST(
        root, 
        split, 
        n=None, 
        transform=None, 
        device='cpu', 
        use_tqdm=True, 
        resolution=28, 
        dataset_dtype='float32', 
        rank=1, 
        world_size=1, 
        seed=42
    ):
    # Load data
    assert split in ['train', 'val', 'test']
    assert world_size == 1, 'ddp not implemented for MNIST'
    assert resolution == 28, 'resolution must be 28 for MNIST'
    assert dataset_dtype == 'float32', 'dataset_dtype must be float32 for MNIST'

    train = split in ['train', 'val']
    dataset = datasets.MNIST(root=root, train=train, transform=transforms.ToTensor(), download=True)
    if transform is None:
        transform = transforms.ToTensor()

    if split == 'train':
        # Build train dataset
        dataset = torch.utils.data.Subset(dataset, range(0, len(dataset) - 10000))
        dataset = PreloadedDataset.from_dataset(dataset, transform, device, use_tqdm)
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
        dataset = PreloadedDataset.from_dataset(dataset, transforms.ToTensor(), device, use_tqdm)
    
    elif split == 'test':
        if n is not None:
            raise NotImplementedError('n not implemented for test_set')
        dataset = PreloadedDataset.from_dataset(dataset, transform, device, use_tqdm)

    return dataset