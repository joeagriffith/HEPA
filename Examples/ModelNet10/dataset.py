import torch
from torchvision import transforms
import numpy as np
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import time


class ModelNet10(torch.utils.data.Dataset):
    def __init__(
            self, 
            root, 
            split, 
            n=None, 
            transform=None, 
            device='cpu', 
            use_tqdm=True, 
            resolution=128, 
            dataset_dtype='uint8', 
            rank=1, 
            world_size=1, 
            seed=42
        ):
        assert split in ['train', 'val', 'test']
        if split == 'val':
            assert n is not None, 'n must be specified for val split, or use .split_set()'
        self.device = device
        self.root = root
        file_split = 'train' if split == 'val' else split
        self.split = split
        self.classes = os.listdir(self.root + 'ModelNet10/' + file_split)
        self.class_n = {}
        self.transform = transform
        self.resolution = resolution
        self.dataset_dtype = dataset_dtype
        self.use_tqdm = use_tqdm
        self.rank = rank
        self.world_size = world_size
        self.seed = seed

        for c in self.classes:
            self.class_n[c] = len(os.listdir(self.root + 'ModelNet10/' + file_split + '/' + c))
        if n is None:
            self.length = sum(self.class_n.values())
        else:
            self.length = n * len(self.classes)

        self.n = n

        path = self.root + 'ModelNet10/tensors/' + split + f'_dataset_n{self.n}_{self.resolution}.pth' if self.n is not None else self.root + 'ModelNet10/tensors/' + split + f'_dataset_{self.resolution}.pth'
        path_224 = self.root + 'ModelNet10/tensors/' + split + f'_dataset_n{self.n}_224.pth' if self.n is not None else self.root + 'ModelNet10/tensors/' + split + f'_dataset_{224}.pth'

        if os.path.exists(path):
            tensors = torch.load(path)
            self.data = tensors['data']
            self.rotations = tensors['rotations']
            self.labels = tensors['labels']
        elif os.path.exists(path_224):
            tensors = torch.load(path_224)
            self.data = tensors['data']
            self.rotations = tensors['rotations']
            self.labels = tensors['labels']
            self.data = torch.nn.functional.interpolate(self.data, size=(1, self.resolution, self.resolution))
            torch.save({'data': self.data, 'rotations': self.rotations, 'labels': self.labels}, path)
        else:
            print('Building dataset...')

            self.data = torch.empty((self.length, 64, 1, 224, 224), dtype=torch.uint8)
            self.rotations = torch.empty((self.length, 64, 3), dtype=torch.float32)
            self.labels = torch.empty(self.length, dtype=torch.long)

            # load info csv and convert RotX, RotY, RotZ columns to a (n,3) pytorch tensor
            info = pd.read_csv(self.root + 'ModelNet10/' + 'train_info.csv')[['RotX', 'RotY', 'RotZ']].values.astype(np.float32)
            info = torch.Tensor(info)

            guid = 0 if split in ['train', 'test'] else (sum(self.class_n.values()) * 64) - 1 # start from end if val split
            idx = 0
            if use_tqdm:
                c_loop = tqdm(enumerate(self.classes), total=len(self.classes)) if split in ['train', 'test'] else tqdm(reversed(list(enumerate(self.classes))), total=len(self.classes))
            else:
                c_loop = enumerate(self.classes) if split in ['train', 'test'] else reversed(list(enumerate(self.classes)))
            for c_i, c in c_loop:
                num = 0
                o_loop = os.listdir(self.root + 'ModelNet10/' + file_split + '/' + c) if split in ['train', 'test'] else reversed(os.listdir(self.root + 'ModelNet10/' + file_split + '/' + c))
                for o in o_loop:
                    # enforce 'n' items per class if specified.
                    if n is not None and num >= self.n:
                        guid += 64 if split in ['train', 'test'] else -64
                    else:
                        i_loop = range(64) if split in ['train', 'test'] else reversed(range(64))
                        for i in i_loop:
                            img = Image.open(self.root + 'ModelNet10/' + file_split + '/' + c + '/' + o + '/' + str(guid) + '.png').convert('RGB')
                            self.data[idx][i] = (transforms.ToTensor()(img)[0]*255.0).to(torch.uint8).unsqueeze(0)
                            self.rotations[idx][i] = info[guid]
                            guid += 1 if split in ['train', 'test'] else -1
                        self.labels[idx] = c_i
                        idx += 1
                    num += 1

            if split == 'val':
                self.data = self.data.flip(0)
                self.data = self.data.flip(1)
                self.labels = self.labels.flip(0)
            
            torch.save({'data': self.data, 'rotations': self.rotations, 'labels': self.labels}, path_224)

            if self.resolution != 224:
                self.data = torch.nn.functional.interpolate(self.data, size=(1, self.resolution, self.resolution))

            torch.save({'data': self.data, 'rotations': self.rotations, 'labels': self.labels}, path)
        
        # # remove data not accessed by process
        if world_size > 1:
            self.shuffle(seed)
            proc_len = self.length // world_size
            lo = proc_len * rank
            hi = proc_len * rank + proc_len
            self.data = self.data[lo:hi]
            self.rotations = self.rotations[lo:hi]
            self.labels = self.labels[lo:hi]
            self.length = proc_len

        assert self.dataset_dtype in ['uint8', 'float32'], 'dataset_dtype must be uint8 or float32'
        if self.dataset_dtype == 'float32':
            self.data = self.data.to(torch.float32) / 255.0

        self.data = self.data.to(self.device)
        self.rotations = self.rotations.to(self.device)
        self.labels = self.labels.to(self.device)

    def split_set(self, ratio):
        assert 0 < ratio < 1, 'ratio must be between 0 and 1'
        assert self.n is None, 'n must be None to split dataset'

        self.shuffle()

        # build val dataset
        val_dataset = ModelNet10(root=self.root, split='val', n=0, device=self.device, use_tqdm=self.use_tqdm, resolution=self.resolution, dataset_dtype=self.dataset_dtype, rank=self.rank, world_size=self.world_size, seed=self.seed)
        val_dataset.data = self.data[int(self.length * ratio):]
        val_dataset.labels = self.labels[int(self.length * ratio):]
        val_dataset.rotations = self.rotations[int(self.length * ratio):]
        val_dataset.length = self.length - int(self.length * ratio)
    
        # edit self
        self.data = self.data[:int(self.length * ratio)]
        self.labels = self.labels[:int(self.length * ratio)]
        self.rotations = self.rotations[:int(self.length * ratio)]
        self.length = int(self.length * ratio)

        return self, val_dataset

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):

        idx1 = np.random.randint(64)
        idx2 = (idx1 + 1) % 64
        # idx2 = np.random.randint(64)

        img1 = self.data[idx][idx1]
        img2 = self.data[idx][idx2]

        if self.dataset_dtype == 'uint8':
            img1 = img1.to(torch.float32) / 255.0
            img2 = img2.to(torch.float32) / 255.0

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        rot1 = self.rotations[idx][idx1]
        rot2 = self.rotations[idx][idx2]
    
        lab1 = self.labels[idx]
        lab2 = self.labels[idx]

        return (img1, rot1, lab1), (img2, rot2, lab2)

    def to(self, device):
        self.device = device
        self.data = self.data.to(device)
        self.labels = self.labels.to(device)
        self.rotations = self.rotations.to(device)
        return self

    def apply_transform(self, batch_size):
        pass

    def shuffle(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        idx = torch.randperm(self.length)
        self.data = self.data[idx]
        self.labels = self.labels[idx]
        self.rotations = self.rotations[idx]
        
class ModelNet10Simple(ModelNet10):
    def __init__(
            self, 
            root, 
            split, 
            n=None, 
            transform=None,
            device='cpu',
            use_tqdm=True,
            resolution=128,
            dataset_dtype='uint8',
            rank=1,
            world_size=1,
            seed=42
        ):
        super().__init__(root, split, n, transform, device, use_tqdm, resolution, dataset_dtype, rank, world_size, seed)

    def __getitem__(self, idx):
        idx1 = np.random.randint(64)

        if self.dataset_dtype == 'uint8':
            img = self.data[idx][idx1].to(torch.float32) / 255.0

        if self.transform is not None:
            img = self.transform(img)
        
        lab = self.labels[idx]

        return img, lab