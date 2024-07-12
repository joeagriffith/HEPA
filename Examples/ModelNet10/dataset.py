import torch
from torchvision import transforms
import numpy as np
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm


class ModelNet10(torch.utils.data.Dataset):
    def __init__(self, root, split, n=None, transform=None, device=torch.device('cpu')):
        assert split in ['train', 'val', 'test']
        if split == 'val':
            assert n is not None, 'n must be specified for val split, or use .split_set()'
        self.device = device
        self.root = root
        file_split = 'train' if split == 'val' else split
        self.split = split
        self.classes = os.listdir(root + 'ModelNet10/' + file_split)
        self.class_n = {}
        self.transform = transform
        for c in self.classes:
            self.class_n[c] = len(os.listdir(root + 'ModelNet10/' + file_split + '/' + c))
        if n is None:
            self.length = sum(self.class_n.values())
        else:
            self.length = n * len(self.classes)

        self.n = n

        path = root + 'ModelNet10/tensors/' + split + f'_dataset_n{self.n}.pth' if self.n is not None else root + 'ModelNet10/tensors/' + split + '_dataset.pth'
        if os.path.exists(path):
            print('Loading data...')
            tensors = torch.load(path)
            self.data = tensors['data']
            self.rotations = tensors['rotations']
            self.labels = tensors['labels']
        else:
            print('Building dataset...')

            self.data = torch.empty((self.length, 64, 1, 224, 224), dtype=torch.uint8, device=device)
            self.rotations = torch.empty((self.length, 64, 3), dtype=torch.float32, device=device)
            self.labels = torch.empty(self.length, dtype=torch.long, device=device)

            # load info csv and convert RotX, RotY, RotZ columns to a (n,3) pytorch tensor
            info = pd.read_csv(root + 'ModelNet10/' + 'train_info.csv')[['RotX', 'RotY', 'RotZ']].values.astype(np.float32)
            info = torch.Tensor(info).to(device)

            guid = 0 if split in ['train', 'test'] else (sum(self.class_n.values()) * 64) - 1 # start from end if val split
            idx = 0
            c_loop = tqdm(enumerate(self.classes), total=len(self.classes)) if split in ['train', 'test'] else tqdm(reversed(list(enumerate(self.classes))), total=len(self.classes))
            for c_i, c in c_loop:
                num = 0
                o_loop = range(self.class_n[c]) if split in ['train', 'test'] else reversed(range(self.class_n[c]))
                for o in o_loop:
                    # enforce 'n' items per class if specified.
                    if n is not None and num >= self.n:
                        guid += 64 if split in ['train', 'test'] else -64
                    else:
                        i_loop = range(64) if split in ['train', 'test'] else reversed(range(64))
                        for i in i_loop:
                            img = Image.open(root + 'ModelNet10/' + file_split + '/' + c + '/' + str(o) + '/' + str(guid) + '.png').convert('RGB')
                            self.data[idx][i] = (transforms.ToTensor()(img)[0]*255.0).to(torch.uint8).unsqueeze(0).to(device)
                            self.rotations[idx][i] = info[guid]
                            guid += 1 if split in ['train', 'test'] else -1
                        self.labels[idx] = c_i
                        idx += 1
                    num += 1

            if split == 'val':
                self.data = self.data.flip(0)
                self.data = self.data.flip(1)
                self.labels = self.labels.flip(0)

            torch.save({'data': self.data, 'rotations': self.rotations, 'labels': self.labels}, path)
            
    def split_set(self, ratio):
        assert 0 < ratio < 1, 'ratio must be between 0 and 1'
        assert self.n is None, 'n must be None to split dataset'

        # shuffle data
        idx = torch.randperm(self.length)
        self.data = self.data[idx]
        self.labels = self.labels[idx]
        self.rotations = self.rotations[idx]

        # build val dataset
        val_dataset = ModelNet10(self.root, split='val', device=self.device, n=0)
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
        idx2 = np.random.randint(64)

        img1 = self.data[idx][idx1].to(torch.float32) / 255.0
        img2 = self.data[idx][idx2].to(torch.float32) / 255.0

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


class ModelNet10Simple(ModelNet10):
    def __init__(self, root, split, device='cpu', n=None, transform=None):
        assert split in ['train', 'val', 'test']
        if split == 'val':
            assert n is not None, 'n must be specified for val split, or use .split_set()'
        self.device = device
        self.root = root
        file_split = 'train' if split == 'val' else split
        self.split = split
        self.classes = os.listdir(root + 'ModelNet10/' + file_split)
        self.class_n = {}
        self.transform = transform
        for c in self.classes:
            self.class_n[c] = len(os.listdir(root + 'ModelNet10/' + file_split + '/' + c))
        if n is None:
            self.length = sum(self.class_n.values()) * 64
        else:
            self.length = n * len(self.classes) * 64

        self.n = n

        path = root + 'ModelNet10/tensors/simple_' + split + f'_n{self.n}.pth' if self.n is not None else root + 'ModelNet10/tensors/simple_' + split + '.pth'
        if os.path.exists(path):
            print('Loading data...')
            tensors = torch.load(path)
            self.data = tensors['data']
            self.rotations = tensors['rotations']
            self.labels = tensors['labels']
        else:
            print('Building dataset...')
            print(f'path: {path}')

            self.data = torch.empty((self.length, 1, 224, 224), dtype=torch.uint8, device=device)
            self.rotations = torch.empty((self.length, 3), dtype=torch.float32, device=device)
            self.labels = torch.empty(self.length, dtype=torch.long, device=device)

            # load info csv and convert RotX, RotY, RotZ columns to a (n,3) pytorch tensor
            info = pd.read_csv(root + 'ModelNet10/' + 'train_info.csv')[['RotX', 'RotY', 'RotZ']].values.astype(np.float32)
            info = torch.Tensor(info).to(device)

            guid = 0 if split in ['train', 'test'] else (sum(self.class_n.values()) * 64) - 1 # start from end if val split
            idx = 0
            c_loop = tqdm(enumerate(self.classes), total=len(self.classes)) if split in ['train', 'test'] else tqdm(reversed(list(enumerate(self.classes))), total=len(self.classes))
            for c_i, c in c_loop:
                num = 0
                o_loop = range(self.class_n[c]) if split in ['train', 'test'] else reversed(range(self.class_n[c]))
                for o in o_loop:
                    # enforce 'n' items per class if specified.
                    if n is not None and num >= self.n:
                        guid += 64 if split in ['train', 'test'] else -64
                    else:
                        for _ in range(64):
                            img = Image.open(root + 'ModelNet10/' + file_split + '/' + c + '/' + str(o) + '/' + str(guid) + '.png').convert('RGB')
                            self.data[idx] = (transforms.ToTensor()(img)[0]*255.0).to(torch.uint8).unsqueeze(0).to(device)
                            self.rotations[idx] = info[guid]
                            self.labels[idx] = c_i

                            guid += 1 if split in ['train', 'test'] else -1
                            idx += 1
                    num += 1

            if split == 'val':
                self.data = self.data.flip(0, 1)
                self.rotations = self.rotations.flip(0, 1)
                self.labels = self.labels.flip(0)

            torch.save({'data': self.data, 'rotations': self.rotations, 'labels': self.labels}, path)
    
    def __getitem__(self, idx):
        img = self.data[idx].to(torch.float32) / 255.0
        lab = self.labels[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, lab