import torch
from torchvision import transforms
import numpy as np
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm


class ModelNet10(torch.utils.data.Dataset):
    def __init__(self, root, split, device='cpu', n=None):
        assert split in ['train', 'val', 'test']
        self.device = device
        self.root = root
        file_split = 'train' if split == 'val' else split
        self.split = split
        self.classes = os.listdir(root + file_split)
        self.class_n = {}
        for c in self.classes:
            self.class_n[c] = len(os.listdir(root + file_split + '/' + c))
        if n is None:
            self.length = sum(self.class_n.values())
        else:
            self.length = n * len(self.classes)

        self.n = n

        # load info csv and convert RotX, RotY, RotZ columns to a (n,3) pytorch tensor
        info = pd.read_csv(root + 'train_info.csv')[['RotX', 'RotY', 'RotZ']].values.astype(np.float32)
        info = torch.Tensor(info).to(device)
        self.rotations = torch.empty((self.length, 64, 3), dtype=torch.float32, device=device)
        
        # load images
        data_path = root + split + f'_dataset_data_{self.n}.pth' if self.n is not None else root + split + '_dataset_data.pth'
        if os.path.exists(data_path):
            self.data = torch.load(data_path).to(device)
            print('Loaded pre-built dataset.')
        else:
            print('Building dataset...')
            self.data = torch.empty((self.length, 64, 3, 224, 224), dtype=torch.uint8, device=device)
            guid = 0 if split in ['train', 'test'] else (sum(self.class_n.values()) * 64) - 1 # start from end if val split
            idx = 0
            c_loop = tqdm(self.classes, total=len(self.classes)) if split in ['train', 'test'] else tqdm(reversed(self.classes), total=len(self.classes))
            for c in c_loop:
                num = 0
                o_loop = range(self.class_n[c]) if split in ['train', 'test'] else reversed(range(self.class_n[c]))
                for o in o_loop:
                    # enforce 'n' items per class if specified.
                    if n is not None and num >= self.n:
                        guid += 64 if split in ['train', 'test'] else -64
                    else:
                        i_loop = range(64) if split in ['train', 'test'] else reversed(range(64))
                        for i in i_loop:
                            img = Image.open(root + file_split + '/' + c + '/' + str(o) + '/' + str(guid) + '.png').convert('RGB')
                            self.data[idx][i] = (transforms.ToTensor()(img)*255.0).to(torch.uint8).to(device)
                            self.rotations[idx][i] = info[guid]
                            guid += 1 if split in ['train', 'test'] else -1
                        idx += 1
                    num += 1
            torch.save(self.data, data_path)

        # load labels
        label_path = root + split + f'_dataset_labels_{self.n}.pth' if self.n is not None else root + split + '_dataset_labels.pth'
        if os.path.exists(label_path):
            self.labels = torch.load(label_path).to(device)
            print('Loaded pre-built labels')
        else:
            print('Building labels...')
            self.labels = torch.empty(self.length, dtype=torch.long, device=device)
            idx = 0
            c_loop = tqdm(enumerate(self.classes), total=len(self.classes)) if split in ['train', 'test'] else tqdm(reversed(list(enumerate(self.classes))), total=len(self.classes))
            for i, c in c_loop:
                num = 0
                o_loop = range(self.class_n[c]) if split in ['train', 'test'] else reversed(range(self.class_n[c]))
                for o in o_loop:
                    # enforce 'n' items per class if specified.
                    if n is not None and num >= self.n:
                        pass
                    else:
                        self.labels[idx] = i
                        idx += 1
                    num += 1

            if split == 'val':
                self.data = self.data.flip(0)
                self.data = self.data.flip(1)
                self.labels = self.labels.flip(0)

            torch.save(self.labels, label_path)
            
    def split(self, ratio):
        assert 0 < ratio < 1, 'ratio must be between 0 and 1'
        assert self.n is None, 'n must be None to split dataset'

        # shuffle data
        idx = torch.randperm(self.length)
        self.data = self.data[idx]
        self.labels = self.labels[idx]
        self.rotations = self.rotations[idx]

        # build val dataset
        val_dataset = ModelNet10(self.root, train=self.train, device=self.device, n=0)
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

        rot1 = self.rotations[idx][idx1]
        rot2 = self.rotations[idx][idx2]
    
        lab1 = self.labels[idx]
        lab2 = self.labels[idx]

        return (img1, rot1, lab1), (img2, rot2, lab2)