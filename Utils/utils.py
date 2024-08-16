import torch
from torch.utils.tensorboard import SummaryWriter
import os

from Models import iGPA, BYOL, BYOPL, iJEPA, AE, VAE, MAE, Supervised

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from Examples.ModelNet10.dataset import ModelNet10, ModelNet10Simple
from Examples.MNIST.dataset import MNIST
from Utils.dataset import PreloadedDataset

def get_model(cfg:dict):
    if cfg['model_type'] == 'iGPA':
        return iGPA(
            in_features=cfg['in_features'],
            num_actions=cfg['num_actions'],
            stop_at=cfg['stop_at'],
            resolution=cfg['resolution'],
            p=cfg['p'],
            consider_actions=cfg['consider_actions']
        ).to(cfg['device'])

    elif cfg['model_type'] == 'BYOL':
        return BYOL(
            in_features=cfg['in_features'],
            resolution=cfg['resolution'],
        ).to(cfg['device'])
    
    elif cfg['model_type'] == 'BYOPL':
        return BYOPL(
            in_features=cfg['in_features'],
            num_actions=cfg['num_actions'],
            resolution=cfg['resolution'],
            p=cfg['p'],
            consider_actions=cfg['consider_actions']
        ).to(cfg['device'])

    elif cfg['model_type'] == 'iJEPA':
        return iJEPA(
            in_features=cfg['in_features'],
            input_size=(cfg['resolution'], cfg['resolution']),
            patch_size=cfg['patch_size'],
            min_keep=cfg['min_keep'],
        ).to(cfg['device'])

    elif cfg['model_type'] == 'AE':
        return AE(
            in_features=cfg['in_features'],
            resolution=cfg['resolution'],
        ).to(cfg['device'])

    elif cfg['model_type'] == 'VAE':
        return VAE(
            in_features=cfg['in_features'],
            z_dim=cfg['z_dim'],
            resolution=cfg['resolution'],
        ).to(cfg['device'])

    elif cfg['model_type'] == 'MAE':
        return MAE(
            in_features=cfg['in_features'],
            resolution=cfg['resolution'],
        ).to(cfg['device'])
    
    elif cfg['model_type'] == 'Supervised':
        return Supervised(
            in_features=cfg['in_features'],
            resolution=cfg['resolution'],
        ).to(cfg['device'])

    else:
        raise ValueError(f"Model type '{cfg['model_type']}' is not supported.")

def get_optimiser(model, cfg):
    # placeholder values, actually set in train.py
    tmp_lr = 0.01
    tmp_wd = 0.002

    non_decay_parameters = []
    decay_parameters = []   
    for n, p in model.named_parameters():
        if cfg['exclude_bias'] and 'bias' in n:
            non_decay_parameters.append(p)
        elif cfg['exclude_bn'] and 'bn' in n:
            non_decay_parameters.append(p)
        else:
            decay_parameters.append(p)
    non_decay_parameters = [{'params': non_decay_parameters, 'weight_decay': 0.0}]
    decay_parameters = [{'params': decay_parameters}]

    assert cfg['optimiser'] in ['AdamW', 'SGD'], 'optimiser must be one of ["AdamW", "SGD"]'
    if cfg['optimiser'] == 'AdamW':
        optimiser = torch.optim.AdamW(decay_parameters + non_decay_parameters, lr=tmp_lr, weight_decay=tmp_wd, betas=cfg['betas'])
    elif cfg['optimiser'] == 'SGD':
        optimiser = torch.optim.SGD(decay_parameters + non_decay_parameters, lr=tmp_lr, weight_decay=tmp_wd, momentum=cfg['momentum'])
    
    return optimiser

def get_datasets(cfg):
    if cfg['dataset'] == 'mnist':
        dataset = datasets.MNIST(root=cfg['root'], train=True, transform=transforms.ToTensor(), download=True)

        val_ratio = 1.0 - cfg['train_ratio']
        n_val = round(len(dataset) * val_ratio)
        n_train = len(dataset) - n_val
        train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

        device = torch.device(cfg['device'])
        train_set = PreloadedDataset.from_dataset(train_set, None, device, use_tqdm=cfg['local'])
        val_set = PreloadedDataset.from_dataset(val_set, None, device, use_tqdm=cfg['local'])

    elif cfg['dataset'] == 'modelnet10':
        train_set = ModelNet10(cfg['root'], 'train', device=cfg['device'], use_tqdm=cfg['local'], resolution=cfg['resolution'], dataset_dtype=cfg['dataset_dtype'], rank=cfg['ddp_rank'], world_size=cfg['ddp_world_size'], seed=cfg['seed'])
        train_set, val_set = train_set.split_set(cfg['train_ratio'])

    return train_set, val_set
    

def get_ss_datasets(cfg):
    device = torch.device(cfg['device'])
    if cfg['dataset'] == 'mnist':
        ss_train_dataset = MNIST(cfg['root'], split='train', n=1, transform=transforms.ToTensor(), device=cfg['device'], use_tqdm=cfg['local'])
        ss_val_dataset = MNIST(cfg['root'], split='val', transform=transforms.ToTensor(), device=cfg['device'], use_tqdm=cfg['local'])
    elif cfg['dataset'] == 'modelnet10':
        ss_train_dataset = ModelNet10Simple(cfg['root'], 'train', n=10, transform=None, device=cfg['device'], use_tqdm=cfg['local'], rank=cfg['ddp_rank'], world_size=cfg['ddp_world_size'], seed=cfg['seed'])
        ss_val_dataset = ModelNet10Simple(cfg['root'], 'val', n=10, transform=None, device=cfg['device'], use_tqdm=cfg['local'], rank=cfg['ddp_rank'], world_size=cfg['ddp_world_size'], seed=cfg['seed'])
    else:
        raise ValueError(f'Dataset {cfg["dataset"]} not implemented')
    
    return ss_train_dataset, ss_val_dataset


def get_writer(cfg, n=None):
    if n is None:
        trial_log_dir = cfg['log_dir'] + f'raw/{cfg["experiment"]}/Encoder/{cfg["trial"]}'
        agg_log_dir = cfg['log_dir'] + f'agg/{cfg["experiment"]}/Encoder/{cfg["trial"]}'
    else:
        trial_log_dir = cfg['log_dir'] + f'raw/{cfg["experiment"]}/Classifier-n{n}/{cfg["trial"]}'
        agg_log_dir = cfg['log_dir'] + f'agg/{cfg["experiment"]}/Classifier-n{n}/{cfg["trial"]}'

    # remove aggregation as it needs to be recalculated with this run.
    if os.path.exists(trial_log_dir + '/reduction.csv'):
        os.remove(trial_log_dir + '/reduction.csv')
    if os.path.exists(agg_log_dir):
        for f in os.listdir(agg_log_dir):
            os.remove(agg_log_dir + '/' + f)

    if 'run_no' not in cfg.keys():
        run_no = 0
        while os.path.exists(trial_log_dir + f'/run_{run_no}'):
            run_no += 1
        cfg['run_no'] = run_no

    return SummaryWriter(trial_log_dir + f'/run_{cfg["run_no"]}')
