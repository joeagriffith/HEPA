import os
import time
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter
from Utils.dataset import PreloadedDataset

from Methods.iGPA.model import iGPA
from Methods.BYOL.model import BYOL
from Methods.AE.model import AE
from Methods.MAE.model import MAE
from Methods.GPAViT.model import GPAViT
from Methods.GPAMAE.model import GPAMAE
from Methods.VAE.model import VAE
from Methods.Supervised.model import Supervised

from Utils.train import train
from Utils.evals import linear_probing
from Utils.functional import get_optimiser
from Utils.cfg import mnist_cfg, get_model_kwargs

# Experiments to run
cfgs = [
    mnist_cfg('delme', 'AE', AE, save=False, num_epochs=3),
    mnist_cfg('delme', 'iGPA', iGPA, save=False, num_epochs=3),
]

for cfg in cfgs:
    print(f'Running experiment: {cfg["experiment"]} - {cfg["trial"]}')
    start_time = time.time()

    device = torch.device(cfg['device'])
    if cfg['device'] == 'cuda':
        torch.backends.cudnn.benchmark = True

    print(f'Loading data...')
    dataset = datasets.MNIST(root=cfg['root'], train=True, transform=transforms.ToTensor(), download=True)
    t_dataset = datasets.MNIST(root=cfg['root'], train=False, transform=transforms.ToTensor(), download=True)

    VAL_RATIO = 0.2
    n_val = int(len(dataset) * VAL_RATIO)
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_set = PreloadedDataset.from_dataset(train_set, None, device)
    val_set = PreloadedDataset.from_dataset(val_set, None, device)
    test_set = PreloadedDataset.from_dataset(t_dataset, None, device)

    print(f'Initialising...')
    # Initialise Model
    model_kwargs = get_model_kwargs(cfg)
    model = cfg['model_type'](**model_kwargs)

    # Initialise Optimiser
    optimiser = get_optimiser(
        model, 
        cfg['optimiser'], 
        lr=cfg['lr'], 
        wd=cfg['wd'], 
        exclude_bias=cfg['exclude_bias'],
        exclude_bn=cfg['exclude_bn'],
    )

    # Initialise Logging
    writer = None
    if cfg['log']:
        enc_log_dir = f'Examples/MNIST/out/logs/{cfg["experiment"]}/Encoder/{cfg["trial"]}'
        run_no = 0
        while os.path.exists(enc_log_dir + f'/run_{run_no}'):
            run_no += 1
        writer = SummaryWriter(enc_log_dir + f'/run_{run_no}')
        # remove reduction if exists
        if os.path.exists(enc_log_dir + '/reduction.csv'):
            os.remove(enc_log_dir + '/reduction.csv')

    # Initialise Saving
    save_dir = None
    if cfg['save']:
        save_dir = f'Examples/MNIST/out/models/{cfg["experiment"]}/{cfg["trial"]}/run_{run_no}.pth'
    
    train_set.transform = cfg['train_transform']
    val_set.transform = cfg['val_transform']

    print(f'Training...')
    train(
        model,
        optimiser,
        train_set,
        val_set,
        num_epochs=cfg['num_epochs'],
        batch_size=cfg['batch_size'],
        dataset=cfg['dataset'],
        has_teacher=cfg['has_teacher'],
        aug_mode=cfg['aug_mode'],
        augment=cfg['augment'],
        writer=writer,
        save_dir=save_dir,
        save_every=cfg['save_every'],
        resolution=cfg['resolution'],
        root=cfg['root'],
        decay_lr=cfg['decay_lr'],
        warmup=cfg['warmup'],
        flat=cfg['flat'],
    )

    # linear probing
    if cfg['log']:
        print(f'Evaluating...')
        # for n in [1, 10, 100, 1000]:
        for n in [1, 10]:
            class_log_dir = f'Examples/MNIST/out/logs/{cfg["experiment"]}/Classifier-n{n}/{cfg["trial"]}'
            run_no = 1
            while os.path.exists(class_log_dir + f'/run_{run_no}'):
                run_no += 1
            writer = SummaryWriter(class_log_dir + f'/run_{run_no}')
            if os.path.exists(class_log_dir + '/reduction.csv'):
                os.remove(class_log_dir + '/reduction.csv')
            linear_probing(model, 'mnist', cfg['root'], n, writer, flatten=False, test=True)
    else:
        print('No logging, skipping linear probing')
    
    print(f'Done. Time taken: {time.time() - start_time:.2f}s')