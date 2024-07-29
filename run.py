import os
import sys
import time
import json
import yaml

import torch
import torchvision.transforms as transforms

from Utils.train import train
from Utils.evals import linear_probing
from Utils.utils import get_optimiser, get_model, get_datasets, get_writer
from Utils.cfg import mnist_cfg, modelnet10_cfg

if __name__ == '__main__':
    args = sys.argv[1:]
    assert len(args) == 1, 'Please provide a config file'
    filepath = 'configs/' + args[0]
    if not filepath.endswith('.yaml'):
        filepath += '.yaml'
    with open(filepath, 'r') as file:
        yaml_cfgs = yaml.safe_load(file)
    
    cfgs = []
    for yaml_cfg in yaml_cfgs['cfgs']:
        if yaml_cfg['dataset'] == 'modelnet10':
            cfgs.append(modelnet10_cfg(**yaml_cfg))
        elif yaml_cfg['dataset'] == 'mnist':
            cfgs.append(mnist_cfg(**yaml_cfg))
        else:
            raise ValueError(f'Dataset {yaml_cfg["dataset"]} not supported')

    for (cfg, specified_cfg) in cfgs:

        writer = None
        if cfg['log']:
            if not os.path.exists(cfg['log_dir']):
                os.makedirs(cfg['log_dir'])
            writer = get_writer(cfg)

        print(f'\n======================  Experiment: {cfg["experiment"]} == Trial: {cfg["trial"]} == Run: {cfg["run_no"]} ======================')
        start_time = time.time()

        device = torch.device(cfg['device'])
        if cfg['device'] == 'cuda':
            torch.backends.cudnn.benchmark = True

        print(f'Initialising...')
        train_set, val_set = get_datasets(cfg)
        if cfg['data_aug']:
            train_set.transform = transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10)
        model = get_model(cfg)
        optimiser = get_optimiser(model, cfg)

        if cfg['save']:
            trial_save_dir = cfg['save_dir'] + cfg['experiment'] + '/' + cfg['trial'] + '/'
            if not os.path.exists(trial_save_dir):
                os.makedirs(trial_save_dir)
            cfg['save_dir'] = trial_save_dir + str(cfg['run_no']) + '.pth'

        writer.add_text('model', str(model).replace('\n', '<br/>').replace(' ', '&nbsp;'))
        writer.add_text('config', json.dumps(cfg, indent=4).replace('\n', '<br/>').replace(' ', '&nbsp;'))
        writer.add_text('specified_config', json.dumps(specified_cfg, indent=4).replace('\n', '<br/>').replace(' ', '&nbsp;'))
        
        print(f'Training...')
        train(
            model,
            optimiser,
            train_set,
            val_set,
            writer=writer,
            cfg=cfg,
        )

        # linear probing
        if cfg['log']:
            print(f'Evaluating...')
            # for n in [1, 10, 100, 1000]:
            for n in cfg['classifier_subset_sizes']:
                writer = get_writer(cfg, n)
                linear_probing(model, writer, n, cfg)
        else:
            print('No logging, skipping linear probing')
        
        print(f'Done - Time taken: {time.time() - start_time:.2f}s')