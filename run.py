import os
import sys
import time
import json
import yaml

import torch
import torchvision.transforms as transforms

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from Utils.train import train
from Utils.evals import linear_probing
from Utils.utils import get_optimiser, get_model, get_datasets, get_writer
from Utils.cfg import mnist_cfg, modelnet10_cfg

if __name__ == '__main__':

    # ======================== Handle configs =======================
    args = sys.argv[1:]
    assert len(args) == 1, 'Please provide a config file'
    filepath = 'experiments/' + args[0]
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

    # ======================== Handle devices =======================
    # DDP Code from Karpathy @ https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    else:
        # vanilla, non-DDP run
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        # attempt to autodetect device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"using device: {device}")
    

    for (cfg, specified_cfg) in cfgs:
        assert cfg['device'] == device.split(':')[0], f'Device mismatch: {cfg["device"]} != {device}'

        torch.manual_seed(cfg['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg['seed'])

        writer = None
        if master_process:
            if cfg['log']:
                if not os.path.exists(cfg['log_dir']):
                    os.makedirs(cfg['log_dir'])
                writer = get_writer(cfg)
            if cfg['save']:
                trial_save_dir = cfg['save_dir'] + cfg['experiment'] + '/' + cfg['trial'] + '/'
                if not os.path.exists(trial_save_dir):
                    os.makedirs(trial_save_dir)
                cfg['save_dir'] = trial_save_dir + str(cfg['run_no']) + '.pth'

            print(f'\n======================  Experiment: {cfg["experiment"]} == Trial: {cfg["trial"]} == Run: {cfg["run_no"]} ======================')
            start_time = time.time()

        cfg['ddp'] = ddp
        cfg['ddp_rank'] = ddp_rank
        cfg['ddp_local_rank'] = ddp_local_rank
        cfg['ddp_world_size'] = ddp_world_size
        cfg['master_process'] = master_process

        if cfg['device'] == 'cuda':
            torch.backends.cudnn.benchmark = True

        # Init Model
        model = get_model(cfg)
        if cfg['use_compile']:
            model = torch.compile(model)
        if ddp:
            model = DDP(model, device_ids=[ddp_local_rank])
        raw_model = model.module if ddp else model

        # Init Optimiser
        optimiser = get_optimiser(raw_model, cfg)
        
        # Init Dataset
        train_set, val_set = get_datasets(cfg)
        if cfg['data_aug']:
            train_set.transform = transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10)

        # Log hyperparameters
        if master_process:
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

        if master_process:
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