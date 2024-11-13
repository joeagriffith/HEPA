#=========================== shortcut cfg initialisers ===========================

def mnist_cfg(
        experiment:str, 
        trial:str, 
        model_type:str, 
        **kwargs
    ):

    # The subset of non-default values
    specified_cfg = kwargs.copy()
    specified_cfg['experiment'] = experiment
    specified_cfg['trial'] = trial
    specified_cfg['model_type'] = model_type

    enforce_cfg = {
        'dataset': 'mnist',
        'root': '../../datasets/',
        'log_dir': 'out/MNIST/logs/',
        'save_dir': 'out/MNIST/models/',
        'batch_size': 256,
        'in_features': 1,
        'resolution': 28,
        'num_actions': 5,
        # 'num_actions': 4,
        # 'num_actions': 2,
        'patch_size': 7,
        'min_keep': 1,
        'classifier_subset_sizes': [1, 10, 100, 1000],
    }

    for key, value in enforce_cfg.items():
        if key in kwargs:
            assert kwargs[key] == value, f"Specified {key} does not agree with enforced configuration. Must be {value}."
        else:
            kwargs[key] = value
    
    return base_cfg(experiment, trial, model_type, **kwargs), specified_cfg

def modelnet10_cfg(
        experiment:str, 
        trial:str, 
        model_type:str, 
        resolution:int=128, 
        dataset_dtype:str='uint8',
        action_type:str='quaternion_delta',
        **kwargs
    ):

    # The subset of non-default values
    specified_cfg = kwargs.copy()
    specified_cfg['experiment'] = experiment
    specified_cfg['trial'] = trial
    specified_cfg['model_type'] = model_type
    specified_cfg['resolution'] = resolution
    specified_cfg['dataset_dtype'] = dataset_dtype
    specified_cfg['action_type'] = action_type

    enforce_cfg = {
        'dataset': 'modelnet10',
        'root': '../../datasets/',
        'log_dir': 'out/ModelNet10/logs/',
        'save_dir': 'out/ModelNet10/models/',
        'batch_size': 64,
        'in_features': 1,
        'patch_size': 16,
        'min_keep': 4,
        'classifier_subset_sizes': [1, 10, 50],
    }
    if action_type == 'euler_delta':
        enforce_cfg['num_actions'] = 3
    elif action_type == 'quaternion_delta':
        enforce_cfg['num_actions'] = 4
    elif action_type == 'axis_angle':
        enforce_cfg['num_actions'] = 4
    else:
        raise NotImplementedError(f'Action type {action_type} not implemented')

    for key, value in enforce_cfg.items():
        if key in kwargs:
            assert kwargs[key] == value, f"Specified {key} does not agree with enforced configuration. Must be {value}."
        else:
            kwargs[key] = value

    kwargs['resolution'] = resolution
    kwargs['dataset_dtype'] = dataset_dtype
    kwargs['action_type'] = action_type
    
    return base_cfg(experiment, trial, model_type, **kwargs), specified_cfg

def voxceleb1_cfg(
        experiment:str, 
        trial:str, 
        model_type:str, 
        **kwargs
    ):

    # The subset of non-default values
    specified_cfg = kwargs.copy()
    specified_cfg['experiment'] = experiment
    specified_cfg['trial'] = trial
    specified_cfg['model_type'] = model_type

    enforce_cfg = {
        'dataset': 'voxceleb1',
        'root': '../../datasets/',
        'log_dir': 'out/VoxCeleb1/logs/',
        'save_dir': 'out/VoxCeleb1/models/',
        'batch_size': 256,
        'in_features': 1,
        'resolution': 1,
        'num_actions': 5,
        # 'num_actions': 4,
        # 'num_actions': 2,
        'patch_size': 7,
        'min_keep': 1,
        'classifier_subset_sizes': [1, 10, 100, 1000],
    }

    for key, value in enforce_cfg.items():
        if key in kwargs:
            assert kwargs[key] == value, f"Specified {key} does not agree with enforced configuration. Must be {value}."
        else:
            kwargs[key] = value
    
    return base_cfg(experiment, trial, model_type, **kwargs), specified_cfg


#=========================== base cfg initialiser ===========================

def base_cfg(
        experiment:str, 
        trial:str,
        model_type:str, 
        dataset:str,
        root:str,
        log_dir:str,
        save_dir:str,
        batch_size:int,
        in_features: int,
        resolution: int,
        num_actions: int,
        patch_size: int,
        min_keep: int,
        **kwargs,
    ):
    cfg = {
        'experiment': experiment,
        'trial': trial,
        'device': 'cuda',
        'use_compile': False,
        'seed': -1,
        'local': True,
        'profile': False,

        'dataset': dataset,
        'dataset_dtype': 'float32',
        'action_type': 'euler_delta',
        'root': root,
        'log_dir': log_dir,
        'save_dir': save_dir,
        'resolution': resolution,
        'train_ratio': 0.8,
        'data_aug': False,

        'model_type': model_type,
        'in_features': in_features,

        'log': True,
        'save': True,
        'save_every': 10,
        'save_copy_every': None,

        'optimiser': 'AdamW',
        'start_lr': 3e-5 if model_type in ['BYOL'] else 3e-4,
        'end_lr': 1e-6,
        'start_wd': 0.04,
        'end_wd': 0.4,
        'batch_size': batch_size,
        'num_epochs': 250 if dataset == 'mnist' else 1500,
        'stop_learning_at': 250 if dataset == 'mnist' else 1000,
        'exclude_bias': True,
        'exclude_bn': True,
        'decay_lr': True,
        'warmup': 10,
        'flat': 0,

        'has_teacher': False,
        'bn_output': False,

        'track_feature_corrs': True,
        'track_feature_stds': True,
        'track_feature_entropy': True,
        'classifier_subset_sizes': [1, 10, 100, 1000],

        'ddp_rank': 0,
        'ddp_world_size': 1,
        'master_process': True,
        'local': True,
        'profile': False,
    }

    if cfg['optimiser'] == 'AdamW':
        cfg['betas'] = (0.9, 0.999)
    elif cfg['optimiser'] == 'SGD':
        cfg['momentum'] = 0.9
    else:
        raise ValueError(f"Optimiser '{cfg['optimiser']}' is not supported.")

    enforce_cfg = {}
    if cfg['model_type'] == 'GPA':
        enforce_cfg['has_teacher'] = True

        # GPA specific optionals
        cfg['num_actions'] = num_actions
        cfg['stop_at'] = 0
        cfg['start_tau'] = 0.996
        cfg['end_tau'] = 1.0
        cfg['consider_actions'] = True
        cfg['p'] = 0.25
        cfg['save_metric'] = 'val_loss'
    
    elif cfg['model_type'] == 'BYOL':
        enforce_cfg['has_teacher'] = True

        cfg['start_tau'] = 0.996
        cfg['end_tau'] = 1.0
        cfg['bn_output'] = True
        cfg['save_metric'] = 'none'
    
    elif cfg['model_type'] == 'JEPA':
        enforce_cfg['has_teacher'] = True

        cfg['start_tau'] = 0.996
        cfg['end_tau'] = 1.0
        cfg['bn_output'] = True
        cfg['num_actions'] = num_actions
        cfg['consider_actions'] = True
        cfg['p'] = 0.25
        cfg['save_metric'] = 'none'

    elif cfg['model_type'] == 'iJEPA':
        enforce_cfg['has_teacher'] = True
        
        cfg['start_tau'] = 0.996
        cfg['end_tau'] = 1.0
        cfg['patch_size'] = patch_size
        cfg['min_keep'] = min_keep
        cfg['save_metric'] = 'none'
   
    elif cfg['model_type'] in ['AE', 'MAE', 'Supervised']:
        enforce_cfg['has_teacher'] = False

        # class specific optionals
        cfg['data_aug'] = True
        cfg['save_metric'] = 'val_loss'

    elif cfg['model_type'] == 'VAE':
        enforce_cfg['has_teacher'] = False

        # VAE specific optionals
        cfg['z_dim'] = 256 
        cfg['data_aug'] = True
        cfg['save_metric'] = 'val_loss'

    for key, value in enforce_cfg.items():
        if key in kwargs:
            assert kwargs[key] == value, f"Specified {key} does not agree with enforced configuration. Must be {value}."
        else:
            kwargs[key] = value

    for key, value in kwargs.items():
        if key not in cfg:
            raise ValueError(f"'{key}' is not a valid configuration parameter.")
        cfg[key] = value

    # Conditionals
    if cfg['model_type'] == 'GPA':
        if cfg['stop_at'] == 0:
            cfg['has_teacher'] = False

    return cfg
