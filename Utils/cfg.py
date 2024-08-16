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
        'root': '../Datasets/',
        'log_dir': 'out/MNIST/logs/',
        'save_dir': 'out/MNIST/models/',
        'batch_size': 256,
        'num_epochs': 250,
        'stop_learning_at': None,
        'in_features': 1,
        'resolution': 28,
        'num_actions': 5,
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
        **kwargs
    ):

    # The subset of non-default values
    specified_cfg = kwargs.copy()
    specified_cfg['experiment'] = experiment
    specified_cfg['trial'] = trial
    specified_cfg['model_type'] = model_type
    specified_cfg['resolution'] = resolution
    specified_cfg['dataset_dtype'] = dataset_dtype

    enforce_cfg = {
        'dataset': 'modelnet10',
        'root': '../Datasets/',
        'log_dir': 'out/ModelNet10/logs/',
        'save_dir': 'out/ModelNet10/models/',
        'batch_size': 64,
        'num_epochs': 1500,
        'stop_learning_at': 1000,
        'in_features': 1,
        'num_actions': 4,
        'patch_size': 16,
        'min_keep': 4,
        'classifier_subset_sizes': [1, 10, 50],
    }

    for key, value in enforce_cfg.items():
        if key in kwargs:
            assert kwargs[key] == value, f"Specified {key} does not agree with enforced configuration. Must be {value}."
        else:
            kwargs[key] = value

    kwargs['resolution'] = resolution
    kwargs['dataset_dtype'] = dataset_dtype
    
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
        num_epochs:int,
        stop_learning_at:int,
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
        'save_every': 1,

        'optimiser': 'AdamW',
        'start_lr': 3e-5 if model_type in ['BYOL'] else 3e-4,
        'end_lr': 1e-6,
        'start_wd': 0.04,
        'end_wd': 0.4,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'stop_learning_at': stop_learning_at,
        'exclude_bias': True,
        'exclude_bn': True,
        'decay_lr': True,
        'warmup': 10,
        'flat': 0,

        'has_teacher': False,
        'bn_output': False,

        'track_feature_corrs': True,
        'track_feature_stds': True,
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
    if cfg['model_type'] == 'iGPA':
        enforce_cfg['has_teacher'] = True

        # iGPA specific optionals
        cfg['num_actions'] = num_actions
        cfg['stop_at'] = 0
        cfg['start_tau'] = 0.996
        cfg['end_tau'] = 1.0
        cfg['consider_actions'] = True
        cfg['p'] = 0.25
    
    elif cfg['model_type'] == 'BYOL':
        enforce_cfg['has_teacher'] = True

        cfg['start_tau'] = 0.996
        cfg['end_tau'] = 1.0
        cfg['bn_output'] = True
    
    elif cfg['model_type'] == 'BYOPL':
        enforce_cfg['has_teacher'] = True

        cfg['start_tau'] = 0.996
        cfg['end_tau'] = 1.0
        cfg['bn_output'] = True
        cfg['num_actions'] = num_actions
        cfg['consider_actions'] = True
        cfg['p'] = 0.25

    elif cfg['model_type'] == 'iJEPA':
        enforce_cfg['has_teacher'] = True
        
        cfg['start_tau'] = 0.996
        cfg['end_tau'] = 1.0
        cfg['patch_size'] = patch_size
        cfg['min_keep'] = min_keep
   
    elif cfg['model_type'] in ['AE', 'MAE', 'Supervised']:
        enforce_cfg['has_teacher'] = False

        # class specific optionals
        cfg['data_aug'] = True

    elif cfg['model_type'] == 'VAE':
        enforce_cfg['has_teacher'] = False

        # VAE specific optionals
        cfg['z_dim'] = 256 
        cfg['data_aug'] = True


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
    if cfg['model_type'] == 'iGPA':
        if cfg['stop_at'] == 0:
            cfg['has_teacher'] = False

    return cfg
