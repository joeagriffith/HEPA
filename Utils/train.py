import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from tqdm import tqdm
from Utils.functional import cosine_schedule, aug_interact, aug_transform
from Utils.evals import one_step_linear_probing, eval_representations, get_rep_metrics
from Utils.functional import quaternion_delta, axis_angle
from Utils.utils import get_ss_datasets

import os

def train(
        model,
        optimiser,
        train_dataset,
        val_dataset,
        writer,
        cfg:dict,
):

    device = cfg['device'] + ':' + str(cfg['ddp_rank'])

    if cfg['transformation_fn'] == 'interact':
        transform = aug_interact
    elif cfg['transformation_fn'] == 'perturb':
        transform = aug_transform
    else:
        assert cfg['transformation_fn'] is None, 'transformation function must be "interact", "perturb" or None'

    assert cfg['aug_mode'] in ['none', 'transform', 'sample']
    if cfg['aug_mode'] == 'transform':
        assert cfg['transformation_fn'] is not None, 'transformation must be provided if aug_mode is "transform"'
    assert cfg['dataset'] in ['mnist', 'modelnet10'], 'dataset must be one of ["mnist", "modelnet10"]'

#============================== Online Model Learning Parameters ==============================
    # LR schedule, warmup then cosine
    assert cfg['warmup'] + cfg['flat'] <= cfg['num_epochs'], f'warmup must be less than or equal to num_epochs, got {cfg["warmup"]} and {cfg["num_epochs"]}'
    start_lr = cfg['start_lr'] * cfg['batch_size'] / 256
    end_lr = cfg['end_lr'] * cfg['batch_size'] / 256
    warm_up_lrs = torch.linspace(0, start_lr, cfg['warmup']+1)[1:]
    if cfg['num_epochs'] > cfg['warmup']+cfg['flat']:
        if cfg['decay_lr']:
            cosine_lrs = cosine_schedule(start_lr, end_lr, cfg['num_epochs']-cfg['warmup']-cfg['flat'])
        else:
            cosine_lrs = torch.ones(cfg['num_epochs']-cfg['warmup']-cfg['flat']) * start_lr
        lrs = torch.cat([warm_up_lrs, cosine_lrs])
    if cfg['flat'] > 0:
        lrs = torch.cat([lrs, torch.ones(cfg['flat']) * cfg['end_lr']])
    assert len(lrs) == cfg['num_epochs']

    # WD schedule, cosine 
    wds = cosine_schedule(cfg['start_wd'], cfg['end_wd'], cfg['num_epochs'])

#============================== Target Model Learning Parameters ==============================
    if cfg['has_teacher']:
        # Initialise target model
        teacher = model.copy()
        # EMA schedule, cosine
        taus = cosine_schedule(cfg['start_tau'], cfg['end_tau'], cfg['num_epochs'])
    else:
        teacher = None

# ============================== Data Handling ==============================
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False)

    ss_train_dataset, ss_val_dataset = get_ss_datasets(cfg)
    ss_train_loader = DataLoader(ss_train_dataset, batch_size=cfg['batch_size'], shuffle=True)
    ss_val_loader = DataLoader(ss_val_dataset, batch_size=1000, shuffle=False)

# ============================== Training Stuff ==============================

    # Initialise training variables
    last_train_loss = -1
    last_val_loss = -1
    best_val_loss = float('inf')
    postfix = {}

# ============================== Training Loop ==============================
    for epoch in range(cfg['num_epochs']):
        model.train()

        if teacher:
            teacher.train()

        train_dataset.apply_transform(batch_size=cfg['batch_size'])

        # Update lr
        for param_group in optimiser.param_groups:
            param_group['lr'] = lrs[epoch].item()
        # Update wd
        for param_group in optimiser.param_groups:
            if param_group['weight_decay'] != 0:
                param_group['weight_decay'] = wds[epoch].item()

        # Training Pass
        epoch_train_losses = torch.zeros(len(train_loader), device=device)
        epoch_train_norms = torch.zeros(len(train_loader), device=device)
        if cfg['master_process'] and cfg['local']:
            loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
            loop.set_description(f'Epoch [{epoch}/{cfg["num_epochs"]}]')
            if epoch > 0:
                loop.set_postfix(postfix)
        else:
            loop = enumerate(train_loader)
        for i, data in loop:

            images2 = None
            actions = None
            if cfg['dataset'] == 'mnist':
                images1, _ = data
            elif cfg['dataset'] == 'modelnet10':
                (images1, rot1, _), (images2, rot2, _) = data
                actions = (rot2 - rot1) / 360.0
                # actions = quaternion_delta(rot1, rot2)
                # actions = axis_angle(rot1, rot2)
            else:
                raise NotImplementedError(f'Dataset {cfg["dataset"]} not implemented')

            if images1.device != device:
                images1 = images1.to(device)
                if images2 is not None:
                    images2 = images2.to(device)
                if actions is not None:
                    actions = actions.to(device)

            if cfg['aug_mode'] == 'transform':
                images2, actions = transform(images1, 0.25)
            elif cfg['aug_mode'] == 'sample':
                assert cfg['dataset'] == 'modelnet10', 'sample mode only implemented for modelnet10'
            else:
                images2 = None
                actions = None

            loss = model.train_step(images1, images2, actions, teacher, epoch)
        
            loss.backward()

            epoch_train_norms[i] = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).detach()

            optimiser.step()

            optimiser.zero_grad(set_to_none=True)

            if cfg['has_teacher']:
                # Update target model
                with torch.no_grad():
                    for o_param, t_param in zip(model.parameters(), teacher.parameters()):
                        t_param.data = taus[epoch] * t_param.data + (1 - taus[epoch]) * o_param.data

            epoch_train_losses[i] = loss.detach()

        # Validation Pass
        model.eval()
        if cfg['has_teacher']:
            teacher.eval()
        with torch.no_grad():
            epoch_val_losses = torch.zeros(len(val_loader), device=device)
            for i, data in enumerate(val_loader):
                images2 = None
                actions = None
                if cfg['dataset'] == 'mnist':
                    images1, _ = data
                elif cfg['dataset'] == 'modelnet10':
                    (images1, rot1, _), (images2, rot2, _) = data
                    actions = (rot2 - rot1) / 360.0
                    # actions = quaternion_delta(rot1, rot2)
                    # actions = axis_angle(rot1, rot2)

                if images1.device != device:
                    images1 = images1.to(device)
                    if images2 is not None:
                        images2 = images2.to(device)
                    if actions is not None:
                        actions = actions.to(device)

                if cfg['aug_mode'] == 'transform':
                    images2, actions = transform(images1, 0.25)
                elif cfg['aug_mode'] == 'sample':
                    assert cfg['dataset'] == 'modelnet10', 'sample mode only implemented for modelnet10'
                else:
                    images2 = None
                    actions = None

                if cfg['model_type'] == 'iGPA' and not cfg['consider_actions']:
                    actions = None

                loss = model.train_step(images1, images2, actions, teacher, epoch)
                epoch_val_losses[i] = loss.detach()

        last_train_loss = epoch_train_losses.mean().item()
        last_train_norm = epoch_train_norms.mean().item()
        last_val_loss = epoch_val_losses.mean().item()
        if cfg['ddp']:
            dist.all_reduce(last_train_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(last_val_loss, op=dist.ReduceOp.AVG)
        
        if cfg['master_process']:
            # single step linear classification eval
            ss_val_acc, ss_val_loss = one_step_linear_probing(model, ss_train_loader, ss_val_loader)

            # evaluate representations
            if writer is not None:
                rep_metrics = get_rep_metrics(model, val_dataset, cfg)
                writer.add_scalar('val/feature_corr', rep_metrics['corr'], epoch)
                writer.add_scalar('val/feature_std', rep_metrics['std'], epoch)

            postfix = {'train_loss': last_train_loss, 'val_loss': last_val_loss}
            if writer is not None:
                writer.add_scalar('train/loss', last_train_loss, epoch)
                writer.add_scalar('train/norm', last_train_norm, epoch)
                writer.add_scalar('val/loss', last_val_loss, epoch)
                writer.add_scalar('val/1step_accuracy', ss_val_acc, epoch)
                writer.add_scalar('val/1step_loss', ss_val_loss, epoch)

            if ss_val_loss < best_val_loss and cfg['save'] and epoch % cfg['save_every'] == 0:
                best_val_loss = ss_val_loss
                torch.save(model.state_dict(), cfg['save_dir'])

    if cfg['master_process']:
        if writer is not None:
            rep_metrics = eval_representations(model, cfg)
            writer.add_scalar('test/feature_corr', rep_metrics['corr'], epoch)
            writer.add_scalar('test/feature_std', rep_metrics['std'], epoch)