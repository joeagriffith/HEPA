import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as F_v2
from tqdm import tqdm
from Examples.MNIST.mnist_linear_1k import single_step_classification_eval, get_mnist_subset_loaders, get_rep_metrics
from Utils.functional import smooth_l1_loss, cosine_schedule

import os

def train(
        model,
        optimiser,
        train_dataset,
        val_dataset,
        num_epochs,
        batch_size,
        dataset,
        has_teacher=False,
        aug_mode='none',
        augment=None,
        writer=None,
        save_dir=None,
        save_every=1,
):

    device = next(model.parameters()).device
    assert aug_mode in ['none', 'augment', 'sample']
    if aug_mode == 'augment':
        assert augment is not None, 'augment must be provided if aug_mode is "augment"'
    elif aug_mode == 'sample':
        assert augment is None, 'augment must be None if aug_mode is "sample"'
    assert dataset in ['mnist', 'modelnet10'], 'dataset must be one of ["mnist", "modelnet10"]'

#============================== Online Model Learning Parameters ==============================
    # LR schedule, warmup then cosine
    base_lr = optimiser.param_groups[0]['lr'] * batch_size / 256
    end_lr = 1e-6
    warm_up_lrs = torch.linspace(0, base_lr, 11)[1:]
    cosine_lrs = cosine_schedule(base_lr, end_lr, num_epochs-10)
    lrs = torch.cat([warm_up_lrs, cosine_lrs])
    assert len(lrs) == num_epochs

    # WD schedule, cosine 
    start_wd = 0.04
    end_wd = 0.4
    wds = cosine_schedule(start_wd, end_wd, num_epochs)

#============================== Target Model Learning Parameters ==============================
    if has_teacher:
        # Initialise target model
        teacher = model.copy()
        # EMA schedule, cosine
        start_tau=0.996
        end_tau = 1.0
        taus = cosine_schedule(start_tau, end_tau, num_epochs)
    else:
        teacher = None

# ============================== Data Handling ==============================
    ss_train_loader, ss_val_loader = get_mnist_subset_loaders(1, batch_size, device=device)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ============================== Training Stuff ==============================

    train_options = {
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'has_teacher': has_teacher,
        'train_transform': str(train_dataset.transform),
        'val_transform': str(val_dataset.transform),
    }

    # Log training options, model details, and optimiser details
    if writer is not None:
        writer.add_text('Encoder/options', str(train_options))
        writer.add_text('Encoder/model', str(model).replace('\n', '<br/>').replace(' ', '&nbsp;'))
        writer.add_text('Encoder/optimiser', str(optimiser).replace('\n', '<br/>').replace(' ', '&nbsp;'))

    # Initialise training variables
    last_train_loss = -1
    last_val_loss = -1
    best_val_loss = float('inf')
    postfix = {}

    if save_dir is not None:# and not os.path.exists(save_dir):
        parent_dir = save_dir.rsplit('/', 1)[0]
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

# ============================== Training Loop ==============================
    for epoch in range(num_epochs):
        model.train()
        if teacher:
            teacher.train()
        train_dataset.apply_transform(batch_size=batch_size)
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
        if epoch > 0:
            loop.set_postfix(postfix)

        # Update lr
        for param_group in optimiser.param_groups:
            param_group['lr'] = lrs[epoch].item()
        # Update wd
        for param_group in optimiser.param_groups:
            if param_group['weight_decay'] != 0:
                param_group['weight_decay'] = wds[epoch].item()

        # Training Pass
        epoch_train_losses = torch.zeros(len(train_loader), device=device)
        for i, data in loop:
            actions = None
            if dataset == 'mnist':
                images1, _ = data
            elif dataset == 'modelnet10':
                (images1, rot1, _), (images2, rot2, _) = data
                actions = (rot2 - rot1) / 360.0

            if aug_mode == 'augment':
                images2, actions = augment(images1, 0.25)
            elif aug_mode == 'sample':
                assert dataset == 'modelnet10', 'sample mode only implemented for modelnet10'
            else:
                images2 = None

            loss = model.train_step(images1, images2, actions, teacher, epoch)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)

            if has_teacher:
                # Update target model
                with torch.no_grad():
                    for o_param, t_param in zip(model.parameters(), teacher.parameters()):
                        t_param.data = taus[epoch] * t_param.data + (1 - taus[epoch]) * o_param.data

            epoch_train_losses[i] = loss.detach()

        # Validation Pass
        model.eval()
        if has_teacher:
            teacher.eval()
        with torch.no_grad():
            epoch_val_losses = torch.zeros(len(val_loader), device=device)
            for i, data in enumerate(val_loader):
                actions = None
                if dataset == 'mnist':
                    images1, _ = data
                elif dataset == 'modelnet10':
                    (images1, rot1, _), (images2, rot2, _) = data
                    actions = (rot2 - rot1) / 360.0

                if aug_mode == 'augment':
                    images2, actions = augment(images1, 0.25)
                elif aug_mode == 'sample':
                    assert dataset == 'modelnet10', 'sample mode only implemented for modelnet10'
                else:
                    images2 = None

                loss = model.train_step(images1, images2, actions, teacher, epoch)
                epoch_val_losses[i] = loss.detach()

        # evaluate representations
        if writer is not None:
            rep_metrics = get_rep_metrics(model, val_dataset)
            writer.add_scalar('Encoder/feature_corr', rep_metrics['corr'], epoch)
            writer.add_scalar('Encoder/feature_std', rep_metrics['std'], epoch)
        
        # single step linear classification eval
        ss_val_acc, ss_val_loss = single_step_classification_eval(model, ss_train_loader, ss_val_loader)
        
        last_train_loss = epoch_train_losses.mean().item()
        last_val_loss = epoch_val_losses.mean().item()
        postfix = {'train_loss': last_train_loss, 'val_loss': last_val_loss}
        if writer is not None:
            writer.add_scalar('Encoder/train_loss', last_train_loss, epoch)
            writer.add_scalar('Encoder/val_loss', last_val_loss, epoch)
            writer.add_scalar('Encoder/1step_val_acc', ss_val_acc, epoch)
            writer.add_scalar('Encoder/1step_val_loss', ss_val_loss, epoch)

        if ss_val_loss < best_val_loss and save_dir is not None and epoch % save_every == 0:
            best_val_loss = ss_val_loss
            torch.save(model.state_dict(), save_dir)