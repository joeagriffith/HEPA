import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as F_v2
from tqdm import tqdm
from Examples.MNIST.mnist_linear_1k import single_step_classification_eval, get_ss_mnist_loaders
from Utils.functional import smooth_l1_loss, cosine_schedule

import os


def train(
        model,
        optimiser,
        train_dataset,
        val_dataset,
        num_epochs,
        batch_size,
        beta=None,
        train_aug_scaler='none',
        val_aug_scaler='none',
        learn_on_ss=False,
        writer=None,
        save_dir=None,
        save_every=1,
):

    device = next(model.parameters()).device

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

#============================== Augmentation Parameters ==============================
    # Initialise augmentation probabilty schedule
    assert train_aug_scaler in ['linear', 'exp', 'cosine', 'zeros', 'none'], 'aug_scaler must be one of ["linear", "exp", "cosine", "zeros", "none"]'
    if train_aug_scaler == 'linear':
        aug_ps = torch.linspace(0.0, 0.25, num_epochs)
    elif train_aug_scaler == 'exp':
        aug_ps = 0.25 * (1.0 - torch.exp(torch.linspace(0, -5, num_epochs)))
    elif train_aug_scaler == 'cosine':
        aug_ps = cosine_schedule(0.0, 0.25, num_epochs)
    elif train_aug_scaler == 'zeros':
        aug_ps = torch.zeros(num_epochs)
    elif train_aug_scaler == 'none':
        aug_ps = 0.25 * torch.ones(num_epochs)
    
    # Initialise validation augmentation probabilty schedule
    assert val_aug_scaler in ['linear', 'exp', 'cosine', 'none', 'zeros'], 'aug_scaler must be one of ["linear", "exp", "cosine", "zeros", "none"]'
    if val_aug_scaler == 'linear':
        val_aug_ps = torch.linspace(0, 0.30, num_epochs)
    elif val_aug_scaler == 'exp':
        val_aug_ps = 0.25 * (1.0 - torch.exp(torch.linspace(0, -5, num_epochs)))
    elif val_aug_scaler == 'cosine':
        val_aug_ps = cosine_schedule(0.0, 0.30, num_epochs)
    elif val_aug_scaler == 'zeros':
        val_aug_ps = torch.zeros(num_epochs)
    elif val_aug_scaler == 'none':
        val_aug_ps = 0.25 * torch.ones(num_epochs)
    
# ============================== Data Handling ==============================
    ss_train_loader, ss_val_loader = get_ss_mnist_loaders(batch_size, device=device)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ============================== Training Stuff ==============================
    scaler = torch.cuda.amp.GradScaler()

    train_options = {
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'beta': beta,
        'train_aug_scaler': train_aug_scaler,
        'val_aug_scaler': val_aug_scaler,
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
        for i, (images, _) in loop:
            # Sample Action
            act_p = torch.rand(5) # whether to apply each augmentation
            angle = torch.rand(1).item() * 360 - 180 if act_p[0] < aug_ps[epoch] else 0
            translate_x = torch.randint(-8, 9, (1,)).item() if act_p[1] < aug_ps[epoch] else 0
            translate_y = torch.randint(-8, 9, (1,)).item() if act_p[2] < aug_ps[epoch] else 0
            scale = torch.rand(1).item() * 0.5 + 0.75 if act_p[3] < aug_ps[epoch] else 1.0
            shear = torch.rand(1).item() * 50 - 25 if act_p[4] < aug_ps[epoch] else 0
            targets = F_v2.affine(images, angle=angle, translate=(translate_x, translate_y), scale=scale, shear=shear)
            action = torch.tensor([angle/180, translate_x/8, translate_y/8, (scale-1.0)/0.25, shear/25], dtype=torch.float32, device=images.device).unsqueeze(0).repeat(images.shape[0], 1)

            with torch.cuda.amp.autocast():
                preds = model.predict(images, action)

                if beta is None:
                    loss = F.mse_loss(preds, targets)
                else:
                    loss = smooth_l1_loss(preds, targets, beta)

            # Update model
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad(set_to_none=True)

            epoch_train_losses[i] = loss.detach()
        
        # Validation Pass
        model.eval()
        with torch.no_grad():
            epoch_val_losses = torch.zeros(len(val_loader), device=device)
            for i, (images, _) in enumerate(val_loader):

                # Create Target Image and Action vector
                act_p = torch.rand(5) # whether to apply each augmentation
                angle = torch.rand(1).item() * 360 - 180 if act_p[0] < val_aug_ps[epoch] else 0
                translate_x = torch.randint(-8, 9, (1,)).item() if act_p[1] < val_aug_ps[epoch] else 0
                translate_y = torch.randint(-8, 9, (1,)).item() if act_p[2] < val_aug_ps[epoch] else 0
                scale = torch.rand(1).item() * 0.5 + 0.75 if act_p[3] < val_aug_ps[epoch] else 1.0
                shear = torch.rand(1).item() * 50 - 25 if act_p[4] < val_aug_ps[epoch] else 0
                targets = F_v2.affine(images, angle=angle, translate=(translate_x, translate_y), scale=scale, shear=shear)
                action = torch.tensor([angle/180, translate_x/8, translate_y/8, (scale-1.0)/0.25, shear/25], dtype=torch.float32, device=images.device).unsqueeze(0).repeat(images.shape[0], 1)

                with torch.cuda.amp.autocast():
                    preds = model.predict(images, action)

                    if beta is None:
                        loss = F.mse_loss(preds, targets)
                    else:
                        loss = smooth_l1_loss(preds, targets, beta)

                epoch_val_losses[i] = loss.detach()

        # single step linear classification eval
        ss_val_acc, ss_val_loss = single_step_classification_eval(model, ss_train_loader, ss_val_loader, scaler, learn_on_ss)
        if learn_on_ss:
            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad(set_to_none=True)
        
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