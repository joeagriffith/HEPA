import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as F_v2
from tqdm import tqdm
from Examples.MNIST.mnist_linear_1k import single_step_classification_eval, get_mnist_subset_loaders, get_rep_metrics
from Utils.functional import smooth_l1_loss, cosine_schedule
from Utils.functional import augment

import os

class DINOLoss(torch.nn.Module):

    def __init__(self, num_epochs, num_features, C_mom=0.9, scale_temps=1.0, device='cpu'):
        super().__init__()
        # Temperature schedule
        self.tmp_s = torch.ones(num_epochs) * 0.1 * scale_temps
        self.tmp_t = torch.cat([torch.linspace(0.04, 0.07, 30) * scale_temps, torch.ones(num_epochs-30) * 0.07 * scale_temps])

        # Initialise C
        self.C = torch.zeros((1, num_features), device=device)
        self.C_mom = C_mom
    
    def update_C(self, t):
        # t: (batch_size, num_features)
        target = t.mean(0, keepdim=True)
        self.C = self.C_mom * self.C + (1 - self.C_mom) * target

    def forward(self, s, t, epoch):
        # s, t: (batch_size, num_features)

        tmp_s, tmp_t = self.tmp_s[epoch], self.tmp_t[epoch]

        # Convert to probabilities
        # (batch_size, num_features) -> (batch_size, num_features)
        s = F.softmax(s / tmp_s, dim=-1)
        t = F.softmax((t - self.C) / tmp_t, dim=-1)

        # Update C
        self.update_C(t)

        # # Calculate loss for CLS tokens across different images
        # (batch_size, num_features) -> (1,)
        loss = -(t * s.log()).sum(-1).mean()

        return loss

def train(
        online_model,
        optimiser,
        train_dataset,
        val_dataset,
        num_epochs,
        batch_size,
        stop_at,
        train_aug_scaler='none',
        val_aug_scaler='none',
        loss_fn='mse',
        learn_on_ss=False,
        writer=None,
        save_dir=None,
        save_every=1,
):

    device = next(online_model.parameters()).device

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
    assert train_aug_scaler in ['linear', 'exp', 'cosine', 'zeros', 'none', 'ones'], 'aug_scaler must be one of ["linear", "exp", "cosine", "zeros", "none", "ones"]'
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
    elif train_aug_scaler == 'ones':
        aug_ps = torch.ones(num_epochs)
    
    # Initialise validation augmentation probabilty schedule
    assert val_aug_scaler in ['linear', 'exp', 'cosine', 'none', 'zeros', 'ones'], 'aug_scaler must be one of ["linear", "exp", "cosine", "zeros", "none", "ones"]'
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
    elif val_aug_scaler == 'ones':
        val_aug_ps = torch.ones(num_epochs)
    
#============================== Target Model Learning Parameters ==============================
    # Initialise target model
    target_model = online_model.copy()
    # EMA schedule, cosine
    start_tau=0.996
    end_tau = 1.0
    taus = cosine_schedule(start_tau, end_tau, num_epochs)

# ============================== Data Handling ==============================
    ss_train_loader, ss_val_loader = get_mnist_subset_loaders(1, batch_size, device=device)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ============================== Training Stuff ==============================
    if loss_fn == 'mse':
        loss_fn = lambda x, y, _: F.mse_loss(x, y, reduction='none').sum(dim=(-1)).mean()
    elif loss_fn == 'normalised_mse':
        loss_fn = lambda x, y, _: F.mse_loss(F.normalize(x, dim=-1), F.normalize(y, dim=-1), reduction='none').sum(dim=(-1)).mean()
    elif loss_fn == 'dino':
        loss_fn = DINOLoss(num_epochs, online_model.num_features, C_mom=0.9, scale_temps=1.0, device=device)
    else:
        raise ValueError('loss_fn must be one of ["mse", "normalised_mse", "dino"]')


    train_options = {
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'train_aug_scaler': train_aug_scaler,
        'val_aug_scaler': val_aug_scaler,
        'stop_at': stop_at,
        'loss_fn': loss_fn,
        'transform': train_dataset.transform,
    }

    # Log training options, model details, and optimiser details
    if writer is not None:
        writer.add_text('Encoder/options', str(train_options))
        writer.add_text('Encoder/model', str(online_model).replace('\n', '<br/>').replace(' ', '&nbsp;'))
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
        online_model.train()
        target_model.train()
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
            images_aug, action = augment(images, aug_ps[epoch])
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                with torch.no_grad():
                    targets = target_model(images_aug, stop_at)
                preds = online_model.predict(images, action, stop_at)
                loss = loss_fn(preds, targets, epoch)

            # Update model
            loss.backward()
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)

            # Update target model
            with torch.no_grad():
                for o_param, t_param in zip(online_model.parameters(), target_model.parameters()):
                    t_param.data = taus[epoch] * t_param.data + (1 - taus[epoch]) * o_param.data

            epoch_train_losses[i] = loss.detach()
        
        # Validation Pass
        online_model.eval()
        target_model.eval()
        with torch.no_grad():
            epoch_val_losses = torch.zeros(len(val_loader), device=device)
            for i, (images, _) in enumerate(val_loader):
                # Create Target Image and Action vector
                images_aug, action = augment(images, val_aug_ps[epoch])
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    targets = target_model(images_aug, stop_at)
                    preds = online_model.predict(images, action, stop_at)
                    loss = loss_fn(preds, targets, epoch)

                epoch_val_losses[i] = loss.detach()

        # evaluate representations
        if writer is not None:
            rep_metrics = get_rep_metrics(online_model, val_dataset)
            writer.add_scalar('Encoder/feature_corr', rep_metrics['corr'], epoch)
            writer.add_scalar('Encoder/feature_std', rep_metrics['std'], epoch)

        # single step linear classification eval
        ss_val_acc, ss_val_loss = single_step_classification_eval(online_model, ss_train_loader, ss_val_loader)
        
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
            torch.save(online_model.state_dict(), save_dir)