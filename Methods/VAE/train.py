import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as F_v2
from tqdm import tqdm
from Examples.MNIST.mnist_linear_1k import single_step_classification_eval, get_mnist_subset_loaders
from Utils.functional import smooth_l1_loss, cosine_schedule

import os

def vae_loss(recon_x, x, mu, logVar, beta=1.0):
    reconstruction_loss = F.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')
    mse = F.mse_loss(F.sigmoid(recon_x), x) * x.shape[0]
    kl_loss = -0.5 * torch.sum(1 + logVar - mu.pow(2) - logVar.exp())
    return reconstruction_loss + beta * kl_loss, mse

def train(
        model,
        optimiser,
        train_dataset,
        val_dataset,
        num_epochs,
        batch_size,
        beta=None,
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

# ============================== Data Handling ==============================
    ss_train_loader, ss_val_loader = get_mnist_subset_loaders(1, batch_size, device)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ============================== Training Stuff ==============================
    scaler = torch.cuda.amp.GradScaler()

    train_options = {
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'beta': beta,
        'transform': train_dataset.transform,
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
        epoch_train_mses = torch.zeros(len(train_loader), device=device)
        for i, (images, _) in loop:

            with torch.cuda.amp.autocast():
                x_hat, mu, logVar = model.reconstruct(images)
                loss, mse = vae_loss(x_hat, images, mu, logVar, beta)

            # Update model
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad(set_to_none=True)

            epoch_train_losses[i] = loss.detach()
            epoch_train_mses[i] = mse.detach()
        
        # Validation Pass
        with torch.no_grad():
            epoch_val_losses = torch.zeros(len(val_loader), device=device)
            epoch_val_mses = torch.zeros(len(val_loader), device=device)
            for i, (images, _) in enumerate(val_loader):

                with torch.cuda.amp.autocast():
                    x_hat, mu, logVar = model.reconstruct(images)
                    loss, mse = vae_loss(x_hat, images, mu, logVar, beta)

                epoch_val_losses[i] = loss.detach()
                epoch_val_mses[i] = mse.detach()

        # single step linear classification eval
        ss_val_acc, ss_val_loss = single_step_classification_eval(model, ss_train_loader, ss_val_loader, scaler, learn_on_ss)
        if learn_on_ss:
            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad(set_to_none=True)
        
        last_train_loss = epoch_train_losses.sum().item() / len(train_dataset)
        last_train_mse = epoch_train_mses.sum().item() / len(train_dataset)
        last_val_loss = epoch_val_losses.sum().item() / len(val_dataset)
        last_val_mse = epoch_val_mses.sum().item() / len(val_dataset)
        postfix = {'train_loss': last_train_loss, 'train_mse': last_train_mse, 'val_loss': last_val_loss, 'val_mse': last_val_mse, '1step_val_acc': ss_val_acc, '1step_val_loss': ss_val_loss}
        loop.set_postfix(postfix)
        if writer is not None:
            writer.add_scalar('Encoder/train_loss', last_train_loss, epoch)
            writer.add_scalar('Encoder/val_loss', last_val_loss, epoch)
            writer.add_scalar('Encoder/1step_val_acc', ss_val_acc, epoch)
            writer.add_scalar('Encoder/1step_val_loss', ss_val_loss, epoch)

        if ss_val_loss < best_val_loss and save_dir is not None and epoch % save_every == 0:
            best_val_loss = ss_val_loss
            torch.save(model.state_dict(), save_dir)