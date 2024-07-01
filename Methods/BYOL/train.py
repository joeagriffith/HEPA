import torch
import torch.nn.functional as F
from tqdm import tqdm

from Utils.functional import cosine_schedule, smooth_l1_loss
from Examples.MNIST.mnist_linear_1k import single_step_classification_eval, get_mnist_subset_loaders

def regression_loss(pred, target):
    return 2 - 2 * (pred * target).sum(dim=-1)

def train(
        online_model,
        optimiser,
        train_dataset,
        val_dataset,
        num_epochs,
        batch_size,
        augmentation,
        beta=None,
        writer=None,
        save_dir=None,
        save_every=1,
):
    device = next(online_model.parameters()).device

#============================== Online Model Learning Parameters ==============================
    # LR schedule, warmup then cosine
    base_lr = optimiser.param_groups[0]['lr'] * batch_size / 256
    end_lr = 1e-6
    warm_up_lrs = torch.linspace(0, base_lr, 10)
    cosine_lrs = cosine_schedule(base_lr, end_lr, num_epochs-10)
    lrs = torch.cat([warm_up_lrs, cosine_lrs])
    assert len(lrs) == num_epochs

    # WD schedule, cosine 
    start_wd = 0.04
    end_wd = 0.4
    wds = cosine_schedule(start_wd, end_wd, num_epochs)
    
#============================== Target Model Learning Parameters ==============================
    # Initialise target model
    target_model = online_model.copy()
    # EMA schedule, cosine
    start_tau=0.996
    end_tau = 1.0
    taus = cosine_schedule(start_tau, end_tau, num_epochs)

# ============================== Data Handling ==============================
    # Initialise dataloaders for single step classification eval
    ss_train_loader, ss_val_loader = get_mnist_subset_loaders(1, batch_size, device=device)

    # Initialise dataloaders for training and validation
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ============================== Training Stuff ==============================
    # Initialise scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Log training options
    train_options = {
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'augmentation': str(augmentation),
        'beta': beta,
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
            with torch.no_grad():
                # Augment images
                x1, x2 = augmentation(images), augmentation(images)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    y1_t, y2_t = target_model(x1), target_model(x2)
                    z1_t, z2_t = target_model.project(y1_t), target_model.project(y2_t)
                    z1_t, z2_t = F.normalize(z1_t, dim=-1), F.normalize(z2_t, dim=-1)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                y1_o, y2_o = online_model(x1), online_model(x2)
                z1_o, z2_o = online_model.project(y1_o), online_model.project(y2_o)
                p1_o, p2_o = online_model.predict(z1_o), online_model.predict(z2_o)
                p1_o, p2_o = F.normalize(p1_o, dim=-1), F.normalize(p2_o, dim=-1)

                if beta is None:
                    loss = 0.5 * (F.mse_loss(p1_o, z2_t) + F.mse_loss(p2_o, z1_t))
                else:
                    loss = 0.5 * (smooth_l1_loss(p1_o, z2_t, beta) + smooth_l1_loss(p2_o, z1_t, beta))


            # Update online model
            # scaler.scale(loss).backward()
            # scaler.step(optimiser)
            # scaler.update()
            loss.backward()
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)

            # Update target model
            with torch.no_grad():
                for o_param, t_param in zip(online_model.parameters(), target_model.parameters()):
                    t_param.data = taus[epoch] * t_param.data + (1 - taus[epoch]) * o_param.data

            epoch_train_losses[i] = loss.detach()
        
        # Validation Pass
        with torch.no_grad():
            online_model.eval()
            target_model.eval()
            epoch_val_losses = torch.zeros(len(val_loader), device=device)
            for i, (images, _) in enumerate(val_loader):
                # with torch.cuda.amp.autocast():
                    # Augment images
                x1, x2 = augmentation(images), augmentation(images)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    y1_t, y2_t = target_model(x1), target_model(x2)
                    z1_t, z2_t = target_model.project(y1_t), target_model.project(y2_t)
                    z1_t, z2_t = F.normalize(z1_t, dim=-1), F.normalize(z2_t, dim=-1)

                    y1_o, y2_o = online_model(x1), online_model(x2)
                    z1_o, z2_o = online_model.project(y1_o), online_model.project(y2_o)
                    p1_o, p2_o = online_model.predict(z1_o), online_model.predict(z2_o)
                    p1_o, p2_o = F.normalize(p1_o, dim=-1), F.normalize(p2_o, dim=-1)

                    if beta is None:
                        loss = 0.5 * (F.mse_loss(p1_o, z2_t) + F.mse_loss(p2_o, z1_t))
                    else:
                        loss = 0.5 * (smooth_l1_loss(p1_o, z2_t, beta) + smooth_l1_loss(p2_o, z1_t, beta))

                    epoch_val_losses[i] = loss.detach()

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