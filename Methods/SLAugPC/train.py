import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as F_v2
from tqdm import tqdm

from Utils.functional import cosine_schedule
from Examples.MNIST.mnist_linear_1k import single_step_classification_eval, get_ss_mnist_loaders

class DINOLoss(torch.nn.Module):

    def __init__(self, num_epochs, num_features, C_mom=0.9, device='cpu'):
        super().__init__()
        # Temperature schedule
        self.tmp_s = torch.ones(num_epochs) * 0.1
        self.tmp_t = torch.cat([torch.linspace(0.04, 0.07, 30), torch.ones(num_epochs-30) * 0.07])
        # self.tmp_s = torch.ones(num_epochs) * 1.0
        # self.tmp_t = torch.cat([torch.linspace(0.4, 0.7, 30), torch.ones(num_epochs-30) * 0.7])

        # Initialise C
        self.C = torch.zeros((1, num_features), device=device)
        self.C_mom = C_mom
    
    def update_C(self, target):
        # t1, t2: (batch_size, num_features)
        # update C
        self.C = self.C_mom * self.C + (1 - self.C_mom) * target.mean(0, keepdim=True)

    def forward(self, pred, target, epoch):
        # s1, s2, t1, t2: (batch_size, num_features)

        tmp_s, tmp_t = self.tmp_s[epoch], self.tmp_t[epoch]

        # Convert to probabilities
        # (batch_size, num_features) -> (batch_size, num_features)
        pred = F.softmax(pred / tmp_s, dim=-1)
        target = F.softmax((target - self.C) / tmp_t, dim=-1)

        # Update C
        self.update_C(target)

        # # Calculate loss for CLS tokens across different images
        # (batch_size, num_features) -> (1,)
        loss = - (target * pred.log()).sum(-1).mean()
        if loss.isnan():
            raise ValueError("NaN Loss after loss calculation, Exiting Training")

        return loss

def train(
        online_model,
        optimiser,
        train_dataset,
        val_dataset,
        num_epochs,
        batch_size,
        train_aug_scaler='none',
        val_aug_scaler='none',
        start_tau=0.996,
        end_tau =1.0,
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
    start_wd = 0.1 # 0.04
    end_wd = 0.1 # 0.4
    wds = cosine_schedule(start_wd, end_wd, num_epochs)
    
#============================== Target Model Learning Parameters ==============================
    # Initialise target model
    target_model = online_model.copy()
    # EMA schedule, cosine
    start_tau=start_tau
    end_tau =end_tau
    taus = cosine_schedule(start_tau, end_tau, num_epochs)

#============================== Augmentation Parameters ==============================
    # Initialise augmentation probabilty schedule
    assert train_aug_scaler in ['linear', 'exp', 'cosine', 'zeros', 'none'], 'aug_scaler must be one of ["linear", "exp"]'
    if train_aug_scaler == 'linear':
        aug_ps = torch.linspace(0, 0.30, num_epochs)
    elif train_aug_scaler == 'exp':
        aug_ps = 0.25 * (1.0 - torch.exp(torch.linspace(0, -5, num_epochs)))
    elif train_aug_scaler == 'cosine':
        aug_ps = cosine_schedule(0.0, 0.30, num_epochs)
    elif train_aug_scaler == 'zeros':
        aug_ps = torch.zeros(num_epochs)
    elif train_aug_scaler == 'none':
        aug_ps = 0.25 * torch.ones(num_epochs)
    
    # Initialise validation augmentation probabilty schedule
    assert val_aug_scaler in ['linear', 'exp', 'cosine', 'none', 'zeros'], 'aug_scaler must be one of ["linear", "exp"]'
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
    # Initialise dataloaders for single step classification eval
    ss_train_loader, ss_val_loader = get_ss_mnist_loaders(batch_size, device)

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
        'train_aug_scaler': train_aug_scaler,
        'val_aug_scaler': val_aug_scaler,
        'learn_on_ss': learn_on_ss,
        'start_tau': start_tau,
        'end_tau': end_tau,
    }

    dino = DINOLoss(num_epochs, online_model.num_features, C_mom=0.9, device=device)
    loss_fn = lambda x, y: dino(x, y, epoch)

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
            with torch.cuda.amp.autocast():
                # Sample Action
                act_p = torch.rand(5) # whether to apply each augmentation
                angle = torch.rand(1).item() * 360 - 180 if act_p[0] < aug_ps[epoch] else 0
                translate_x = torch.randint(-8, 9, (1,)).item() if act_p[1] < aug_ps[epoch] else 0
                translate_y = torch.randint(-8, 9, (1,)).item() if act_p[2] < aug_ps[epoch] else 0
                scale = torch.rand(1).item() * 0.5 + 0.75 if act_p[3] < aug_ps[epoch] else 1.0
                shear = torch.rand(1).item() * 50 - 25 if act_p[4] < aug_ps[epoch] else 0
                images_aug = F_v2.affine(images, angle=angle, translate=(translate_x, translate_y), scale=scale, shear=shear)
                action = torch.tensor([angle/180, translate_x/8, translate_y/8, (scale-1.0)/0.25, shear/25], dtype=torch.float32, device=images.device).unsqueeze(0).repeat(images.shape[0], 1)
                with torch.no_grad():
                    # Augment images and encode
                    targets = target_model(images_aug)
                    # targets = target_model.predict(images_aug)
                preds = online_model.predict(images, action)

                preds = F.normalize(preds, dim=1)
                targets = F.normalize(targets, dim=1)
                loss = loss_fn(preds, targets)

            # Update online model
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad(set_to_none=True)

            # Update target model
            with torch.no_grad():
                for o_param, t_param in zip(online_model.parameters(), target_model.parameters()):
                    t_param.data = taus[epoch] * t_param.data + (1 - taus[epoch]) * o_param.data

            epoch_train_losses[i] = loss.detach()
        
        # Validation Pass
        online_model.eval()
        with torch.no_grad():
            epoch_val_losses = torch.zeros(len(val_loader), device=device)
            for i, (images, _) in enumerate(val_loader):
                with torch.cuda.amp.autocast():
                    # Sample Action
                    act_p = torch.rand(5) # whether to apply each augmentation
                    angle = torch.rand(1).item() * 360 - 180 if act_p[0] < val_aug_ps[epoch] else 0
                    translate_x = torch.randint(-8, 9, (1,)).item() if act_p[1] < val_aug_ps[epoch] else 0
                    translate_y = torch.randint(-8, 9, (1,)).item() if act_p[2] < val_aug_ps[epoch] else 0
                    scale = torch.rand(1).item() * 0.5 + 0.75 if act_p[3] < val_aug_ps[epoch] else 1.0
                    shear = torch.rand(1).item() * 50 - 25 if act_p[4] < val_aug_ps[epoch] else 0
                    images_aug = F_v2.affine(images, angle=angle, translate=(translate_x, translate_y), scale=scale, shear=shear)
                    action = torch.tensor([angle/180, translate_x/8, translate_y/8, (scale-1.0)/0.25, shear/25], dtype=torch.float32, device=images.device).unsqueeze(0).repeat(images.shape[0], 1)

                    targets = target_model(images_aug)
                    # targets = target_model.predict(images_aug)

                    preds = online_model.predict(images, action)

                    preds = F.normalize(preds, dim=1)
                    targets = F.normalize(targets, dim=1)
                    loss = loss_fn(preds, targets)

                    epoch_val_losses[i] = loss

        # single step linear classification eval
        ss_val_acc, ss_val_loss = single_step_classification_eval(online_model, ss_train_loader, ss_val_loader, scaler, learn_on_ss)
        if learn_on_ss:
            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad(set_to_none=True)
        
        if epoch_train_losses.mean().isnan():
            print("NaN Loss, Exiting Training")
            break
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