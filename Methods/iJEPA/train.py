import torch
import torch.nn.functional as F
from tqdm import tqdm

from Utils.functional import cosine_schedule
from Examples.MNIST.mnist_linear_1k import single_step_classification_eval, get_ss_mnist_loaders


class iBOTLoss(torch.nn.Module):

    def __init__(self, num_epochs, num_tokens, num_features, device):
        super().__init__()
        # Temperature schedule
        self.tmp_s = torch.ones(num_epochs) * 0.1
        self.tmp_t = torch.cat([torch.linspace(0.04, 0.07, 30), torch.ones(num_epochs-30) * 0.07])

        # Initialise C
        self.C = torch.zeros((1, num_tokens, num_features), device=device)
        self.C_mom = 0.9
    
    def update_C(self, t1, t2):
        # t1, t2: (batch_size, num_tokens, num_features)
        # update C
        target = 0.5 * torch.cat([t1, t2], dim=0).mean(0, keepdim=True)
        self.C = self.C_mom * self.C + (1 - self.C_mom) * target

    def forward(self, s1, s2, t1, t2, mask1, mask2):
        # s1, s2, t1, t2: (batch_size, num_tokens, num_features)
        # mask1, mask2: (batch_size, num_tokens-1)

        # Convert to probabilities
        # (batch_size, num_tokens, num_features) -> (batch_size, num_tokens, num_features)
        s1, s2 = F.softmax(s1 / self.tmp_s, dim=-1), F.softmax(s2 / self.tmp_s, dim=-1)
        t1, t2 = F.softmax((t1 - self.C) / self.tmp_t, dim=-1), F.softmax((t2 - self.C) / self.tmp_t, dim=-1)

        # Update C
        self.update_C(t1, t2)

        # Split CLS and patches
        # (batch_size, num_tokens, num_features) -> (batch_size, num_features), (batch_size, num_tokens-1, num_features)
        s1_cls, s1_patches = s1[:, 0], s1[:, 1:]
        s2_cls, s2_patches = s2[:, 0], s2[:, 1:]
        t1_cls, t1_patches = t1[:, 0], t1[:, 1:]
        t2_cls, t2_patches = t2[:, 0], t2[:, 1:]

        # # Calculate loss for CLS tokens across different images
        # (batch_size, num_features) -> (1,)
        loss_cls1 = - (t2_cls * s1_cls.log()).sum(-1).mean()
        loss_cls2 = - (t1_cls * s2_cls.log()).sum(-1).mean()
        loss_cls = 0.5 * (loss_cls1 + loss_cls2)

        # # Calculate loss for between masked patches of same image
        # Calculate entropies
        # (batch_size, num_tokens-1, num_features) -> (batch_size, num_tokens-1)
        entropies1 = -(t1_patches * s1_patches.log()).sum(-1)
        entropies2 = -(t2_patches * s2_patches.log()).sum(-1)
        # Mask and mean Entropies
        # (batch_size, num_tokens-1) -> (batch_size,)
        mean_masked_entropies1 = (entropies1 * ~mask1).sum(-1) / (~mask1).sum(-1)
        mean_masked_entropies2 = (entropies2 * ~mask2).sum(-1) / (~mask2).sum(-1)
        # Mean over batches to evaluate loss
        # (batch_size,) -> (1,)
        loss_patches = 0.5 * (mean_masked_entropies1.mean() + mean_masked_entropies2.mean())

        return loss_cls + loss_patches


def train(
        student_model,
        optimiser,
        train_dataset,
        val_dataset,
        num_epochs,
        batch_size,
        augmentation,
        learn_on_ss=False,
        writer=None,
        save_dir=None,
        save_every=1,
):
    device = next(student_model.parameters()).device

#============================== Student Learning Parameters ==============================
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
    
#============================== Teacher Learning Parameters ==============================
    # Initialise target model
    teacher_model = student_model.copy()
    # EMA schedule, cosine
    start_tau=0.996
    end_tau = 1.0
    taus = cosine_schedule(start_tau, end_tau, num_epochs)

#============================== Loss Parameters ==============================
    # Temperature schedule
    assert num_epochs >= 30, 'num_epochs must be >= 30 because of the temperature schedule'
    tmp_s = 0.1
    tmp_ts = torch.cat([torch.linspace(0.04, 0.07, 30), torch.ones(num_epochs-30) * 0.07])

    # Initialise C
    C = torch.zeros(1, student_model.num_features, device=device)
    C_mom = 0.9

# ============================== Data Handling ==============================
    # Initialise dataloaders for single step classification eval
    ss_train_loader, ss_val_loader = get_ss_mnist_loaders(batch_size, device)

    # Initialise dataloaders for training and validation
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ============================== Training Stuff ==============================
    # Initialise scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Initialise loss function
    loss_fn = iBOTLoss(num_epochs, student_model.num_features, device)

    # Log training options
    train_options = {
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'augmentation': str(augmentation),
        'learn_on_ss': learn_on_ss,
    }

    # Log training options, model details, and optimiser details
    if writer is not None:
        writer.add_text('Encoder/options', str(train_options))
        writer.add_text('Encoder/model', str(student_model).replace('\n', '<br/>').replace(' ', '&nbsp;'))
        writer.add_text('Encoder/optimiser', str(optimiser).replace('\n', '<br/>').replace(' ', '&nbsp;'))

    # Initialise training variables
    last_train_loss = -1
    last_val_loss = -1
    best_val_loss = float('inf')
    postfix = {}

# ============================== Training Loop ==============================
    for epoch in range(num_epochs):
        train_dataset.apply_transform(batch_size=batch_size)
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
        if epoch > 0:
            loop.set_postfix(postfix)

        # Training Pass
        epoch_train_losses = torch.zeros(len(train_loader), device=device)
        for i, (images, _) in loop:
            with torch.cuda.amp.autocast():
                # Augment images
                with torch.no_grad():
                    x1, x2 = augmentation(images), augmentation(images)
                    t1, t2 = teacher_model.encode(x1), teacher_model.encode(x2)
                    p_t1, p_t2 = teacher_model.project(t1), teacher_model.project(t2)

                # Sample Mask (Batch_Size, Num_Patches)
                mask1 = torch.rand(x1.shape[0], student_model.num_patches, device=device) > 0.5
                mask2 = torch.rand(x2.shape[0], student_model.num_patches, device=device) > 0.5

                # Encode tokens and project
                s1, s2 = student_model.encode(x1, mask1), student_model.encode(x2, mask2)
                p_s1, p_s2 = student_model.project(s1), student_model.project(s2)

                loss = loss_fn(p_s1, p_s2, p_t1, p_t2, mask1, mask2)


            # Update lr
            for param_group in optimiser.param_groups:
                param_group['lr'] = lrs[epoch]
            # Update wd
            for param_group in optimiser.param_groups:
                if param_group['weight_decay'] != 0:
                    param_group['weight_decay'] = wds[epoch]

            # Update online model
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad(set_to_none=True)

            # Update teacher
            with torch.no_grad():
                for s_param, t_param in zip(student_model.parameters(), teacher_model.parameters()):
                    t_param.data = taus[epoch] * t_param.data + (1 - taus[epoch]) * s_param.data

            epoch_train_losses[i] = loss.detach()
        
        # Validation Pass
        with torch.no_grad():
            epoch_val_losses = torch.zeros(len(val_loader), device=device)
            for i, (images, _) in enumerate(val_loader):
                with torch.cuda.amp.autocast():
                    # Augment images
                    x1, x2 = augmentation(images), augmentation(images)
                    t1, t2 = teacher_model.encode(x1), teacher_model.encode(x2)
                    p_t1, p_t2 = teacher_model.project(t1), teacher_model.project(t2)

                    # Encode tokens and project
                    s1, s2 = student_model.encode(x1), student_model.encode(x2)
                    p_s1, p_s2 = student_model.project(s1), student_model.project(s2)

                    loss = loss_fn(p_s1, p_s2, p_t1, p_t2, mask1, mask2)

                    epoch_val_losses[i] = loss.detach()

        # single step linear classification eval
        ss_val_acc, ss_val_loss = single_step_classification_eval(student_model, ss_train_loader, ss_val_loader, scaler, learn_on_ss)
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
            torch.save(student_model.state_dict(), save_dir)