import torch
import torch.nn.functional as F
from tqdm import tqdm

from Utils.functional import cosine_schedule
from Examples.MNIST.mnist_linear_1k import single_step_classification_eval, get_ss_mnist_loaders

class DINOLoss(torch.nn.Module):

    def __init__(self, num_epochs, num_features, C_mom=0.9, scale_temps=1.0, device='cpu'):
        super().__init__()
        # Temperature schedule
        self.tmp_s = torch.ones(num_epochs) * 0.1 * scale_temps
        self.tmp_t = torch.cat([torch.linspace(0.04, 0.07, 30) * scale_temps, torch.ones(num_epochs-30) * 0.07 * scale_temps])

        # Initialise C
        self.C = torch.zeros((1, num_features), device=device)
        self.C_mom = C_mom
    
    def update_C(self, t1, t2):
        # t1, t2: (batch_size, num_features)
        # update C
        target = 0.5 * torch.cat([t1, t2], dim=0).mean(0, keepdim=True)
        self.C = self.C_mom * self.C + (1 - self.C_mom) * target

    def forward(self, s1, s2, t1, t2, epoch):
        # s1, s2, t1, t2: (batch_size, num_features)

        tmp_s, tmp_t = self.tmp_s[epoch], self.tmp_t[epoch]

        # Convert to probabilities
        # (batch_size, num_features) -> (batch_size, num_features)
        s1, s2 = F.softmax(s1 / tmp_s, dim=-1), F.softmax(s2 / tmp_s, dim=-1)
        t1, t2 = F.softmax((t1 - self.C) / tmp_t, dim=-1), F.softmax((t2 - self.C) / tmp_t, dim=-1)

        # Update C
        self.update_C(t1, t2)

        # # Calculate loss for CLS tokens across different images
        # (batch_size, num_features) -> (1,)
        loss1 = - (t2 * s1.log()).sum(-1).mean()
        loss2 = - (t1 * s2.log()).sum(-1).mean()
        loss = 0.5 * (loss1 + loss2)

        return loss

def train(
        student_model,
        optimiser,
        train_dataset,
        val_dataset,
        num_epochs,
        batch_size,
        augmentation,
        scale_temps=1.0,
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
    warm_up_lrs = torch.linspace(0, base_lr, 11)[1:]
    cosine_lrs = cosine_schedule(base_lr, end_lr, num_epochs-10)
    lrs = torch.cat([warm_up_lrs, cosine_lrs])
    lrs = torch.ones(num_epochs) * base_lr
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
    loss_fn = DINOLoss(num_epochs, student_model.num_features, C_mom=0.9, scale_temps=scale_temps, device=device)

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
                with torch.no_grad():
                    # Augment images
                    x1, x2 = augmentation(images), augmentation(images)
                    t1, t2 = teacher_model(x1), teacher_model(x2)
                    t1_p, t2_p = teacher_model.project(t1), teacher_model.project(t2)

                s1, s2 = student_model(x1), student_model(x2)
                s1_p, s2_p = student_model.project(s1), student_model.project(s2)

                loss = loss_fn(s1_p, s2_p, t1_p, t2_p, epoch)


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

                    # Encode images
                    s1, s2 = student_model(x1), student_model(x2)
                    t1, t2 = teacher_model(x1), teacher_model(x2)

                    # Project encodings
                    s1_p, s2_p = student_model.project(s1), student_model.project(s2)
                    t1_p, t2_p = teacher_model.project(t1), teacher_model.project(t2)

                    loss = loss_fn(s1_p, s2_p, t1_p, t2_p, epoch)

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