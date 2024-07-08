import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from Utils.dataset import PreloadedDataset
from tqdm import tqdm
import torch.nn.functional as F

from Examples.ModelNet10.dataset import ModelNet10
from Utils.functional import feature_correlation, feature_std

def linear_probing(
    model: nn.Module,
    root: str,
    n_per_class: int,
    writer: SummaryWriter = None,
    flatten: bool = False,
    test: bool = False,
    finetune: bool = False,
):
    device = next(model.parameters()).device

    # Create classifier and specify training parameters
    classifier = nn.Sequential(
        nn.BatchNorm1d(model.num_features, affine=False),
        nn.Linear(model.num_features, 10, bias=False),
    ).to(device)
    batch_size = max(n_per_class, 10)
    num_epochs = 100
    lr = 0.01

    if finetune:
        encoder = model.copy()
        encoder.train()
        
        param_dict = {pn: p for pn, p in encoder.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nondecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': 0.01}, 
            {'params': nondecay_params, 'weight_decay': 0.0},
            {'params': classifier.parameters(), 'weight_decay': 0.0}
        ]
    else:
        encoder = model
        encoder.eval()
        optim_groups = [
            {'params': classifier.parameters(), 'weight_decay': 0.0}
        ]

    optimiser = torch.optim.AdamW(optim_groups, lr=lr)

    last_train_loss = torch.tensor(-1, device=device)
    last_train_acc = torch.tensor(-1, device=device)
    last_val_loss = torch.tensor(-1, device=device)
    last_val_acc = torch.tensor(-1, device=device)
    best_val_acc = torch.tensor(-1, device=device)

    train_set = ModelNet10(root, 'train', n=n_per_class, device=device)
    val_set = ModelNet10(root, 'val', n=55, device=device)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    postfix = {}
    for epoch in range(num_epochs):
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
        if epoch > 0:
            loop.set_postfix(postfix)
        epoch_train_loss = torch.zeros(len(train_loader), device=device)
        epoch_train_acc = torch.zeros(len(train_loader), device=device)
        for i, ((x, _, y), _) in loop:
            if flatten:
                x = x.flatten(1)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                z = encoder(x)
                y_pred = classifier(z)
                loss = F.cross_entropy(y_pred, y)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()

            epoch_train_loss[i] = loss.detach()
            epoch_train_acc[i] = (y_pred.argmax(dim=1) == y).float().mean().detach()

        last_train_loss = epoch_train_loss.mean()
        last_train_acc = epoch_train_acc.mean()
        
        with torch.no_grad():
            epoch_val_loss = torch.zeros(len(val_loader), device=device)
            epoch_val_acc = torch.zeros(len(val_loader), device=device)
            for i, ((x, _, y), _) in enumerate(val_loader):
                if flatten:
                    x = x.flatten(1)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    z = encoder(x)
                    y_pred = classifier(z)
                    loss = F.cross_entropy(y_pred, y)
                epoch_val_loss[i] += loss.detach()
                epoch_val_acc[i] += (y_pred.argmax(dim=1) == y).float().mean().detach()

            last_val_loss = epoch_val_loss.mean().detach() 
            last_val_acc = epoch_val_acc.mean().detach()
            if last_val_acc > best_val_acc:
                best_val_acc = last_val_acc
        
        if writer is not None:
            if finetune:
                writer.add_scalar('Classifier/ft_train_loss', last_train_loss.item(), epoch)
                writer.add_scalar('Classifier/ft_train_acc', last_train_acc.item(), epoch)
                writer.add_scalar('Classifier/ft_val_loss', last_val_loss.item(), epoch)
                writer.add_scalar('Classifier/ft_val_acc', last_val_acc.item(), epoch)
            else:
                writer.add_scalar('Classifier/train_loss', last_train_loss.item(), epoch)
                writer.add_scalar('Classifier/train_acc', last_train_acc.item(), epoch)
                writer.add_scalar('Classifier/val_loss', last_val_loss.item(), epoch)
                writer.add_scalar('Classifier/val_acc', last_val_acc.item(), epoch)
        
        postfix = {
            'train_loss': last_train_loss.item(),
            'train_acc': last_train_acc.item(),
            'val_loss': last_val_loss.item(),
            'val_acc': last_val_acc.item(),
        }
        loop.set_postfix(postfix)
        loop.close()

    if test:
        t_dataset = datasets.MNIST(root='../Datasets/', train=False, transform=transforms.ToTensor(), download=True)
        test = PreloadedDataset.from_dataset(t_dataset, transforms.ToTensor(), device)
        test_loader = DataLoader(test, batch_size=100, shuffle=False)

        test_accs = torch.zeros(len(test_loader), device=device)
        with torch.no_grad():
            for i, ((x, _, y), _) in enumerate(test_loader):
                x.to(device)
                y.to(device)
                if flatten:
                    x = x.flatten(1)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    z = encoder(x)
                    y_pred = classifier(z)
                test_accs[i] = (y_pred.argmax(dim=1) == y).float().mean()

        test_acc = test_accs.mean().item()
        print(f'Test accuracy: {test_acc}')
        if writer is not None:
            if finetune:
                writer.add_scalar('Classifier/ft_test_acc', test_acc)
            else:
                writer.add_scalar('Classifier/test_acc', test_acc)

    print(f'Best validation accuracy: {best_val_acc.item()}')

def single_step_classification_eval(
        encoder,
        train_loader,
        val_loader,
        flatten=False,
):
    encoder.eval()
    device = next(encoder.parameters()).device

    classifier = nn.Sequential(
        nn.BatchNorm1d(encoder.num_features, affine=False),
        nn.Linear(encoder.num_features, 10, bias=False),
    ).to(device)
    optimiser = torch.optim.AdamW(classifier.parameters(), lr=1e-1, weight_decay=0.0)

    for i, ((x, _, y), _) in enumerate(train_loader):
        x.to(device)
        y.to(device)
        if flatten:
            x = x.flatten(1)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            with torch.no_grad():
                z = encoder(x)

            y_pred = classifier(z)
            loss = F.cross_entropy(y_pred, y)

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()

    val_accs = torch.zeros(len(val_loader), device=device)
    val_losses = torch.zeros(len(val_loader), device=device)
    with torch.no_grad():
        for i, ((x, _, y), _) in enumerate(val_loader):
            x.to(device)
            y.to(device)
            if flatten:
                x = x.flatten(1)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                with torch.no_grad():
                    z = encoder(x)        
                y_pred = classifier(z)
            val_accs[i] = (y_pred.argmax(dim=1) == y).float().mean()
            val_losses[i] = F.cross_entropy(y_pred, y)

    val_acc = val_accs.mean().item()
    val_loss = val_losses.mean().item()

    return val_acc, val_loss

def get_rep_metrics(
    model: nn.Module,
    dataset: PreloadedDataset,
    flatten: bool = False,
    corr: bool = True,
    std: bool = True,
):
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=100, shuffle=False)

    embeddings = torch.empty(len(dataset), model.num_features, device=device)
    with torch.no_grad():
        for i, ((x, _, _), _) in enumerate(loader):
            x.to(device)
            if flatten:
                x = x.flatten(1)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                z = model(x)
            embeddings[i * 100:(i + 1) * 100] = z

    metrics = {}
    if corr:
        metrics['corr'] = feature_correlation(embeddings).item()
    if std:
        metrics['std'] = feature_std(embeddings).item()

    return metrics

def eval_representations(
    model: nn.Module,
    flatten: bool = False,
    writer: SummaryWriter = None,
):
    device = next(model.parameters()).device

    root = '../Datasets/ModelNet10'
    test = ModelNet10(root, 'test', device=device)

    metrics = get_rep_metrics(model, test, flatten=flatten, corr=True, std=True)
    
    if writer is not None:
        for key, value in metrics.items():
            writer.add_scalar(f'Encoder/test-{key}', value)
    
    return metrics