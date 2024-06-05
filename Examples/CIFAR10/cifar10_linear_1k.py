import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from Utils.dataset import PreloadedDataset
from tqdm import tqdm
import torch.nn.functional as F



def cifar10_linear_1k_eval(
    model: nn.Module,
    writer: SummaryWriter = None,
):
    device = next(model.parameters()).device

    # Create classifier and specify training parameters
    classifier = nn.Linear(model.num_features, 10, bias=False).to(device)
    num_epochs = 100
    batch_size = 50
    lr = 0.01
    optimiser = torch.optim.AdamW(classifier.parameters(), lr=lr)

    # Load data
    t_dataset = datasets.CIFAR10(root='../Datasets/', train=False, transform=transforms.ToTensor(), download=True)
    dataset = datasets.CIFAR10(root='../Datasets/', train=True, transform=transforms.ToTensor(), download=True)
    train1k = PreloadedDataset.from_dataset(dataset, transforms.ToTensor(), device)
    test = PreloadedDataset.from_dataset(t_dataset, transforms.ToTensor(), device)

    # Reduce to 1000 samples, 100 from each class.
    indices = []
    for i in range(10):
        idx = train1k.targets == i
        indices.append(torch.where(idx)[0][:100])
    indices = torch.cat(indices)
    train1k.images = train1k.images[indices]
    train1k.transformed_images = train1k.transformed_images[indices]
    train1k.targets = train1k.targets[indices]

    # Build data loaders
    train_loader = DataLoader(train1k, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    scaler = torch.cuda.amp.GradScaler()

    last_train_loss = torch.tensor(-1, device=device)
    last_train_acc = torch.tensor(-1, device=device)
    last_val_loss = torch.tensor(-1, device=device)
    last_val_acc = torch.tensor(-1, device=device)
    best_val_acc = torch.tensor(-1, device=device)

    postfix = {}
    for epoch in range(num_epochs):
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
        if epoch > 0:
            loop.set_postfix(postfix)
        epoch_train_loss = torch.zeros(len(train_loader), device=device)
        epoch_train_acc = torch.zeros(len(train_loader), device=device)
        for i, (x, y) in loop:
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    z = model(x)
                y_pred = classifier(z)
                loss = F.cross_entropy(y_pred, y)
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()

            epoch_train_loss[i] = loss.detach()
            epoch_train_acc[i] = (y_pred.argmax(dim=1) == y).float().mean().detach()

        last_train_loss = epoch_train_loss.mean()
        last_train_acc = epoch_train_acc.mean()
        
        with torch.no_grad():
            epoch_val_loss = torch.zeros(len(val_loader), device=device)
            epoch_val_acc = torch.zeros(len(val_loader), device=device)
            for i, (x, y) in enumerate(val_loader):
                with torch.cuda.amp.autocast():
                    z = model(x)
                    y_pred = classifier(z)
                    loss = F.cross_entropy(y_pred, y)
                epoch_val_loss[i] += loss.detach()
                epoch_val_acc[i] += (y_pred.argmax(dim=1) == y).float().mean().detach()

            last_val_loss = epoch_val_loss.mean().detach() 
            last_val_acc = epoch_val_acc.mean().detach()
            if last_val_acc > best_val_acc:
                best_val_acc = last_val_acc
        
        if writer is not None:
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

    print(f'Best validation accuracy: {best_val_acc.item()}')
