import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from Utils.dataset import PreloadedDataset
from tqdm import tqdm
import torch.nn.functional as F

def get_mnist_subset_datasets(n_per_class, transform=None, device=torch.device('cpu')):
    # Load data
    dataset = datasets.MNIST(root='../Datasets/', train=True, transform=transforms.ToTensor(), download=True)
    if transform is None:
        transform = transforms.ToTensor()

    # Build train dataset
    train_indices = []
    for i in range(10):
        idxs = torch.where(dataset.targets == i)[0][:n_per_class]
        train_indices.append(idxs)
    train_indices = torch.cat(train_indices)
    train = PreloadedDataset.from_dataset(dataset, transform, device)
    train.images = train.images[train_indices]
    train.transformed_images = train.transformed_images[train_indices]
    train.targets = train.targets[train_indices]

    # Build val dataset
    _, val_dataset = torch.utils.data.random_split(dataset, [50000, 10000])
    val = PreloadedDataset.from_dataset(val_dataset, transforms.ToTensor(), device)

    return train, val

def get_mnist_subset_loaders(n_per_class, batch_size, transform=None, device=torch.device('cpu')):

    train, val = get_mnist_subset_datasets(n_per_class, transform, device)

    # Build data loaders
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=1000, shuffle=False)

    return train_loader, val_loader

def mnist_linear_eval(
    model: nn.Module,
    n_per_class: int,
    writer: SummaryWriter = None,
    flatten: bool = False,
    test: bool = False,
):
    device = next(model.parameters()).device
    model.eval()

    # Create classifier and specify training parameters
    classifier = nn.Linear(model.num_features, 10, bias=False).to(device)
    num_epochs = 100
    batch_size = n_per_class
    lr = 0.01
    optimiser = torch.optim.AdamW(classifier.parameters(), lr=lr)

    scaler = torch.cuda.amp.GradScaler()

    last_train_loss = torch.tensor(-1, device=device)
    last_train_acc = torch.tensor(-1, device=device)
    last_val_loss = torch.tensor(-1, device=device)
    last_val_acc = torch.tensor(-1, device=device)
    best_val_acc = torch.tensor(-1, device=device)

    train_loader, val_loader = get_mnist_subset_loaders(n_per_class, batch_size, device=device)

    postfix = {}
    for epoch in range(num_epochs):
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
        if epoch > 0:
            loop.set_postfix(postfix)
        epoch_train_loss = torch.zeros(len(train_loader), device=device)
        epoch_train_acc = torch.zeros(len(train_loader), device=device)
        for i, (x, y) in loop:
            if flatten:
                x = x.flatten(1)
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
                if flatten:
                    x = x.flatten(1)
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

    if test:
        t_dataset = datasets.MNIST(root='../Datasets/', train=False, transform=transforms.ToTensor(), download=True)
        test = PreloadedDataset.from_dataset(t_dataset, transforms.ToTensor(), device)
        test_loader = DataLoader(test, batch_size=100, shuffle=False)

        test_accs = torch.zeros(len(test_loader), device=device)
        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                if flatten:
                    x = x.flatten(1)
                with torch.cuda.amp.autocast():
                    z = model(x)
                    y_pred = classifier(z)
                test_accs[i] = (y_pred.argmax(dim=1) == y).float().mean()

        test_acc = test_accs.mean().item()
        print(f'Test accuracy: {test_acc}')
        writer.add_scalar('Classifier/test_acc', test_acc)

    print(f'Best validation accuracy: {best_val_acc.item()}')

def single_step_classification_eval(
        encoder,
        train_loader,
        val_loader,
        scaler,
        learn_encoder=False,
        flatten=False,
):
    encoder.eval()
    device = next(encoder.parameters()).device

    classifier = torch.nn.Linear(encoder.num_features, 10, bias=False).to(device)
    optimiser = torch.optim.AdamW(classifier.parameters(), lr=1e-1, weight_decay=1e-4)

    for i, (images, labels) in enumerate(train_loader):
        if flatten:
            images = images.flatten(1)
        with torch.cuda.amp.autocast():
            if learn_encoder:
                z = encoder(images)        
            else:
                with torch.no_grad():
                    z = encoder(images)

            y_pred = classifier(z)
            loss = F.cross_entropy(y_pred, labels)

        optimiser.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()

    val_accs = torch.zeros(len(val_loader), device=device)
    val_losses = torch.zeros(len(val_loader), device=device)
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            if flatten:
                images = images.flatten(1)
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    z = encoder(images)        
                y_pred = classifier(z)
            val_accs[i] = (y_pred.argmax(dim=1) == labels).float().mean()
            val_losses[i] = F.cross_entropy(y_pred, labels)

    val_acc = val_accs.mean().item()
    val_loss = val_losses.mean().item()

    return val_acc, val_loss