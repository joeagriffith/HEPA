import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from Utils.dataset import PreloadedDataset
from tqdm import tqdm
import torch.nn.functional as F

from Examples.MNIST.dataset import MNIST
from Examples.ModelNet10.dataset import ModelNet10, ModelNet10Simple
from Examples.VoxCeleb1.dataset import VoxCeleb1, VoxCeleb1Triplet
from Utils.utils import get_ss_datasets
from Utils.functional import feature_correlation, feature_std, feature_entropy

def linear_probing(
    model: nn.Module,
    writer: SummaryWriter,
    n_per_class: int,
    cfg: dict,
    finetune: bool = False,
    train_set = None,
    val_set = None,
):
    assert finetune is False, 'Finetuning is not supported for linear probing'
    device = torch.device(cfg['compute_device'])
    model.eval()
    enc_fn = torch.compile(model.forward)

    # Create classifier and specify training parameters
    if cfg['dataset'] == 'voxceleb1':
        classifier = nn.Sequential(
            nn.BatchNorm1d(model.num_features, affine=False) if cfg['bn_output'] else nn.Identity(),
            nn.Linear(model.num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(device)
    else:
        classifier = nn.Sequential(
            nn.BatchNorm1d(model.num_features, affine=False) if cfg['bn_output'] else nn.Identity(),
            nn.Linear(model.num_features, 10, bias=False),
        ).to(device)
    # batch_size = max(n_per_class, 10)
    batch_size = 64
    num_epochs = 100 if cfg['dataset'] == 'mnist' else 200
    lr = 0.1

    if cfg['dataset'] == 'mnist':
        train = MNIST(cfg['root'], split='train', n=n_per_class, device=cfg['data_device'], use_tqdm=cfg['local'])
        val = MNIST(cfg['root'], split='val', device=cfg['data_device'], use_tqdm=cfg['local'])

    elif cfg['dataset'] == 'modelnet10':
        train = ModelNet10Simple(cfg['root'], split='train', n=n_per_class, device=cfg['data_device'], use_tqdm=cfg['local'], rank=cfg['ddp_rank'], world_size=cfg['ddp_world_size'], seed=cfg['seed'])
        val = ModelNet10Simple(cfg['root'], split='val', n=10, device=cfg['data_device'], use_tqdm=cfg['local'], rank=cfg['ddp_rank'], world_size=cfg['ddp_world_size'], seed=cfg['seed'])

    elif cfg['dataset'] == 'voxceleb1':
        assert train_set is not None and val_set is not None, 'train_set and val_set must be provided for voxceleb1'
        train, val = get_ss_datasets(cfg, train_set, val_set, subset_ratio=0.1)
    
    else:
        raise ValueError(f'Dataset {cfg["dataset"]} not supported')

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)

    param_dict = {pn: p for pn, p in classifier.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if 'bn' not in n and 'bias' not in n]
    nondecay_params = [p for n, p in param_dict.items() if 'bn' in n or 'bias' in n]

    optim_groups = [
        {'params': decay_params, 'weight_decay': 0.005}, 
        {'params': nondecay_params, 'weight_decay': 0.0},
    ]

    optimiser = torch.optim.AdamW(optim_groups, lr=lr)
    sched_step_size = 30 if cfg['dataset'] == 'mnist' else 60
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=sched_step_size, gamma=0.1) 

    last_train_loss = torch.tensor(-1, device=device)
    last_train_acc = torch.tensor(-1, device=device)
    last_val_loss = torch.tensor(-1, device=device)
    last_val_acc = torch.tensor(-1, device=device)
    best_val_acc = torch.tensor(-1, device=device)

    postfix = {}
    for epoch in range(num_epochs):
        classifier.train()

        if cfg['local']:
            loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
            loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
            if epoch > 0:
                loop.set_postfix(postfix)
        else:
            loop = enumerate(train_loader)
        epoch_train_loss = torch.zeros(len(train_loader), device=device)
        epoch_train_acc = torch.zeros(len(train_loader), device=device)
        for i, (x, y) in loop:
            if type(x) == torch.tensor:
                x = x.to(device)
                y = y.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    with torch.no_grad():   
                        z = enc_fn(x)
                    y_pred = classifier(z)
                    loss = F.cross_entropy(y_pred, y)
                epoch_train_acc[i] = (y_pred.argmax(dim=1) == y).float().mean().detach()
            elif type(x) == list or type(x) == tuple:
                x = (x[0].to(device), x[1].to(device), x[2].to(device))
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    with torch.no_grad():
                        z1, z2, z3 = enc_fn(x[0]), enc_fn(x[1]), enc_fn(x[2])
                        pos_z = z1 - z2
                        neg_z = z1 - z3
                    pos_y_pred = classifier(pos_z)
                    pos_y = torch.ones_like(pos_y_pred, device=device)
                    neg_y_pred = classifier(neg_z)
                    neg_y = torch.zeros_like(neg_y_pred, device=device)

                    loss = 0.5 * (F.binary_cross_entropy_with_logits(pos_y_pred, pos_y) + F.binary_cross_entropy_with_logits(neg_y_pred, neg_y))
                y_pred = torch.cat([(pos_y_pred.sigmoid() > 0.5).float(), (neg_y_pred.sigmoid() > 0.5).float()])
                y = torch.cat((pos_y, neg_y))
                epoch_train_acc[i] = (y_pred == y).float().mean()
                
            classifier.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()

            epoch_train_loss[i] = loss.detach()

        last_train_loss = epoch_train_loss.mean()
        last_train_acc = epoch_train_acc.mean()

        scheduler.step()
        
        with torch.no_grad():
            classifier.eval()

            epoch_val_loss = torch.zeros(len(val_loader), device=device)
            epoch_val_acc = torch.zeros(len(val_loader), device=device)
            for i, (x, y) in enumerate(val_loader):
                if type(x) == torch.tensor:
                    x = x.to(device)
                    y = y.to(device)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        with torch.no_grad():
                            z = enc_fn(x)
                            y_pred = classifier(z)
                            loss = F.cross_entropy(y_pred, y)
                    epoch_val_acc[i] = (y_pred.argmax(dim=1) == y).float().mean().detach()
                elif type(x) == list or type(x) == tuple:
                    x = (x[0].to(device), x[1].to(device), x[2].to(device))
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        with torch.no_grad():
                            z1, z2, z3 = enc_fn(x[0]), enc_fn(x[1]), enc_fn(x[2])
                            pos_z = z1 - z2
                            neg_z = z1 - z3
                            pos_y_pred = classifier(pos_z)
                            pos_y = torch.ones_like(pos_y_pred, device=device)
                            neg_y_pred = classifier(neg_z)
                            neg_y = torch.zeros_like(neg_y_pred, device=device)
                            loss = 0.5 * (F.binary_cross_entropy_with_logits(pos_y_pred, pos_y) + F.binary_cross_entropy_with_logits(neg_y_pred, neg_y))
                    y_pred = torch.cat([(pos_y_pred.sigmoid() > 0.5).float(), (neg_y_pred.sigmoid() > 0.5).float()])
                    y = torch.cat((pos_y, neg_y))
                    epoch_val_acc[i] = (y_pred == y).float().mean().detach()
                epoch_val_loss[i] += loss.detach()

            last_val_loss = epoch_val_loss.mean().detach() 
            last_val_acc = epoch_val_acc.mean().detach()
            if last_val_acc > best_val_acc:
                best_val_acc = last_val_acc
        
        if writer is not None:
            writer.add_scalar('train/loss', last_train_loss.item(), epoch)
            writer.add_scalar('train/accuracy', last_train_acc.item(), epoch)
            writer.add_scalar('val/loss', last_val_loss.item(), epoch)
            writer.add_scalar('val/accuracy', last_val_acc.item(), epoch)
        
        postfix = {
            'train_loss': last_train_loss.item(),
            'train_accuracy': last_train_acc.item(),
            'val_loss': last_val_loss.item(),
            'val_accuracy': last_val_acc.item(),
        }

    if cfg['dataset'] == 'mnist':
        t_dataset = datasets.MNIST(root='../Datasets/', train=False, transform=transforms.ToTensor(), download=True)
    elif cfg['dataset'] == 'modelnet10':
        t_dataset = ModelNet10Simple(cfg['root'], split='test', device=cfg['data_device'], use_tqdm=cfg['local'], rank=cfg['ddp_rank'], world_size=cfg['ddp_world_size'], seed=cfg['seed'])
    elif cfg['dataset'] == 'voxceleb1':
        print('Warning: linear probing for voxceleb1 testset is not yet implemented.')
        return
        t_dataset = VoxCeleb1Triplet(cfg['root'], split='test', device=cfg['data_device'], use_tqdm=cfg['local'], rank=cfg['ddp_rank'], world_size=cfg['ddp_world_size'], seed=cfg['seed'])

    test = PreloadedDataset.from_dataset(t_dataset, transforms.ToTensor(), device, use_tqdm=cfg['local'])
    test_loader = DataLoader(test, batch_size=100, shuffle=False)

    test_accs = torch.zeros(len(test_loader), device=device)
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                with torch.no_grad():
                    z = enc_fn(x)
                    y_pred = classifier(z)
            test_accs[i] = (y_pred.argmax(dim=1) == y).float().mean()

    test_acc = test_accs.mean().item()
    if writer is not None:
        if finetune:
            writer.add_scalar('test/finetuning_accuracy', test_acc)
        else:
            writer.add_scalar('test/accuracy', test_acc)
    print(f'N: {n_per_class} - Test accuracy: {test_acc}')

def one_step_linear_probing(
        enc_fn,
        num_features,
        train_loader,
        val_loader,
        device,
        bn_output: bool = False,
):
    device = torch.device(device)
    classifier = nn.Sequential(
        nn.BatchNorm1d(num_features, affine=False) if bn_output else nn.Identity(),
        nn.Linear(num_features, 10, bias=False),
    ).to(device)
    optimiser = torch.optim.AdamW(classifier.parameters(), lr=1e-1, weight_decay=0.0)
    checked = False

    for i, (x, y) in enumerate(train_loader):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            if type(x) == tuple or type(x) == list:
                if not checked:
                    classifier = nn.Sequential(
                        nn.BatchNorm1d(num_features, affine=False) if bn_output else nn.Identity(),
                        nn.Linear(num_features, 512),
                        nn.ReLU(),
                        nn.Linear(512, 128),
                        nn.ReLU(),
                        nn.Linear(128, 1)
                    ).to(device)
                    optimiser = torch.optim.AdamW(classifier.parameters(), lr=1e-1, weight_decay=0.0)
                    checked = True
                
                x = (x[0].to(device), x[1].to(device), x[2].to(device))
                with torch.no_grad():
                    z1, z2, z3 = enc_fn(x[0]), enc_fn(x[1]), enc_fn(x[2])
                    pos_z = z1 - z2
                    neg_z = z1 - z3

                pos_y_pred = classifier(pos_z)
                pos_y = torch.ones_like(pos_y_pred, device=device)
                neg_y_pred = classifier(neg_z)
                neg_y = torch.zeros_like(neg_y_pred, device=device)

                loss = 0.5 * (F.binary_cross_entropy_with_logits(pos_y_pred, pos_y) + F.binary_cross_entropy_with_logits(neg_y_pred, neg_y))
            else:
                x = x.to(device)
                y = y.to(device)
                with torch.no_grad():
                    z = enc_fn(x)
                y_pred = classifier(z)
                loss = F.cross_entropy(y_pred, y)

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()

    val_accs = torch.zeros(len(val_loader), device=device)
    val_losses = torch.zeros(len(val_loader), device=device)
    classifier.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                with torch.no_grad():
                    if type(x) == tuple or type(x) == list:
                        x = (x[0].to(device), x[1].to(device), x[2].to(device))

                        z1, z2, z3 = enc_fn(x[0]), enc_fn(x[1]), enc_fn(x[2])
                        pos_z = z1 - z2
                        neg_z = z1 - z3
                        pos_y_pred = classifier(pos_z)
                        neg_y_pred = classifier(neg_z)
                        pos_y = torch.ones_like(pos_y_pred, device=device)
                        neg_y = torch.zeros_like(neg_y_pred, device=device)
                        val_losses[i] = 0.5 * (F.binary_cross_entropy_with_logits(pos_y_pred, pos_y) + F.binary_cross_entropy_with_logits(neg_y_pred, neg_y))
                        preds = torch.cat([(pos_y_pred.sigmoid() > 0.5).float(), (neg_y_pred.sigmoid() > 0.5).float()])
                        y = torch.cat((pos_y, neg_y))
                        val_accs[i] = (preds == y).float().mean()
                    else:
                        x = x.to(device)
                        y = y.to(device)
                        
                        z = enc_fn(x)
                        y_pred = classifier(z)
                        val_losses[i] = F.cross_entropy(y_pred, y)
                        val_accs[i] = (y_pred.argmax(dim=1) == y).float().mean()

    val_acc = val_accs.mean().item()
    val_loss = val_losses.mean().item()

    return val_acc, val_loss


def get_rep_metrics(
    model: nn.Module,
    dataset: PreloadedDataset,
    cfg: dict,
):
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=100, shuffle=False)

    embeddings = torch.empty(len(dataset), model.num_features, device=device)
    with torch.no_grad():
        # for i, ((x, _, _), _) in enumerate(loader):
        loop = enumerate(loader)
        for i in range(len(loader)):
            if cfg['dataset'] == 'modelnet10':
                try:
                    _, ((x, _, _), _) = next(loop)
                except Exception as e:
                    print(f"An error occurred: {e}. Using default value instead.")
                    _, (x, _) = next(loop)
            else:
                _, (x, _) = next(loop)

            x = x.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                z = model(x)
            embeddings[i * 100:(i + 1) * 100] = z

    metrics = {}
    if cfg['track_feature_corrs']:
        metrics['corr'] = feature_correlation(embeddings).item()
    if cfg['track_feature_stds']:
        metrics['std'] = feature_std(embeddings).item()
    if cfg['track_feature_entropy']:
        metrics['entropy'] = feature_entropy(embeddings).item()

    return metrics

def eval_representations(
    model: nn.Module,
    cfg: dict
):
    if cfg['dataset'] == 'mnist':
        test = MNIST(cfg['root'], 'test', transform=transforms.ToTensor(), device=cfg['compute_device'], use_tqdm=cfg['local'])
    elif cfg['dataset'] == 'modelnet10':
        test = ModelNet10(cfg['root'], 'test', device=cfg['compute_device'], use_tqdm=cfg['local'], resolution=cfg['resolution'], dataset_dtype=cfg['dataset_dtype'])
    elif cfg['dataset'] == 'voxceleb1':
        print("Warning: eval_representations for voxceleb1 testset is not yet implemented.")
        return None

    # dataset type is mnist for both as get() returns (x,y)
    metrics = get_rep_metrics(model, test, cfg)
    
    return metrics