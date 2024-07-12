import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as F_v2
import torchvision.transforms as transforms
import math

def NTXent(z:torch.Tensor, temperature:float=0.5):
    """
    Compute the normalized temperature-scaled cross entropy loss for the given batch of samples.
    Args:
        z: torch.Tensor, the batch of samples to compute the loss for.
        temperature: float, the temperature scaling factor.
    Returns:
        torch.Tensor, the computed loss.
    """
    # Compute the cosine similarity matrix
    z = F.normalize(z, dim=-1)
    similarity_matrix = torch.exp(torch.matmul(z, z.T) / temperature)

    # Compute the positive and negative samples
    with torch.no_grad():
        batch_size = z.size(0)
        mask = torch.zeros((batch_size, batch_size), device=z.device, dtype=torch.float32)
        mask[range(1, batch_size, 2), range(0, batch_size, 2)] = 1.0
        mask[range(0, batch_size, 2), range(1, batch_size, 2)] = 1.0
    numerator = similarity_matrix * mask
    denominator = (similarity_matrix * (torch.ones_like(mask) - torch.eye(batch_size, device=z.device))).sum(dim=-1, keepdim=True)

    # prevent nans
    with torch.no_grad():
        numerator[~mask.bool()] = 1.0


    # calculate loss
    losses = -torch.log(numerator / denominator)
    loss = losses[mask.bool()].mean()

    return loss

def smooth_l1_loss(input:torch.Tensor, target:torch.Tensor, beta:float=1.0):
    """
    Compute the smooth L1 loss for the given input and target tensors.
    Args:
        input: torch.Tensor, the input tensor.
        target: torch.Tensor, the target tensor.
        beta: float, the beta parameter.
    Returns:
        torch.Tensor, the computed loss.
    """
    diff = torch.abs(input - target)
    loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    return loss.mean()

def negative_cosine_similarity(x1:torch.Tensor, x2:torch.Tensor):
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)
    return -torch.matmul(x1, x2.T).sum(dim=-1).mean()


def get_optimiser(model, optimiser, lr, wd, exclude_bias=True, exclude_bn=True, momentum=0.9, betas=(0.9, 0.999)):
    non_decay_parameters = []
    decay_parameters = []   
    for n, p in model.named_parameters():
        if exclude_bias and 'bias' in n:
            non_decay_parameters.append(p)
        elif exclude_bn and 'bn' in n:
            non_decay_parameters.append(p)
        else:
            decay_parameters.append(p)
    non_decay_parameters = [{'params': non_decay_parameters, 'weight_decay': 0.0}]
    decay_parameters = [{'params': decay_parameters}]

    assert optimiser in ['AdamW', 'SGD'], 'optimiser must be one of ["AdamW", "SGD"]'
    if optimiser == 'AdamW':
        if momentum != 0.9:
            print('Warning: AdamW does not accept momentum parameter. Ignoring it. Please specify betas instead.')
        optimiser = torch.optim.AdamW(decay_parameters + non_decay_parameters, lr=lr, weight_decay=wd, betas=betas)
    elif optimiser == 'SGD':
        if betas != (0.9, 0.999):
            print('Warning: SGD does not accept betas parameter. Ignoring it. Please specify momentum instead.')
        optimiser = torch.optim.SGD(decay_parameters + non_decay_parameters, lr=lr, weight_decay=wd, momentum=momentum)
    
    return optimiser

def cosine_schedule(base, end, T):
    return end - (end - base) * ((torch.arange(0, T, 1) * math.pi / T).cos() + 1) / 2

def aug_interact(images, p):    
    # Sample Action
    act_p = torch.rand(5) # whether to apply each augmentation
    angle = torch.rand(1).item() * 360 - 180 if act_p[0] < p else 0
    translate_x = torch.randint(-8, 9, (1,)).item() if act_p[1] < p else 0
    translate_y = torch.randint(-8, 9, (1,)).item() if act_p[2] < p else 0
    scale = torch.rand(1).item() * 0.5 + 0.75 if act_p[3] < p else 1.0
    shear = torch.rand(1).item() * 50 - 25 if act_p[4] < p else 0
    images_aug = F_v2.affine(images, angle=angle, translate=(translate_x, translate_y), scale=scale, shear=shear)
    action = torch.tensor([angle/180, translate_x/8, translate_y/8, (scale-1.0)/0.25, shear/25], dtype=torch.float32, device=images.device).unsqueeze(0).repeat(images.shape[0], 1)

    return images_aug, action

def aug_transform(images, *_):
    t = transforms.Compose([
        transforms.RandomCrop(20),
        transforms.Resize(28, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.75, 1.25), shear=25),
    ])
    return t(images), None

def feature_correlation(x):
    # x: (N, C)
    # Normalize
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True)
    x = (x - mean) / (std + 1e-8)
    # Compute correlation
    corr = torch.matmul(x.T, x) / (x.size(0)-1)
    # select above diagonal
    corr = torch.triu(corr, diagonal=1)
    corr = corr[corr != 0]
    # average
    corr = corr.mean()

    return corr

def feature_std(x):
    # x: (N, C)
    x = F.normalize(x, dim=1)
    return x.std(dim=0, keepdim=True).mean()

def create_sine_cosine_embeddings(height, width, channels):
    """
    Create sine-cosine positional embeddings for the given dimensions.
    
    Args:
        height (int): Height of the embedding grid.
        width (int): Width of the embedding grid.
        channels (int): Number of channels for the embeddings.
        
    Returns:
        torch.Tensor: Sine-cosine positional embeddings.
    """
    assert channels % 4 == 0, "Channels must be divisible by 4 for sine-cosine embeddings."
    
    quarter_dim = channels // 4
    emb_h = torch.arange(height, dtype=torch.float32)
    emb_w = torch.arange(width, dtype=torch.float32)
    
    emb_h = emb_h.unsqueeze(1) / (10000 ** (torch.arange(quarter_dim, dtype=torch.float32) / quarter_dim))
    emb_w = emb_w.unsqueeze(1) / (10000 ** (torch.arange(quarter_dim, dtype=torch.float32) / quarter_dim))
    
    emb_h = torch.cat((torch.sin(emb_h), torch.cos(emb_h)), dim=1)
    emb_w = torch.cat((torch.sin(emb_w), torch.cos(emb_w)), dim=1)
    
    emb_h = emb_h.unsqueeze(1).repeat(1, width, 1)
    emb_w = emb_w.unsqueeze(0).repeat(height, 1, 1)
    
    embeddings = torch.cat((emb_h, emb_w), dim=2).view(height * width, -1).contiguous()
    
    return embeddings

def repeat_interleave_batch(x, B, repeat):
    N = len(x) // B
    x = torch.cat([
        torch.cat([x[i*B:(i+1)*B] for _ in range(repeat)], dim=0)
        for i in range(N)
    ], dim=0)
    return x