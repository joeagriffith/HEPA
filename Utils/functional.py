import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as F_v2
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

def augment(images, p):    
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