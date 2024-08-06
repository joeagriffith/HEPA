import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from scipy.spatial.transform import Rotation as R
import numpy as np
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

def cosine_schedule(base, end, T):
    return end - (end - base) * ((torch.arange(0, T, 1) * math.pi / T).cos() + 1) / 2


def aug_transform(images, *_):
    B, C, H, W = images.shape

    new_H = int(H*0.71)
    t = transforms.Compose([
        transforms.RandomCrop(new_H),
        transforms.Resize(H, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.75, 1.25), shear=25),
    ])
    return t(images)

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

def repeat_interleave_batch(x, B, repeat):
    N = len(x) // B
    x = torch.cat([
        torch.cat([x[i*B:(i+1)*B] for _ in range(repeat)], dim=0)
        for i in range(N)
    ], dim=0)
    return x

def euler_to_quaternion(euler):
    euler = euler.cpu()
    return R.from_euler('xyz', euler, degrees=True).as_quat()

def quaternion_delta(euler_from, euler_to):
    quat1 = euler_to_quaternion(euler_from)
    quat2 = euler_to_quaternion(euler_to)

    delta_quat = R.from_quat(quat2) * R.from_quat(quat1).inv()
    return torch.tensor(delta_quat.as_quat(), dtype=torch.float32, device=euler_from.device)

def delta_axis_angle(euler_from, euler_to):
    rot1 = R.from_euler('xyz', euler_from, degrees=True)
    rot2 = R.from_euler('xyz', euler_to, degrees=True)

    delta_rot = rot2 * rot1.inv()

    delta_axis_angle = delta_rot.as_rotvec()

    return delta_axis_angle

def axis_angle(euler_from, euler_to):

    del_axis_angle = delta_axis_angle(euler_from, euler_to)

    angle = np.linalg.norm(del_axis_angle, axis=1)
    axis = del_axis_angle / angle[:, None]

    angle = torch.tensor(angle, dtype=torch.float32, device=euler_from.device).unsqueeze(1)
    axis = torch.tensor(axis, dtype=torch.float32, device=euler_from.device)

    out = torch.cat([axis, angle], dim=1)

    return out
