import torch


def masked_mse(pred, target, mask):
    # mask shape: (C, H, W) or (1, C, H, W)
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)
    diff = (pred - target) * mask
    denom = mask.sum().clamp_min(1.0)
    return (diff.pow(2).sum() / denom).clamp_min(0.0)
