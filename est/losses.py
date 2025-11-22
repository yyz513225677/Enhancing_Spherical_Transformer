from typing import Optional

import torch
from torch import nn


def ohem_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -1, top_k: float = 0.25) -> torch.Tensor:
    """Online Hard Example Mining for Cross Entropy.

    Args:
        logits: [N, C]
        labels: [N]
        ignore_index: label value to ignore
        top_k: fraction of pixels to keep
    """
    criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=ignore_index)
    per_example = criterion(logits, labels)
    valid_mask = labels != ignore_index
    valid_losses = per_example[valid_mask]
    if valid_losses.numel() == 0:
        return per_example.mean()
    k = max(1, int(valid_losses.numel() * top_k))
    hard_loss, _ = torch.topk(valid_losses, k)
    return hard_loss.mean()
