import math
from typing import Callable

from torch.optim import Optimizer


class CosineWarmup:
    """Cosine decay with warmup learning rate scheduler."""

    def __init__(self, optimizer: Optimizer, warmup_steps: int, total_steps: int, min_lr: float = 0.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.current_step = 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self) -> None:
        self.current_step += 1
        for i, base_lr in enumerate(self.base_lrs):
            if self.current_step < self.warmup_steps:
                lr = base_lr * self.current_step / max(1, self.warmup_steps)
            else:
                progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
            self.optimizer.param_groups[i]["lr"] = lr
