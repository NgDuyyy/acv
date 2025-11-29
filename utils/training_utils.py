"""Utility helpers shared across the training and evaluation scripts."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from config import CHECKPOINT_DIR


def clip_gradient(optimizer: torch.optim.Optimizer, grad_clip: float) -> None:
    """Clamp gradients to the provided range to avoid exploding values."""

    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(
    data_name: str,
    epoch: int,
    epochs_since_improvement: int,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    bleu4: float,
    is_best: bool,
) -> None:
    """Persist both encoder and decoder state dictionaries."""

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    state = {
        'epoch': epoch,
        'epochs_since_improvement': epochs_since_improvement,
        'bleu-4': bleu4,
        'encoder': encoder,
        'decoder': decoder,
        'encoder_optimizer': encoder_optimizer,
        'decoder_optimizer': decoder_optimizer,
    }
    filename = CHECKPOINT_DIR / f'checkpoint_{data_name}.pth.tar'
    torch.save(state, filename)
    if is_best:
        torch.save(state, CHECKPOINT_DIR / f'BEST_checkpoint_{data_name}.pth.tar')


@dataclass
class AverageMeter:
    """Track running averages for scalar metrics."""

    val: float = 0.0
    avg: float = 0.0
    sum: float = 0.0
    count: int = 0

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0


def adjust_learning_rate(optimizer: torch.optim.Optimizer, shrink_factor: float) -> None:
    """Scale every learning rate in the optimizer by ``shrink_factor``."""

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] *= shrink_factor
    print(f"The new learning rate is {optimizer.param_groups[0]['lr']:.6f}\n")


def accuracy(scores: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """Compute the top-k accuracy for a batch."""

    batch_size = targets.size(0)
    _, topk = scores.topk(k, 1, True, True)
    correct = topk.eq(targets.view(-1, 1).expand_as(topk))
    return correct.reshape(-1).float().sum().item() * (100.0 / batch_size)
