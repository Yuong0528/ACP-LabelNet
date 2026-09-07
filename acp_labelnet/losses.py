"""Asymmetric multi-label loss and GradNorm task balancing."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def asymmetric_loss_per_label(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma_positive: float = 1.0,
    gamma_negative: float = 4.0,
    probability_clip: float = 0.05,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    positive = targets * (1.0 - probabilities).pow(gamma_positive) * F.logsigmoid(logits)
    clipped = (probabilities - probability_clip).clamp_min(0.0)
    negative = (1.0 - targets) * clipped.pow(gamma_negative) * torch.log1p(-clipped + 1e-8)
    return -(positive + negative).mean(dim=0)


class GradNormBalancer(nn.Module):
    """Learn task weights from relative training rates (Chen et al., 2018)."""

    def __init__(self, num_tasks: int, alpha: float = 1.5):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_tasks))
        self.alpha = alpha
        self.register_buffer("initial_losses", torch.zeros(num_tasks))
        self.initialized = False

    def initialize(self, losses: torch.Tensor) -> None:
        if not self.initialized:
            self.initial_losses.copy_(losses.detach())
            self.initialized = True

    def weighted_loss(self, losses: torch.Tensor) -> torch.Tensor:
        return (self.weights * losses).sum()

    def gradnorm_objective(self, losses: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
        gradient_norms = []
        for task in range(len(losses)):
            gradient = torch.autograd.grad(
                self.weights[task] * losses[task],
                shared,
                retain_graph=True,
                create_graph=True,
            )[0]
            gradient_norms.append(gradient.norm(2))
        gradient_norms = torch.stack(gradient_norms)
        with torch.no_grad():
            loss_ratio = losses.detach() / self.initial_losses.clamp_min(1e-8)
            inverse_rate = loss_ratio / loss_ratio.mean().clamp_min(1e-8)
            target = gradient_norms.detach().mean() * inverse_rate.pow(self.alpha)
        return torch.abs(gradient_norms - target).sum()

    @torch.no_grad()
    def renormalize(self) -> None:
        self.weights.clamp_(min=1e-3)
        self.weights.mul_(len(self.weights) / self.weights.sum().clamp_min(1e-6))

