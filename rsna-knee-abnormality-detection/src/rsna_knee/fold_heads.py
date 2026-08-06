"""Lightweight masked heads for precomputed MRI slice features."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MaskedAttentionHead(nn.Module):
    """Pool valid slice features independently for each target label."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        label_count: int,
    ) -> None:
        super().__init__()
        if feature_dim < 1 or hidden_dim < 1 or label_count < 1:
            raise ValueError("model dimensions must be positive")
        self.normalization = nn.LayerNorm(feature_dim)
        self.projection = nn.Linear(feature_dim, hidden_dim)
        self.queries = nn.Parameter(torch.empty(label_count, hidden_dim))
        self.output_weights = nn.Parameter(
            torch.empty(label_count, hidden_dim)
        )
        self.output_bias = nn.Parameter(torch.zeros(label_count))
        nn.init.normal_(self.queries, std=hidden_dim**-0.5)
        nn.init.normal_(self.output_weights, std=hidden_dim**-0.5)

    def forward(self, embeddings: Tensor, valid_mask: Tensor) -> Tensor:
        if embeddings.ndim != 3:
            raise ValueError("embeddings must have shape [batch, slices, features]")
        if valid_mask.shape != embeddings.shape[:2]:
            raise ValueError("valid_mask shape must match batch and slices")

        hidden = torch.tanh(
            self.projection(self.normalization(embeddings))
        )
        scores = torch.einsum(
            "bsh,lh->bsl", hidden, self.queries
        ) / math.sqrt(hidden.shape[-1])
        mask = valid_mask.to(dtype=torch.bool).unsqueeze(-1)
        scores = scores.masked_fill(~mask, -1e4)
        attention = torch.softmax(scores, dim=1) * mask.to(scores.dtype)
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(
            1e-8
        )
        pooled = torch.einsum("bsl,bsh->blh", attention, hidden)
        return (
            torch.einsum("blh,lh->bl", pooled, self.output_weights)
            + self.output_bias
        )


def weighted_soft_bce(
    logits: Tensor,
    targets: Tensor,
    weights: Tensor,
) -> Tensor:
    """Binary cross entropy for confidence-weighted soft targets."""
    if logits.shape != targets.shape or logits.shape != weights.shape:
        raise ValueError("logits, targets, and weights must have equal shape")
    if torch.any((targets < 0) | (targets > 1)):
        raise ValueError("targets must be between 0 and 1")
    if torch.any(weights < 0):
        raise ValueError("weights must be non-negative")
    losses = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )
    denominator = weights.sum()
    if denominator <= 0:
        raise ValueError("at least one weight must be positive")
    return (losses * weights).sum() / denominator
