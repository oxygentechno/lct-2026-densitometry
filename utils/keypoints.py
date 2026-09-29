"""Soft-argmax по heatmap. Порядок координат как на evaluate: пара (x, y) после flatten."""

from __future__ import annotations

import torch
from torch import Tensor


def extract_coordinates_softargmax(heatmap: Tensor, start_dim: int = 2, temperature: int = 1) -> Tensor:
    flattened_heatmap = heatmap.flatten(start_dim)
    log_probs = torch.log_softmax(flattened_heatmap, dim=-1) / temperature
    log_probs = log_probs - torch.logsumexp(log_probs, dim=-1, keepdims=True)
    log_probs = log_probs.unflatten(dim=-1, sizes=heatmap.shape[start_dim:])

    coords = []
    for i in range(start_dim, len(heatmap.shape)):
        shape = [1 for _ in heatmap.shape]
        shape[i] = heatmap.shape[i]
        coords.append(
            torch.exp(torch.log(torch.arange(heatmap.shape[i], device=heatmap.device).view(shape)) + log_probs).sum(
                dim=tuple(range(start_dim, len(heatmap.shape)))
            )
        )
    return torch.stack(coords, dim=-1)
