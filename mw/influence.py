"""
Metrics for measuring influence a head has on a spatially positioned datum.
"""

import torch
from typing import Optional


def softmax_w_threshold(similarities: torch.Tensor, threshold: float):
    return (
        similarities.masked_fill(similarities < threshold, -float("inf"))
        .softmax(dim=0)
        .nan_to_num(0)
    )


def great_distance(
    similarities: torch.Tensor,
    threshold: Optional[float] = None,
    _denominator: float = torch.pi * 0.5,
):
    return 1 - torch.acos(similarities) / _denominator


def great_distance_w_threshold(similarities: torch.Tensor, threshold: float):
    return great_distance(similarities, _denominator=threshold).clip(0)
