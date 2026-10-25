"""Semantic source pyramids and distance-conditioned, masked fusion."""
from typing import Sequence

import torch
from torch import nn

from .rcda import conv_block


def sample_modality_mask(batch_size, drop_probability, training, device):
    """With total probability p, drop exactly one non-fire source per sample.

    The returned mask is reused at every scale and applied both to source inputs
    and fusion weights.  It is independent of the gate ablation switch.
    """
    if not 0 <= drop_probability <= 1:
        raise ValueError("drop_probability must be in [0,1]")
    mask = torch.ones(batch_size, 4, dtype=torch.bool, device=device)
    if training and drop_probability:
        drop = torch.rand(batch_size, device=device) < drop_probability
        source = torch.randint(1, 4, (batch_size,), device=device)
        mask[torch.arange(batch_size, device=device)[drop], source[drop]] = False
    return mask


class SourcePyramid(nn.Module):
    def __init__(self, in_channels: int, widths: Sequence[int]):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        for channels in widths:
            self.blocks.append(conv_block(in_channels, channels))
            in_channels = channels

    def forward(self, x):
        pyramid = []
        for index, block in enumerate(self.blocks):
            if index:
                x = self.pool(x)
            x = block(x)
            pyramid.append(x)
        return pyramid


class DistanceConditionedGate(nn.Module):
    def __init__(self, channels, use_gate=True, use_distance=True):
        super().__init__()
        self.use_gate = use_gate
        self.use_distance = use_distance
        hidden = max(channels, 4)
        # Four independent output rows can respond differently to distance.
        # There is no shared scalar distance bias that softmax cancels.
        self.scores = nn.Sequential(
            nn.Conv2d(channels * 4 + 1, hidden, 1), nn.SiLU(),
            nn.Conv2d(hidden, 4, 1),
        )

    def forward(self, features, distance, modality_mask):
        if len(features) != 4:
            raise ValueError("fusion requires exactly four source features")
        if modality_mask.dtype != torch.bool or modality_mask.shape != (features[0].shape[0], 4):
            raise ValueError("modality_mask must be a [B,4] boolean tensor")
        if not modality_mask[:, 0].all():
            raise ValueError("fire source must remain present")
        if self.use_gate:
            if not self.use_distance:
                distance = torch.zeros_like(distance)
            scores = self.scores(torch.cat(list(features) + [distance], dim=1))
        else:
            scores = torch.zeros(features[0].shape[0], 4, *features[0].shape[-2:],
                                 device=features[0].device, dtype=features[0].dtype)
        scores = scores.masked_fill(~modality_mask[:, :, None, None], float("-inf"))
        weights = scores.softmax(dim=1)
        fused = (torch.stack(list(features), dim=1) * weights[:, :, None]).sum(dim=1)
        return fused, weights
