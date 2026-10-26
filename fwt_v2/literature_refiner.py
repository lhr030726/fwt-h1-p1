"""Small frozen-decoder residual head and explicit masked BCE/Tversky objective.

The head changes no H1 parameters or data semantics. Tversky is a region-overlap
auxiliary objective with explicit FP/FN weights, not an AP surrogate.
"""
from collections.abc import Mapping
import math

import torch
from torch import nn
from torch.nn import functional as F


class DecoderFeatureRefiner(nn.Module):
    """Add a zero-initialized spatial correction to detached H1 raw logits."""

    schema = "fwt_decoder_feature_refiner_v1"

    def __init__(self, decoder_channels=64, hidden_channels=32):
        super().__init__()
        if any(type(value) is not int or value < 1
               for value in (decoder_channels, hidden_channels)):
            raise ValueError("decoder_channels and hidden_channels must be positive integers")
        self.decoder_channels = decoder_channels
        self.hidden_channels = hidden_channels
        self.features = nn.Sequential(
            nn.Conv2d(decoder_channels + 2, hidden_channels, 3, padding=1),
            nn.GroupNorm(1, hidden_channels), nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(1, hidden_channels), nn.SiLU())
        self.output = nn.Conv2d(hidden_channels, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def configuration(self):
        return dict(schema=self.schema, decoder_channels=self.decoder_channels,
                    hidden_channels=self.hidden_channels, convolution_kernels=[3, 3, 1],
                    normalization="group", normalization_groups=1, activation="silu",
                    condition_order=["decoder_feature", "logits_det", "distance"],
                    conditions_detached=True, output_initialization="zero",
                    output="detached_h1_logit_plus_correction", sampling="none")

    def forward(self, outputs, physics):
        if not isinstance(outputs, Mapping) or not isinstance(physics, Mapping):
            raise ValueError("outputs and physics must be mappings")
        logits = outputs.get("logits_det")
        if (not isinstance(logits, torch.Tensor) or logits.ndim != 4
                or logits.shape[0] < 1 or logits.shape[1] != 1
                or min(logits.shape[-2:]) < 1 or not logits.is_floating_point()
                or not bool(torch.isfinite(logits).all())):
            raise ValueError("logits_det must be finite floating [B,1,H,W]")
        feature, distance = outputs.get("decoder_feature"), physics.get("distance")
        for name, value, channels in (("decoder_feature", feature, self.decoder_channels),
                                       ("distance", distance, 1)):
            expected = (logits.shape[0], channels, *logits.shape[-2:])
            if (not isinstance(value, torch.Tensor) or tuple(value.shape) != expected
                    or value.device != logits.device or value.dtype != logits.dtype
                    or not bool(torch.isfinite(value).all())):
                raise ValueError(f"{name} must be finite and match logit shape/device/dtype")
        if bool((distance.abs() > 1.00001).any()):
            raise ValueError("distance must use the frozen [-1,1] scale")
        conditions = torch.cat((feature.detach(), logits.detach(), distance.detach()), dim=1)
        return logits.detach() + self.output(self.features(conditions))

    def probability(self, outputs, physics):
        return self(outputs, physics).sigmoid()


def masked_refinement_loss(logits, target, mask, aux_weight=0., alpha_fp=.3, beta_fn=.7):
    """Return total/BCE/Tversky; only valid pixels contribute or receive gradients.

    Tversky = 1 - (TP + eps)/(TP + alpha_fp*FP + beta_fn*FN + eps),
    pooled across all valid pixels in the batch, eps=1e-6. A batch without a
    positive label has a differentiable zero auxiliary term, retaining BCE.
    Unknown labels outside the binary validity mask are ignored before arithmetic.
    """
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 4
            or logits.shape[0] < 1 or logits.shape[1] != 1
            or min(logits.shape[-2:]) < 1 or not logits.is_floating_point()
            or not bool(torch.isfinite(logits).all())):
        raise ValueError("logits must be finite floating [B,1,H,W]")
    if (not isinstance(target, torch.Tensor) or target.shape != logits.shape
            or target.device != logits.device or target.dtype != logits.dtype):
        raise ValueError("target must match logits shape/device/dtype")
    if (not isinstance(mask, torch.Tensor) or mask.shape != logits.shape
            or mask.device != logits.device or not bool(torch.isfinite(mask).all())
            or bool(((mask != 0) & (mask != 1)).any())):
        raise ValueError("mask must be binary and match logits shape/device")
    if not bool(mask.any()):
        raise ValueError("mask has no valid supervision")
    for name, value in (("aux_weight", aux_weight), ("alpha_fp", alpha_fp), ("beta_fn", beta_fn)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if alpha_fp + beta_fn <= 0:
        raise ValueError("alpha_fp and beta_fn must have positive total weight")
    valid = mask.detach().bool()
    labels = target.detach()[valid]
    if not bool(torch.isfinite(labels).all()) or bool(((labels != 0) & (labels != 1)).any()):
        raise ValueError("target at valid pixels must be finite and binary")
    selected = logits[valid]
    bce = F.binary_cross_entropy_with_logits(selected, labels, reduction="mean")
    probability = selected.sigmoid()
    positives = labels.sum()
    tp = (probability * labels).sum()
    fp = (probability * (1 - labels)).sum()
    fn = ((1 - probability) * labels).sum()
    epsilon = 1e-6
    overlap = 1 - (tp + epsilon) / (tp + alpha_fp * fp + beta_fn * fn + epsilon)
    tversky = torch.where(positives > 0, overlap, tp * 0.)
    return dict(total=bce + aux_weight * tversky, bce=bce, tversky=tversky)
