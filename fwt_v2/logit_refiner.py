"""Deterministic clean-logit control for the frozen-H1 residual experiment.

This model shares P1's denoiser and its zero-output initialization, but never
reads a noisy label residual or performs diffusion sampling. The caller must
freeze H1 and use eval() when extracting conditions. All received conditions
and the original H1 logits are detached here as an additional gradient guard.
"""

from dataclasses import asdict
from typing import Mapping, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .diffusion import ConditionalResidualDiffusion, DiffusionConfig, _ConditionalUNet


class ConditionalLogitRefiner(nn.Module):
    """Predict sigmoid(z_H1 + delta) with the original multiscale denoiser.

    The loss choice is ordinary metadata, not part of state_dict, so BCE and
    focal controls can load exactly the same initial parameters. Config fields
    describing diffusion time are retained for architecture/init provenance;
    this model always supplies a zero residual tensor and timestep zero.
    """

    schema = "fwt_conditional_logit_refiner_v1"

    def __init__(self, config: Optional[DiffusionConfig] = None, *, loss="bce"):
        super().__init__()
        if loss not in ("bce", "focal"):
            raise ValueError("loss must be bce or focal")
        self.config = config if config is not None else DiffusionConfig(
            prediction_type="clean_probability", training_timestep_sampling="antithetic",
            output_initialization="zero")
        if not isinstance(self.config, DiffusionConfig):
            raise TypeError("config must be a DiffusionConfig")
        if (self.config.prediction_type != "clean_probability"
                or self.config.output_initialization != "zero"):
            raise ValueError("logit refiner requires clean_probability and zero output initialization")
        self.loss_name = loss
        # The diffusion constructor builds this same UNet before constructing
        # deterministic schedule buffers (which consume no RNG), then zeros
        # these same output tensors. Preserve all of its constructor RNG draws.
        self.denoiser = _ConditionalUNet(self.config)
        nn.init.zeros_(self.denoiser.output.weight)
        nn.init.zeros_(self.denoiser.output.bias)

    def configuration(self):
        """Explicit experiment metadata; no diffusion checkpoint compatibility claim."""
        return dict(schema=self.schema, diffusion_config=asdict(self.config),
                    loss=self.loss_name, residual_input="zeros", timestep=0,
                    output="sigmoid(detached_h1_logit_plus_correction)",
                    sampling="none", output_initialization="zero")

    def forward(self, outputs: Mapping, physics: Mapping):
        if not isinstance(outputs, Mapping) or not isinstance(physics, Mapping):
            raise ValueError("outputs and physics must be mappings")
        logits = outputs.get("logits_det")
        if (not isinstance(logits, torch.Tensor) or logits.ndim != 4
                or logits.shape[0] < 1 or logits.shape[1] != 1
                or min(logits.shape[-2:]) < 8 or not logits.is_floating_point()):
            raise ValueError("logits_det must be floating [B,1,H,W] with B>=1 and H,W>=8")
        if "distance" not in physics:
            raise ValueError("physics must contain distance")
        if self.config.condition_mode == "multiscale" and not all(
                name in outputs for name in ("fused_pyramid", "relational_feature")):
            raise ValueError("multiscale conditions require fused_pyramid and relational_feature")
        conditions = ConditionalResidualDiffusion.prepare_conditions(self, outputs, physics)
        p_det = conditions[0]
        residual = torch.zeros_like(p_det)
        timesteps = torch.zeros(p_det.shape[0], dtype=torch.long, device=p_det.device)
        correction = self.denoiser(residual, timesteps, conditions)
        return ConditionalResidualDiffusion.clean_probability_logits(correction, logits)

    def probability(self, outputs, physics):
        return self(outputs, physics).sigmoid()

    def training_loss(self, target, outputs, physics, mask=None):
        logits = self(outputs, physics)
        if (not isinstance(target, torch.Tensor) or target.shape != logits.shape
                or target.device != logits.device or target.dtype != logits.dtype
                or not torch.isfinite(target).all()
                or bool(((target < 0) | (target > 1)).any())):
            raise ValueError("target must match logit shape/device/dtype with finite values in [0,1]")
        target = target.detach()
        values = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        if self.loss_name == "focal":
            probability = logits.sigmoid()
            pt = torch.where(target > .5, probability, 1 - probability)
            alpha = torch.where(target > .5, .75, .25)
            values = alpha * (1 - pt).square() * values
        if mask is None:
            return values.mean()
        if (not isinstance(mask, torch.Tensor) or mask.shape != values.shape
                or mask.device != values.device or not torch.isfinite(mask).all()
                or bool(((mask < 0) | (mask > 1)).any()) or float(mask.sum()) <= 0):
            raise ValueError("mask must match target shape/device with finite [0,1] values and positive mass")
        weights = mask.detach()
        return (values * weights).sum() / weights.sum()
