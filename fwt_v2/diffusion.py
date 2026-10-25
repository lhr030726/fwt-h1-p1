"""Probability-residual diffusion with versioned prediction objectives.

The schedule and DDPM posterior follow Ho et al. (2006.11239) and
openai/improved-diffusion; deterministic skip sampling follows DDIM
(2010.02502, eta=0). This is a new model version, not a compatible loader for
the earlier FWT diffusion weights or their linear noise schedule.

The optional v target follows Salimans & Ho (2202.00512), avoiding division
by sqrt(alpha_bar) during clean-residual reconstruction at near-zero SNR.
It changes the training target/weighting and requires newly trained weights.

The clean_probability branch instead predicts a clean Bernoulli probability
with unweighted BCE, then uses its residual in the same Gaussian DDIM/DDPM
process. It is a distinct clean-prediction objective, not Bernoulli diffusion
or a calibration guarantee, and requires newly trained weights.
"""

from dataclasses import dataclass
import math
from typing import Mapping, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DiffusionConfig:
    timesteps: int = 200
    backbone_feature_channels: Tuple[int, ...] = (64, 128, 256, 512)
    relational_channels: int = 64
    hidden_channels: Tuple[int, ...] = (32, 64, 96, 128)
    condition_channels: int = 16
    condition_mode: str = "multiscale"
    prediction_type: str = "epsilon"
    training_timestep_sampling: str = "uniform"
    output_initialization: str = "random"

    def __post_init__(self):
        if not isinstance(self.timesteps, int) or self.timesteps < 2:
            raise ValueError("timesteps must be an integer >= 2")
        if len(self.backbone_feature_channels) != 4 or len(self.hidden_channels) != 4:
            raise ValueError("exactly four backbone and denoiser feature scales are required")
        if any(c < 1 for c in (*self.backbone_feature_channels, *self.hidden_channels,
                               self.relational_channels, self.condition_channels)):
            raise ValueError("all channel counts must be positive")
        if self.condition_mode not in {"multiscale", "probability_distance"}:
            raise ValueError("unknown diffusion condition_mode")
        if self.prediction_type not in {"epsilon", "v", "clean_probability"}:
            raise ValueError("prediction_type must be epsilon, v or clean_probability")
        if self.training_timestep_sampling not in {"uniform", "antithetic"}:
            raise ValueError("training_timestep_sampling must be uniform or antithetic")
        if self.output_initialization not in {"random", "zero"}:
            raise ValueError("output_initialization must be random or zero")
        if self.output_initialization == "zero" and self.prediction_type != "clean_probability":
            raise ValueError("zero output initialization requires clean_probability")


def _normal(shape, like, generator=None):
    # A CPU generator remains reproducible even when the destination is CUDA.
    device = like.device if generator is None else generator.device
    return torch.randn(shape, device=device, dtype=like.dtype, generator=generator).to(like.device)


def _extract(vector, timesteps, reference):
    return vector[timesteps].to(reference.dtype).reshape(-1, 1, 1, 1)


def _block(cin, cout):
    # GroupNorm avoids updating running statistics during sampling or on tiny batches.
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(1, cout), nn.SiLU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(1, cout), nn.SiLU())


class _ConditionalUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        widths, cc = config.hidden_channels, config.condition_channels
        self.time_mlp = nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 64))
        self.time_projections = nn.ModuleList([nn.Linear(64, c) for c in widths])
        self.feature_projections = nn.ModuleList()
        self.rel_projection = None
        if config.condition_mode == "multiscale":
            self.feature_projections.extend(nn.Conv2d(c, cc, 1)
                                            for c in config.backbone_feature_channels)
            self.rel_projection = nn.Conv2d(config.relational_channels, cc, 1)
        self.encoders = nn.ModuleList()
        for scale, width in enumerate(widths):
            base_channels = 3 if scale == 0 else widths[scale-1] + 2
            extra = cc if config.condition_mode == "multiscale" else 0
            if scale == 2 and self.rel_projection is not None:
                extra += cc
            self.encoders.append(_block(base_channels + extra, width))
        self.decoders = nn.ModuleList([_block(widths[s+1] + widths[s], widths[s])
                                       for s in range(3)])
        self.output = nn.Conv2d(widths[0], 1, 1)

    def forward(self, rt, timesteps, conditions):
        p_det, distance, pyramid, relational = conditions
        frequencies = torch.exp(torch.arange(16, device=rt.device, dtype=rt.dtype)
                                * (-math.log(10000.0) / 15))
        angles = timesteps.to(rt.dtype)[:, None] * frequencies[None]
        time = self.time_mlp(torch.cat([angles.sin(), angles.cos()], dim=1))
        skips = []
        h = rt
        for scale, encoder in enumerate(self.encoders):
            if scale:
                h = F.avg_pool2d(h, 2)
            size = h.shape[-2:]
            parts = [h, F.interpolate(p_det, size=size, mode="bilinear", align_corners=False),
                     F.interpolate(distance, size=size, mode="bilinear", align_corners=False)]
            if self.config.condition_mode == "multiscale":
                feature = self.feature_projections[scale](pyramid[scale])
                parts.append(F.interpolate(feature, size=size, mode="bilinear", align_corners=False))
                if scale == 2:
                    feature = self.rel_projection(relational)
                    parts.append(F.interpolate(feature, size=size, mode="bilinear", align_corners=False))
            h = encoder(torch.cat(parts, dim=1))
            h = h + self.time_projections[scale](time)[:, :, None, None]
            skips.append(h)
        for scale in (2, 1, 0):
            h = F.interpolate(h, size=skips[scale].shape[-2:], mode="bilinear", align_corners=False)
            h = self.decoders[scale](torch.cat([h, skips[scale]], dim=1))
        return self.output(h)


class ConditionalResidualDiffusion(nn.Module):
    """Learn R0 = Y - sigmoid(logits_det); backbone tensors are detached.

    The caller must also freeze the backbone parameters and call its eval()
    before extracting conditions, to prevent changing its weights or BN state.
    Projection layers here remain trainable. Features do not receive gradients.
    """

    def __init__(self, config: Optional[DiffusionConfig] = None):
        super().__init__()
        self.config = config or DiffusionConfig()
        self.denoiser = _ConditionalUNet(self.config)
        t = torch.linspace(0, 1, self.config.timesteps + 1, dtype=torch.float64)
        curve = torch.cos((t + 0.008) / 1.008 * math.pi / 2).square()
        betas = (1 - curve[1:] / curve[:-1]).clamp(max=0.999)
        alphas = 1 - betas
        bars = torch.cumprod(alphas, 0)
        previous = torch.cat([torch.ones(1, dtype=torch.float64), bars[:-1]])
        values = {
            "betas": betas, "alphas": alphas, "alpha_bars": bars,
            "alpha_bars_previous": previous,
            "posterior_variance": betas * (1-previous) / (1-bars),
            "posterior_coef_r0": betas * previous.sqrt() / (1-bars),
            "posterior_coef_rt": (1-previous) * alphas.sqrt() / (1-bars),
        }
        for name, value in values.items():
            self.register_buffer(name, value.float())
        if self.config.output_initialization == "zero":
            # Construct normally first, preserving every constructor RNG draw.
            # The initial clean-logit correction is zero; it is not a checkpoint candidate.
            nn.init.zeros_(self.denoiser.output.weight)
            nn.init.zeros_(self.denoiser.output.bias)

    @property
    def terminal_snr(self):
        return self.alpha_bars[-1] / (1-self.alpha_bars[-1])

    def _check_timesteps(self, timesteps, reference, allow_final=False):
        low = -1 if allow_final else 0
        if (timesteps.shape != (reference.shape[0],) or timesteps.dtype != torch.long
                or timesteps.device != reference.device
                or bool(((timesteps < low) | (timesteps >= self.config.timesteps)).any())):
            raise ValueError("timesteps must be a device-matched LongTensor[B] in the schedule range")

    def q_sample(self, r0, timesteps, noise=None, generator=None):
        self._check_timesteps(timesteps, r0)
        noise = _normal(r0.shape, r0, generator) if noise is None else noise
        if noise.shape != r0.shape or not torch.isfinite(noise).all():
            raise ValueError("noise must be finite and match residual shape")
        ab = _extract(self.alpha_bars, timesteps, r0)
        return ab.sqrt() * r0 + (1-ab).sqrt() * noise, noise

    def training_target(self, r0, epsilon, timesteps):
        """v = sqrt(alpha_bar)*epsilon - sqrt(1-alpha_bar)*R0.

        v-MSE and epsilon-MSE have different SNR weightings; the target is
        checkpoint metadata, never a post-hoc reinterpretation of old weights.
        """
        self._check_timesteps(timesteps, r0)
        if self.config.prediction_type == "clean_probability":
            raise ValueError("clean_probability uses clean-label BCE, not a noise/v target")
        if self.config.prediction_type == "epsilon":
            return epsilon
        ab = _extract(self.alpha_bars, timesteps, r0)
        return ab.sqrt() * epsilon - (1-ab).sqrt() * r0

    @staticmethod
    def clean_probability_logits(model_output, logits_det):
        """Frozen backbone logit plus a learned correction, without hard clipping.

        Use the original finite logits rather than logit(sigmoid(logits)), which
        would lose information when the sigmoid rounds to an endpoint in FP32.
        """
        if (logits_det is None or logits_det.shape != model_output.shape
                or logits_det.device != model_output.device
                or logits_det.dtype != model_output.dtype
                or not logits_det.is_floating_point()
                or not torch.isfinite(logits_det).all()
                or not torch.isfinite(model_output).all()):
            raise ValueError("clean_probability requires finite matching original logits_det")
        return logits_det.detach() + model_output

    def predict_clean_residual(self, rt, timesteps, model_output, logits_det=None):
        """Unclipped R0; expose it so diagnostics cannot hide clipping blow-up."""
        self._check_timesteps(timesteps, rt)
        if model_output.shape != rt.shape or not torch.isfinite(model_output).all():
            raise ValueError("model_output must be finite and match noisy residual shape")
        if self.config.prediction_type == "clean_probability":
            clean_logits = self.clean_probability_logits(model_output, logits_det)
            return clean_logits.sigmoid() - logits_det.detach().sigmoid()
        ab = _extract(self.alpha_bars, timesteps, rt)
        if self.config.prediction_type == "v":
            return ab.sqrt() * rt - (1-ab).sqrt() * model_output
        return (rt - (1-ab).sqrt() * model_output) / ab.sqrt()

    def posterior_mean_variance(self, rt, timesteps, epsilon, logits_det=None):
        """Standard adjacent DDPM posterior, with predicted clean residual clipped."""
        self._check_timesteps(timesteps, rt)
        # The positional argument retains its legacy name but represents the
        # configured model output (epsilon, v or a clean-probability logit correction).
        r0 = self.predict_clean_residual(rt, timesteps, epsilon, logits_det).clamp(-1, 1)
        mean = (_extract(self.posterior_coef_r0, timesteps, rt) * r0
                + _extract(self.posterior_coef_rt, timesteps, rt) * rt)
        return mean, _extract(self.posterior_variance, timesteps, rt), r0

    def ddim_step(self, rt, timesteps, previous_timesteps, epsilon, logits_det=None):
        """Deterministic DDIM (eta=0), including the final alpha_previous=1 step."""
        self._check_timesteps(timesteps, rt)
        self._check_timesteps(previous_timesteps, rt, allow_final=True)
        if bool((previous_timesteps >= timesteps).any()):
            raise ValueError("DDIM previous timesteps must be strictly earlier")
        ab = _extract(self.alpha_bars, timesteps, rt)
        ab_prev = _extract(self.alpha_bars, previous_timesteps.clamp_min(0), rt)
        ab_prev = torch.where(previous_timesteps[:, None, None, None] < 0, 1.0, ab_prev)
        r0 = self.predict_clean_residual(rt, timesteps, epsilon, logits_det).clamp(-1, 1)
        # Recompute epsilon after clipping for consistency with the predicted r0.
        epsilon = (rt - ab.sqrt() * r0) / (1-ab).sqrt()
        return ab_prev.sqrt() * r0 + (1-ab_prev).sqrt() * epsilon

    def prepare_conditions(self, outputs, physics):
        logits = outputs["logits_det"]
        if (logits.ndim != 4 or logits.shape[1] != 1 or min(logits.shape[-2:]) < 8
                or not logits.is_floating_point()):
            raise ValueError("logits_det must have shape [B,1,H,W], H,W>=8")
        if not torch.isfinite(logits).all():
            raise ValueError("logits_det contains non-finite values")
        distance = physics["distance"]
        if (distance.shape != logits.shape or not torch.isfinite(distance).all()
                or distance.device != logits.device or distance.dtype != logits.dtype
                or bool((distance.abs() > 1.00001).any())):
            raise ValueError("distance must be finite and match logits_det")
        pyramid, relational = (), None
        if self.config.condition_mode == "multiscale":
            pyramid = outputs["fused_pyramid"]
            if len(pyramid) != 4:
                raise ValueError("four fused_pyramid scales required")
            for scale, (feature, channels) in enumerate(zip(pyramid, self.config.backbone_feature_channels)):
                expected = (logits.shape[0], channels, logits.shape[-2] // (2**scale),
                            logits.shape[-1] // (2**scale))
                if (tuple(feature.shape) != expected
                        or feature.device != logits.device or feature.dtype != logits.dtype
                        or not torch.isfinite(feature).all()):
                    raise ValueError("fused feature shape or values do not match diffusion config")
            relational = outputs["relational_feature"]
            expected = (logits.shape[0], self.config.relational_channels,
                        logits.shape[-2] // 4, logits.shape[-1] // 4)
            if (tuple(relational.shape) != expected
                    or relational.device != logits.device or relational.dtype != logits.dtype
                    or not torch.isfinite(relational).all()):
                raise ValueError("relational feature does not match diffusion config")
            pyramid = tuple(feature.detach() for feature in pyramid)
            relational = relational.detach()
        return logits.detach().sigmoid(), distance.detach(), pyramid, relational

    def training_loss(self, target, outputs, physics, mask=None, timesteps=None,
                      noise=None, generator=None):
        conditions = self.prepare_conditions(outputs, physics)
        p_det = conditions[0]
        if (target.shape != p_det.shape or not torch.isfinite(target).all()
                or bool(((target < 0) | (target > 1)).any())):
            raise ValueError("target must be finite [B,1,H,W] values in [0,1]")
        if timesteps is None:
            device = p_det.device if generator is None else generator.device
            timesteps = torch.randint(self.config.timesteps, (target.shape[0],),
                                      device=device, generator=generator).to(p_det.device)
        r0 = target.detach() - p_det
        rt, epsilon = self.q_sample(r0, timesteps, noise, generator)
        prediction = self.denoiser(rt, timesteps, conditions)
        if self.config.prediction_type == "clean_probability":
            squared = F.binary_cross_entropy_with_logits(
                self.clean_probability_logits(prediction, outputs["logits_det"]),
                target.detach(), reduction="none")
        else:
            squared = (prediction-self.training_target(r0, epsilon, timesteps)).square()
        groups = [timesteps.detach().cpu().tolist()]
        if self.config.training_timestep_sampling == "antithetic":
            # Both (t, epsilon) and (T-1-t, -epsilon) have the original uniform
            # timestep / standard Gaussian marginals. Averaging their losses
            # preserves the expected objective, with two denoiser evaluations.
            # It guarantees complementary timestep coverage, not lower gradient
            # variance or improved calibration. No extra RNG draws are used.
            paired_t = self.config.timesteps - 1 - timesteps
            paired_rt, paired_noise = self.q_sample(r0, paired_t, noise=-epsilon)
            paired_prediction = self.denoiser(paired_rt, paired_t, conditions)
            if self.config.prediction_type == "clean_probability":
                paired_squared = F.binary_cross_entropy_with_logits(
                    self.clean_probability_logits(paired_prediction, outputs["logits_det"]),
                    target.detach(), reduction="none")
            else:
                paired_squared = (paired_prediction-self.training_target(r0, paired_noise, paired_t)).square()
            squared = .5 * (squared + paired_squared)
            groups.append(paired_t.detach().cpu().tolist())
        # Plain transient metadata: not parameters/buffers, never a checkpoint
        # selection signal. The engine records it beside the sample identities.
        self.last_training_diagnostics = dict(
            timestep_sampling=self.config.training_timestep_sampling,
            timestep_groups=groups, denoiser_evaluations=len(groups),
            noise_draws=int(noise is None))
        if mask is None:
            return squared.mean()
        if (mask.shape != squared.shape or not torch.isfinite(mask).all()
                or bool(((mask < 0) | (mask > 1)).any()) or float(mask.sum()) <= 0):
            raise ValueError("mask must match target with finite [0,1] values and positive mass")
        return (squared * mask).sum() / mask.sum()

    def forward(self, target, outputs, physics, **kwargs):
        return self.training_loss(target, outputs, physics, **kwargs)

    @torch.no_grad()
    def sample(self, outputs, physics, K=4, steps=20, method="ddim", generator=None):
        if not isinstance(K, int) or K < 1:
            raise ValueError("K must be a positive integer")
        if not isinstance(steps, int) or not 1 <= steps <= self.config.timesteps:
            raise ValueError("steps must lie in [1,timesteps]")
        if method not in {"ddim", "ddpm"}:
            raise ValueError("method must be 'ddim' or 'ddpm'")
        if method == "ddpm" and steps != self.config.timesteps:
            raise ValueError("DDPM requires all adjacent steps; use DDIM for skip sampling")
        p_det, distance, pyramid, relational = self.prepare_conditions(outputs, physics)
        batch, _, height, width = p_det.shape
        repeated = (p_det.repeat_interleave(K, 0), distance.repeat_interleave(K, 0),
                    tuple(f.repeat_interleave(K, 0) for f in pyramid),
                    None if relational is None else relational.repeat_interleave(K, 0))
        logits_det = (outputs["logits_det"].detach().repeat_interleave(K, 0)
                      if self.config.prediction_type == "clean_probability" else None)
        rt = _normal((batch*K, 1, height, width), p_det, generator)
        indices = torch.linspace(self.config.timesteps-1, 0, steps).round().long().tolist()
        for position, step in enumerate(indices):
            timesteps = torch.full((batch*K,), step, dtype=torch.long, device=rt.device)
            epsilon = self.denoiser(rt, timesteps, repeated)
            if method == "ddpm":
                mean, variance, _ = self.posterior_mean_variance(rt, timesteps, epsilon, logits_det)
                rt = mean if step == 0 else mean + variance.sqrt() * _normal(rt.shape, rt, generator)
            else:
                previous = indices[position+1] if position+1 < len(indices) else -1
                previous = torch.full_like(timesteps, previous)
                rt = self.ddim_step(rt, timesteps, previous, epsilon, logits_det)
        return rt.clamp(-1, 1).reshape(batch, K, 1, height, width)


class DeterministicResidualHead(nn.Module):
    """Same-condition residual correction control, trained on the same Tr labels.

    This auxiliary control shares the denoiser architecture and conditioning
    channels, but uses zero residual input/time and no noise schedule or sampling.
    It separates ordinary residual fitting from diffusion's stochastic modeling.
    """

    def __init__(self, config: Optional[DiffusionConfig] = None):
        super().__init__()
        self.config = config or DiffusionConfig()
        if self.config.output_initialization != "random":
            raise ValueError("zero output initialization is only supported by the diffusion head")
        self.denoiser = _ConditionalUNet(self.config)

    def forward(self, outputs, physics):
        conditions = ConditionalResidualDiffusion.prepare_conditions(self, outputs, physics)
        p_det = conditions[0]
        zeros = torch.zeros_like(p_det)
        timesteps = torch.zeros(p_det.shape[0], dtype=torch.long, device=p_det.device)
        return self.denoiser(zeros, timesteps, conditions).tanh()

    def probability(self, outputs, physics):
        residual = self(outputs, physics)
        return (outputs["logits_det"].detach().sigmoid() + residual).clamp(0, 1)

    def training_loss(self, target, outputs, physics, mask=None):
        residual = self(outputs, physics)
        p_det = outputs["logits_det"].detach().sigmoid()
        if (target.shape != residual.shape or not torch.isfinite(target).all()
                or bool(((target < 0) | (target > 1)).any())):
            raise ValueError("target must be finite [B,1,H,W] values in [0,1]")
        squared = (residual - (target.detach() - p_det)).square()
        if mask is None:
            return squared.mean()
        if (mask.shape != squared.shape or not torch.isfinite(mask).all()
                or bool(((mask < 0) | (mask > 1)).any()) or float(mask.sum()) <= 0):
            raise ValueError("mask must match target with finite [0,1] values and positive mass")
        return (squared * mask).sum() / mask.sum()


def _sample_probabilities(outputs: Mapping, residual_samples, kappa):
    if not math.isfinite(kappa) or kappa < 0:
        raise ValueError("kappa must be finite and nonnegative")
    probability = outputs["logits_det"].detach().sigmoid()
    if (residual_samples.ndim != 5 or residual_samples.shape[1] < 1
            or residual_samples.shape[0] != probability.shape[0]
            or residual_samples.shape[2:] != probability.shape[1:]
            or not torch.isfinite(residual_samples).all()):
        raise ValueError("residual samples must be finite [B,K,1,H,W] matching logits")
    return (probability[:, None] + kappa * residual_samples).clamp(0, 1)


def compute_probability(outputs, residual_samples, kappa=1.0):
    """Clip each reconstructed sample first, then average across K."""
    return _sample_probabilities(outputs, residual_samples, kappa).mean(dim=1)


def sampling_disagreement(outputs, residual_samples, kappa=1.0):
    """Conditional sampling variance, not an epistemic/aleatoric decomposition."""
    return _sample_probabilities(outputs, residual_samples, kappa).var(dim=1, unbiased=False)
