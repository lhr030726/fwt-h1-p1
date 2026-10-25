"""FWT-Det v2: semantic multiscale fusion and physical directional propagation.

The 18-slot transport schema includes the unused raw angle at slot 7.  The model
actually encodes 17 channels in explicit groups; their legacy order is NEVER
interpreted as four contiguous slices.  Physical projections use an independent
signed interface, not the standardized network-input channels.
The explicit POWER schema adds four weather covariates at slots 18..21. Its
daily control uses the identical architecture with these four inputs zeroed.
"""
from dataclasses import dataclass
from typing import Mapping, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .fusion import DistanceConditionedGate, SourcePyramid, sample_modality_mask
from .propagation import DirectionalPropagation
from .rcda import RCDADecoder, conv_block
from .normalization import convert_batch_to_group


GROUP_INDICES = ((0, 12), (1, 15, 16, 17), (2, 3, 4, 5),
                 (6, 8, 9, 10, 11, 13, 14))


@dataclass(frozen=True)
class FWTConfig:
    schema_version: str = "rcda18-semantic17-v2"
    widths: Tuple[int, int, int, int] = (16, 32, 64, 128)
    decoder_widths: Tuple[int, int, int, int] = (64, 128, 256, 512)
    prop_channels: int = 64
    use_gate: bool = True
    use_distance: bool = True
    use_propagation: bool = True
    use_physics_prior: bool = True
    drop_modal: float = 0.15
    propagation_steps: int = 3
    propagation_radius: int = 3
    wind_scale: float = 1.0
    slope_scale: float = 1.0
    normalization: str = 'batch'
    norm_max_groups: int = 8
    use_wind_prior: bool = True
    extra_weather_channels: int = 0

    def __post_init__(self):
        if type(self.extra_weather_channels) is not int or self.extra_weather_channels not in (0, 4):
            raise ValueError("extra_weather_channels must be 0 (legacy) or 4 (POWER hourly schema)")
        expected_schema = "rcda22-semantic21-v3" if self.extra_weather_channels == 4 else "rcda18-semantic17-v2"
        if self.schema_version != expected_schema:
            raise ValueError("unsupported FWT input schema")
        if len(self.widths) != 4 or len(self.decoder_widths) != 4:
            raise ValueError("FWT requires exactly four pyramid levels")
        if min(*self.widths, *self.decoder_widths, self.prop_channels) < 1:
            raise ValueError("all model widths must be positive")
        if not 0 <= self.drop_modal <= 1:
            raise ValueError("drop_modal must be in [0,1]")
        if self.normalization not in ('batch', 'group'):
            raise ValueError('normalization must be batch or group')
        if type(self.norm_max_groups) is not int or self.norm_max_groups < 1:
            raise ValueError('norm_max_groups must be a positive integer')
        if not isinstance(self.use_wind_prior, bool):
            raise ValueError("use_wind_prior must be a boolean")


def validate_wind_prior_config(data_config, model_config):
    """Reject hard wind priors for learned-only encodings, without mutating defaults.

    Called wherever stored or requested data and model configurations meet.
    Even disabled propagation/physics does not waive the explicit false flag.
    """
    if data_config.covariate_only_wind and model_config.use_wind_prior:
        raise ValueError(f"{data_config.wind_convention} requires model.use_wind_prior=false")
    if data_config.input_channels != 18 + model_config.extra_weather_channels:
        raise ValueError("model extra weather channels disagree with data input schema")


class FWTDet(RCDADecoder):
    """First formal main model; decoder is local and outputs raw logits.

    `use_distance=False` disables explicit distance conditioning in the gate and
    propagation only.  A complete explicit-distance ablation must ALSO zero slot
    12 before encoding and all distance conditions of a diffusion branch.
    """

    def __init__(self, config: FWTConfig = None):
        config = config or FWTConfig()
        super().__init__(config.decoder_widths)
        self.config = config
        self.group_indices = GROUP_INDICES[:-1] + (GROUP_INDICES[-1] + tuple(range(18, 18 + config.extra_weather_channels)),)
        self.group_encoders = nn.ModuleList(
            [SourcePyramid(len(indices), config.widths) for indices in self.group_indices])
        self.gates = nn.ModuleList([
            DistanceConditionedGate(channels, config.use_gate, config.use_distance)
            for channels in config.widths])
        self.fusion_projections = nn.ModuleList([
            nn.Conv2d(source, target, 1)
            for source, target in zip(config.widths, config.decoder_widths)])
        self.prop_compress = nn.Conv2d(config.decoder_widths[2], config.prop_channels, 1)
        self.prop_expand = nn.Conv2d(config.prop_channels, config.decoder_widths[2], 1,
                                     bias=False)
        self.propagation = DirectionalPropagation(
            config.prop_channels, config.propagation_steps, config.propagation_radius,
            config.use_distance, config.use_physics_prior, config.wind_scale,
            config.slope_scale, use_wind_prior=config.use_wind_prior)
        if config.normalization == 'group':
            convert_batch_to_group(self, config.norm_max_groups)

    def _validate_inputs(self, x, physics: Mapping[str, torch.Tensor]):
        channels = 18 + self.config.extra_weather_channels
        if x.ndim != 4 or x.shape[1] != channels:
            raise ValueError(f"FWTDet expects x_norm [B,{channels},H,W] with the declared schema")
        if min(x.shape[-2:]) < 8:
            raise ValueError("four pyramid levels require H and W >= 8")
        if not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError("network input must be floating point and finite")
        for name, channels in (("distance", 1), ("wind_xy", 2), ("slope", 1),
                               ("upslope_xy", 2)):
            value = physics.get(name)
            expected = (x.shape[0], channels, *x.shape[-2:])
            if value is None or tuple(value.shape) != expected:
                raise ValueError("physics.%s must have shape %s" % (name, expected))
            if value.device != x.device or value.dtype != x.dtype:
                raise ValueError("physics and x_norm must share device and floating dtype")
            if not torch.isfinite(value).all():
                raise ValueError("physics.%s contains a non-finite value" % name)
        if (physics["distance"].abs() > 1 + 1e-5).any():
            raise ValueError("signed distance must use the frozen [-1,1] scale")
        if (physics["slope"] < 0).any():
            raise ValueError("physical slope magnitude must be nonnegative")

    def forward(self, x_norm, physics, modality_mask=None):
        self._validate_inputs(x_norm, physics)
        if modality_mask is None:
            modality_mask = sample_modality_mask(
                x_norm.shape[0], self.config.drop_modal, self.training, x_norm.device)
        if (modality_mask.dtype != torch.bool
                or modality_mask.shape != (x_norm.shape[0], 4)
                or modality_mask.device != x_norm.device or not modality_mask[:, 0].all()):
            raise ValueError("modality_mask must be [B,4] boolean with fire present")
        source_pyramids = []
        for group, (encoder, indices) in enumerate(zip(self.group_encoders, self.group_indices)):
            source = x_norm[:, list(indices)] * modality_mask[:, group, None, None, None]
            source_pyramids.append(encoder(source))
        fused_pyramid, gate_weights = [], []
        for scale, (gate, projection) in enumerate(zip(self.gates, self.fusion_projections)):
            features = [pyramid[scale] for pyramid in source_pyramids]
            distance = F.interpolate(physics["distance"], size=features[0].shape[-2:],
                                     mode="area")
            fused, weights = gate(features, distance, modality_mask)
            fused_pyramid.append(projection(fused))
            gate_weights.append(weights)
        compressed = self.prop_compress(fused_pyramid[2])
        lowres = {name: F.interpolate(value, size=compressed.shape[-2:], mode="area")
                  for name, value in physics.items() if name in
                  ("distance", "wind_xy", "slope", "upslope_xy")}
        # Source removal must also remove its direct physical shortcut.
        lowres['wind_xy'] = lowres['wind_xy'] * modality_mask[:, 3, None, None, None]
        for name in ('slope', 'upslope_xy'):
            lowres[name] = lowres[name] * modality_mask[:, 1, None, None, None]
        direction_weights = None
        relational_feature = compressed
        if self.config.use_propagation:
            relational_feature, direction_weights = self.propagation(
                compressed, lowres["distance"], lowres["wind_xy"], lowres["slope"],
                lowres["upslope_xy"])
            # beta=0 makes this entire skip update exactly zero at initialization.
            fused_pyramid[2] = fused_pyramid[2] + self.prop_expand(relational_feature - compressed)
        return {"logits_det": self.decode(fused_pyramid),
                "fused_pyramid": fused_pyramid, "relational_feature": relational_feature,
                "gate_weights": gate_weights, "direction_weights": direction_weights,
                "modality_mask": modality_mask}


class RCDABaseline(RCDADecoder):
    """Auxiliary RCDA input/architecture control, not a third main model.

    `raw12` reads slots 0..11 including the original wind angle. `derived17`
    reads all transport slots except that raw angle at slot 7. Both receive the
    same versioned 18-slot batch transport interface as FWT-Det. They use one
    shared encoder with local original-style conv blocks and the same decoder
    widths, and deliberately never read `physics`. Changed normalization/target
    semantics still require their own audited checkpoint provenance.
    """

    def __init__(self, config: FWTConfig = None, input_mode="raw12"):
        config = config or FWTConfig()
        if config.extra_weather_channels:
            raise ValueError('RCDA auxiliary baselines do not support external hourly wind channels')
        if config.normalization != 'batch':
            raise ValueError('RCDA auxiliary baselines retain batch normalization')
        if input_mode not in ("raw12", "derived17"):
            raise ValueError("RCDA input_mode must be raw12 or derived17")
        super().__init__(config.decoder_widths)
        self.config = config
        self.input_mode = input_mode
        self.input_indices = (tuple(range(12)) if input_mode == "raw12"
                              else tuple(index for index in range(18) if index != 7))
        self.encoders = nn.ModuleList()
        self.Maxpool = nn.MaxPool2d(2)
        channels = len(self.input_indices)
        for width in config.decoder_widths:
            self.encoders.append(conv_block(channels, width))
            channels = width

    def forward(self, x_norm, physics=None):
        if x_norm.ndim != 4 or x_norm.shape[1] != 18 or min(x_norm.shape[-2:]) < 8:
            raise ValueError("RCDABaseline expects transport [B,18,H,W], H/W >= 8")
        if not x_norm.is_floating_point() or not torch.isfinite(x_norm).all():
            raise ValueError("RCDA transport input must be floating point and finite")
        x = x_norm[:, list(self.input_indices)]
        pyramid = []
        for index, encoder in enumerate(self.encoders):
            if index:
                x = self.Maxpool(x)
            x = encoder(x)
            pyramid.append(x)
        return {"logits_det": self.decode(pyramid), "fused_pyramid": pyramid,
                "relational_feature": None, "gate_weights": [],
                "direction_weights": None, "modality_mask": None}
