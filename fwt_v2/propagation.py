"""Low-resolution wind/terrain-conditioned upstream half-line propagation.

Image coordinates are x right, y down.  For direction e, the operator reads
F(q-r*e); a source pulse therefore influences points downstream along +e.
The explicit physical terms add a nonnegative bias, not a monotonicity guarantee
for the complete learned predictor.

wind_scale and slope_scale are positive feature divisors, not learned spread speeds.
Their unit defaults are transparent development starting points; any alternative
must be fitted/selected on the permitted training/development roles and frozen.
use_wind_prior=False disables only the explicit wind term; terrain and learned
directional propagation remain available.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


DIRECTIONS_XY = ((1, 0), (1, 1), (0, 1), (-1, 1),
                 (-1, 0), (-1, -1), (0, -1), (1, -1))


class DepthwiseHalfLine(nn.Module):
    """Eight depthwise rays in a centered (2R+1)^2 support, excluding center."""

    def __init__(self, channels, radius=3):
        super().__init__()
        if channels < 1 or radius < 1:
            raise ValueError("channels and radius must be positive")
        self.channels = channels
        self.radius = radius
        self.ray_logits = nn.Parameter(torch.zeros(channels, 8, radius))
        masks = torch.zeros(8, radius, radius * 2 + 1, radius * 2 + 1)
        for index, (dx, dy) in enumerate(DIRECTIONS_XY):
            for step in range(1, radius + 1):
                # Conv2d is cross-correlation: negative offsets read upstream.
                masks[index, step - 1, radius - step * dy, radius - step * dx] = 1
        self.register_buffer("ray_masks", masks)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError("half-line input channel count differs from configuration")
        weights = torch.einsum("ckr,krhw->ckhw", self.ray_logits.softmax(dim=-1),
                               self.ray_masks)
        weights = weights.reshape(self.channels * 8, 1, self.radius * 2 + 1,
                                   self.radius * 2 + 1)
        result = F.conv2d(x, weights, padding=self.radius, groups=self.channels)
        return result.reshape(x.shape[0], self.channels, 8, *x.shape[-2:]).permute(0, 2, 1, 3, 4)


class DirectionalPropagation(nn.Module):
    def __init__(self, channels, steps=3, radius=3, use_distance=True,
                 use_physics_prior=True, wind_scale=1.0, slope_scale=1.0, *,
                 use_wind_prior=True):
        super().__init__()
        if (steps < 1 or not math.isfinite(wind_scale) or wind_scale <= 0
                or not math.isfinite(slope_scale) or slope_scale <= 0):
            raise ValueError("steps and physical normalization scales must be finite and positive")
        if not isinstance(use_wind_prior, bool):
            raise ValueError("use_wind_prior must be a boolean")
        self.steps = steps
        self.use_distance = use_distance
        self.use_physics_prior = use_physics_prior
        self.use_wind_prior = use_wind_prior
        self.wind_scale = wind_scale
        self.slope_scale = slope_scale
        self.halfline = DepthwiseHalfLine(channels, radius)
        self.data_scores = nn.Conv2d(channels + 1, 8, 1)
        nn.init.zeros_(self.data_scores.weight)
        nn.init.zeros_(self.data_scores.bias)
        self.channel_mix = nn.Conv2d(channels, channels, 1)
        self.gamma_w = nn.Parameter(torch.tensor(0.0))
        self.gamma_h = nn.Parameter(torch.tensor(0.0))
        self.beta = nn.Parameter(torch.tensor(0.0))
        directions = torch.tensor(DIRECTIONS_XY, dtype=torch.float32)
        directions = directions / directions.norm(dim=1, keepdim=True)
        self.register_buffer("directions_xy", directions)

    def forward(self, feature, distance, wind_xy, slope, upslope_xy):
        if not self.use_distance:
            distance = torch.zeros_like(distance)
        prior = torch.zeros(feature.shape[0], 8, *feature.shape[-2:],
                            device=feature.device, dtype=feature.dtype)
        if self.use_physics_prior:
            uphill = torch.einsum("bchw,kc->bkhw", upslope_xy, self.directions_xy)
            terrain_prior = F.softplus(self.gamma_h) * F.relu(uphill) * slope / self.slope_scale
            if self.use_wind_prior:
                wind = torch.einsum("bchw,kc->bkhw", wind_xy, self.directions_xy)
                prior = F.softplus(self.gamma_w) * F.relu(wind / self.wind_scale) + terrain_prior
            else:
                prior = terrain_prior
        state = feature
        for _ in range(self.steps):
            scores = self.data_scores(torch.cat([state, distance], dim=1)) + prior
            direction_weights = scores.softmax(dim=1)
            messages = self.halfline(state)
            aggregate = (messages * direction_weights[:, :, None]).sum(dim=1)
            state = F.silu(state + self.channel_mix(aggregate))
        return feature + self.beta * (state - feature), direction_weights
