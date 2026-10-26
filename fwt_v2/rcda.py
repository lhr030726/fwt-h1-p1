"""Local RCDA decoder components; no import from the legacy experiment.

Adapted from E:/xiaolunwen/FWT-Net-main/Models/RCDA.py, SHA256
dbb5f77f0e22bcdcc6ceec998f5e184e876a142952b181734994b58156c5e9ff.
The module/parameter names and default decoder operations are preserved for
audited partial migration.  The decoder accepts an explicit feature pyramid,
returns logits, handles odd image sizes, and permits tiny widths for CPU tests.
Encoder/input semantics have changed and are NOT implied to be compatible.
"""
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class AGCA(nn.Module):
    def __init__(self, in_channel, ratio=16):
        super().__init__()
        hidden = max(1, in_channel // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channel, hidden, 1, bias=False)
        self.softmax = nn.Softmax(2)
        self.A0 = nn.Parameter(torch.eye(hidden))
        self.A2 = nn.Parameter(torch.full((hidden, hidden), 1e-6))
        self.conv2 = nn.Conv1d(1, 1, 1, bias=False)
        self.conv3 = nn.Conv1d(1, 1, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(hidden, in_channel, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv1(self.avg_pool(x))
        batch, channels, _, _ = y.shape
        y = y.flatten(2).transpose(1, 2)
        attention = self.softmax(self.conv2(y)).expand(batch, channels, channels)
        y = torch.matmul(y, self.A0 * attention + self.A2)
        y = self.relu(self.conv3(y)).transpose(1, 2).reshape(batch, channels, 1, 1)
        return x * self.sigmoid(self.conv4(y))


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, groups=1, norm_cfg=None, act_cfg=None):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                            padding, groups=groups, bias=norm_cfg is None)]
        if norm_cfg:
            layers.append(nn.BatchNorm2d(out_channels,
                                         momentum=norm_cfg.get("momentum", 0.1),
                                         eps=norm_cfg.get("eps", 1e-5)))
        if act_cfg:
            activation = nn.SiLU if act_cfg["type"] == "SiLU" else nn.ReLU
            layers.append(activation(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CAA(nn.Module):
    def __init__(self, channels, h_kernel_size=11, v_kernel_size=11):
        super().__init__()
        norm = {"type": "BN", "momentum": 0.03, "eps": 0.001}
        act = {"type": "SiLU"}
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.max_pool = nn.MaxPool2d(7, 1, 3)
        self.conv1 = ConvModule(channels * 2, channels, 1, norm_cfg=norm, act_cfg=act)
        self.h_conv = ConvModule(channels, channels, (1, h_kernel_size),
                                 padding=(0, h_kernel_size // 2), groups=channels)
        self.v_conv = ConvModule(channels, channels, (v_kernel_size, 1),
                                 padding=(v_kernel_size // 2, 0), groups=channels)
        self.conv2 = ConvModule(channels, channels, 1, norm_cfg=norm, act_cfg=act)
        self.act = nn.Sigmoid()

    def forward(self, x):
        pooled = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)
        return self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(pooled)))))


class DualAttention(nn.Module):
    def __init__(self, ch, ratio=16):
        super().__init__()
        self.W_g = nn.Conv2d(ch, ch, 1)
        self.agca = AGCA(ch, ratio)
        self.caa = CAA(ch)

    def forward(self, g, x):
        return x * self.caa(self.agca(self.W_g(g) + x)) + x


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, padding=1), nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True), nn.Conv2d(ch_out, ch_out, 3, padding=1),
            nn.BatchNorm2d(ch_out), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(ch_in, ch_out, 3, padding=1), nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, size):
        # Explicit size makes skip fusion valid for odd source dimensions.
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=True)
        return self.up[1:](x)


class RCDADecoder(nn.Module):
    """Decoder whose default keys retain their old RCDA decoder meaning."""

    def __init__(self, widths: Sequence[int]):
        super().__init__()
        self.upsamples = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for index in range(len(widths) - 1, 0, -1):
            higher, lower = widths[index], widths[index - 1]
            self.upsamples.append(up_conv(higher, lower))
            self.attentions.append(DualAttention(lower))
            self.decoders.append(conv_block(lower * 2, lower))
        self.Conv_1x1 = nn.Conv2d(widths[0], 1, 1)

    def decode(self, pyramid, *, return_decoder_features=False):
        x = pyramid[-1]
        for index, up in enumerate(self.upsamples):
            skip = pyramid[-2 - index]
            x_up = up(x, skip.shape[-2:])
            attended = self.attentions[index](x_up, skip)
            x = self.decoders[index](torch.cat([attended, x_up], dim=1))
        logits = self.Conv_1x1(x)
        return (logits, x) if return_decoder_features else logits
