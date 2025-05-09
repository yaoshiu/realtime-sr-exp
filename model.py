# Copyright 2020 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import math

import torch
import torchvision
from math import sqrt
from torch import nn, Tensor

from modules import *


class ESPCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: int,
        upscale_factor: int,
    ) -> None:
        super(ESPCN, self).__init__()
        hidden_channels = channels // 2
        out_channels = int(out_channels * (upscale_factor**2))

        # Feature mapping
        self.feature_maps = nn.Sequential(
            nn.Conv2d(in_channels, channels, (5, 5), (1, 1), (2, 2)),
            nn.Tanh(),
            nn.Conv2d(channels, hidden_channels, (3, 3), (1, 1), (1, 1)),
            nn.Tanh(),
        )

        # Sub-pixel convolution layer
        self.sub_pixel = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(upscale_factor),
        )

        # Initial model weights
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if module.in_channels == 32:
                    nn.init.normal_(module.weight.data, 0.0, 0.001)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias.data)
                else:
                    nn.init.normal_(
                        module.weight.data,
                        0.0,
                        math.sqrt(
                            2 / (module.out_channels * module.weight.data[0][0].numel())
                        ),
                    )
                    if module.bias is not None:
                        nn.init.zeros_(module.bias.data)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support torch.script function.
    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.feature_maps(x)
        x = self.sub_pixel(x)

        x = torch.clamp_(x, 0.0, 1.0)

        return x


def espcn_x2(**kwargs) -> ESPCN:
    model = ESPCN(upscale_factor=2, **kwargs)

    return model


def espcn_x3(**kwargs) -> ESPCN:
    model = ESPCN(upscale_factor=3, **kwargs)

    return model


def espcn_x4(**kwargs) -> ESPCN:
    model = ESPCN(upscale_factor=4, **kwargs)

    return model


def espcn_x8(**kwargs) -> ESPCN:
    model = ESPCN(upscale_factor=8, **kwargs)

    return model


class FSRCNN(nn.Module):
    """

    Args:
        upscale_factor (int): Image magnification factor.
    """

    def __init__(self, upscale_factor: int) -> None:
        super(FSRCNN, self).__init__()
        # Feature extraction layer.
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(1, 56, (5, 5), (1, 1), (2, 2)), nn.PReLU(56)
        )

        # Shrinking layer.
        self.shrink = nn.Sequential(
            nn.Conv2d(56, 12, (1, 1), (1, 1), (0, 0)), nn.PReLU(12)
        )

        # Mapping layer.
        self.map = nn.Sequential(
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12),
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12),
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12),
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12),
        )

        # Expanding layer.
        self.expand = nn.Sequential(
            nn.Conv2d(12, 56, (1, 1), (1, 1), (0, 0)), nn.PReLU(56)
        )

        # Deconvolution layer.
        self.deconv = nn.ConvTranspose2d(
            56,
            1,
            (9, 9),
            (upscale_factor, upscale_factor),
            (4, 4),
            (upscale_factor - 1, upscale_factor - 1),
        )

        # Initialize model weights.
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # Support torch.script function.
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out = self.feature_extraction(x)
        out = self.shrink(out)
        out = self.map(out)
        out = self.expand(out)
        out = self.deconv(out)

        return out

    # The filter weight of each layer is a Gaussian distribution with zero mean and standard deviation initialized by random extraction 0.001 (deviation is 0).
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(
                    m.weight.data,
                    mean=0.0,
                    std=sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())),
                )
        nn.init.normal_(self.deconv.weight.data, mean=0.0, std=0.001)
        if self.deconv.bias is not None:
            nn.init.zeros_(self.deconv.bias.data)
        # This line is redundant as we already initialized weights above
        # nn.init.normal_(self.deconv.weight.data, mean=0.0, std=0.001)
        # Check if bias exists before initializing
        if self.deconv.bias is not None:
            nn.init.zeros_(self.deconv.bias.data)


def fsrcnn_x2(**kwargs) -> FSRCNN:
    model = FSRCNN(upscale_factor=2)

    return model


class RT4KSR_Rep(nn.Module):
    def __init__(
        self,
        num_channels,
        num_feats,
        num_blocks,
        upscale,
        act,
        eca_gamma,
        is_train,
        forget,
        layernorm,
        residual,
    ) -> None:
        super().__init__()
        self.forget = forget
        self.gamma = nn.Parameter(torch.zeros(1))
        self.gaussian = torchvision.transforms.GaussianBlur(kernel_size=5, sigma=1)

        self.down = nn.PixelUnshuffle(2)
        self.up = nn.PixelShuffle(2)
        self.head = nn.Sequential(
            nn.Conv2d(num_channels * (2**2), num_feats, 3, padding=1)
        )

        hfb = []
        if is_train:
            hfb.append(ResBlock(num_feats, ratio=2))
        else:
            hfb.append((RepResBlock(num_feats)))
        hfb.append(act)
        self.hfb = nn.Sequential(*hfb)

        body = []
        for i in range(num_blocks):
            if is_train:
                body.append(
                    SimplifiedNAFBlock(
                        in_c=num_feats,
                        act=act,
                        exp=2,
                        eca_gamma=eca_gamma,
                        layernorm=layernorm,
                        residual=residual,
                    )
                )
            else:
                body.append(
                    SimplifiedRepNAFBlock(
                        in_c=num_feats,
                        act=act,
                        exp=2,
                        eca_gamma=eca_gamma,
                        layernorm=layernorm,
                        residual=residual,
                    )
                )

        self.body = nn.Sequential(*body)

        tail: list[nn.Module] = [LayerNorm2d(num_feats)]
        if is_train:
            tail.append(ResBlock(num_feats, ratio=2))
        else:
            tail.append(RepResBlock(num_feats))
        self.tail = nn.Sequential(*tail)

        self.upsample = nn.Sequential(
            nn.Conv2d(num_feats, num_channels * ((2 * upscale) ** 2), 3, padding=1),
            nn.PixelShuffle(upscale * 2),
        )

    def forward(self, x):
        # stage 1
        hf = x - self.gaussian(x)

        # unshuffle to save computation
        x_unsh = self.down(x)
        hf_unsh = self.down(hf)

        shallow_feats_hf = self.head(hf_unsh)
        shallow_feats_lr = self.head(x_unsh)

        # stage 2
        deep_feats = self.body(shallow_feats_lr)
        hf_feats = self.hfb(shallow_feats_hf)

        # stage 3
        if self.forget:
            deep_feats = self.tail(self.gamma * deep_feats + hf_feats)
        else:
            deep_feats = self.tail(deep_feats)

        out = self.upsample(deep_feats)
        return out


def rt4ksr_x2(**kwargs):
    act = activation("gelu")
    model = RT4KSR_Rep(
            num_channels=1,
            num_feats=24,
            num_blocks=4,
            upscale=2,
            act=act,
            eca_gamma=0,
            forget=False,
            is_train=True,
            layernorm=True,
            residual=False,
        )
    return model
