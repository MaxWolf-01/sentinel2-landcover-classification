# refactored & adapted from: https://github.com/zhoudaxia233/EfficientUnet-PyTorch/tree/master
from __future__ import annotations

import functools
import math
import re
import typing
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import nn

from utils import initialize_classification_layer_bias


@dataclass
class EfficientNetConfig:
    version: typing.Literal["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"]
    in_channels: int
    num_classes: int
    bn_momentum: float = 0.99
    bn_epsilon: float = 1e-3
    depth_divisor: int | None = 8
    drop_connect_rate: float | None = 0.2
    min_depth: int | None = None
    class_distribution: list[float] | None = None  # optional initial probability for weight initialization
    # these are set based on the version (but can be overwritten):
    dropout_rate: float | None = None
    width_coefficient: float | None = None
    depth_coefficient: float | None = None

    def __post_init__(self) -> None:
        params_dict: dict[str, tuple[float, float, int, float]] = {
            # (width_coefficient, depth_coefficient, resolution, dropout_rate)
            # Note: the resolution here is just for reference, its values won't be used.
            "b0": (1.0, 1.0, 224, 0.2),
            "b1": (1.0, 1.1, 240, 0.2),
            "b2": (1.1, 1.2, 260, 0.3),
            "b3": (1.2, 1.4, 300, 0.3),
            "b4": (1.4, 1.8, 380, 0.4),
            "b5": (1.6, 2.2, 456, 0.4),
            "b6": (1.8, 2.6, 528, 0.5),
            "b7": (2.0, 3.1, 600, 0.5),
        }
        if self.version not in params_dict:
            raise ValueError(f"There is no model version {self.version}")
        width_coefficient, depth_coefficient, _, dropout_rate = params_dict[self.version]
        self.width_coefficient = self.width_coefficient or width_coefficient
        self.depth_coefficient = self.depth_coefficient or depth_coefficient
        self.dropout_rate = self.dropout_rate or dropout_rate
        self.bn_momentum = 1 - self.bn_momentum


@dataclass
class BlockConfig:
    kernel_size: int
    num_repeat: int
    input_filters: int
    output_filters: int
    expand_ratio: int
    skip_connection: bool
    stride: tuple[int, int] | int
    se_ratio: float  # todo rename / comment

    def __str__(self) -> str:
        args = [
            "r%d" % self.num_repeat,
            "k%d" % self.kernel_size,
            "s%d%d" % (self.stride[0], self.stride[1]),
            "e%s" % self.expand_ratio,
            "i%d" % self.input_filters,
            "o%d" % self.output_filters,
        ]
        if 0 < self.se_ratio <= 1:
            args.append("se%s" % self.se_ratio)
        if self.skip_connection is False:
            args.append("noskip")
        return "_".join(args)

    @staticmethod
    def from_str(block_string: str) -> BlockConfig:
        assert isinstance(block_string, str)
        ops = block_string.split("_")
        options = {}
        for op in ops:
            splits = re.split(r"(\d.*)", op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value
        if "s" not in options or len(options["s"]) != 2:
            raise ValueError("Strides options should be a pair of integers.")
        return BlockConfig(
            kernel_size=int(options["k"]),
            num_repeat=int(options["r"]),
            input_filters=int(options["i"]),
            output_filters=int(options["o"]),
            expand_ratio=int(options["e"]),
            skip_connection=("noskip" not in block_string),
            se_ratio=float(options["se"]) if "se" in options else None,
            stride=(int(options["s"][0]), int(options["s"][1])),
        )


class EfficientnetUnet(nn.Module):
    def __init__(self, config: EfficientNetConfig, concat_input: bool = True) -> None:
        super().__init__()
        self.encoder: EfficientNet = EfficientNet(config)
        self.up_convs: nn.ModuleList = nn.ModuleList([])
        self.double_convs: nn.ModuleList = nn.ModuleList([])
        in_channels_up: list[int] = [self.n_channels] + [512 // (2**i) for i in range(3)]
        out_channels_up: list[int] = [512 // (2**i) for i in range(4)]
        for in_channel, out_channel, cat_channel_size in zip(in_channels_up, out_channels_up, self.size[:4]):
            self.up_convs.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2))
            self.double_convs.append(_double_conv(cat_channel_size, out_channel))
        self.concat_input: bool = concat_input
        if self.concat_input:
            self.input_up_conv: nn.ConvTranspose2d = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
            self.input_double_conv: nn.Sequential = _double_conv(self.size[4], 32)
        self.out_conv1x1: nn.Conv2d = nn.Conv2d(self.size[5], config.num_classes, kernel_size=1)
        self.apply(init_weights)
        initialize_classification_layer_bias(self.out_conv1x1, class_distribution=config.class_distribution)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        _, encoder_feature_maps = self.encoder.encode(x)
        x = encoder_feature_maps.pop(0)
        for up_conv, double_conv, feature_map in zip(self.up_convs, self.double_convs, encoder_feature_maps):
            x = up_conv(x)
            x = torch.cat([x, feature_map], dim=1)
            x = double_conv(x)
        if self.concat_input:
            x = self.input_up_conv(x)
            x = torch.cat([x, identity], dim=1)
            x = self.input_double_conv(x)
        x = self.out_conv1x1(x)
        return x

    @property
    def n_channels(self) -> int:
        return {
            "b0": 1280,
            "b1": 1280,
            "b2": 1408,
            "b3": 1536,
            "b4": 1792,
            "b5": 2048,
            "b6": 2304,
            "b7": 2560,
        }[self.encoder.name]

    @property
    def size(self) -> tuple[int, int, int, int, int, int]:
        """NOTE: originally, size[4] was 35, but strangely, it only works with 38 _(ツ)_/¯"""
        return {
            "b0": [592, 296, 152, 80, 38, 32],
            "b1": [592, 296, 152, 80, 38, 32],
            "b2": [600, 304, 152, 80, 38, 32],
            "b3": [608, 304, 160, 88, 38, 32],
            "b4": [624, 312, 160, 88, 38, 32],
            "b5": [640, 320, 168, 88, 38, 32],
            "b6": [656, 328, 168, 96, 38, 32],
            "b7": [672, 336, 176, 96, 38, 32],
        }[self.encoder.name]


def _double_conv(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class EfficientNet(nn.Module):
    def __init__(self, config: EfficientNetConfig) -> None:
        super().__init__()
        self.name: str = config.version
        self.drop_connect_rate: float | None = config.drop_connect_rate
        round_filters_ = functools.partial(
            _round_filters,
            width_coefficient=config.width_coefficient,
            depth_divisor=config.depth_divisor,
            min_depth=config.min_depth,
        )
        out_channels: int = round_filters_(filters=32)
        self.stem: nn.Sequential = nn.Sequential(
            Conv2dSamePadding(
                in_channels=config.in_channels, out_channels=out_channels, kernel_size=3, stride=2, bias=False
            ),
            nn.BatchNorm2d(out_channels, momentum=config.bn_momentum, eps=config.bn_epsilon),
            nn.SiLU(),
        )
        self.blocks: nn.ModuleList[MBConvBlock] = nn.ModuleList([])
        block_configs: list[BlockConfig] = [
            BlockConfig.from_str(s)
            for s in [
                "r1_k3_s11_e1_i32_o16_se0.25",
                "r2_k3_s22_e6_i16_o24_se0.25",
                "r2_k5_s22_e6_i24_o40_se0.25",
                "r3_k3_s22_e6_i40_o80_se0.25",
                "r3_k5_s11_e6_i80_o112_se0.25",
                "r4_k5_s22_e6_i112_o192_se0.25",
                "r1_k3_s11_e6_i192_o320_se0.25",
            ]
        ]
        for block_config in block_configs:
            # Update block input and output filters based on depth multiplier.
            block_config.input_filters = round_filters_(filters=block_config.input_filters)
            block_config.output_filters = round_filters_(filters=block_config.output_filters)
            block_config.num_repeat = (
                int(math.ceil(config.depth_coefficient * block_config.num_repeat))
                if config.depth_coefficient is not None
                else block_config.num_repeat
            )
            # The first block needs to take care of stride and filter size increase.
            self.blocks += [MBConvBlock(config=block_config, efficient_net_config=config)]
            if block_config.num_repeat > 1:
                block_config.input_filters = block_config.output_filters
                block_config.stride = 1
            for _ in range(block_config.num_repeat - 1):
                self.blocks += [MBConvBlock(config=block_config, efficient_net_config=config)]

        out_channels = round_filters_(filters=1280)  # todo magic num
        self.conv_head: nn.Sequential = nn.Sequential(
            Conv2dSamePadding(
                in_channels=self.blocks[-1].output_filters,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels, momentum=config.bn_momentum, eps=config.bn_epsilon),
            nn.SiLU(),
        )
        self.fc: nn.Sequential = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange("... 1 1 -> ..."),  # (B, C, 1, 1) -> (B, C)
            nn.Dropout(p=config.dropout_rate),
            nn.Linear(out_channels, config.num_classes),
        )  # todo encode flag to pop?

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.encode(x)
        x = self.fc(x)
        return x

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.stem(x)
        feature_maps: list[torch.Tensor] = []  # stores the feature maps from deepest to shallowest for upsampling path
        drop_connect_rate = None
        for i, block in enumerate(self.blocks):
            if self.drop_connect_rate is not None:
                drop_connect_rate = self.drop_connect_rate * (i / len(self.blocks))
            x = block(x, drop_connect_rate=drop_connect_rate)
            if x.shape[-2:] not in [fm.shape[-2:] for fm in feature_maps] and x.shape[-2:] != (7, 7):
                feature_maps.insert(0, x)
        x = self.conv_head(x)
        feature_maps.insert(0, x)
        return x, feature_maps


def _round_filters(
    filters: int, width_coefficient: float | None, depth_divisor: int | None, min_depth: int | None
) -> int:
    assert min_depth is not None or depth_divisor is not None, "min_depth or depth_divisor should be supplied"
    if width_coefficient is None:
        return filters
    filters *= width_coefficient
    min_depth = min_depth or depth_divisor
    new_filters = max(min_depth, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


class Conv2dSamePadding(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation, groups=groups, bias=bias
        )
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        _, _, input_h, input_w = x.size()
        kernel_h, kernel_w = self.weight.size()[2:]
        stride_h, stride_w = self.stride
        output_h, output_w = math.ceil(input_h / stride_h), math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * self.stride[0] + (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * self.stride[1] + (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block"""

    def __init__(self, config: BlockConfig, efficient_net_config: EfficientNetConfig) -> None:
        super().__init__()
        self.skip_connection: bool = config.skip_connection
        self.stride: int = config.stride
        self.input_filters: int = config.input_filters
        self.output_filters: int = config.output_filters

        out_channels = config.input_filters * config.expand_ratio
        stem_modules: list[nn.Module] = []
        if config.expand_ratio != 1:
            # expansion phase
            stem_modules += [
                Conv2dSamePadding(
                    in_channels=config.input_filters,
                    out_channels=out_channels,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    num_features=out_channels,
                    momentum=efficient_net_config.bn_momentum,
                    eps=efficient_net_config.bn_epsilon,
                ),
                nn.SiLU(),
            ]
        # Depth-wise convolution phase
        stem_modules += [
            Conv2dSamePadding(
                in_channels=out_channels,
                out_channels=out_channels,
                groups=out_channels,
                kernel_size=config.kernel_size,
                stride=config.stride,
                bias=False,
            ),
            nn.BatchNorm2d(
                num_features=out_channels,
                momentum=efficient_net_config.bn_momentum,
                eps=efficient_net_config.bn_epsilon,
            ),
            nn.SiLU(),
        ]
        self.stem: nn.Sequential = nn.Sequential(*stem_modules)
        self.has_squeeze_excitation: bool = (config.se_ratio is not None) and (0 < config.se_ratio <= 1)
        if self.has_squeeze_excitation:
            num_squeezed_channels = max(1, int(config.input_filters * config.se_ratio))
            self.squeeze_excitation: nn.Sequential = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Conv2dSamePadding(
                    in_channels=out_channels,
                    out_channels=num_squeezed_channels,
                    kernel_size=1,
                ),
                nn.SiLU(),
                Conv2dSamePadding(
                    in_channels=num_squeezed_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                ),
            )
        self.final_layer: nn.Sequential = nn.Sequential(
            Conv2dSamePadding(
                in_channels=out_channels,
                out_channels=config.output_filters,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                num_features=config.output_filters,
                momentum=efficient_net_config.bn_momentum,
                eps=efficient_net_config.bn_epsilon,
            ),
        )

    def forward(self, x: torch.Tensor, drop_connect_rate: float | None = None) -> torch.Tensor:
        identity = x
        x = self.stem(x)
        if self.has_squeeze_excitation:
            x = x * torch.sigmoid(self.squeeze_excitation(x))  # "SiLU with prev layer output"
        x = self.final_layer(x)
        if self.skip_connection and self.stride == 1 and self.input_filters == self.output_filters:
            if drop_connect_rate:
                x = _drop_connect(x, drop_connect_rate=drop_connect_rate, training=self.training)
            x = x + identity
        return x


def _drop_connect(inputs: torch.Tensor, drop_connect_rate: float, training: bool) -> torch.Tensor:
    if not training:
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1.0 - drop_connect_rate
    random_tensor = keep_prob + torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


def init_weights(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def _test() -> None:
    imagenet_weights = {
        "b0": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth",
        "b1": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth",
        "b2": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth",
        "b3": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth",
        "b4": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth",
        "b5": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth",
        "b6": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth",
        "b7": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth",
    }
    for version, weights_url in list(imagenet_weights.items()):
        B, C, H, W, n_classes = 2, 6, 224, 224, 4
        config = EfficientNetConfig(version=version, in_channels=C, num_classes=n_classes)  # type: ignore
        model = EfficientNet(config)
        state_dict = torch.hub.load_state_dict_from_url(weights_url)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        x = torch.rand(B, C, H, W)
        y = model(x)
        assert y.shape == (B, n_classes)
        unet = EfficientnetUnet(config)
        y = unet(x)
        assert y.shape == (B, n_classes, H, W)
        print(f"{version} unet passed")


if __name__ == "__main__":
    _test()
