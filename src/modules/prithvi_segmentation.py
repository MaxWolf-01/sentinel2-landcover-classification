from dataclasses import dataclass

import einops
import torch
from torch import nn

from modules.prithvi import MaskedAutoencoderViT
from utils import load_prithvi


class Norm2d(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        x = einops.rearrange(x, "b c h w -> b h w c")
        x = self.ln(x)
        x = einops.rearrange(x, "b h w c -> b c h w")
        return x


class ConvTransformerTokensToEmbeddingNeck(nn.Module):
    """Neck that transforms the token-based output of transformer into a single embedding suitable for processing with
    standard layers. Performs 4 ConvTranspose2d operations on the rearranged input with kernel_size=2 and stride=2
    """

    def __init__(
        self,
        embed_dim: int,
        output_embed_dim: int,
        # num_frames: int = 1,
        patch_height: int = 14,
        patch_width: int = 14,
        drop_cls_token: bool = True,
    ) -> None:
        """

        Args:
            embed_dim (int): Input embedding dimension
            output_embed_dim (int): Output embedding dimension
            patch_height (int, optional): Height (in patches) of embedding to be upscaled. Defaults to 14.
            patch_width (int, optional): Width (in patches) of embedding to be upscaled. Defaults to 14.
            drop_cls_token (bool, optional): Whether there is a cls_token, which should be dropped. This assumes the cls token is the first token. Defaults to True.
        """
        super().__init__()
        self.drop_cls_token = drop_cls_token
        self.patch_height = patch_height
        self.patch_width = patch_width
        # self.num_frames = num_frames

        conv_t2d = lambda inp, out: nn.ConvTranspose2d(  # noqa: E731
            inp, out, kernel_size=2, stride=2, dilation=1, padding=0, output_padding=0
        )

        self.feature_pyramid_net = nn.Sequential(
            conv_t2d(embed_dim, output_embed_dim),
            Norm2d(output_embed_dim),
            nn.GELU(),
            conv_t2d(output_embed_dim, output_embed_dim),  # fixme: is no norm & act correct?
            conv_t2d(output_embed_dim, output_embed_dim),
            Norm2d(output_embed_dim),
            nn.GELU(),
            conv_t2d(output_embed_dim, output_embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_cls_token:
            x = x[:, 1:, :]
        x = einops.rearrange(x, "b (h w) emb -> b emb h w", h=self.patch_height, w=self.patch_width)
        x = self.feature_pyramid_net(x)
        return x


class FCNHead(nn.Module):
    """Fully Convolution Networks for Semantic Segmentation.

    Implementation of `FCNNet <https://arxiv.org/abs/1411.4038>`.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        out_channels: int,
        num_convs: int,
        dropout: float,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            *[
                layer
                for i in range(num_convs)
                for layer in [
                    nn.Conv2d(
                        in_channels=in_channels if i == 0 else out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            ],
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class PrithviSegmentationNetConfig:
    num_frames: int  # input frames per prediction
    num_classes: int
    fcn_out_channels: int
    fcn_num_convs: int
    fcn_dropout: float
    frozen_backbone: bool

    embed_dim: int = 768  # fixed prithvi output embedding dim
    output_embed_dim: int = -1  # to be set in __post_init__
    patch_height: int = 14
    patch_width: int = 14

    def __post_init__(self) -> None:
        self.output_embed_dim = self.embed_dim * self.num_frames


class PrithviSegmentationNet(nn.Module):
    def __init__(self, config: PrithviSegmentationNetConfig) -> None:
        super().__init__()
        self.backbone: MaskedAutoencoderViT = load_prithvi(num_frames=config.num_frames)
        self.neck: nn.Module = ConvTransformerTokensToEmbeddingNeck(
            embed_dim=config.embed_dim * config.num_frames,
            output_embed_dim=config.output_embed_dim,
            patch_height=config.patch_height,
            patch_width=config.patch_width,
        )
        self.head: nn.Module = FCNHead(
            num_classes=config.num_classes,
            in_channels=config.output_embed_dim,
            out_channels=config.fcn_out_channels,
            num_convs=config.fcn_num_convs,
            dropout=config.fcn_dropout,
        )
        self.head.apply(initialize_head_or_neck_weights)
        self.neck.apply(initialize_head_or_neck_weights)

        if config.frozen_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input shape: (B, T, C, H, W); Output shape: (B, num_classes, H, W)"""
        # no mae mask | features: (B, tokens, emb_d)
        features, _, _ = self.backbone.forward_encoder(x, mask_ratio=0.0)
        neck_output = self.neck(features)
        output = self.head(neck_output)
        return output


def initialize_head_or_neck_weights(m: nn.Module) -> None:
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


if __name__ == "__main__":

    def t() -> None:
        m = PrithviSegmentationNet(
            PrithviSegmentationNetConfig(
                num_frames=1,
                output_embed_dim=256,
                fcn_out_channels=256,
                fcn_num_convs=1,
                fcn_dropout=0.1,
                frozen_backbone=True,
                num_classes=2,
            )
        )
        B, T, C, H, W = 2, 1, 6, 224, 224
        x = torch.randn(B, C, T, H, W)
        y = m(x)
        print(y.shape)
        m = load_prithvi(1, no_decoder=False)
        loss, pred, mask = m.forward(x)
        print(loss.shape, pred.shape, mask.shape)

    t()
