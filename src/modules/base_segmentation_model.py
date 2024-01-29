import einops
import torch
from torch import nn

from src.modules.prithvi import MaskedAutoencoderViT
from src.utils import load_prithvi


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
        out_channels: int = 256,
        num_convs: int = 1,
        kernel_size: int = 3,
        dropout_ratio: float = 0.1,
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
            nn.Dropout2d(dropout_ratio),
            nn.Conv2d(out_channels, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PrithviSegmentationModel(nn.Module):
    def __init__(
        self,
        neck: nn.Module,
        head: nn.Module,
        num_frames: int = 1,
    ) -> None:
        super().__init__()
        # TODO we don't need to load the entire thing if we only want the encoder!!
        self.backbone: MaskedAutoencoderViT = load_prithvi(num_frames=num_frames)
        self.neck: nn.Module = neck
        self.head: nn.Module = head

        self.head.apply(initialize_head_or_neck_weights)
        self.neck.apply(initialize_head_or_neck_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input shape: (B, T, C, H, W); Output shape: (B, num_classes, H, W)"""
        features, _, _ = self.backbone.forward_encoder(x, mask_ratio=0.0)  # no mae mask | features: (B, tokens, emb_d)

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
        m = PrithviSegmentationModel(
            neck=ConvTransformerTokensToEmbeddingNeck(embed_dim=768, output_embed_dim=256),
            head=FCNHead(num_classes=10, in_channels=256, out_channels=256),
        )
        B, T, C, H, W = 2, 1, 6, 224, 224
        x = torch.randn(B, C, T, H, W)
        y = m(x)
        print(y.shape)

    t()
