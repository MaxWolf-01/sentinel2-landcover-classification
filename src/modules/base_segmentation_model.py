import einops
import torch
from torch import nn

from src.prithvi import MaskedAutoencoderViT
from src.utils import load_prithvi


def _conv_transpos2d_output(
    input_size: int,
    stride: int,
    padding: int,
    dilation: int,
    kernel_size: int,
    output_padding: int,
):
    """Calculate the output size of a ConvTranspose2d.
    Taken from: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
    """
    return (input_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1


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
    todo this could be refactored some more (einops, sequential, more general, ...) & tested
    """

    def __init__(
        self,
        embed_dim: int,
        output_embed_dim: int,
        # num_frames: int = 1,
        Hp: int = 14,
        Wp: int = 14,
        drop_cls_token: bool = True,
    ) -> None:
        """

        Args:
            embed_dim (int): Input embedding dimension
            output_embed_dim (int): Output embedding dimension
            Hp (int, optional): Height (in patches) of embedding to be upscaled. Defaults to 14.
            Wp (int, optional): Width (in patches) of embedding to be upscaled. Defaults to 14.
            drop_cls_token (bool, optional): Whether there is a cls_token, which should be dropped. This assumes the cls token is the first token. Defaults to True.
        """
        super().__init__()
        self.drop_cls_token = drop_cls_token
        self.Hp = Hp
        self.Wp = Wp
        self.H_out = Hp
        self.W_out = Wp
        # self.num_frames = num_frames

        kernel_size = 2
        stride = 2
        dilation = 1
        padding = 0
        output_padding = 0
        for _ in range(4):
            self.H_out = _conv_transpos2d_output(self.H_out, stride, padding, dilation, kernel_size, output_padding)
            self.W_out = _conv_transpos2d_output(self.W_out, stride, padding, dilation, kernel_size, output_padding)

        self.embed_dim = embed_dim
        self.output_embed_dim = output_embed_dim
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(
                self.embed_dim,
                self.output_embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                output_padding=output_padding,
            ),
            Norm2d(self.output_embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(
                self.output_embed_dim,
                self.output_embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                output_padding=output_padding,
            ),
        )
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(
                self.output_embed_dim,
                self.output_embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                output_padding=output_padding,
            ),
            Norm2d(self.output_embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(
                self.output_embed_dim,
                self.output_embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                output_padding=output_padding,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_cls_token:
            x = x[:, 1:, :]
        x = x.permute(0, 2, 1).reshape(x.shape[0], -1, self.Hp, self.Wp)

        x = self.fpn1(x)
        x = self.fpn2(x)

        x = x.reshape((-1, self.output_embed_dim, self.H_out, self.W_out))

        return x


class FCNHead(nn.Module):
    """Fully Convolution Networks for Semantic Segmentation.

    Implementation of `FCNNet <https://arxiv.org/abs/1411.4038>`_.
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
        self.convs = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels if i == 0 else out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
                for i in range(num_convs)
            ],
            nn.Dropout2d(dropout_ratio),
            nn.Conv2d(out_channels, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convs(x)
        return x


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

    def forward(self, x):
        features, _, _ = self.backbone.forward_encoder(x, mask_ratio=0.0)  # no mae mask

        neck_output = self.neck(features)

        output = self.head(neck_output)
        return output


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
