import typing
from dataclasses import dataclass

import torch
from torch import nn

from configs.simple_finetune import Config
import torch.nn.functional as F

Loss = typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def get_loss(config: Config) -> Loss:
    match config.train.loss_type:
        case "ce":
            return nn.CrossEntropyLoss(label_smoothing=config.train.label_smoothing)
        case "focal":
            return FocalLoss(
                alpha=config.train.focal_loss_alpha,
                gamma=config.train.focal_loss_gamma,
                label_smoothing=config.train.label_smoothing,
            )


ReduceType = typing.Literal["mean", "sum"]  # sum can give more weight to less common classes / samples with higher loss


@dataclass
class FocalLoss:
    alpha: float
    gamma: float
    label_smoothing: float
    reduce_type: ReduceType = "mean"

    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        # https://discuss.pytorch.org/t/focal-loss-for-imbalanced-multi-class-classification-in-pytorch/61289
        ce_loss = F.cross_entropy(
            y,
            y_hat,
            label_smoothing=self.label_smoothing,
            reduction="none",  # keep per-batch-item loss
        )
        pt = torch.exp(-ce_loss)  # log likelihood to likelihood, [0,1]
        return _reduce((self.alpha * (1 - pt) ** self.gamma * ce_loss), self.reduce_type)


@dataclass
class DiceLoss:  # todo this also exists as a torchmetric
    # https://arxiv.org/pdf/1606.04797.pdf
    eps: float = 1e-8
    reduce_type: ReduceType = "mean"

    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        B, num_classes, H, W = y.shape  # assert shape is correct
        input_soft = F.softmax(y, dim=1)
        target_one_hot = F.one_hot(y, num_classes=y.shape[1])

        dims = (1, 2, 3)
        intersection = (input_soft * target_one_hot).sum(dims)
        union = (input_soft + target_one_hot).sum(dims)

        dice_coefficient = (2.0 * intersection + self.eps) / (union + self.eps)
        return _reduce((1.0 - dice_coefficient), self.reduce_type)


@dataclass
class CombinedLoss:
    l1_weight: float
    l2_weight: float
    l1: Loss
    l2: Loss

    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Returns the weighted sum of the two losses and the individual losses, for logging purposes."""
        part1 = self.l1_weight * self.l1(y, y_hat)
        part2 = self.l2_weight * self.l2(y, y_hat)
        return (part1 + part2), (part1, part2)


def _reduce(loss: torch.Tensor, reduction: ReduceType = "mean") -> torch.Tensor:
    match reduction:
        case "mean":
            return loss.mean()
        case "sum":
            return loss.sum()
        case _:
            raise ValueError(f"Invalid reduction: {reduction}.")
