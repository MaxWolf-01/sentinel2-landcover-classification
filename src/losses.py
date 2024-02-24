import enum
import typing
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

Loss = typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class LossType(str, enum.Enum):
    CE = "ce"
    FOCAL = "focal"
    DICE = "dice"
    DICE_FOCAL = "dice_focal"


def get_loss(config) -> Loss:
    match config.train.loss_type:
        case LossType.CE:
            return nn.CrossEntropyLoss(label_smoothing=config.train.label_smoothing)
        case LossType.FOCAL:
            return FocalLoss(
                alpha=config.train.focal_loss_alpha,
                gamma=config.train.focal_loss_gamma,
                label_smoothing=config.train.label_smoothing,
            )
        case LossType.DICE:
            return DiceLoss(eps=config.train.dice_eps)
        case LossType.DICE_FOCAL:
            return CombinedLoss(
                l1_weight=config.train.dice_focal_dice_weight,
                l2_weight=config.train.dice_focal_focal_weight,
                l1=DiceLoss(eps=config.train.dice_eps),
                l2=FocalLoss(
                    alpha=config.train.focal_loss_alpha,
                    gamma=config.train.focal_loss_gamma,
                    label_smoothing=config.train.label_smoothing,
                ),
            )
        case _:
            raise ValueError(f"Unknown loss type: {config.train.loss_type}.\nValid options: {list(LossType)}.")


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
    # store losses for logging
    l1_value: float | None = None
    l2_value: float | None = None

    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        """Returns the weighted sum of two losses and updates the stored losses for logging."""
        part1 = self.l1_weight * self.l1(y, y_hat)
        part2 = self.l2_weight * self.l2(y, y_hat)
        self.l1_value, self.l2_value = part1.item(), part2.item()
        return part1 + part2


def _reduce(loss: torch.Tensor, reduction: ReduceType = "mean") -> torch.Tensor:
    match reduction:
        case "mean":
            return loss.mean()
        case "sum":
            return loss.sum()
        case _:
            raise ValueError(f"Invalid reduction: {reduction}.")
