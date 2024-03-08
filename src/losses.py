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


# TODO write binary verisions!
def get_loss(config) -> Loss:
    class_weights = torch.tensor(config.train.class_distribution) if config.train.weighted_loss else None
    assert (
        class_weights is None or len(class_weights) == config.num_classes
    ), f"{len(class_weights)}!={config.num_classes}"
    match config.train.loss_type:
        case LossType.CE:
            return nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=config.train.label_smoothing,
                ignore_index=0 if config.train.masked_loss else -100,
            )
        case LossType.FOCAL:
            return FocalLoss(
                alpha=class_weights if class_weights is not None else torch.tensor([1.0] * config.num_classes),
                gamma=config.train.focal_loss_gamma,
                label_smoothing=config.train.label_smoothing,
                ignore_index=0 if config.train.masked_loss else -100,
            )
        case LossType.DICE:
            return DiceLoss(eps=config.train.dice_eps)
        case LossType.DICE_FOCAL:
            return CombinedLoss(
                l1_weight=config.train.dice_focal_dice_weight,
                l2_weight=config.train.dice_focal_focal_weight,
                l1=DiceLoss(eps=config.train.dice_eps),
                l2=FocalLoss(
                    alpha=class_weights if class_weights is not None else torch.tensor([1.0] * config.num_classes),
                    gamma=config.train.focal_loss_gamma,
                    label_smoothing=config.train.label_smoothing,
                    ignore_index=0 if config.train.masked_loss else -100,
                ),
            )
        case _:
            raise ValueError(f"Unknown loss type: {config.train.loss_type}.\nValid options: {list(LossType)}.")


ReduceType = typing.Literal["mean", "sum"]  # sum can give more weight to less common classes / samples with higher loss


@dataclass
class FocalLoss:
    alpha: torch.Tensor  # (C,)
    gamma: float
    label_smoothing: float
    ignore_index: int = -100
    reduce_type: ReduceType = "mean"

    def __call__(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # https://discuss.pytorch.org/t/focal-loss-for-imbalanced-multi-class-classification-in-pytorch/61289
        ce_loss = F.cross_entropy(
            y_hat,
            y,
            label_smoothing=self.label_smoothing,
            ignore_index=self.ignore_index,
            reduction="none",  # keep per-batch-item loss
        )
        pt = torch.exp(-ce_loss)  # log likelihood to likelihood (probabilities), [0,1]
        alpha = self.alpha.to(y.device).gather(0, y.view(-1)).view(*y.shape)  # map class to alpha
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss
        return _reduce(focal_loss, self.reduce_type)


@dataclass
class DiceLoss:  # todo this also exists as a torchmetric
    # https://arxiv.org/pdf/1606.04797.pdf
    eps: float = 1e-8
    ignore_index: int = -100
    reduce_type: ReduceType = "mean"

    def __call__(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        B, num_classes, H, W = y_hat.shape
        mask = (y_hat != self.ignore_index).long().unsqueeze(1)  # type: ignore
        input_soft = F.softmax(y_hat, dim=1) * mask
        target_one_hot = F.one_hot(y, num_classes=num_classes) * mask

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
