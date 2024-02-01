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


@dataclass
class FocalLoss:
    alpha: float
    gamma: float
    label_smoothing: float

    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        # https://discuss.pytorch.org/t/focal-loss-for-imbalanced-multi-class-classification-in-pytorch/61289
        ce_loss = F.cross_entropy(
            y,
            y_hat,
            label_smoothing=self.label_smoothing,
            reduction="none",  # keep per-batch-item loss
        )
        pt = torch.exp(-ce_loss)  # log likelihood to likelihood, [0,1]
        return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()  # mean over batch
