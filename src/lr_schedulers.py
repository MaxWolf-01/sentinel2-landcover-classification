import enum

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from utils import get_logger

logger = get_logger(__name__)


class LRSchedulerType(str, enum.Enum):
    STEP = "step"
    COSINE_WARM_RESTARTS = "cosine_warm_restarts"


def get_lr_scheduler(config, optim: Optimizer) -> LRScheduler | None:
    match config.train.lr_scheduler_type:
        case LRSchedulerType.STEP:
            return torch.optim.lr_scheduler.StepLR(
                optim, step_size=config.train.step_lr_sched_step_size, gamma=config.train.step_lr_sched_gamma
            )
        case LRSchedulerType.COSINE_WARM_RESTARTS:
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optim, T_0=config.train.cosine_warm_restarts_T_0, eta_min=config.train.cosine_warm_restarts_eta_min
            )
        case _:
            logger.info(f"Not using a learning rate scheduler. (Supplied with: {config.train.lr_scheduler_type})")
            return None
