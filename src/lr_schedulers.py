import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from src.utils import get_logger
import src.configs.segmentation as segment_cfg
import src.configs.prithvi_mae_finetune as mae_cfg

logger = get_logger(__name__)


def get_lr_scheduler(config: mae_cfg.Config | segment_cfg.Config, optim: Optimizer) -> LRScheduler | None:
    match config.train.lr_scheduler_type:
        case "step":
            return torch.optim.lr_scheduler.StepLR(
                optim, step_size=config.train.step_lr_sched_step_size, gamma=config.train.step_lr_sched_gamma
            )
        case "cosine_warm_restarts":
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optim, T_0=config.train.cosine_warm_restarts_T_0, eta_min=config.train.cosine_warm_restarts_eta_min
            )
        case _:
            logger.info(f"Not using a learning rate scheduler. (Supplied with: {config.train.lr_scheduler_type})")
            return None
