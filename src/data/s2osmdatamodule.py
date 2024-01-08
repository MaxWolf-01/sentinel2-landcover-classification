from __future__ import annotations

import copy
import functools
import random
from dataclasses import dataclass

import lightning.pytorch as pl
import torch
from torch.utils.data import Subset

from src.data.s2osmdataset import S2OSMDataset, S2OsmDatasetConfig
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class S2OSMDatamoduleConfig:
    dataset_cfg: S2OsmDatasetConfig
    batch_size: int
    num_workers: int
    pin_memory: bool
    use_transforms: bool
    data_split: tuple[float, float, float]
    # Increase the batch size for validation by this factor. Possible if no_grad is used in validation step.
    val_batch_size_multiplier: int


class S2OSMDatamodule(pl.LightningDataModule):
    def __init__(self, cfg: S2OSMDatamoduleConfig) -> None:
        super().__init__()
        self.dataset_cfg: S2OsmDatasetConfig = cfg.dataset_cfg

        self.batch_size: int = cfg.batch_size
        self.num_workers: int = cfg.num_workers
        self.pin_memory: bool = cfg.pin_memory
        self.use_transforms: bool = cfg.use_transforms

        self.data_split: tuple[float, float, float] = cfg.data_split
        assert sum(self.data_split) == 1.0, "Data split must sum to 1.0"

        self.train: Subset[S2OSMDataset] | None = None
        self.val: Subset[S2OSMDataset] | None = None
        self.test: Subset[S2OSMDataset] | None = None

        self.val_batch_size_multiplier: int = cfg.val_batch_size_multiplier
        self.dataloader_partial = functools.partial(
            torch.utils.data.DataLoader,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )

    def setup(self, stage: str | None = None) -> None:
        dataset: S2OSMDataset = S2OSMDataset(self.dataset_cfg)

        train_len = int(self.data_split[0] * len(dataset))
        val_len = int(self.data_split[1] * len(dataset))
        test_len = len(dataset) - train_len - val_len if self.data_split[2] > 0 else 0

        all_indicies = list(range(len(dataset)))
        random.shuffle(all_indicies)
        train_indices = all_indicies[:train_len]
        val_indices = all_indicies[train_len : train_len + val_len]
        test_indices = all_indicies[train_len + val_len :]

        self.train = copy.deepcopy(dataset)
        self.train.indices = train_indices
        self.val = copy.deepcopy(dataset)
        self.val.indices = val_indices
        self.test = copy.deepcopy(dataset)
        self.test.indices = test_indices

        logger.info(f"Datamodule setup with {train_len} train, {val_len} val and {test_len} test samples.")

        if not self.use_transforms:
            return

        logger.info("Configuring data augmentations ...")

        train_transforms = []
        val_test_transforms = []

        # Avoid data-leakage through augmentation -> aplpy transforms to train, val and test separately
        self.train.transform = train_transforms
        self.val.transform = val_test_transforms
        self.test.transform = val_test_transforms

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.dataloader_partial(self.train, shuffle=True)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return self.dataloader_partial(self.val, batch_size=self.batch_size * self.val_batch_size_multiplier)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return self.dataloader_partial(self.test)
