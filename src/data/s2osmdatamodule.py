from __future__ import annotations

import copy
import functools
import random
from dataclasses import dataclass

import lightning.pytorch as pl
import torch
import albumentations as A

from src.data.s2osmdataset import S2OSMDataset, S2OSMDatasetConfig
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class S2OSMDatamoduleConfig:
    dataset_cfg: S2OSMDatasetConfig
    batch_size: int
    num_workers: int
    pin_memory: bool
    augment: bool
    data_split: tuple[float, float, float]
    # Increase the batch size for validation by this factor. Possible if no_grad is used in validation step.
    val_batch_size_multiplier: int

    # transform params
    random_crop_size: int

    # dataset_statistics
    mean: torch.Tensor = torch.tensor(
        [799.2815, 992.0511, 844.3999, 3567.4006, 2048.2227, 1259.0723], dtype=torch.float32
    )
    std: torch.Tensor = torch.tensor(
        [1000.5408, 928.0647, 979.9531, 1121.2145, 970.6279, 852.6375], dtype=torch.float32
    )


class S2OSMDatamodule(pl.LightningDataModule):
    def __init__(self, cfg: S2OSMDatamoduleConfig) -> None:
        super().__init__()
        self.cfg: S2OSMDatamoduleConfig = cfg

        self.batch_size: int = cfg.batch_size
        self.num_workers: int = cfg.num_workers
        self.pin_memory: bool = cfg.pin_memory
        self.augment: bool = cfg.augment

        self.data_split: tuple[float, float, float] = cfg.data_split
        assert sum(self.data_split) == 1.0, "Data split must sum to 1.0"

        self.train: S2OSMDataset | None = None
        self.val: S2OSMDataset | None = None
        self.test: S2OSMDataset | None = None

        self.val_batch_size_multiplier: int = cfg.val_batch_size_multiplier
        self.dataloader_partial = functools.partial(
            torch.utils.data.DataLoader,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )

        self.mean = cfg.mean
        self.std = cfg.std

    def setup(self, stage: str | None = None) -> None:
        dataset: S2OSMDataset = S2OSMDataset(self.cfg.dataset_cfg)

        train_len: int = int(self.data_split[0] * len(dataset))
        val_len: int = int(self.data_split[1] * len(dataset))
        test_len: int = len(dataset) - train_len - val_len if self.data_split[2] > 0 else 0

        all_indicies: list[int] = list(range(len(dataset)))
        random.shuffle(all_indicies)
        train_indices: list[int] = all_indicies[:train_len]
        val_indices: list[int] = all_indicies[train_len : train_len + val_len]
        test_indices: list[int] = all_indicies[train_len + val_len :]

        self.train = copy.deepcopy(dataset)
        self.train.indices = train_indices
        self.val = copy.deepcopy(dataset)
        self.val.indices = val_indices
        self.test = copy.deepcopy(dataset)
        self.test.indices = test_indices

        random_transforms_and_augments = [
            A.RandomCrop(width=self.cfg.random_crop_size, height=self.cfg.random_crop_size, always_apply=True),
            # todo add transforms after evaluation pipeline is set up
            # A.HorizontalFlip(p=self.cfg.random_horizontal_flip_p),
            # A.VerticalFlip(p=self.cfg.random_vertical_flip_p),
            A.Normalize(mean=self.cfg.mean, std=self.cfg.std),  # Normalize comes last!
        ]
        # necessary transforms
        deterministic_base_transforms = [
            A.CenterCrop(width=self.cfg.random_crop_size, height=self.cfg.random_crop_size, always_apply=True),
            A.Normalize(mean=self.mean, std=self.std),  # Normalize comes last!
        ]
        train_transforms = A.Compose(deterministic_base_transforms if self.augment else random_transforms_and_augments)
        val_test_transforms: A.Compose = A.Compose(deterministic_base_transforms)

        # Avoid data-leakage through augmentation -> aplpy transforms to train, val and test separately
        self.train.transform = train_transforms
        self.val.transform = val_test_transforms
        self.test.transform = val_test_transforms

        logger.info(f"Datamodule setup with {train_len} train, {val_len} val and {test_len} test samples.")

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.dataloader_partial(self.train, shuffle=True)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return self.dataloader_partial(self.val, batch_size=self.batch_size * self.val_batch_size_multiplier)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return self.dataloader_partial(self.test)
