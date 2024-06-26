from __future__ import annotations

import functools
from dataclasses import dataclass

import albumentations as A
import lightning.pytorch as pl
import torch

from data.s2osm_dataset import S2OSMDataset, S2OSMDatasetConfig
from utils import Subset, get_logger, get_sample_weights, train_val_test_split

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
    random_horizontal_flip_p: float
    random_vertical_flip_p: float
    random_crop_size: int = 224

    class_distribution: list[float] | None = None  # set dynamically from dataset for weighted sampling if enabled


class S2OSMDatamodule(pl.LightningDataModule):
    def __init__(self, cfg: S2OSMDatamoduleConfig, masked_loss: bool) -> None:
        super().__init__()
        self.cfg: S2OSMDatamoduleConfig = cfg

        self.batch_size: int = cfg.batch_size
        self.num_workers: int = cfg.num_workers
        self.pin_memory: bool = cfg.pin_memory
        self.augment: bool = cfg.augment
        self.masked_loss: bool = masked_loss

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
        dataset: S2OSMDataset = S2OSMDataset(self.cfg.dataset_cfg)
        self.train, self.test, self.val = train_val_test_split(dataset, self.cfg.data_split, deepcopy=True)

        stats_file_path = dataset.data_dirs.base_path / "mean_std.pt"
        if not stats_file_path.exists():
            raise FileNotFoundError(
                f"Statistics file not found at {stats_file_path} You can create it with the flag: --recompute-mean-std"
            )

        stats = torch.load(stats_file_path)
        mean = stats["mean"]
        std = stats["std"]

        augmentation_transforms = [
            A.RandomCrop(width=self.cfg.random_crop_size, height=self.cfg.random_crop_size, always_apply=True),
            A.HorizontalFlip(p=self.cfg.random_horizontal_flip_p),
            A.VerticalFlip(p=self.cfg.random_vertical_flip_p),
            A.Normalize(mean=mean, std=std),  # Normalize comes last!
        ]
        # necessary transforms
        deterministic_base_transforms = [
            A.CenterCrop(width=self.cfg.random_crop_size, height=self.cfg.random_crop_size, always_apply=True),
            A.Normalize(mean=mean, std=std),  # Normalize comes last!
        ]
        train_transforms = A.Compose(augmentation_transforms if self.augment else deterministic_base_transforms)
        val_test_transforms: A.Compose = A.Compose(deterministic_base_transforms)

        # Avoid data-leakage through augmentation -> aplpy transforms to train, val and test separately
        self.train.dataset.transform = train_transforms
        self.val.dataset.transform = val_test_transforms
        self.test.dataset.transform = val_test_transforms

        logger.info(
            f"Datamodule setup with {len(self.train)} train, {len(self.val)} val and {len(self.test)} test samples."
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        sampler = None
        if self.cfg.class_distribution is not None:
            sample_weights: torch.Tensor = get_sample_weights(
                self.train, class_distribution=self.cfg.class_distribution, ignore_zero_label=self.masked_loss
            )
            sampler = torch.utils.data.WeightedRandomSampler(
                sample_weights, num_samples=len(self.train), replacement=True
            )
        shuffle = True if sampler is None else False
        return self.dataloader_partial(self.train, shuffle=shuffle, sampler=sampler)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return self.dataloader_partial(self.val, batch_size=self.batch_size * self.val_batch_size_multiplier)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return self.dataloader_partial(self.test)
