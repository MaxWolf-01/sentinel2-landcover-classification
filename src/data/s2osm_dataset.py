import random
import typing
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import einops
import numpy.typing as npt
import rasterio
import torch
from torch.utils.data import Dataset

from src.configs.data_config import DataDirs
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class S2OSMDatasetConfig:
    aoi: str  # vie/test/at/...
    label_map: str  # multiclass/binary
    n_time_frames: int = 1  # number of time frames to use for a single prediction
    squeeze_time_dim: bool = False  # if True, output will have shape (c, h, w), else (c, 1, h, w) if n_time_frames == 1


class S2OSMSample(typing.NamedTuple):
    x: torch.Tensor
    y: torch.LongTensor


class S2OSMDataset(Dataset):
    transform: A.Compose | None = None  # to be set in the datamodule

    def __init__(self, cfg: S2OSMDatasetConfig) -> None:
        super().__init__()
        self.n_time_frames: int = cfg.n_time_frames
        self.squeeeze_time_dim: bool = cfg.squeeze_time_dim
        self.data_dirs = DataDirs(aoi=cfg.aoi, map_type=cfg.label_map)
        self.sentinel_files: dict[int, Path] = self.data_dirs.sentinel_files
        self.osm_files: dict[int, Path] = self.data_dirs.osm_files
        assert len(self) > 0, "No data found. Did you run `download_s2_osm_data.py`?"
        logger.info(f"Initialized {self} with {len(self)} samples.")

    def __len__(self) -> int:
        return len(self.sentinel_files)

    def __getitem__(self, idx: int) -> S2OSMSample:
        with rasterio.open(self.sentinel_files[idx]) as f:
            sentinel_data: npt.NDArray = f.read()
        osm_idx = get_mask_file_idx(self.sentinel_files[idx])
        with rasterio.open(self.osm_files[osm_idx]) as f:
            osm_data: npt.NDArray = f.read(1)  # read first band

        if self.transform is not None:
            sentinel_data = einops.rearrange(sentinel_data, "c h w -> h w c")  # albumentations uses chan last
            transformed: dict[str, typing.Any] = self.transform(image=sentinel_data, mask=osm_data)
            sentinel_data = transformed["image"]
            osm_data = transformed["mask"]
            sentinel_data = einops.rearrange(sentinel_data, "h w c -> c h w")

        sentinel_tensor = torch.from_numpy(sentinel_data).float()
        sentinel_tensor = (
            sentinel_tensor.unsqueeze(1) if not self.squeeeze_time_dim and self.n_time_frames == 1 else sentinel_tensor
        )
        osm_tensor = torch.from_numpy(osm_data).long()
        return S2OSMSample(x=sentinel_tensor, y=osm_tensor)

    def compute_class_weights(self, ignore_zero: bool) -> torch.Tensor:
        """Computes class weights inversely proportional to class frequencies in the dataset. Uses random sample if the
        dataset is large.
        Returns:
            class_weights (torch.Tensor): class weights to be used in the loss function, sorted by class index.
        """
        sample_labels = torch.cat([self[i][1] for i in random.sample(range(len(self)), k=min(2500, len(self)))])
        unique, counts = torch.unique(sample_labels, return_counts=True)
        if ignore_zero:
            unique, counts = unique[unique != 0], counts[unique != 0]
        class_weights = counts.sum() / (len(unique) * counts)
        return class_weights


def get_mask_file_idx(sentinel_file: Path) -> int:
    return int(sentinel_file.stem.split("_")[0])
