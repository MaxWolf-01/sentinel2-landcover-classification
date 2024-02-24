import typing
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import einops
import numpy.typing as npt
import rasterio
import torch
from torch.utils.data import Dataset

from data.download_data import S2OSMDataDirs
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class S2OSMDatasetConfig:
    aoi: str  # vie/test/at/...
    label_map: str  # multiclass/binary


class S2OSMSample(typing.NamedTuple):
    x: torch.Tensor
    y: torch.LongTensor


class S2OSMDataset(Dataset):
    transform: A.Compose | None = None  # to be set in the datamodule

    def __init__(self, cfg: S2OSMDatasetConfig) -> None:
        super().__init__()
        self.data_dirs = S2OSMDataDirs(aoi=cfg.aoi, map_type=cfg.label_map)
        self.sentinel_files = self.data_dirs.sentinel_files(sort=True)  # sort would not need to be set
        self.osm_files = self.data_dirs.osm_files(sort=True)  # sort needs to be set
        assert len(self) > 0, "No data found. Did you run `download_data.py`?"
        logger.info(f"Initialized {self} with {len(self)} samples.")

    def __len__(self) -> int:
        return len(self.sentinel_files)

    def __getitem__(self, idx: int) -> S2OSMSample:
        with rasterio.open(self.sentinel_files[idx]) as f:
            sentinel_data: npt.NDArray = f.read()
        osm_idx = get_mask_file_idx(self.sentinel_files[idx])
        with rasterio.open(self.osm_files[osm_idx]) as f:
            osm_data: npt.NDArray = f.read(1)  # read first band
        print(self.sentinel_files[idx], self.osm_files[osm_idx])

        if self.transform is not None:
            sentinel_data = einops.rearrange(sentinel_data, "c h w -> h w c")  # albumentations uses chan last
            transformed: dict[str, typing.Any] = self.transform(image=sentinel_data, mask=osm_data)
            sentinel_data = transformed["image"]
            osm_data = transformed["mask"]
            sentinel_data = einops.rearrange(sentinel_data, "h w c -> c h w")

        sentinel_tensor = torch.from_numpy(sentinel_data).float().unsqueeze(1)  # add time dim (1, for now)
        osm_tensor = torch.from_numpy(osm_data).long()

        return S2OSMSample(x=sentinel_tensor, y=osm_tensor)


def get_mask_file_idx(sentinel_file: Path) -> int:
    return int(sentinel_file.stem.split("_")[0])
