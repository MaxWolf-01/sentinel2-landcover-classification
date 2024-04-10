import typing
from dataclasses import dataclass
from pathlib import Path

import einops
import rasterio
import torch
from torch.utils.data import Dataset

from src.data.download_mae_data import get_mae_data_dir
from src.utils import get_logger
import albumentations as A
import numpy.typing as npt

logger = get_logger(__name__)


@dataclass
class MAEDatasetConfig:
    aoi: str  # vie/test/at/...


class MAESample(typing.NamedTuple):
    x: torch.Tensor


class MAEDataset(Dataset):
    transform: A.Compose | None = None  # to be set in the datamodule

    def __init__(self, cfg: MAEDatasetConfig) -> None:
        super().__init__()
        self.data_dir: Path = get_mae_data_dir(aoi=cfg.aoi)
        self.sentinel_files = list(self.data_dir.glob("*.tif"))
        assert len(self) > 0, "No data found. Did you run `download_mae_data.py`?"

        logger.info(f"Initialized {self} with {len(self)} samples.")

    def __len__(self) -> int:
        return len(self.sentinel_files)

    def __getitem__(self, idx: int) -> MAESample:
        with rasterio.open(self.sentinel_files[idx]) as f:
            sentinel_data: npt.NDArray = f.read()

        if self.transform is not None:
            sentinel_data = einops.rearrange(sentinel_data, "c h w -> h w c")  # albumentations uses chan last
            transformed: dict[str, typing.Any] = self.transform(image=sentinel_data)
            sentinel_data = transformed["image"]
            sentinel_data = einops.rearrange(sentinel_data, "h w c -> c h w")

        # TODO add (random? or fixed?) bigger time dimension!
        sentinel_tensor = torch.from_numpy(sentinel_data).float().unsqueeze(1)  # add time dim (1, for now)

        return MAESample(x=sentinel_tensor)
