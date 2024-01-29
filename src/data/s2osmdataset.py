import typing
from dataclasses import dataclass

import einops
import rasterio
import torch
from torch.utils.data import Dataset

from src.configs.paths import SENTINEL_DIR, OSM_DIR
from src.utils import get_logger
import albumentations as A
import numpy.typing as npt

logger = get_logger(__name__)


@dataclass
class S2OSMDatasetConfig(typing.NamedTuple):
    ...


class S2OSMSample(typing.NamedTuple):
    x: torch.Tensor
    y: torch.LongTensor


class S2OSMDataset(Dataset):
    def __init__(self, cfg: S2OSMDatasetConfig) -> None:
        super().__init__()
        self.transform: A.Compose | None = None  # to be set in the datamodule TODO why? can we remove this hack?
        self.sentinel_files = list(SENTINEL_DIR.glob("*.tif"))
        self.osm_files = list(OSM_DIR.glob("*.tif"))
        assert len(self.sentinel_files) == len(self.osm_files), (
            f"There are different amounts of input data and labels:\n"
            f"Input Data:{self.sentinel_files}\nLabels: {self.osm_files}"
        )
        assert len(self) > 0, "No data found. Did you run `download_data.py`?"

        logger.info(f"Initialized {self} with {len(self)} samples.")

    def __len__(self) -> int:
        return len(self.sentinel_files)

    def __getitem__(self, idx: int) -> S2OSMSample:
        with rasterio.open(self.sentinel_files[idx]) as f:
            sentinel_data: npt.NDArray = f.read()
        with rasterio.open(self.osm_files[idx]) as f:
            osm_data: npt.NDArray = f.read(1)  # read first band

        if self.transform is not None:
            sentinel_data = einops.rearrange(sentinel_data, "c h w -> h w c")  # albumentations uses chan last
            transformed: dict[str, typing.Any] = self.transform(image=sentinel_data, mask=osm_data)
            sentinel_data = transformed["image"]
            osm_data = transformed["mask"]
            sentinel_data = einops.rearrange(sentinel_data, "h w c -> c h w")

        sentinel_tensor = torch.from_numpy(sentinel_data).float().unsqueeze(1)  # add time dim (1, for now)
        osm_tensor = torch.from_numpy(osm_data).long()

        return S2OSMSample(x=sentinel_tensor, y=osm_tensor)
