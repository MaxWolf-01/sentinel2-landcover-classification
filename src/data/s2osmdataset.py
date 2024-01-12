import os
import typing
from dataclasses import dataclass

import rasterio
import torch
from torch.utils.data import Dataset

from src.utils import get_logger, load_prithvi

logger = get_logger(__name__)


@dataclass
class S2OsmDatasetConfig:
    sentinel_dir: str
    osm_dir: str


class S2OsmSample(typing.NamedTuple):
    x: torch.Tensor
    y: torch.LongTensor


class S2OSMDataset(Dataset):
    def __init__(self, cfg: S2OsmDatasetConfig) -> None:
        self.sentinel_dir = cfg.sentinel_dir
        self.osm_dir = cfg.osm_dir

        self.sentinel_files = [os.path.join(self.sentinel_dir, file) for file in os.listdir(self.sentinel_dir)]
        self.osm_files = [os.path.join(self.osm_dir, file) for file in os.listdir(self.osm_dir)]

        logger.info(f"Initialized {self} with {len(self)} samples.")

    def __len__(self) -> int:
        len_sen = len(self.sentinel_files)
        len_osm = len(self.osm_files)

        if len_sen != len_osm:
            raise Exception(
                f"There are different amounts of features and labels: Features:{len_sen}, Labels: {len_osm}")

        return len_sen

    def __getitem__(self, idx: int) -> S2OsmSample:
        with rasterio.open(self.sentinel_files[idx]) as src:
            sentinel_data = src.read()
            sentinel_tensor = torch.from_numpy(sentinel_data / 255.0).float().unsqueeze(0).unsqueeze(0) #TODO: Deal with time & batch dimension

        with rasterio.open(self.osm_files[idx]) as src:
            osm_data = src.read(1)
            osm_tensor = torch.from_numpy(osm_data).long().unsqueeze(0)

        return S2OsmSample(x=sentinel_tensor, y=osm_tensor)


#TESTING
if __name__ == "__main__":
    __ds = S2OSMDataset(S2OsmDatasetConfig(sentinel_dir="sentinel", osm_dir="osm"))
    print(__ds[0].x.shape, __ds[0].y.shape)
    print(__ds[1].x.shape, __ds[1].y.shape)
    print(__ds[2].x.shape, __ds[2].y.shape)
    print(__ds[0].x)

    ds = S2OSMDataset(S2OsmDatasetConfig(sentinel_dir="sentinel", osm_dir="osm"))
    model = load_prithvi(512)
    out = model(ds[0].x)
    print(out)