import typing
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class S2OsmDatasetConfig:
    ...


class S2OsmSample(typing.NamedTuple):
    x: torch.Tensor
    y: torch.LongTensor


class S2OSMDataset(Dataset):
    def __init__(self, cfg: S2OsmDatasetConfig) -> None:
        ...
        logger.info(f"Initialized {self} with {len(self)} samples.")

    def __len__(self) -> int:
        ...

    def __getitem__(self, idx: int) -> S2OsmSample:
        ...


if __name__ == "__main__":
    __ds = S2OSMDataset(S2OsmDatasetConfig())
    print(__ds[0].x.shape, __ds[0].y.shape)
    print(__ds[1].x.shape, __ds[1].y.shape)
    print(__ds[2].x.shape, __ds[2].y.shape)
    print(__ds[0].x)
