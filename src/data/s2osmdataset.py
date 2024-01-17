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
        self._sentinel_files = list(SENTINEL_DIR.glob("*.tif"))
        self._osm_files = list(OSM_DIR.glob("*.tif"))
        assert len(self._sentinel_files) == len(self._osm_files), (
            f"There are different amounts of input data and labels:\n"
            f"Input Data:{self._sentinel_files}\nLabels: {self._osm_files}"
        )
        assert len(self) > 0, "No data found. Did you run `download_data.py`?"

        logger.info(f"Initialized {self} with {len(self)} samples.")

    def __len__(self) -> int:
        return len(self._sentinel_files)

    def __getitem__(self, idx: int) -> S2OSMSample:
        with rasterio.open(self._sentinel_files[idx]) as f:
            sentinel_data: npt.NDArray = f.read()
        with rasterio.open(self._osm_files[idx]) as f:
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


if __name__ == "__main__":

    def t() -> None:
        from src.utils import load_prithvi

        ds = S2OSMDataset(S2OSMDatasetConfig())
        model = load_prithvi(num_frames=1)
        x = ds[0].x  # (1, 1, 3, 512, 512)
        # TODO don't forget target needs to be cropped to the same location as the input as well
        print("Original Data: ", x.shape, " Target: ", ds[0].y.shape)
        # TODO remove this after data is loaded with the necessary 6 bands...
        x = einops.rearrange(x, "b t c h w  -> b c t h w")
        x = x[:, :, :, :224, :224]  # "random crop"
        print("Pritvhi Input:  ", x.shape)
        features, _, _ = model.forward_encoder(x, mask_ratio=0)
        print("Pritvhi Output: ", features.shape)

    t()
