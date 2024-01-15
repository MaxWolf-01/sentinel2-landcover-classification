import typing
from dataclasses import dataclass

import einops
import rasterio
import torch
from torch.utils.data import Dataset

from src.data.paths import SENTINEL_DIR, OSM_DIR
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class S2OSMDatasetConfig(typing.NamedTuple):
    ...


class S2OSMSample(typing.NamedTuple):
    x: torch.Tensor
    y: torch.LongTensor


class S2OSMDataset(Dataset):
    def __init__(self, transform: typing.Callable[[torch.Tensor], torch.Tensor] | None = None) -> None:
        super().__init__()
        self.transform = transform
        self.sentinel_files = list(SENTINEL_DIR.glob("*.tif"))
        self.osm_files = list(OSM_DIR.glob("*.tif"))
        assert len(self.sentinel_files) == len(self.osm_files), (
            f"There are different amounts of input data and labels:\n"
            f"Input Data:{self.sentinel_files}\nLabels: {self.osm_files}"
        )

        logger.info(f"Initialized {self} with {len(self)} samples.")

    def __len__(self) -> int:
        return len(self.sentinel_files)

    def __getitem__(self, idx: int) -> S2OSMSample:
        # TODO normalization (see notebook) [not sure if we should norm before or after transforms]
        with rasterio.open(self.sentinel_files[idx]) as f:
            sentinel_data = f.read()
            sentinel_tensor = torch.from_numpy(sentinel_data).float()
            sentinel_tensor = sentinel_tensor.unsqueeze(0)  # add time dimension (left as 1 for initial experiments)

        with rasterio.open(self.osm_files[idx]) as f:
            osm_data = f.read(1)  # read first band
            osm_tensor = torch.from_numpy(osm_data).long().unsqueeze(0)

        if self.transform is not None:
            sentinel_tensor = self.transform(sentinel_tensor)
            # todo some transforms like random crop need to be applied to the target as well

        return S2OSMSample(x=sentinel_tensor, y=osm_tensor)


if __name__ == "__main__":

    def t() -> None:
        from src.utils import load_prithvi

        ds = S2OSMDataset()
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
