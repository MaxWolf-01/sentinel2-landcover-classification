import typing
from dataclasses import dataclass

import einops
import rasterio
import torch
from torch.utils.data import Dataset

from src.configs.french_label_mappings import map_labels_to_simplified_categories
from src.configs.download_config import DataDirs
from src.utils import get_logger
import albumentations as A
import numpy.typing as npt

logger = get_logger(__name__)


@dataclass
class S2CNESDatasetConfig:
    aoi: str  # Area of Interest: fr/
    label_map: str  # Type of label map: multiclass/binary


class S2CNESLandCoverSample(typing.NamedTuple):
    x: torch.Tensor  # Sentinel-2 data
    y: torch.Tensor  # CNES Land Cover data


class S2CNESDataset(Dataset):
    def __init__(self, cfg: S2CNESDatasetConfig) -> None:
        super().__init__()
        self.transform: A.Compose | None = None  # Transformations for data augmentation
        data_dirs = DataDirs(aoi=cfg.aoi, map_type=cfg.label_map)
        self.sentinel_files = list(data_dirs.sentinel.glob("*.tif"))
        self.land_cover_files = list(data_dirs.land_cover.glob("*.tif"))

        assert len(self.sentinel_files) == len(
            self.land_cover_files
        ), "Mismatch between the number of Sentinel and CNES Land Cover files."
        assert len(self) > 0, "No data found. Ensure data has been downloaded."

        logger.info(f"Initialized {self} with {len(self)} samples.")

    def __len__(self) -> int:
        return len(self.sentinel_files)

    def __getitem__(self, idx: int) -> S2CNESLandCoverSample:
        with rasterio.open(self.sentinel_files[idx]) as f:
            sentinel_data: npt.NDArray = f.read()  # Read all bands

        with rasterio.open(self.land_cover_files[idx]) as f:
            # Read all bands: OCS, OCS_Confidence, OCS_Validity
            land_cover_data: npt.NDArray = f.read()

        land_cover_data = map_labels_to_simplified_categories(land_cover_data)

        # Apply transformations if any
        if self.transform is not None:
            sentinel_data = einops.rearrange(sentinel_data, "c h w -> h w c")  # Change to HWC for albumentations
            land_cover_data = einops.rearrange(land_cover_data, "c h w -> h w c")  # Same for land cover data

            transformed = self.transform(image=sentinel_data, mask=land_cover_data[:, :, 0])  # Only use OCS for mask
            sentinel_data = transformed["image"]
            land_cover_data = transformed["mask"]

            sentinel_data = einops.rearrange(sentinel_data, "h w c -> c h w")  # Back to CHW
            #TODO: Use more than just the OCS band. (f.e. confidence levels, which can (in case of low conf) be masked for training)

        sentinel_tensor = torch.from_numpy(sentinel_data).float().unsqueeze(1)  # add time dim (1, for now)

        land_cover_tensor = torch.from_numpy(land_cover_data).long()

        return S2CNESLandCoverSample(x=sentinel_tensor, y=land_cover_tensor)
