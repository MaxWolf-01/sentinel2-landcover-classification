import torch
from torch.utils.data import DataLoader
from src.data.s2osmdataset import S2OSMDataset, S2OSMDatasetConfig


def calculate_mean_std(aoi: str, label_map: str):
    cfg = S2OSMDatasetConfig(aoi=aoi, label_map=label_map)
    dataset = S2OSMDataset(cfg)
    dataloader = DataLoader(dataset, shuffle=False)

    welford = WelfordsMethod()
    for batch in dataloader:
        data = batch.x
        data = data.squeeze(2)
        welford.update(data)

    mean, std = welford.finalize()
    # Adjustments for channel-wise mean and std
    mean = mean.mean(dim=[0, 2, 3])
    std = std.mean(dim=[0, 2, 3])

    torch.save({"mean": mean, "std": std}, str(dataset.data_dirs.base_path) + "\\mean_std.pt")


class WelfordsMethod:
    def __init__(self, shape=None):
        self.count = 0
        self.mean = None
        self.M2 = None
        self.shape = shape

    def update(self, x):
        if self.mean is None:
            # Initialize mean and M2 with the shape of x if not already done
            self.shape = x.shape
            self.mean = torch.zeros(self.shape, dtype=x.dtype, device=x.device)
            self.M2 = torch.zeros(self.shape, dtype=x.dtype, device=x.device)

        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def finalize(self):
        if self.count < 2:
            # Avoid division by zero
            return self.mean, torch.sqrt(self.M2)
        variance = self.M2 / (self.count - 1)
        std_dev = torch.sqrt(variance)
        return self.mean, std_dev
