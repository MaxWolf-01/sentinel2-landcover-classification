from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from typing import Tuple


def calculate_mean_std(dataset: torch.utils.data.Dataset, save_path: Path) -> None:
    dataloader: DataLoader = DataLoader(dataset, shuffle=False)
    welford: WelfordsMethod = WelfordsMethod(dim=(0, 1, 3, 4))
    for batch in tqdm(dataloader):
        data: torch.Tensor = batch.x
        welford.update(data)
    mean, std = welford.finalize(keepdim=False)
    torch.save({"mean": mean, "std": std}, save_path)


class WelfordsMethod:
    def __init__(self, dim: tuple[int, ...] | int | None = None) -> None:
        self.dim: tuple[int, ...] | int | None = dim
        self.count: int = 0
        self.mean: torch.Tensor | None = None
        self.M2: torch.Tensor | None = None

    def update(self, x: torch.Tensor) -> None:
        if self.mean is None:
            self.mean = torch.zeros_like(x)
            self.M2 = torch.zeros_like(x)
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def finalize(self, keepdim: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.count < 2:
            return self.mean, torch.zeros_like(self.mean)
        return (
            self.mean.mean(dim=self.dim, keepdim=keepdim),
            (self.M2 / (self.count - 1)).sqrt().mean(dim=self.dim, keepdim=keepdim),  # unbiased std dev
        )


def _test_welfords_method():
    dim = (0, 1, 3, 4)  # reduce B, T, C, H, W -> C
    data = torch.randn(100000, 5, 3, 32, 32)  # large value neededd for exact prec (idk if bug; but it good engh ig)
    expected_mean = data.mean(dim=dim, keepdim=True)
    expected_std = data.std(dim=dim, keepdim=True)

    welford = WelfordsMethod(dim)
    for b in range(data.size(0)):  # Simulate processing batches
        welford.update(data[b, :, :, :, :].unsqueeze(0))
    mean, std = welford.finalize()
    print(expected_mean, expected_std)
    print(mean, std)
    print(mean.shape, std.shape)
    assert torch.allclose(mean, expected_mean, atol=1e-5)
    assert torch.allclose(std, expected_std, atol=1e-5)


if __name__ == "__main__":
    _test_welfords_method()
