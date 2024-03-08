from __future__ import annotations

import copy
import logging
import random
import string
import typing
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch import nn

from src.configs.paths import CONFIG_DIR, LOG_DIR, PRE_TRAINED_WEIGHTS_DIR
from src.modules.prithvi import MaskedAutoencoderViT

PRITHVI_WEIGHTS: Path = PRE_TRAINED_WEIGHTS_DIR / "Prithvi_100M.pt"
PRITHVI_CONFIG: Path = CONFIG_DIR / "prithvi_config.yaml"


def get_logger(name: str, log_level: int = logging.INFO) -> logging.Logger:
    log_directory = LOG_DIR / "system"
    log_directory.mkdir(parents=True, exist_ok=True)

    log_filename = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'
    log_filepath = log_directory / log_filename

    file_logging_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    console_logging_format = file_logging_format

    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(file_logging_format))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(console_logging_format))

    _logger = logging.getLogger(name)
    if not _logger.handlers:  # Check if handlers already exist for this logger
        _logger.setLevel(log_level)
        _logger.addHandler(file_handler)
        _logger.addHandler(console_handler)

    return _logger


logger = get_logger(__name__)


def load_untrained_prithvi(num_frames: int) -> MaskedAutoencoderViT:
    with PRITHVI_CONFIG.open("r") as f:
        config = yaml.safe_load(f)
    model_args = config["model_args"]
    model_args["num_frames"] = num_frames
    model = MaskedAutoencoderViT(**model_args)
    logger.info("Loaded untrained MaskedAutoencoderViT model.")
    return model


def load_prithvi(
    num_frames: int,
    no_decoder: bool = True,
) -> MaskedAutoencoderViT:
    with PRITHVI_CONFIG.open("r") as f:
        config = yaml.safe_load(f)
    model_args = config["model_args"]
    model_args["num_frames"] = num_frames
    model = MaskedAutoencoderViT(**model_args)

    state_dict = torch.load(PRITHVI_WEIGHTS, map_location="cpu")

    # remove weights that should be left randomly initialized + decoder weights if no_decoder
    weights_to_pop = ["pos_embed", "decoder_pos_embed"]
    attrs_to_pop = []
    if no_decoder:
        decoder_specific_attrs = ["decoder_embed", "mask_token", "decoder_blocks", "decoder_norm", "decoder_pred"]
        weights_to_pop += decoder_specific_attrs
        attrs_to_pop += decoder_specific_attrs

    weights_to_pop_keys = [key for key in state_dict.keys() if any(key.startswith(prefix) for prefix in weights_to_pop)]
    for weight in weights_to_pop_keys:
        state_dict.pop(weight)
    for attr in attrs_to_pop:
        delattr(model, attr)

    model.load_state_dict(state_dict, strict=False)
    model.reinitialize_pos_embed()  # apply initialization for the removed (decoder_)pos_embed
    logger.info(f"Loaded pre-trained weights from {PRITHVI_WEIGHTS}")
    logger.info(
        f"The following weights were left randomly initialized: "
        f"{[w for w in weights_to_pop_keys if not any(w.startswith(prefix) for prefix in attrs_to_pop)]}"
    )
    logger.info(f"The following attributes were removed: {attrs_to_pop}")
    return model


def load_prithvi_mean_std() -> tuple[list[float], list[float]]:
    with PRITHVI_CONFIG.open("r") as f:
        config = yaml.safe_load(f)["train_params"]
    return config["data_mean"], config["data_std"]


def get_unique_run_name(name: str | None, postfix: str | None = None) -> str:
    run_name = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    if postfix is not None:
        run_name += f"_{postfix}"
    if name is not None:
        run_name = f"{name}_{run_name}"
    return run_name


Dataset = typing.TypeVar("Dataset", bound=torch.utils.data.Dataset)


class Subset(typing.Generic[Dataset]):
    def __init__(self, dataset: Dataset, indices: typing.Sequence[int]) -> None:
        self.dataset: Dataset = dataset
        self.indices: typing.Sequence[int] = indices

    def __getitem__(self, index: int) -> typing.Any:
        return self.dataset[self.indices[index]]

    def __len__(self) -> int:
        return len(self.indices)


def train_val_test_split(
    dataset: torch.utils.data.Dataset,
    data_split: tuple[float, float, float],
    deepcopy: bool = False,
) -> tuple[Subset, Subset, Subset]:
    assert sum(data_split) == 1.0, "Data split must sum to 1.0"
    # noinspection PyTypeChecker
    total_length: int = len(dataset)  # ignore "expected Sized, got Dataset"
    train_len: int = int(data_split[0] * total_length)
    val_len: int = int(data_split[1] * total_length)

    all_indicies: list[int] = list(range(total_length))
    random.shuffle(all_indicies)
    train_indices: list[int] = all_indicies[:train_len]
    val_indices: list[int] = all_indicies[train_len : train_len + val_len]
    test_indices: list[int] = all_indicies[train_len + val_len :]

    train, val, test = Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)
    if deepcopy:
        train, val, test = copy.deepcopy(train), copy.deepcopy(val), copy.deepcopy(test)
    return train, val, test


def get_class_probabilities(dataset: torch.utils.data.Dataset, ignore_label: int = -1) -> torch.Tensor:
    """Calculate dataset class frequency as probabilities. Uses random sample if the dataset is large.
    Args:
        dataset (torch.utils.data.Dataset): dataset to calculate class probabilities for.
        ignore_label (int): class label to ignore when calculating class frequencies.
    Returns:
        class_weights (torch.Tensor): class weights to be used in the loss function, sorted by class index.
    """
    sample_labels = torch.cat([dataset[i][1] for i in random.sample(range(len(dataset)), k=min(2500, len(dataset)))])
    unique, counts = torch.unique(sample_labels, return_counts=True, sorted=True)  # (C,), (C,)
    if ignore_label != -1:
        counts = counts[unique != ignore_label]  # (C-1,)
    class_weights = counts / counts.sum()
    return class_weights


def initialize_classification_layer_bias(layer: nn.Linear | nn.Conv2d, class_distribution: list[float]) -> None:
    """Initialize the bias of the classification layer to reflect the class distribution.
    Args:
        layer (nn.Linear | nn.Conv2d): classification layer to initialize.
        class_distribution (list[float]): class distribution to initialize the bias with.
    """
    distribution = torch.tensor(class_distribution, dtype=torch.float32)
    assert torch.isclose(
        distribution.sum(), torch.tensor(1.0)
    ), f"Must sum to 1, got distribution: {class_distribution}"
    assert len(class_distribution) > 1, "Class distribution must have at least 2 classes"
    if len(distribution) == 2:
        layer.bias.data.fill_((distribution[1] / distribution[0]).log())  # positive/negative class
    else:
        layer.bias.data = distribution.log()
