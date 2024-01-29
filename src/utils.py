from __future__ import annotations

import logging
import string
import random
from datetime import datetime
from pathlib import Path

import torch
import yaml

from src.configs.paths import PRE_TRAINED_WEIGHTS_DIR, CONFIG_DIR
from src.modules.prithvi import MaskedAutoencoderViT

PRITHVI_WEIGHTS: Path = PRE_TRAINED_WEIGHTS_DIR / "Prithvi_100M.pt"
PRITHVI_CONFIG: Path = CONFIG_DIR / "prithvi_config.yaml"


def load_prithvi(
    num_frames: int,
) -> MaskedAutoencoderViT:
    with PRITHVI_CONFIG.open("r") as f:
        config = yaml.safe_load(f)
    model_args = config["model_args"]
    model_args["num_frames"] = num_frames
    model = MaskedAutoencoderViT(**model_args)

    state_dict = torch.load(PRITHVI_WEIGHTS, map_location="cpu")
    state_dict.pop("pos_embed", None)
    state_dict.pop("decoder_pos_embed", None)
    model.load_state_dict(state_dict, strict=False)

    return model


def load_prithvi_mean_std() -> tuple[list[float], list[float]]:
    with PRITHVI_CONFIG.open("r") as f:
        config = yaml.safe_load(f)["train_params"]
    return config["data_mean"], config["data_std"]


def load_pritvhi_bands() -> list[str]:
    with PRITHVI_CONFIG.open("r") as f:
        config = yaml.safe_load(f)["train_params"]
    return config["bands"]


def get_unique_run_name(name: str, postfix: str | None = None) -> str:
    short_uuid: str = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    if postfix is not None:
        return f"{name}_{short_uuid}_{postfix}"
    return f"{name}_{short_uuid}"


def get_logger(name: str, log_level: int = logging.INFO) -> logging.Logger:
    log_directory = Path(__file__).parent.parent / "logs" / "system"
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

    logger = logging.getLogger(name)
    if not logger.handlers:  # Check if handlers already exist for this logger
        logger.setLevel(log_level)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
