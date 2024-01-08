import string
import random
from pathlib import Path

import torch
import yaml

from src.prithvi import MaskedAutoencoderViT

SRC_DIR: Path = Path(__file__).parent
WEIGHTS_DIR: Path = SRC_DIR.parent / "weights"
PRITHVI_WEIGHTS: Path = WEIGHTS_DIR / "Prithvi_100M.pt"
PRITHVI_CONFIG: Path = SRC_DIR / "prithvi_config.yaml"


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


def get_run_name(project_name: str, prefix: str | None = None) -> str:
    short_uuid: str = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    if prefix is not None:
        return f"{project_name}_{prefix}_{short_uuid}"
    return f"{project_name}_{short_uuid}"
