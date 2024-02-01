from pathlib import Path

CONFIG_DIR: Path = Path(__file__).parent  # /src/configs
ROOT_DIR: Path = CONFIG_DIR.parent.parent  # /

PRE_TRAINED_WEIGHTS_DIR: Path = ROOT_DIR / "weights"
LOG_DIR: Path = ROOT_DIR / "logs"
CKPT_DIR: Path = ROOT_DIR / "ckpts"
DATA_DIR: Path = ROOT_DIR / "data"

OUT_DIR: Path = ROOT_DIR / "out"
