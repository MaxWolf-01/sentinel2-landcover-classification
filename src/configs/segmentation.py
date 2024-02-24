from __future__ import annotations

import dataclasses
import enum
import typing
from dataclasses import dataclass
from functools import partial

from torch import nn

from modules.efficientnet_unet import EfficientNetConfig, EfficientnetUnet
from modules.prithvi_segmentation import PrithviSegmentationNet, PrithviSegmentationNetConfig
from src.data.s2osm_datamodule import S2OSMDatamoduleConfig
from src.data.s2osm_dataset import S2OSMDatasetConfig

ModelConfig = PrithviSegmentationNetConfig | EfficientNetConfig


class ModelName(str, enum.Enum):  # enum.StrEnum/enum.auto() not supported <3.11
    FC_PRITHVI_BACKBONE = "fc-prithvi-backbone"
    EFFICIENTNET_UNET_B0 = "efficientnet-unet-b0"
    EFFICIENTNET_UNET_B1 = "efficientnet-unet-b1"
    EFFICIENTNET_UNET_B2 = "efficientnet-unet-b2"
    EFFICIENTNET_UNET_B3 = "efficientnet-unet-b3"
    EFFICIENTNET_UNET_B4 = "efficientnet-unet-b4"
    EFFICIENTNET_UNET_B5 = "efficientnet-unet-b5"
    EFFICIENTNET_UNET_B6 = "efficientnet-unet-b6"
    EFFICIENTNET_UNET_B7 = "efficientnet-unet-b7"


@dataclass
class Config:
    model_name: ModelName
    datamodule: S2OSMDatamoduleConfig
    train: TrainConfig

    model: ModelConfig | None = None  # set in `get_model`
    num_classes: int | None = None  # dynamically obtained from label map

    def __post_init__(self) -> None:
        self.train.tags.append(self.model_name)
        if self.model_name.startswith("efficientnet-unet"):
            assert self.datamodule.dataset_cfg.n_time_frames == 1, "EfficientNet-UNet only supports 1 frame input"
            self.datamodule.dataset_cfg.squeeze_time_dim = True

    def get_model(self) -> nn.Module:
        assert self.num_classes is not None, "num_classes must be set before calling `get_model`"
        name_to_model_mapping: dict[
            ModelName, tuple[partial[ModelConfig], typing.Callable[[ModelConfig], nn.Module]]
        ] = {
            ModelName.FC_PRITHVI_BACKBONE: (FC_PRITHVI_BACKBONE, PrithviSegmentationNet),
            ModelName.EFFICIENTNET_UNET_B0: (EFFICIENTNET_UNET_B0, EfficientnetUnet),
            ModelName.EFFICIENTNET_UNET_B1: (EFFICIENTNET_UNET_B1, EfficientnetUnet),
            ModelName.EFFICIENTNET_UNET_B2: (EFFICIENTNET_UNET_B2, EfficientnetUnet),
            ModelName.EFFICIENTNET_UNET_B3: (EFFICIENTNET_UNET_B3, EfficientnetUnet),
            ModelName.EFFICIENTNET_UNET_B4: (EFFICIENTNET_UNET_B4, EfficientnetUnet),
            ModelName.EFFICIENTNET_UNET_B5: (EFFICIENTNET_UNET_B5, EfficientnetUnet),
            ModelName.EFFICIENTNET_UNET_B6: (EFFICIENTNET_UNET_B6, EfficientnetUnet),
            ModelName.EFFICIENTNET_UNET_B7: (EFFICIENTNET_UNET_B7, EfficientnetUnet),
        }
        model_config_partial, instantiator = name_to_model_mapping[self.model_name]
        self.model = model_config_partial(num_classes=self.num_classes)
        return instantiator(self.model)


@dataclass
class TrainConfig:
    float32_matmul_precision: str

    # optimizer
    lr: float
    weight_decay: float
    betas: tuple[float, float]

    # loss
    loss_type: typing.Literal["ce", "focal"]

    # compile
    compile_mode: str
    compile_fullgraph: bool
    compile_disable: bool

    # trainer
    max_epochs: int
    log_interval: int
    devices: int
    precision: str
    overfit_batches: float

    # logger
    use_wandb_logger: bool
    project_name: str
    wandb_entity: str | None = None
    run_name: str | None = None  # set from script / autogenerated
    tags: list[str] = dataclasses.field(default_factory=list)

    seed: int = 42

    # loss_type specific
    label_smoothing: float = 0.0
    focal_loss_alpha: float | None = None
    focal_loss_gamma: float | None = None

    # lr scheduler
    lr_scheduler_type: typing.Literal["step", "cosine_warm_restarts"] | None = None
    step_lr_sched_step_size: int | None = None
    step_lr_sched_gamma: float | None = None
    cosine_warm_restarts_T_0: int | None = None
    cosine_warm_restarts_eta_min: float | None = None


# TODO these are still initial / example values (including model configs etc.)
BASE_CONFIG = partial(
    Config,
    datamodule=S2OSMDatamoduleConfig(
        dataset_cfg=S2OSMDatasetConfig(aoi="at", label_map="multiclass"),
        batch_size=32,
        num_workers=1,
        pin_memory=True,
        augment=True,
        data_split=(0.8, 0.2, 0.0),
        val_batch_size_multiplier=2,
        # transforms
        random_crop_size=224,
    ),
    train=TrainConfig(
        project_name="sentinel-segmentation",
        lr=1.5e-05,
        weight_decay=0.05,
        betas=(0.9, 0.999),
        float32_matmul_precision="high",  # todo set to medium later
        compile_mode="max-autotune",
        compile_fullgraph=True,
        compile_disable=False,
        max_epochs=-1,
        log_interval=50,
        devices=1,
        precision="32-true",  # todo set to bf16 later
        overfit_batches=0.0,
        use_wandb_logger=True,
        # loss
        loss_type="focal",
        focal_loss_alpha=0.25,
        focal_loss_gamma=2,
        # lr scheduler
        lr_scheduler_type=None,
    ),
)

# model defaults (mostly from crop classification cfg)
FC_PRITHVI_BACKBONE = partial(
    PrithviSegmentationNetConfig,
    num_frames=1,
    fcn_out_channels=256,
    fcn_num_convs=1,
    fcn_dropout=0.1,
    frozen_backbone=True,
)

EFFICIENTNET_UNET_B0 = partial(EfficientNetConfig, in_channels=6, version="b0")
EFFICIENTNET_UNET_B1 = partial(EfficientNetConfig, in_channels=6, version="b1")
EFFICIENTNET_UNET_B2 = partial(EfficientNetConfig, in_channels=6, version="b2")
EFFICIENTNET_UNET_B3 = partial(EfficientNetConfig, in_channels=6, version="b3")
EFFICIENTNET_UNET_B4 = partial(EfficientNetConfig, in_channels=6, version="b4")
EFFICIENTNET_UNET_B5 = partial(EfficientNetConfig, in_channels=6, version="b5")
EFFICIENTNET_UNET_B6 = partial(EfficientNetConfig, in_channels=6, version="b6")
EFFICIENTNET_UNET_B7 = partial(EfficientNetConfig, in_channels=6, version="b7")


def set_run_type(config: Config, run_type: typing.Literal["train", "debug", "overfit"]) -> Config:
    return {"train": lambda x: x, "debug": debug, "overfit": overfit, "tune": tune}[run_type](config)


def debug(config: Config) -> Config:
    config.train.devices = 1
    config.datamodule.batch_size = 1
    config.train.compile_disable = True
    config.train.log_img_in_train = True
    config.train.tags.append("debug")
    return config


def overfit(config: Config) -> Config:
    config.train.overfit_batches = 1
    config.datamodule.augment = False
    config.train.log_img_in_train = True
    config.train.tags.append("overfit")
    return config


def tune(config: Config) -> Config:
    return config
