from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass

from src.data.s2osmdatamodule import S2OSMDatamoduleConfig
from src.data.s2osmdataset import S2OSMDatasetConfig


@dataclass
class Config:
    model: ModelConfig
    datamodule: S2OSMDatamoduleConfig
    train: TrainConfig


@dataclass
class ModelConfig:
    num_frames: int  # input frames per prediction
    output_embed_dim: int
    patch_height: int
    patch_width: int
    fcn_out_channels: int
    fcn_num_convs: int
    fcn_dropout: float

    num_classes: int | None = None  # set dynamically from dataset tag mapping
    embed_dim: int = 768  # fixed prithvi output embedding dim


@dataclass
class TrainConfig:
    # optimizer
    lr: float
    weight_decay: float
    betas: tuple[float, float]

    # loss
    loss_type: typing.Literal["ce", "focal"]

    # lr scheduler
    use_lr_scheduler: bool
    lr_scheduler_type: str
    lr_step_size: int
    lr_gamma: float
    weight_decay: float

    float32_matmul_precision: str

    # compile
    compile_mode: str
    compile_fullgraph: bool
    compile_disable: bool

    # trainer
    devices: int
    precision: str
    overfit_batches: float

    # logger
    use_wandb_logger: bool
    log_img_in_train: bool
    project_name: str
    wandb_entity: str | None = None
    run_name: str | None = None  # set from script / autogenerated
    tags: list[str] = dataclasses.field(default_factory=list)

    seed: int = 42

    # loss_type specific
    label_smoothing: float = 0.0
    focal_loss_alpha: float | None = None
    focal_loss_gamma: float | None = None


# TODO these are still initial / example values
CONFIG = Config(
    model=ModelConfig(
        # model defaults (mostly from crop classification cfg)
        num_frames=(num_frames := 1),
        output_embed_dim=ModelConfig.embed_dim * num_frames,
        patch_height=14,
        patch_width=14,
        fcn_out_channels=256,
        fcn_num_convs=1,
        fcn_dropout=0.1,
    ),
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
        project_name="simple-prithvi-finetune",
        lr=1.5e-05,
        weight_decay=0.05,
        betas=(0.9, 0.999),
        float32_matmul_precision="medium",
        compile_mode="max-autotune",
        compile_fullgraph=True,
        compile_disable=False,
        devices=1,
        precision="bf16-mixed",
        overfit_batches=0.0,
        use_wandb_logger=True,
        tags=["frozen-prithvi"],
        log_img_in_train=False,

        loss_type="focal",
        focal_loss_alpha=0.25,
        focal_loss_gamma=2,

        use_lr_scheduler=False,
        lr_scheduler_type="StepLR",
        lr_step_size=10,
        lr_gamma=0.1,
    ),
)


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


# def large_model(config: Config) -> Config:
#     config.model.fcn_num_convs = 3
#     config.model.fcn_out_channels = 512
#     config.train.tags.append("large-model")
#     return config
#
#
# def small_model(config: Config) -> Config:
#     config.model.fcn_out_channels = 128
#     config.train.tags.append("small-model")
#     return config
