from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass

from data.mae_datamodule import MAEDatamoduleConfig
from data.mae_dataset import MAEDatasetConfig


@dataclass
class Config:
    model: ModelConfig
    datamodule: MAEDatamoduleConfig
    train: TrainConfig


@dataclass
class ModelConfig:
    num_frames: int  # input frames per prediction


@dataclass
class TrainConfig:
    float32_matmul_precision: str

    from_scratch: bool

    # optimizer
    lr: float
    weight_decay: float
    betas: tuple[float, float]

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

    # lr scheduler
    lr_scheduler_type: typing.Literal["step", "cosine_warm_restarts"] | None = None
    step_lr_sched_step_size: int | None = None
    step_lr_sched_gamma: float | None = None
    cosine_warm_restarts_T_0: int | None = None
    cosine_warm_restarts_eta_min: float | None = None


CONFIG = Config(
    model=ModelConfig(
        num_frames=1,
    ),
    datamodule=MAEDatamoduleConfig(
        dataset_cfg=MAEDatasetConfig(aoi="at"),
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
        project_name="prithvi-mae-finetune",
        from_scratch=False,
        lr=1.5e-05,
        weight_decay=0.05,
        betas=(0.9, 0.999),
        float32_matmul_precision="high",  # todo set to medium later
        compile_mode="max-autotune",
        compile_fullgraph=True,
        compile_disable=False,
        devices=1,
        precision="32-true",  # todo set to bf-16-mixed later
        overfit_batches=0.0,
        use_wandb_logger=True,
        tags=[],
        log_img_in_train=True,
        # lr scheduler
        lr_scheduler_type=None,
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
