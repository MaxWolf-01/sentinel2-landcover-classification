from __future__ import annotations

import copy
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
    num_frames: int
    embed_dim: int
    output_embed_dim: int
    patch_height: int
    patch_width: int
    fcn_out_channels: int
    num_classess: int


@dataclass
class TrainConfig:
    # optimizer
    lr: float
    weight_decay: float
    betas: tuple[float, float]

    float32_matmul_precision: str

    # compile
    compile_mode: str
    compile_fullgraph: bool
    compile_disable: bool

    # trainer
    devices: int
    precision: str

    # logger
    use_wandb_logger: bool
    project_name: str


# TODO these are still initial / example values
CONFIG = Config(
    model=ModelConfig(
        # model defaults (mostly from crop classification cfg)
        num_frames=(num_frames := 1),
        embed_dim=(embed_dim := 768),
        output_embed_dim=embed_dim * num_frames,
        patch_height=14,
        patch_width=14,
        fcn_out_channels=256,
        num_classess=7,  # todo
    ),
    datamodule=S2OSMDatamoduleConfig(
        dataset_cfg=S2OSMDatasetConfig(),
        batch_size=8,
        num_workers=1,
        pin_memory=True,
        use_transforms=True,
        data_split=(0.8, 0.1, 0.1),
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
        compile_disable=True,
        devices=1,
        precision="bf16-mixed",
        use_wandb_logger=True,
    ),
)


DEBUG_CFG = copy.deepcopy(CONFIG)
DEBUG_CFG.train.use_wandb_logger = False
DEBUG_CFG.train.devices = 1
DEBUG_CFG.datamodule.batch_size = 2

# OVERFIT_CFG = CONFIG