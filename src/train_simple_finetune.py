from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import lightning.pytorch as pl
import optuna
import torch
import torchmetrics
from lightning.pytorch.loggers import WandbLogger

from torch import nn

from src.data.s2osmdatamodule import S2OSMDatamodule, S2OSMDatamoduleConfig
from src.modules.base_segmentation_model import PrithviSegmentationModel, ConvTransformerTokensToEmbeddingNeck, FCNHead

Mode = Literal["train", "val", "test"]


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
    model: ModelConfig
    datamodule: S2OSMDatamoduleConfig  # storing here so it is logged

    # optimizer
    lr: float
    weight_decay: float
    betas: tuple[float, float]

    float32_matmul_precision: str

    # compile
    mode: str
    fullgraph: bool
    disable_compilation: bool

    # trainer
    devices: int
    precision: str

    # logger
    use_wandb_logger: bool
    project_name: str


@dataclass
class Batch:
    x: torch.Tensor
    y: torch.Tensor


class PrithviSegmentationFineTuner(pl.LightningModule):
    def __init__(
        self,
        config: TrainConfig,
        optuna_trial: optuna.Trial | None = None,
    ) -> None:
        super().__init__()
        self.config: TrainConfig = config
        # If u pass asdict(config), we can't load ckpt w/o passing config; Can't log w log_hyperparams bc no logger yet
        self.save_hyperparameters(logger=False, ignore=["net", "optuna_trial"])

        self.net: nn.Module = PrithviSegmentationModel(
            num_frames=config.model.num_frames,
            neck=ConvTransformerTokensToEmbeddingNeck(
                embed_dim=config.model.embed_dim * config.model.num_frames,
                output_embed_dim=config.model.output_embed_dim,
                patch_height=config.model.patch_height,
                patch_width=config.model.patch_width,
            ),
            head=FCNHead(
                in_channels=config.model.output_embed_dim,
                out_channels=config.model.fcn_out_channels,
                num_classes=config.model.num_classess,
            ),
        )
        self.loss_fn = nn.CrossEntropyLoss()  # TODO meaningful label smoothing?

        # TODO mIoU
        self.metrics: dict[Mode, dict[str, torchmetrics.Metric]] = {
            # "train": {
            #     "accuracy": torchmetrics.Accuracy(),
            # },
            # "val": {
            #     "accuracy": torchmetrics.Accuracy(),
            # }
        }
        # TODO, for debugging and monitoring: log images; log prediction of a fixed bbox over time; log attention maps

        torch.set_float32_matmul_precision(self.config.float32_matmul_precision)

        self.net: nn.Module = torch.compile(  # type: ignore
            model=self.net,
            mode=config.mode,
            fullgraph=config.fullgraph,
            disable=config.disable_compilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._model_step(batch, mode="train")

    def on_train_epoch_end(self) -> None:
        for name, metric in self.metrics["train"].keys():
            epoch_metric = metric.compute()
            self.log(f"train/{name}", epoch_metric)
            metric.reset()

    def validation_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        return self._model_step(batch, mode="val")

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            return
        for name, metric in self.metrics["val"].items():
            epoch_metric = metric.compute()
            self.log(f"val/{name}", epoch_metric)
            metric.reset()

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        # TODO scheduler (default in hls-os repo where linear decay with warmup)
        # total_steps = self.trainer.max_epochs * len(self.train_dataloader())
        # scheduler =
        return {
            # "lr_scheduler": {
            # "scheduler": scheduler,
            # "interval": "step",
            # "frequency": 1,
            # },
            "optimizer": optimizer,
        }

    def _model_step(self, batch: Batch, mode: Mode) -> torch.Tensor:
        x = batch.x
        y = batch.y

        logits = self.net(x)

        loss = self.loss_fn(logits, y)

        if self.trainer.sanity_checking:
            return loss

        self.log(f"{mode}/loss", loss)

        # TODO calculate and log metrics

        return loss


def train() -> None:
    # TODO these are initial / example values
    model_config: ModelConfig = ModelConfig(
        num_frames=(num_frames := 1),
        embed_dim=(embed_dim := 768),
        output_embed_dim=embed_dim * num_frames,
        patch_height=14,
        patch_width=14,
        fcn_out_channels=256,
        num_classess=7,  # todo
    )
    datamodule_config: S2OSMDatamoduleConfig = S2OSMDatamoduleConfig(
        batch_size=8,
        num_workers=1,
        pin_memory=True,
        use_transforms=False,
        data_split=(0.8, 0.1, 0.1),
        val_batch_size_multiplier=2,
    )
    config: TrainConfig = TrainConfig(
        model=model_config,
        datamodule=datamodule_config,
        project_name="PrithviFineTune",
        lr=1.5e-05,
        weight_decay=0.05,
        betas=(0.9, 0.999),
        float32_matmul_precision="medium",
        mode="max-autotune",
        fullgraph=True,
        disable_compilation=True,
        devices=1,
        precision="bf16-mixed",
        use_wandb_logger=False,
        # model defaults (from crop classification cfg)
    )

    model: PrithviSegmentationFineTuner = PrithviSegmentationFineTuner(config)

    datamodule: S2OSMDatamodule = S2OSMDatamodule(datamodule_config)

    callbacks: list[pl.Callback] = [
        pl.callbacks.ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1, save_last=True, every_n_epochs=1),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        # pl.callbacks.BackboneFinetuning(), might be helpful?
    ]

    logger: WandbLogger | None = WandbLogger() if config.use_wandb_logger else None

    trainer: pl.Trainer = pl.Trainer(
        callbacks=callbacks,
        devices=config.devices,
        precision=config.precision,
        logger=logger,
    )

    trainer.fit(model=model, datamodule=datamodule)


def tune() -> None:
    ...


def objective(trial: optuna.Trial) -> float:
    ...


if __name__ == "__main__":
    # tune()
    train()
