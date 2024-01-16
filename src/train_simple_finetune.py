from __future__ import annotations

import argparse
from typing import Any, Literal

import lightning.pytorch as pl
import optuna
import torch
import torchmetrics
from lightning.pytorch.loggers import WandbLogger

from torch import nn

from src.configs.paths import LOG_DIR, ROOT_DIR, CKPT_DIR
from src.configs.simple_finetune import Config
from src.data.s2osmdatamodule import S2OSMDatamodule
from src.data.s2osmdataset import S2OSMSample
from src.modules.base_segmentation_model import PrithviSegmentationModel, ConvTransformerTokensToEmbeddingNeck, FCNHead
import src.configs.simple_finetune as cfg
from src.utils import get_run_name

Mode = Literal["train", "val", "test"]


class PrithviSegmentationFineTuner(pl.LightningModule):
    def __init__(
        self,
        config: Config,
        optuna_trial: optuna.Trial | None = None,
    ) -> None:
        super().__init__()
        self.config: Config = config
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
            "train": {
                #     "accuracy": torchmetrics.Accuracy(),
            },
            "val": {
                #     "accuracy": torchmetrics.Accuracy(),
            },
        }
        # TODO, for debugging and monitoring: log images; log prediction of a fixed bbox over time; log attention maps, plot confusion matrix

        torch.set_float32_matmul_precision(self.config.train.float32_matmul_precision)

        self.net: PrithviSegmentationModel = torch.compile(  # type: ignore
            model=self.net,
            mode=config.train.compile_mode,
            fullgraph=config.train.compile_fullgraph,
            disable=config.train.compile_disable,
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

    def validation_step(self, batch: S2OSMSample, batch_idx: int) -> torch.Tensor:
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
            lr=self.config.train.lr,
            weight_decay=self.config.train.weight_decay,
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

    def _model_step(self, batch: S2OSMSample, mode: Mode) -> torch.Tensor:
        x = batch.x
        y = batch.y

        logits = self.net(x)

        print(logits.shape, y.shape)
        loss = self.loss_fn(logits, y)

        if self.trainer.sanity_checking:
            return loss

        self.log(f"{mode}/loss", loss)

        # TODO calculate and log metrics

        return loss


def train(config: Config, trial: optuna.Trial | None = None) -> None:
    model: PrithviSegmentationFineTuner = PrithviSegmentationFineTuner(config, optuna_trial=trial)
    datamodule: S2OSMDatamodule = S2OSMDatamodule(config.datamodule)
    callbacks: list[pl.Callback] = [
        pl.callbacks.ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            every_n_epochs=1,
            dirpath=CKPT_DIR / config.train.project_name,
        ),
        # pl.callbacks.BackboneFinetuning(), might be helpful?
    ]
    callbacks += [pl.callbacks.LearningRateMonitor(logging_interval="step")] if config.train.use_wandb_logger else []
    logger: WandbLogger | None = (
        WandbLogger(  # todo use wandb key from env for team entity
            project=config.train.project_name,
            name=get_run_name(config.train.project_name),
            log_model=False,  # no wandb artifacts
            save_dir=LOG_DIR / "wandb" / config.train.project_name,
        )
        if config.train.use_wandb_logger
        else False
    )
    trainer: pl.Trainer = pl.Trainer(
        default_root_dir=ROOT_DIR,
        callbacks=callbacks,
        devices=config.train.devices,
        precision=config.train.precision,
        logger=logger,
    )
    trainer.fit(model=model, datamodule=datamodule)


def tune() -> None:
    ...


def objective(trial: optuna.Trial) -> float:
    ...


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="base", help="Supply config name. Default: base")
    args = parser.parse_args()
    config = args.config or "base"
    config: Config = {
        "base": cfg.CONFIG,
        "debug": cfg.DEBUG_CFG,
        "tune": ...,
    }[config]
    if config == "tune":
        tune()
    else:
        train(config=config)


if __name__ == "__main__":
    main()
