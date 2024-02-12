from __future__ import annotations

import argparse
import dataclasses
import os
import pprint
import random
from typing import Any, Literal

import dotenv
import numpy as np
import lightning.pytorch as pl
import optuna
import torch
import torchmetrics
import wandb
from lightning.pytorch.loggers import WandbLogger

from data.download_data import AOIs
from data.mae_datamodule import MAEDatamodule
from data.mae_dataset import MAESample
from lr_schedulers import get_lr_scheduler
from modules.prithvi import MaskedAutoencoderViT
from plotting import load_sentinel_tiff_for_plotting
from src.configs.paths import LOG_DIR, ROOT_DIR, CKPT_DIR
import src.configs.prithvi_mae_finetune as cfg
from src.utils import get_unique_run_name, get_logger, load_prithvi, load_untrained_prithvi
from numpy import typing as npt
from src.configs.prithvi_mae_finetune import Config

script_logger = get_logger(__name__)

Mode = Literal["train", "val", "test"]


class PrithviMAETrainer(pl.LightningModule):
    def __init__(self, config: Config, optuna_trial: optuna.Trial | None = None) -> None:
        super().__init__()
        self.config: Config = config
        # If u pass asdict(config), we can't load ckpt w/o passing config; Can't log w log_hyperparams bc no logger yet
        self.save_hyperparameters(logger=False, ignore=["net", "optuna_trial"])

        self.net: MaskedAutoencoderViT = (
            load_untrained_prithvi(num_frames=config.model.num_frames)
            if self.config.train.from_scratch
            else load_prithvi(num_frames=config.model.num_frames, no_decoder=False)
        )

        # todo SSIM / LPIPS?
        metrics = lambda: {}  # noqa: E731
        self.metrics: dict[Mode, dict[str, torchmetrics.Metric]] = {
            "train": metrics(),
            "val": metrics(),
        }

        torch.set_float32_matmul_precision(self.config.train.float32_matmul_precision)

        self.net: MaskedAutoencoderViT = torch.compile(  # type: ignore
            model=self.net,
            mode=config.train.compile_mode,
            fullgraph=config.train.compile_fullgraph,
            disable=config.train.compile_disable,
        )

    def on_fit_start(self) -> None:
        """
        This hook is called at the very beginning of the fit process.
        It is used  to move all metrics to the appropriate device.
        """
        for mode_metrics in self.metrics.values():
            for metric in mode_metrics.values():
                metric.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._model_step(batch, mode="train")

    def on_train_epoch_end(self) -> None:
        epoch_metrics = self.compute_metrics(mode="train")
        self.log_scalar_metrics(epoch_metrics, mode="train")
        # self.log_image_metrics(epoch_metrics, mode="train")  # TODO use center crop

    def validation_step(self, batch: MAESample, batch_idx: int) -> torch.Tensor:
        return self._model_step(batch, mode="val")

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            return

        epoch_metrics = self.compute_metrics(mode="val")
        self.log_scalar_metrics(epoch_metrics, mode="val")
        self.log_image_metrics(epoch_metrics, mode="val")

    def predict_step(self, batch: MAESample, batch_idx: int) -> torch.Tensor:
        return self._model_step(batch, mode="test")

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.config.train.lr,
            weight_decay=self.config.train.weight_decay,
        )
        scheduler = get_lr_scheduler(self.config, optimizer)
        optimizer_config = {"optimizer": optimizer}
        if self.config.train.lr_scheduler_type is not None:
            optimizer_config.update(
                {
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "epoch",
                        "frequency": 1,
                    }
                }
            )
        return optimizer_config

    def _model_step(self, batch: MAESample, mode: Mode) -> torch.Tensor:
        x = batch.x

        loss, logits, mask = self.net(x)  # mask ratio is 0.75

        if mode == "test":
            return logits

        if self.trainer.sanity_checking:
            return loss

        self.log(f"{mode}/loss", loss)

        # self.update_metrics(mode, predictions=torch.argmax(logits, dim=1), labels=y)  TODO

        return loss

    def update_metrics(self, mode: Mode, predictions: torch.Tensor, labels: torch.Tensor) -> None:
        for name, metric in self.metrics[mode].items():
            metric.update(predictions, labels)

    def compute_metrics(self, mode: Mode) -> dict[str, npt.NDArray]:
        computed_metrics = {}
        for metric_name, metric in self.metrics[mode].items():
            computed_value = metric.compute()
            computed_metrics[metric_name] = computed_value.cpu().numpy()
            metric.reset()
        return computed_metrics

    def log_scalar_metrics(self, computed_metrics: dict[str, npt.NDArray], mode: Mode) -> None:
        for metric_name, metric_value in computed_metrics.items():
            if np.prod(metric_value.shape) == 1:
                self.log(f"{mode}/{metric_name}", metric_value.item())

    def log_image_metrics(self, computed_metrics: dict[str, npt.NDArray], mode: Mode) -> None:
        if not isinstance(self.logger, WandbLogger) or (not self.config.train.log_img_in_train and mode == "train"):
            return

        img_idx = random.randint(0, len(self.trainer.train_dataloader.dataset) - 1) if mode == "train" else 0
        dataloader = self.trainer.train_dataloader if mode == "train" else self.trainer.val_dataloaders
        self.log_image_masked_and_reconstructed(
            dataloader=dataloader,
            mode=mode,
            idx=img_idx,
        )
        # log_image_masked_and_reconstructed()

    # TODO improve this plot: e.g. -) masked original image or reconstrcuted image patches merged with unmasked ones
    def log_image_masked_and_reconstructed(
        self,
        dataloader: torch.utils.data.DataLoader,
        mode: Mode,
        idx: int | None = None,
        prefix: str = "",
    ) -> None:
        if idx is None:
            idx = random.randint(0, len(dataloader.dataset) - 1)
        model_input: torch.Tensor = dataloader.dataset[idx].x

        with torch.inference_mode():
            loss, logits, mask = self(model_input.unsqueeze(0).to(self.device))
        pred_image: torch.Tensor = self.net.unpatchify(logits).squeeze(0)  # (C, H, W)
        pred_image = pred_image[:3, 0, ...]  # remove time dim and keep only RGB channels
        pred_image = pred_image[[2, 1, 0], ...]  # BGR to RGB

        original_image, bbox = load_sentinel_tiff_for_plotting(
            dataloader.dataset.dataset.sentinel_files[idx], return_bbox=True
        )
        # crop # fixme? when mode=train, it's a random crop not corresponding to the input
        original_image = dataloader.dataset.dataset.transform[0](image=original_image)["image"]
        # mask = einops.rearrange(mask, '1 (h w) -> h w 1', h=14, w=14)
        # 224x224 & 16x16 patches -> 14x14 grid
        # mask= einops.repeat(mask, "1 (h w) -> (rep_h h) (rep_w w) (rep_c 1)", h=14, w=14, rep_h=16, rep_w=16, rep_c=3)
        # masked_image: torch.Tensor = copy.deepcopy(original_image)
        # masked_image[mask.cpu() == 0] = 0  # Masking applied
        # masked_image shape: 37632

        caption = f"{bbox} | Epoch: {self.current_epoch} | Sample ID: {idx}"
        wandb.log(
            {
                f"{mode}/{prefix}original_image": wandb.Image(original_image, caption=caption),
                # f"{mode}/{prefix}masked_image": wandb.Image(masked_image),
                f"{mode}/{prefix}reconstructed_image": wandb.Image(pred_image.cpu().numpy().transpose(1, 2, 0)),
            }
        )


def train(config: Config, trial: optuna.Trial | None = None) -> None:
    model: PrithviMAETrainer = PrithviMAETrainer(config, optuna_trial=trial)
    datamodule: MAEDatamodule = MAEDatamodule(config.datamodule)
    callbacks: list[pl.Callback] = [
        pl.callbacks.ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            every_n_epochs=1,
            dirpath=CKPT_DIR / config.train.project_name,
            filename=config.train.run_name + "_{epoch:02d}_{val/loss:.2f}_{step}",
        ),
    ]
    callbacks += [pl.callbacks.LearningRateMonitor(logging_interval="epoch")] if config.train.use_wandb_logger else []
    logger: WandbLogger | bool = (
        WandbLogger(
            entity=config.train.wandb_entity,
            project=config.train.project_name,
            name=config.train.run_name,
            log_model=False,  # no wandb artifacts
            save_dir=LOG_DIR / "wandb" / config.train.project_name,
            tags=config.train.tags,
        )
        if config.train.use_wandb_logger
        else False
    )
    if logger:
        logger.watch(model, log="all", log_freq=10)  # todo set lower after debugging
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
    parser.add_argument("--from-scratch", action="store_true", help="Train without pretrained weights. Default: False")
    parser.add_argument("--type", type=str, default="train", help="[train, debug, overfit, ...]. Default: train")
    parser.add_argument("--bs", type=int, default=None, help="batch size.")
    parser.add_argument("--aoi", type=str, default=None, help=f"one of {list(AOIs)}")
    parser.add_argument("--name", type=str, default=None, help="run name prefix. Default: None")
    parser.add_argument("--wandb", action="store_true", help="DISABLE wandb logging.")
    parser.add_argument(  # list of tags
        "--tags", nargs="+", default=[], help="Tags for wandb. Default: None. Example usage: --tags t1 t2 t3"
    )
    parser.add_argument("--no-compile", action="store_true", help="Compile model. Default: True")
    args = parser.parse_args()

    dotenv.load_dotenv()

    config: Config = {
        "train": cfg.CONFIG,
        "debug": cfg.debug(cfg.CONFIG),
        "overfit": cfg.overfit(cfg.CONFIG),
        "tune": ...,
    }[(cfg_key := args.type)]
    config = cfg.pretrain(config) if args.from_scratch else cfg.finetune(config)
    config.datamodule.dataset_cfg.aoi = args.aoi or config.datamodule.dataset_cfg.aoi
    config.datamodule.batch_size = args.bs or config.datamodule.batch_size
    config.train.compile_disable = args.no_compile or config.train.compile_disable
    config.train.use_wandb_logger = False if args.wandb else config.train.use_wandb_logger
    config.train.tags.extend(args.tags)
    config.train.run_name = get_unique_run_name(name=args.name, postfix=config.train.project_name)
    config.train.wandb_entity = os.getenv("WANDB_ENTITY")

    script_logger.info(f"USING CONFIG: '{cfg_key}':\n{pprint.pformat(dataclasses.asdict(config))}")

    pl.seed_everything(config.train.seed)  # after creating run_name
    if cfg_key == "tune":
        tune()
    else:
        train(config=config)


if __name__ == "__main__":
    main()
