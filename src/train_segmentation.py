from __future__ import annotations

import argparse
import dataclasses
import io
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
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from torchmetrics.classification import MulticlassConfusionMatrix
from torchmetrics import JaccardIndex as IoU
from torchmetrics import Accuracy
from torch import nn

from configs.label_mappings import MAPS, LabelMap
from data.download_data import AOIs
from losses import Loss, get_loss
from lr_schedulers import get_lr_scheduler
from plotting import load_sentinel_tiff_for_plotting
from src.configs.paths import LOG_DIR, ROOT_DIR, CKPT_DIR
from src.configs.segmentation import Config
from src.data.s2osm_datamodule import S2OSMDatamodule
from src.data.s2osm_dataset import S2OSMSample
from src.modules.base_segmentation_model import PrithviSegmentationModel, ConvTransformerTokensToEmbeddingNeck, FCNHead
import src.configs.segmentation as cfg
from src.utils import get_unique_run_name, get_logger
from numpy import typing as npt

script_logger = get_logger(__name__)

Mode = Literal["train", "val", "test"]


class PrithviSegmentationFineTuner(pl.LightningModule):
    def __init__(self, config: Config, optuna_trial: optuna.Trial | None = None) -> None:
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
                num_classes=config.model.num_classes,
                num_convs=config.model.fcn_num_convs,
                dropout=config.model.fcn_dropout,
            ),
        )

        self.loss_fn: Loss = get_loss(config)
        self.label_map: LabelMap = MAPS[config.datamodule.dataset_cfg.label_map]
        metrics = lambda: {  # noqa: E731
            "confusion_matrix": MulticlassConfusionMatrix(num_classes=config.model.num_classes),
            "iou": IoU(task="multiclass", num_classes=config.model.num_classes),
            "accuracy": Accuracy(task="multiclass", num_classes=config.model.num_classes),
        }
        self.metrics: dict[Mode, dict[str, torchmetrics.Metric]] = {
            "train": metrics(),
            "val": metrics(),
        }

        torch.set_float32_matmul_precision(self.config.train.float32_matmul_precision)

        self.net: PrithviSegmentationModel = torch.compile(  # type: ignore
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
        self.log_image_metrics(epoch_metrics, mode="train")

    def validation_step(self, batch: S2OSMSample, batch_idx: int) -> torch.Tensor:
        return self._model_step(batch, mode="val")

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            return

        epoch_metrics = self.compute_metrics(mode="val")
        self.log_scalar_metrics(epoch_metrics, mode="val")
        self.log_image_metrics(epoch_metrics, mode="val")

    def predict_step(self, batch: S2OSMSample, batch_idx: int) -> torch.Tensor:
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

    def _model_step(self, batch: S2OSMSample, mode: Mode) -> torch.Tensor:
        x = batch.x
        y = batch.y

        logits = self.net(x)

        if mode == "test":
            return logits

        loss = self.loss_fn(logits, y)

        if self.trainer.sanity_checking:
            return loss

        self.log(f"{mode}/loss", loss)

        self.update_metrics(mode, predictions=torch.argmax(logits, dim=1), labels=y)

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
        class_labels = {i: label for i, label in enumerate(self.label_map)}
        log_confusion_matrix(mode, conf_matrix=computed_metrics["confusion_matrix"], class_labels=class_labels)

        img_idx = random.randint(0, len(self.trainer.train_dataloader.dataset) - 1) if mode == "train" else 0
        dataloader = self.trainer.train_dataloader if mode == "train" else self.trainer.val_dataloaders
        log_segmentation_pred(
            f"{mode}/segmentation",
            model=self,
            dataloader=dataloader,
            idx=img_idx,
            class_labels=class_labels,
            epoch=self.current_epoch,
        )
        # todo this doesn't make sense if the indices are shared...
        log_segmentation_pred(
            f"{mode}/fixed_prediction_dynamics",
            model=self,
            dataloader=dataloader,
            idx=img_idx,
            class_labels=class_labels,
            epoch=self.current_epoch,
        )


# TODO also plot some of the direct input to the model
def log_segmentation_pred(
    plot_name: str,
    model: pl.LightningModule,
    dataloader: torch.utils.data.DataLoader,
    class_labels: dict[int, str],
    idx: int | None = None,
    epoch: int | None = None,
) -> None:
    if idx is None:
        idx = random.randint(0, len(dataloader.dataset) - 1)
    sample: S2OSMSample = dataloader.dataset[idx]
    inp = sample.x.unsqueeze(0).to(model.device)  # (1,c,t,h,w)
    with torch.inference_mode():
        pred = model(inp).squeeze().argmax(dim=0).cpu().numpy()  # (1,n_cls,h,w) -> (h,w)
    orig_img, bbox = load_sentinel_tiff_for_plotting(dataloader.dataset.dataset.sentinel_files[idx], return_bbox=True)
    # crop # fixme? when mode=train, it's a random crop not corresponding to the input
    orig_img = dataloader.dataset.dataset.transform[0](image=orig_img)["image"]
    labels = sample.y.cpu().numpy()
    # Customize colors once https://github.com/wandb/wandb/issues/6637 is resolved.
    masks = {
        "predictions": {"mask_data": pred, "class_labels": class_labels},
        "labels": {"mask_data": labels, "class_labels": class_labels},
    }
    caption = f"{bbox} | Epoch: {epoch} | Sample ID: {idx}"
    wandb.log({f"{plot_name}": wandb.Image(orig_img, masks=masks, caption=caption)})


def log_confusion_matrix(mode: Mode, conf_matrix: np.ndarray, class_labels: dict[int, str]) -> None:
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    normalized_conf_matrix = conf_matrix / row_sums

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(normalized_conf_matrix, cmap="Blues", norm=Normalize(vmin=0, vmax=normalized_conf_matrix.max()))
    fig.colorbar(cax)

    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_xticks((ticks := np.arange(len(class_labels))))
    ax.set_yticks(ticks)
    ax.set_xticklabels((label_texts := list(class_labels.values())), rotation=45)
    ax.set_yticklabels(label_texts)
    plt.tight_layout()

    for (i, j), val in np.ndenumerate(normalized_conf_matrix):
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    wandb.log({f"{mode}/confusion_matrix": wandb.Image(plt.imread(buf))})
    plt.close(fig)


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
            filename=config.train.run_name + "_{epoch:02d}_{val/loss:.2f}_{step}",
        ),
        # pl.callbacks.BackboneFinetuning(), might be helpful?
    ]
    callbacks += [pl.callbacks.LearningRateMonitor(logging_interval="step")] if config.train.use_wandb_logger else []
    logger: WandbLogger | bool = (
        WandbLogger(  # todo use wandb key from env for team entity
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
        wandb.watch(model, log="all", log_freq=10)  # todo set lower after debugging
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
    parser.add_argument("--type", type=str, default="train", help="[train, debug, overfit, ...]. Default: train")
    parser.add_argument("--model", type=str, default="base", help="Model prests.")
    parser.add_argument("--bs", type=int, default=None, help="batch size.")
    parser.add_argument("--aoi", type=str, default=None, help=f"one of {list(AOIs)}")
    parser.add_argument("--labels", type=str, default=None, help=f"one of {list(MAPS)}")
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
    config.datamodule.dataset_cfg.aoi = args.aoi or config.datamodule.dataset_cfg.aoi
    config.datamodule.dataset_cfg.label_map = args.labels or config.datamodule.dataset_cfg.label_map
    config.model.num_classes = len(MAPS[config.datamodule.dataset_cfg.label_map])
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