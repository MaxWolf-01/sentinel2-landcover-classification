from __future__ import annotations

import argparse
import dataclasses
import io
import os
import pprint
import random
from typing import Any, Literal

import dotenv
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import torchmetrics
import wandb
from lightning.pytorch.loggers import WandbLogger
from matplotlib.colors import Normalize
from numpy import typing as npt
from torch import nn
from torchmetrics import Accuracy, JaccardIndex as IoU
from torchmetrics.classification import MulticlassConfusionMatrix

import configs.segmentation as cfg
from configs.data_config import LABEL_MAPS, LabelMap
from data.download_data import AOIs
from losses import Loss, LossType, get_loss
from lr_schedulers import LRSchedulerType, get_lr_scheduler
from plotting import load_sentinel_tiff_for_plotting
from configs.paths import CKPT_DIR, LOG_DIR, ROOT_DIR
from configs.segmentation import Config
from data.calculate_dataset_statistics import calculate_mean_std
from data.s2osm_datamodule import S2OSMDatamodule
from data.s2osm_dataset import S2OSMDataset, S2OSMSample
from utils import get_logger, get_unique_run_name

script_logger = get_logger(__name__)

Mode = Literal["train", "val", "test"]


class SegmentationModule(pl.LightningModule):
    def __init__(self, config: Config, optuna_trial: optuna.Trial | None = None) -> None:
        super().__init__()
        self.config: Config = config
        self.save_hyperparameters(logger=False, ignore=["net", "optuna_trial"])
        self.net: nn.Module = config.get_model()
        self.loss_fn: Loss = get_loss(config)
        label_map: LabelMap = LABEL_MAPS[config.datamodule.dataset_cfg.label_map]
        self.class_labels: dict[int, str] = {i: label for i, label in enumerate(label_map)}
        task: Literal["binary", "multiclass"] = (
            "binary" if "binary" in config.datamodule.dataset_cfg.label_map else "multiclass"
        )
        metrics = lambda: {  # noqa: E731
            "confusion_matrix": MulticlassConfusionMatrix(
                num_classes=config.num_classes,
                ignore_index=0 if config.train.masked_loss else None,
                normalize="true",
            ),
            "iou": IoU(task=task, num_classes=config.num_classes),
            "accuracy": Accuracy(task=task, num_classes=config.num_classes),
            "f1": torchmetrics.F1Score(task=task, num_classes=config.num_classes),
            # "f2": torchmetrics.FBetaScore(task=task, num_classes=config.num_classes, beta=2.),
        }
        self.metrics: dict[Mode, dict[str, torchmetrics.Metric]] = {
            "train": metrics(),
            "val": metrics(),
        }
        torch.set_float32_matmul_precision(self.config.train.float32_matmul_precision)

#        self.net: nn.Module = torch.compile(  # type: ignore
#            model=self.net,
#            mode=config.train.compile_mode,
#            fullgraph=config.train.compile_fullgraph,
#            disable=config.train.compile_disable,
#        )

    def on_fit_start(self) -> None:
        if isinstance(self.logger, WandbLogger):
            self.logger.log_hyperparams(dataclasses.asdict(self.config))
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

        self.update_metrics(mode, predictions=logits.argmax(dim=1), labels=y)

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

    def log_image_metrics(self, computed_metrics: dict[str, np.ndarray], mode: Mode) -> None:
        if not isinstance(self.logger, WandbLogger):
            return
        confusion_matrix = computed_metrics["confusion_matrix"]
        class_labels = self.class_labels
        if self.config.train.masked_loss:  # remove background class from confusion matrix
            confusion_matrix = confusion_matrix[1:, 1:]
            class_labels = {k: v for k, v in class_labels.items() if k != 0}
        log_confusion_matrix(mode, conf_matrix=confusion_matrix, class_labels=class_labels)
        # segmentation predictions
        dataset: S2OSMDataset = self.trainer.val_dataloaders.dataset.dataset
        if mode == "train":
            center_crop_transform = dataset.transform
            dataset = self.trainer.train_dataloader.dataset.dataset
            dataset.transform = center_crop_transform
        self._log_segmentation_pred(
            f"{mode}/segmentation",
            dataset=dataset,
            idx=None,  # random
            class_labels=self.class_labels,
        )
        self._log_segmentation_pred(
            f"{mode}/fixed_prediction_dynamics",
            dataset=dataset,
            idx=0,
            class_labels=self.class_labels,
        )

    def _log_segmentation_pred(
        self,
        plot_name: str,
        dataset: S2OSMDataset,
        class_labels: dict[int, str],
        idx: int | None = None,
    ) -> None:
        if idx is None:
            idx = random.randint(0, len(dataset) - 1)
        sample: S2OSMSample = dataset[idx]
        inp = sample.x.unsqueeze(0).to(self.device)  # (1,c,t,h,w)
        with torch.inference_mode():
            pred = self(inp).squeeze().argmax(dim=0).cpu().numpy()  # (1,n_cls,h,w) -> (h,w)

        orig_img, bbox = load_sentinel_tiff_for_plotting(dataset.sentinel_files[idx], return_bbox=True)
        orig_img = dataset.transform[0](image=orig_img)["image"]  # crop only, no normalization

        labels = sample.y.cpu().numpy()

        # Customize colors once https://github.com/wandb/wandb/issues/6637 is resolved.
        masks = {
            "predictions": {"mask_data": pred, "class_labels": class_labels},
            "labels": {"mask_data": labels, "class_labels": class_labels},
        }
        caption = f"{bbox} | Epoch: {self.current_epoch} | Sample ID: {idx}"
        wandb.log({f"{plot_name}": wandb.Image(orig_img, masks=masks, caption=caption)})


def log_confusion_matrix(mode: Mode, conf_matrix: np.ndarray, class_labels: dict[int, str]) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.colorbar(ax.matshow(conf_matrix, cmap="Blues", norm=Normalize(vmin=0, vmax=conf_matrix.max())))
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_xticks((ticks := np.arange(len(class_labels))))
    ax.set_yticks(ticks)
    ax.set_xticklabels((label_texts := list(class_labels.values())), rotation=45)
    ax.set_yticklabels(label_texts)
    plt.tight_layout()

    for (i, j), val in np.ndenumerate(conf_matrix):
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    wandb.log({f"{mode}/confusion_matrix": wandb.Image(plt.imread(buf))})
    plt.close(fig)


def train(config: Config, trial: optuna.Trial | None = None) -> None:
    model: SegmentationModule = SegmentationModule(config, optuna_trial=trial)
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
        logger.watch(model, log="all", log_freq=30)
    trainer: pl.Trainer = pl.Trainer(
        default_root_dir=ROOT_DIR,
        max_epochs=config.train.max_epochs,
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
    parser.add_argument("--model", type=str, required=True, help=f"Model type. One of: {list(cfg.ModelName)}")
    parser.add_argument("--labels", type=str, default=None, help=f"one of {list(LABEL_MAPS)}.")
    parser.add_argument("--type", type=str, default="train", help="[train, debug, overfit, ...]. Default: train")
    parser.add_argument("--loss-type", type=str, default=None, help=f"Available: {list(LossType)}")
    parser.add_argument("--lr-scheduler", type=str, default=None, help=f"Available: {list(LRSchedulerType)}")
    parser.add_argument("--bs", type=int, default=None, help="batch size.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs. -1 = infinite")
    parser.add_argument("--log-interval", type=int, default=None, help="Log interval. Default: 50")
    parser.add_argument("--aoi", type=str, default=None, help=f"one of {list(AOIs)}")
    parser.add_argument("--recompute-mean-std", action="store_true", help="Recompute dataset mean and std.")
    parser.add_argument("--name", type=str, default=None, help="run name prefix. Default: None")
    parser.add_argument("--wandb", action="store_true", help="DISABLE wandb logging.")
    parser.add_argument(  # list of tags
        "--tags", nargs="+", default=[], help="Tags for wandb. Default: None. Example usage: --tags t1 t2 t3"
    )
    parser.add_argument("--no-compile", action="store_true", help="Compile model. Default: True")
    args = parser.parse_args()

    config: Config = cfg.BASE_CONFIG(model_name=args.model)
    config = cfg.set_run_type(config, args.type)
    config.datamodule.dataset_cfg.label_map = args.labels or config.datamodule.dataset_cfg.label_map
    config.num_classes = len(LABEL_MAPS[config.datamodule.dataset_cfg.label_map])
    config.model_name = args.model or config.model_name
    config.datamodule.dataset_cfg.aoi = args.aoi or config.datamodule.dataset_cfg.aoi
    config.datamodule.batch_size = args.bs or config.datamodule.batch_size
    config.train.max_epochs = args.epochs or config.train.max_epochs
    config.train.log_interval = args.log_interval or config.train.log_interval
    config.train.compile_disable = args.no_compile or config.train.compile_disable
    config.train.use_wandb_logger = False if args.wandb else config.train.use_wandb_logger
    config.train.tags.extend(args.tags)
    config.train.run_name = get_unique_run_name(name=args.name, postfix=config.train.project_name)
    dotenv.load_dotenv()
    config.train.wandb_entity = os.getenv("WANDB_ENTITY")

    script_logger.info(f"Using config in mode'{args.type}':\n{pprint.pformat(dataclasses.asdict(config))}")

    if args.recompute_mean_std:
        script_logger.info("Recomputing mean and std...")
        dataset = S2OSMDataset(config.datamodule.dataset_cfg)
        dim = (0, 2, 3)  # (0, 1, 3, 4) in case we have time dimension
        calculate_mean_std(dataset, dim=dim, save_path=dataset.data_dirs.base_path / "mean_std.pt")

    pl.seed_everything(config.train.seed)  # after creating run_name
    if args.type == "tune":
        tune()
    else:
        train(config=config)


if __name__ == "__main__":
    main()
