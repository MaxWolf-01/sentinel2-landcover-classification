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
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from torchmetrics.classification import MulticlassConfusionMatrix
from torchmetrics import JaccardIndex as IoU
from torchmetrics import Accuracy
from torch import nn
from PIL import Image

from configs.label_mappings import GENERAL_MAP, get_idx_to_label_map
from plotting import load_sentinel_tiff_for_plotting
from src.configs.paths import LOG_DIR, ROOT_DIR, CKPT_DIR
from src.configs.simple_finetune import Config
from src.data.s2osmdatamodule import S2OSMDatamodule
from src.data.s2osmdataset import S2OSMSample
from src.modules.base_segmentation_model import PrithviSegmentationModel, ConvTransformerTokensToEmbeddingNeck, FCNHead
import src.configs.simple_finetune as cfg
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
            ),
        )
        self.loss_fn = nn.CrossEntropyLoss()  # TODO meaningful label smoothing?
        self.metrics: dict[Mode, dict[str, torchmetrics.Metric]] = {
            "train": {
                #     "accuracy": torchmetrics.Accuracy(),
            },
            "val": {
                "confusion_matrix": MulticlassConfusionMatrix(num_classes=config.model.num_classes),
                "iou": IoU(task="multiclass", num_classes=config.model.num_classes),
                "accuracy": Accuracy(task="multiclass", num_classes=config.model.num_classes),
            },
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
        epoch_metrics = compute_metrics(self.metrics, mode="train")
        self.log_scalar_metrics(epoch_metrics, mode="train")
        self.log_image_metrics(epoch_metrics, mode="train")

    def validation_step(self, batch: S2OSMSample, batch_idx: int) -> torch.Tensor:
        logits = self(batch.x)
        predictions = torch.argmax(logits, dim=1)
        labels = batch.y

        for name, metric in self.metrics["val"].items():
            metric.update(predictions, labels)

        return self._model_step(batch, mode="val")

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            return

        epoch_metrics = compute_metrics(self.metrics, mode="val")
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

        if mode == "test":
            return logits

        loss = self.loss_fn(logits, y)

        if self.trainer.sanity_checking:
            return loss

        self.log(f"{mode}/loss", loss)

        if not isinstance(self.logger, WandbLogger):
            return loss

        return loss

    def log_scalar_metrics(self, computed_metrics: dict[str, npt.NDArray], mode: Mode) -> None:
        for metric_name, metric_value in computed_metrics.items():
            if metric_value.prod(metric_value.shape) == 1:
                self.log(f"{mode}/{metric_name}", metric_value.item())

    def log_image_metrics(self, computed_metrics: dict[str, npt.NDArray], mode: Mode) -> None:
        if not isinstance(self.logger, WandbLogger) and (not self.config.train.log_img_in_train and mode == "train"):
            return

        log_confusion_matrix(computed_metrics["confusion_matrix"], get_idx_to_label_map(GENERAL_MAP))

        img_idx = random.randint(0, len(self.trainer.train_dataloader.dataset) - 1) if mode == "train" else 0
        dataloader = self.trainer.train_dataloader if mode == "train" else self.trainer.val_dataloaders
        class_labels = get_idx_to_label_map(GENERAL_MAP)
        log_segmentation_pred(
            f"{mode}/segmentation",
            model=self,
            dataloader=dataloader,
            idx=img_idx,
            class_labels=class_labels,
            epoch=self.current_epoch,
        )
        log_segmentation_pred(
            f"{mode}/fixed_prediction_dynamics",
            model=self,
            dataloader=dataloader,
            idx=img_idx,
            class_labels=class_labels,
            epoch=self.current_epoch,
        )


def compute_metrics(metrics: dict[Mode, dict[str, torchmetrics.Metric]], mode: Mode) -> dict[str, npt.NDArray]:
    computed_metrics = {}
    for metric_name, metric in metrics[mode].items():
        computed_value = metric.compute()
        computed_metrics[metric_name] = computed_value.cpu().numpy()
        metric.reset()
    return computed_metrics


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
    orig_img, bbox = load_sentinel_tiff_for_plotting(dataloader.dataset.sentinel_files[0], return_bbox=True)
    orig_img = dataloader.dataset.transform[0](image=orig_img)["image"]  # center crop
    labels = sample.y.cpu().numpy()
    # Customize colors once https://github.com/wandb/wandb/issues/6637 is resolved.
    masks = {
        "predictions": {"mask_data": pred, "class_labels": class_labels},
        "labels": {"mask_data": labels, "class_labels": class_labels},
    }
    caption = f"{bbox} | Epoch: {epoch} | Sample ID: {idx}"
    wandb.log({f"{plot_name}": wandb.Image(orig_img, masks=masks, caption=caption)})


def log_confusion_matrix(conf_matrix: np.ndarray, class_labels: dict[int, str]) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(conf_matrix, cmap="Blues", norm=Normalize(vmin=0, vmax=np.max(conf_matrix)))
    fig.colorbar(cax)

    ax.set_title("Confusion Matrix", pad=20)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    label_names = list(class_labels.values())

    ax.set_xticks(np.arange(len(label_names)))
    ax.set_yticks(np.arange(len(label_names)))
    ax.set_xticklabels(label_names)
    ax.set_yticklabels(label_names)
    plt.xticks(rotation=45)

    for i in range(len(conf_matrix)):
        for j in range(len(conf_matrix[i])):
            ax.text(j, i, str(conf_matrix[i, j]), ha="center", va="center", color="black")
    fig.canvas.draw()

    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # Adjust for RGBA
    image = Image.fromarray(image[..., :3])  # Convert to RGB  # TODO we can remove the PIL image conversion

    wandb.log({"confusion_matrix": wandb.Image(image)})
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
    logger: WandbLogger | None = (
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
    dotenv.load_dotenv()
    configs: dict[str, Config] = {
        "base": cfg.CONFIG,
        "debug": cfg.DEBUG_CFG,
        "overfit": cfg.OVERFIT_CFG,
        "tune": ...,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="base", help=f"Specify config. Default: base; Available: {list[configs]}"
    )
    parser.add_argument("--name", type=str, default=None, help="Specify run name prefix. Default: None")
    parser.add_argument("--wandb", action="store_true", default=False, help="Force wandb logging. Default: False")
    # list of tags
    parser.add_argument(
        "--tags", nargs="+", default=None, help="Tags for wandb. Default: None. Example usage: --tags t1 t2 t3"
    )
    parser.add_argument("--compile", action="store_true", default=True, help="Compile model. Default: True")
    args = parser.parse_args()
    cfg_key: str = args.config or "base"
    config: Config = configs[cfg_key]
    config.model.num_classes = len(GENERAL_MAP)
    config.train.compile_disable = not args.compile
    config.train.use_wandb_logger = config.train.use_wandb_logger or args.wandb
    config.train.tags.extend(args.tags or [])
    config.train.run_name = get_unique_run_name(name=args.name, postfix=config.train.project_name)
    config.train.wandb_entity = os.getenv("WANDB_ENTITY")

    script_logger.info(f"USING CONFIG: '{cfg_key}':\n{pprint.pformat(dataclasses.asdict(config))}")

    pl.seed_everything(config.train.seed)  # after creating run_name
    if config == "tune":
        tune()
    else:
        train(config=config)


if __name__ == "__main__":
    main()
