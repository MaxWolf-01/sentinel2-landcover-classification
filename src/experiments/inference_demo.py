import typing
from pathlib import Path

import torch
from matplotlib import pyplot as plt

from configs.label_mappings import MULTICLASS_MAP
from configs.paths import OUT_DIR, ROOT_DIR
from data.s2osmdatamodule import S2OSMDatamodule
from src.train_simple_finetune import PrithviSegmentationFineTuner
import lightning.pytorch as pl


class CustomWriter(pl.callbacks.BasePredictionWriter):
    def __init__(self, output_dir: Path, write_interval: typing.Literal["batch", "epoch", "batch_and_epoch"]) -> None:
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            prediction: typing.Any,
            batch_indices: typing.Sequence[int] | None,
            batch: typing.Any,
            batch_idx: int,
            dataloader_idx: int,
    ) -> None:
        torch.save(prediction, self.output_dir / f"batch_{batch_idx}.pt")


ckpt_path = "/home/max/repos/code/sentinel2-landcover-classification/ckpts/simple-prithvi-finetune/last-v10.ckpt"
model = PrithviSegmentationFineTuner.load_from_checkpoint(ckpt_path)

dm = S2OSMDatamodule(cfg=model.config.datamodule)
dm.setup()
predict_dataloader = dm.val_dataloader()  # todo add test set and pass dm directly
OUT_DIR.mkdir(exist_ok=True, parents=True)
pred_writer = CustomWriter(output_dir=OUT_DIR, write_interval="batch")
trainer = pl.Trainer(callbacks=[pred_writer], default_root_dir=ROOT_DIR)
trainer.predict(model=model, dataloaders=predict_dataloader)

if __name__ == '__main__':
    from src.plotting import plot_sentinel_mask_and_pred, load_pred_batch_for_plotting

    batch = load_pred_batch_for_plotting(OUT_DIR / "batch_0.pt")
    sentinel_paths = dm.val.sentinel_files
    mask_paths = dm.val.osm_files
    for (sentinel, mask, pred_img) in zip(sentinel_paths, mask_paths, batch):
        plot_sentinel_mask_and_pred(sentinel=sentinel, mask=mask, pred_img=pred_img, label_map=MULTICLASS_MAP)
        plt.show()
