## Sentinel-2 Landcover Classification (wip)

This repository contains code to train landcover classification models on Sentinel-2 data.
The current pipeline allows for training a U-Net model with an EfficientNet backbone on [Sentinel-2 L2A data from sentinelhub](https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l2a/).
Labels can be fetched from [OpenStreetMap](https://www.openstreetmap.org) all over the world, or from the [CNES Land Cover dataset](https://collections.sentinel-hub.com/cnes-land-cover-map/) for France.
Finetuning Prithvi is WIP, as well as unsupervised pre-training for the U-Net and the Prithvi model.

## Setup

```bash
git clone git@github.com:MaxWolf-01/sentinel2-landcover-classification.git
cd sentinel2-landcover-classification
conda create -n s2lc python="3.10"
conda activate s2lc
pip install -r requirements.txt
```

Before you can run any scripts from the root dir, you will need to:

```bash
export PYTHONPATH="$PYTHONPATH:$(pwd)"
```

To download the training data, run ``src/data/download_sentinel.py`` and ``src/data/download_labels.py`` with the desired parameters.
For more information, run the scripts with the ``--help`` flag or check the documentation in the source code.
Configs for the data download, such as bands, resolution, timespan, evalscripts and pre-defined areas of interest can be found in ``src/configs/data_config.py``.

Example:
```bash
python src/data/download_sentinel.py at  --workers 2 --frequency QS  #  downloads a Sentinel-2 image for Austria every quarter per label
python src/data/download_labels.py at osm-multiclass --workers 4
```
Downloads can be interrupted and resumed by adding the `--resume` flag.
For re-downloading, use the `--overwrite` flag.

Visualize the downloaded data with:
```bash
python src/data/plotting.py at osm-multiclass
```

Once you have the desired data, run the training script with the desired config, e.g.:
```bash
python src/train_segmentation.py at osm-multiclass efficientnet-unet-b5 --bs 32 --epochs 30 --loss-type focal --weighted-loss --name "wandb-run-name"  # disable wandb with "--wandb"
```

Flags like `--type debug` set presets for a batch size of 1, no logger, compiling, etc.

To use Prithvi, download the pretrained weights of the Prithvi foundation model from Hugging Face:

```bash
curl -L "https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M/resolve/main/Prithvi_100M.pt?download=true" -o "weights/Prithvi_100M.pt"
```

### Contributer Infos

Before commiting, make sure to

```bash
pip install pre-commit
pre-commit install
```

Or run `pre-commit run --all-files` to check manually.

To log runs to the wandb team workspace, add the name of the wandb entity in the `.env` file as `WANDB_ENTITY`.

## Acknowledgements

- The repository draws inspiration from the [HLS Foundation](https://github.com/nasa-impact/hls-foundation-os) repository and uses the [IBM NASA Geospatial](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M) model (Prithvi).
- The EfficientNet-Unet implementation is based on the [EfficientNet-PyTorch](https://github.com/zhoudaxia233/EfficientUnet-PyTorch) repository.
