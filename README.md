## Setup

```bash
git clone git@github.com:MaxWolf-01/sentinel2-landcover-classification.git
cd sentinel2-landcover-classification
conda create -n s2lc python="3.10"
conda activate s2lc
pip install -r requirements.txt
```

Download the pretrained weights of the Prithvi foundation model from Hugging Face:

```bash
curl -L "https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M/resolve/main/Prithvi_100M.pt?download=true" -o "weights/Prithvi_100M.pt"
```

Before you can run any scripts from the root dir, you will need to:

```bash
export PYTHONPATH="$PYTHONPATH:$(pwd)"
```

To download the training data, run:

```bash
python src/data/download_s2_osm_data.py  # specify area of interest via "--aoi". List available and default values via "--help"
```

Once that is done, run the training, eg.:

```bash
python src/train_simple_fintune.py  # specify config via "--config". List available and default values via "--help"
```

Configs like `debug` can be used to debug with a lower batch size, no logger, etc.
Give the run name a prefix via `--name` to make it easier to find in the logs / wandb.

### Developer Infos

Before commiting, make sure to

```bash
pip install pre-commit
pre-commit install
```

Or run `pre-commit run --all-files` to check manually.

To log runs to the wandb team workspace, add the name of the wandb entity in the `.env` file as `WANDB_ENTITY`.

## Citations

```bibtex
@software{HLS_Foundation_2023,
    author = {Jakubik, Johannes and Chu, Linsong and Fraccaro, Paolo and Bangalore, Ranjini and Lambhate, Devyani and Das, Kamal and Oliveira Borges, Dario and Kimura, Daiki and Simumba, Naomi and Szwarcman, Daniela and Muszynski, Michal and Weldemariam, Kommy and Zadrozny, Bianca and Ganti, Raghu and Costa, Carlos and Watson, Campbell and Mukkavilli, Karthik and Roy, Sujit and Phillips, Christopher and Ankur, Kumar and Ramasubramanian, Muthukumaran and Gurung, Iksha and Leong, Wei Ji and Avery, Ryan and Ramachandran, Rahul and Maskey, Manil and Olofossen, Pontus and Fancher, Elizabeth and Lee, Tsengdar and Murphy, Kevin and Duffy, Dan and Little, Mike and Alemohammad, Hamed and Cecil, Michael and Li, Steve and Khallaghi, Sam and Godwin, Denys and Ahmadi, Maryam and Kordi, Fatemeh and Saux, Bertrand and Pastick, Neal and Doucette, Peter and Fleckenstein, Rylie and Luanga, Dalton and Corvin, Alex and Granger, Erwan},
    doi = {10.57967/hf/0952},
    month = aug,
    title = {{HLS Foundation}},
    repository-code = {https://github.com/nasa-impact/hls-foundation-os},
    year = {2023}
}
```

```bibtex
@misc{ibm_nasa_geospatial_2023,
    author = { {IBM NASA Geospatial} },
    title = { Prithvi-100M (Revision 489bb56) },
    year = 2023,
    url = { https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M },
    doi = { 10.57967/hf/0952 },
    publisher = { Hugging Face }
}
```
