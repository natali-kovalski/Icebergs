# Iceberg Alley SAR

Detect icebergs off the coast of Newfoundland & Labrador in Sentinel-1 SAR imagery. Classify each detection as iceberg or ship, and show the results on a CesiumJS globe with a timeline.

> Work in progress. The full write-up comes in Milestone 5.

## Setup

Requires Python 3.11+.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"        # add ",ml" for PyTorch + scikit-learn (Milestone 3)
copy .env.example .env         # then fill in Earthdata and Kaggle credentials
python -m iceberg_sar.cli --help
python -m iceberg_sar.cli init-dirs
```

## Milestone 1: first masked scene

One-time setup: log in at https://urs.earthdata.nasa.gov, go to Applications > Authorized Apps, and approve the ASF HyP3 app. Without this, HyP3 refuses API logins.

```powershell
python -m iceberg_sar.cli search --pol HH+HV        # list scenes; footprints -> data/outputs/
python -m iceberg_sar.cli order S1A_IW_GRDH_1SDH_20250508T094925_20250508T094950_059101_075526_EB40
python -m iceberg_sar.cli download                  # waits for HyP3 (often 30-60 min), unzips to data/raw/
python -m iceberg_sar.cli land-mask                 # one-time ~900 MB OSM download, clipped to the AOI
python -m iceberg_sar.cli preprocess data\raw\<product_dir>
```

Outputs:
- `data/interim/<product>/<product>_<POL>_masked.tif`: linear power with land (+500 m), out-of-AOI, and nodata set to NaN. This is the CFAR input.
- `data/outputs/quicklooks/<product>_<POL>_db.png`: dB quicklook. Masked areas are transparent.

Land data: (c) OpenStreetMap contributors, ODbL, via osmdata.openstreetmap.de.

## Layout

- `config.yaml`: all pipeline parameters.
- `aoi.geojson`: area of interest (EPSG:4326).
- `src/iceberg_sar/`: pipeline code.
- `data/`: raw, interim, and output data. It is gitignored.
- `viewer/`: CesiumJS front end.
- `notebooks/`: exploration only.

## Data terms

Sentinel-1 data is from Copernicus via ASF. The Kaggle Statoil/C-CORE dataset is not redistributed in this repo.
