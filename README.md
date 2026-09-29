# Iceberg Alley SAR

Detect icebergs off the coast of Newfoundland & Labrador in Sentinel-1 SAR imagery. Show the results on a CesiumJS globe with a timeline.

> Work in progress. The full write-up comes in Milestone 4.

## Setup

Requires Python 3.11+.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env         # then fill in Earthdata credentials
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

## Milestone 2: CFAR detection

```powershell
python -m iceberg_sar.cli detect data\raw\<product_dir>              # CFAR on HV (config default)
python -m iceberg_sar.cli detect data\raw\<product_dir> --band HH    # compare against HH
```

Method (parameters in `config.yaml` under `cfar` and `detections`):
1. **CA-CFAR** on linear intensity. The clutter level is the mean of a background ring (41×41 window minus an 11×11 guard). A pixel is a detection if it is brighter than `alpha` × that mean. `alpha` comes from `pfa` under a gamma speckle model: the pixel/ring-mean ratio is F(2L, 2NL) distributed. The ENL `L` is estimated from the scene (`enl: auto`). Masked pixels are excluded from the ring.
2. **Connected components** (8-connected), 2–2000 px. Components within 200 m of land, pack ice, or nodata are dropped.
3. **Sea-ice strip rejection.** Each target is grown to the 3×3-smoothed region more than 4 dB above background. If that structure is longer than 300 m, the target sits on a strip of loose ice rather than being a berg or ship.

Outputs in `data/outputs/detections/`:
- `<product>_<BAND>_detections.geojson`: EPSG:4326 points with `id`, `scene_id`, `timestamp`, `lat`/`lon`, `area_px`, `area_m2`, `extent_m`, `structure_m`, `peak_db_*`/`mean_db_*` per band, `background_db`, `contrast_db`, `incidence_deg`, `distance_to_ice_km` (to the edge of the buffered pack-ice mask; NaN if the scene has no pack ice).
- `<product>_<BAND>_detections.png`: dB quicklook with detections circled.
- `<product>_<BAND>_detections.json`: run summary (ENL, threshold, count).

Validation against the North American Ice Service (IIP + Canadian Ice Service) iceberg chart:

```powershell
python -m iceberg_sar.cli ground-truth 2025-05-08     # downloads daily chart GIFs to data/ground_truth/
python -m iceberg_sar.cli validate data\raw\<product_dir> --counts validation\nais_20250508_counts.csv
```

The chart gives icebergs per 1-degree square, and it is only published as an image. For each new date, transcribe the squares covering the scene into a CSV like `validation/nais_20250508_counts.csv`. Findings: [docs/milestone2-cfar-findings.md](docs/milestone2-cfar-findings.md).

## Layout

- `config.yaml`: all pipeline parameters.
- `aoi.geojson`: area of interest (EPSG:4326).
- `src/iceberg_sar/`: pipeline code.
- `data/`: raw, interim, and output data. It is gitignored.
- `viewer/`: CesiumJS front end.
- `notebooks/`: exploration only.

## Data terms

Sentinel-1 data is from Copernicus via ASF.
