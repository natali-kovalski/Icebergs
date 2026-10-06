# Iceberg Alley SAR

Detect icebergs off the coast of Newfoundland & Labrador in Sentinel-1 SAR imagery. Show the results on a CesiumJS globe with a timeline.

> Work in progress. The full write-up comes in Milestone 4.

## Status

| Milestone | State |
|---|---|
| 0. Setup | Done |
| 1. First masked scene (search, HyP3 RTC, land + pack-ice mask, quicklook) | Done |
| 2. CA-CFAR detection + NAIS chart check | Done (OS-CFAR / K-distribution stretch not done) |
| 3. Vue + CesiumJS viewer | Done |
| Point-level validation (IIP sightings, Sentinel-2) | Done: **78% recall** against same-pass IIP labels, see [docs/validation-findings.md](docs/validation-findings.md) |
| Experimental iceberg vs. ship CNN | Works, but treat it as a weak hint (see below) |
| 4. Portfolio polish (write-up, screenshots, end-to-end `run` command) | Not started |

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
1. **CA-CFAR** on linear intensity. The clutter level is the mean of a background ring (400 m half-width minus a 100 m guard; window sizes are set in metres and converted per scene, e.g. 81×81 minus 21×21 px at 10 m). A pixel is a detection if it is brighter than `alpha` × that mean. `alpha` comes from `pfa` under a gamma speckle model: the pixel/ring-mean ratio is F(2L, 2NL) distributed. The ENL `L` is estimated from the scene (`enl: auto`). Masked pixels are excluded from the ring.
2. **Connected components** (8-connected), at least 2 px and at most 0.8 km². Components within 200 m of land, pack ice, or nodata are dropped.
3. **Sea-ice strip rejection.** Each target is grown to the 3×3-smoothed region more than 4 dB above background. If that structure is longer than 300 m, the target sits on a strip of loose ice rather than being a berg or ship.
4. **Co-pol (HH) check.** The target's HH peak must be at least 6 dB above the HH ring mean (`copol_min_contrast_db`). Icebergs are bright in both polarizations; HV speckle spikes are not. This matters at 10 m: Sentinel-1 IW GRD has ~20 m true resolution, so 10 m pixels are oversampled, speckle grains span ~2×2 px, and the 2 px minimum alone lets them through.

Outputs in `data/outputs/detections/`:
- `<product>_<BAND>_detections.geojson`: EPSG:4326 points with `id`, `scene_id`, `timestamp`, `lat`/`lon`, `area_px`, `area_m2`, `extent_m`, `structure_m`, `peak_db_*`/`mean_db_*` per band, `background_db`, `contrast_db`, `copol_contrast_db`, `incidence_deg`, `distance_to_ice_km` (to the edge of the buffered pack-ice mask; NaN if the scene has no pack ice).
- `<product>_<BAND>_detections.png`: dB quicklook with detections circled.
- `<product>_<BAND>_detections.json`: run summary (ENL, threshold, count).

Count check against the North American Ice Service (IIP + Canadian Ice Service) iceberg chart:

```powershell
python -m iceberg_sar.cli ground-truth 2025-05-08     # downloads daily chart GIFs to data/ground_truth/
python -m iceberg_sar.cli validate data\raw\<product_dir> --counts validation\nais_20250508_counts.csv
```

The chart gives icebergs per 1-degree square, and it is only published as an image. For each new date, transcribe the squares covering the scene into a CSV like `validation/nais_20250508_counts.csv`. Findings: [docs/milestone2-cfar-findings.md](docs/milestone2-cfar-findings.md).

## Point-level validation

The NAIS chart only gives counts per 1° square. For per-berg recall, detections are matched one-to-one (within 500 m, drift-corrected when the truth is not simultaneous) against:

- **IIP Iceberg Sightings Database** (NSIDC G00807, seasons up to 2021). Sightings that IIP analysts marked on the same Sentinel-1 pass have no drift, so they give the cleanest recall. Aircraft sightings are hours later.
- **Sentinel-2 L2A optical targets** (Earth Search COGs): isolated bright objects in open water on the same date.

```powershell
python -m iceberg_sar.cli iip-sightings 2019                                            # download a season
python -m iceberg_sar.cli validate-points data\raw\<product_dir> --truth iip-satellite --no-open-water
python -m iceberg_sar.cli validate-points data\raw\<product_dir> --truth iip-aircraft
python -m iceberg_sar.cli s2-targets data\raw\<product_dir>                             # optical targets for the scene date
python -m iceberg_sar.cli validate-points data\raw\<product_dir> --truth s2
```

**Result (2019-04-29, 10 m RTC):** the detector finds **209 of 268** IIP-labelled bergs (78%), median offset 128 m. Recall is 62% for small bergs (15–60 m), 81% for medium bergs (61–120 m), and 0 of 4 for bergs over 120 m. Precision can't be measured without AIS vessel positions. Aircraft sightings and Sentinel-2 were inconclusive (drift, loose sea ice, bergs inshore). Full write-up: [docs/validation-findings.md](docs/validation-findings.md). Outputs go to `data/outputs/validation/`.

## Milestone 3: Cesium viewer

A Vue 3 + CesiumJS app. The timeline steps through the scenes, and you can click a candidate to see its attributes.

```powershell
python -m iceberg_sar.cli czml        # all *_HV_detections.geojson -> viewer/public/data/iceberg_alley.czml
cd viewer
npm install
npm run dev                           # http://localhost:5173
```

- Each scene stays on the timeline until the next scene starts. The last one stays for `viewer.last_interval_days`.
- Candidates within `viewer.near_ice_km` (default 5 km) of the pack-ice mask are grey and flagged `near_ice`. They are likely ice floes (88% of the 2025-05-02 detections). The **Near-ice candidates** checkbox hides them. Nothing is deleted.
- Open-water candidates are coloured by `contrast_db` and sized by `structure_m`. The scene footprint and pack-ice mask can be toggled.
- The basemap is Esri World Imagery, so no Cesium ion token is needed. You can set `VITE_CESIUM_ION_TOKEN` in `viewer/.env` to enable ion services.
- `npm run build` writes a static site to `viewer/dist/` that any static host can serve.

## Experimental: iceberg vs. ship classifier

A small CNN trained on the Kaggle [Statoil/C-CORE Iceberg Classifier Challenge](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge) chips (1,604 labelled 75×75 HH/HV chips in dB). It adds `iceberg_prob` to each detection. **Treat it as a weak hint, not a label** (see below).

```powershell
pip install -e .[ml]                           # torch, scikit-learn, py7zr
# put train.json from Kaggle in data/kaggle/ (Kaggle terms: don't redistribute)
python -m iceberg_sar.cli train-classifier     # 5-fold CV for each variant, ~1 min each on a GPU
python -m iceberg_sar.cli classify data\raw\<product_dir>
```

Output: `<scene>_HV_classified.geojson` (detections + `iceberg_prob`, `iceberg_prob_hhhv`, `iceberg_prob_hh`, chip diagnostics) and `<scene>_HV_classified_chips.png` (most iceberg-like / ship-like / uncertain chips).

**Matching our chips to Kaggle.** We measured both datasets instead of assuming:

- *Resolution matches.* Kaggle chips have the same speckle correlation (lag-1 ≈ 0.6) and ENL (≈ 4) as our 10 m RTC, and 1,470 of 1,471 incidence angles fall in the IW swath (29–46°). So chips are cut at native 10 m, without resampling.
- *Radiometry doesn't match, so we correct it.* HyP3 gives gamma0, Kaggle is sigma0, so we multiply by cos(incidence). HyP3 also removes the HV thermal noise floor (calm sea HV ≈ −40 dB), while Kaggle keeps it (−24 to −29 dB, depending on incidence). We add speckled noise until each chip's HV background matches the Kaggle level at that incidence. After this, open-water chips match Kaggle in background level, ENL and speckle correlation, in both bands.
- Kaggle's missing incidence angles are all ships. They are imputed to the mean so the model can't learn that leak.

**Results.** Two variants: `hhhv` (both bands, fills `iceberg_prob`) and `hh` (HH only, a control without the noise-floor assumption).

| | Kaggle 5-fold out-of-fold log loss | Accuracy |
|---|---|---|
| hhhv | 0.229 | 0.90 |
| hh | 0.266 | 0.89 |
| always predict the class prior | 0.691 | 0.53 |

On Kaggle, the probabilities are well calibrated: of chips scored 0.8–1.0, 96% are icebergs, and of chips scored 0–0.2, 3% are.

**How much to trust it on our scenes.** There is no ship ground truth (no AIS) for our dates, so these are consistency checks, not accuracy:

- *The model is unsure on our data.* On Kaggle, 79% of chips get p < 0.2 or > 0.8. On our 343 open-water detections, only 26% do.
- *The two variants agree on 70% of open-water detections.* Only 32 detections are confidently classified by both, 14 as icebergs and 18 as ships.
- *Size gap:* most of our targets are 2–10 px, while Kaggle targets have a median of ~74 px above background. Small targets are out of distribution.
- *Shifting the HV noise floor by ±2 dB* changes p by 0.07 on average and flips 9–13% of labels.
- *Near pack ice* (within 5 km), chips contain ice texture that Kaggle never shows. p is high there (0.79), but that says nothing useful.
- Visually the ranking makes sense: irregular blobs bright in both bands score high, and sharp point targets with azimuth streaks score low.

Use `iceberg_prob` as a sort key for review, or trust it only where both variants agree with high confidence. Validating it properly needs AIS ship positions for the same passes.

## Layout

- `config.yaml`: all pipeline parameters.
- `aoi.geojson`: area of interest (EPSG:4326).
- `src/iceberg_sar/`: pipeline code (`classify/` is the experimental CNN).
- `data/`: raw, interim, output, ground-truth and model data. It is gitignored.
- `docs/`: findings write-ups ([CFAR](docs/milestone2-cfar-findings.md), [validation](docs/validation-findings.md)) and figures.
- `validation/`: hand-transcribed NAIS chart counts for 2025-05-02, 05-08 and 05-14.
- `viewer/`: Vue + CesiumJS front end (`src/scene.ts` holds the Cesium logic).
- `tests/`: pytest suite (`pytest`).
- `notebooks/`: exploration only.

## Data terms

- Sentinel-1 data is from Copernicus via ASF. Sentinel-2 L2A is from Copernicus via Earth Search (Element 84).
- Iceberg charts: North American Ice Service (International Ice Patrol + Canadian Ice Service).
- Iceberg sightings: International Ice Patrol Iceberg Sightings Database, NSIDC G00807.
- Land polygons: (c) OpenStreetMap contributors, ODbL.
- Kaggle Statoil/C-CORE chips: used under the competition terms and not redistributed.
