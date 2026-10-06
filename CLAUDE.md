# Iceberg Alley SAR: Iceberg Detection off Newfoundland

## What this project is
Detect icebergs off the coast of Newfoundland & Labrador in Sentinel-1 SAR imagery with a CFAR detector, and display the results on an interactive **CesiumJS** globe with a timeline.

This is a portfolio project. It should show hands-on SAR and maritime-domain skills to remote sensing / ocean-tech employers in St. John's, NL. Clarity, reproducibility, and a polished demo matter more than state-of-the-art accuracy.

## About me (the developer)
- Geospatial engineer, 15+ years: photogrammetry (DEM, orthophoto, stereo), 3D reconstruction, geospatial AI.
- Strong in: Python (rasterio, Shapely, GeoPandas, FastAPI), PostGIS, PyTorch, CesiumJS, Vue, Docker, AWS.
- **New to SAR.** Explain SAR-specific concepts briefly when they come up (backscatter, speckle, polarization, incidence angle, CFAR, dB vs. linear). Don't explain general Python or GIS basics.
- I prefer small, hands-on steps that produce a visible result quickly over long setup phases.

## How to work with me (instructions for Claude Code)
- Work one milestone at a time. Before starting a milestone, confirm the plan in a few lines. After finishing, show how to run it and what to look at.
- Keep modules small and testable. Use type hints. Put parameters in `config.yaml` instead of hardcoding them.
- Never commit raw data, credentials, or model weights. Use `.env` for secrets and keep `data/` in `.gitignore`.
- Default CRS for outputs is EPSG:4326 (GeoJSON / CZML). For metric processing, use the scene's native projection or UTM 21N/22N (EPSG:32621 / 32622).
- When a choice has a well-known trade-off (e.g. CFAR variant, window sizes), pick a sensible default, say why in one line, and move on.

## Tech stack
- **Python 3.11+**: numpy, scipy, rasterio, shapely, geopandas, scikit-image, pyyaml, typer (CLI)
- **Data access**: `asf_search` (find scenes), `hyp3_sdk` (order RTC processing)
- **Viz**: CesiumJS (optionally in a small Vue app), CZML for time-dynamic detections
- **Optional**: FastAPI to serve results, Docker for reproducibility

## Data sources
1. **Sentinel-1 SAR imagery** via ASF Vertex (https://search.asf.alaska.edu). Requires a free NASA Earthdata login.
   - Product: GRD, processed with **HyP3 RTC** into analysis-ready GeoTIFFs (gamma0). Request **power (linear)** scale and convert to dB where needed. Note the HyP3 monthly processing credit limit.
   - **Check the polarization of each scene.** Depending on area and date, it may be IW (VV/VH) or EW (HH/HV). Prefer HH/HV: the CFAR and sea-ice mask are tuned on HV.
   - Fallback source: Copernicus Data Space Ecosystem.
2. **Area of interest**: "Iceberg Alley", the NE Newfoundland coast, roughly Bonavista to St. Anthony, extending offshore. Store the AOI as `aoi.geojson`.
3. **Season**: April–June has the most icebergs. Start with 2–3 scenes from one spring.
4. **Ground truth / validation**: Canadian Ice Service iceberg charts and the International Ice Patrol iceberg data for matching dates. These are for visual and approximate validation, not pixel-exact labels.
5. **Land mask**: OSM land polygons or GSHHG coastline, buffered ~500 m to remove coastal clutter.
6. **Point-level validation**: IIP Iceberg Sightings Database (NSIDC G00807, seasons up to 2021) and Sentinel-2 L2A optical targets (Earth Search COGs). Because IIP sightings end in 2021, per-berg validation uses spring 2019 scenes.
7. **Experimental classifier data**: Kaggle Statoil/C-CORE Iceberg Classifier Challenge chips (`data/kaggle/train.json`, don't redistribute).

## Repository structure
```
iceberg-alley-sar/
├── CLAUDE.md
├── README.md
├── config.yaml
├── aoi.geojson
├── data/                 # gitignored
│   ├── raw/              # HyP3 downloads
│   ├── interim/          # masked / calibrated rasters
│   ├── outputs/          # detections (GeoJSON), quicklooks, CZML, validation
│   ├── land/             # OSM land polygons clipped to the AOI
│   ├── ground_truth/     # NAIS chart GIFs, IIP sightings, Sentinel-2 targets
│   ├── kaggle/           # Statoil/C-CORE train.json (experimental classifier)
│   └── models/           # trained classifier weights
├── docs/                 # findings write-ups and figures
├── validation/           # hand-transcribed NAIS chart counts (CSV)
├── src/iceberg_sar/
│   ├── config.py         # config.yaml loading, metre -> pixel conversion
│   ├── search.py         # find scenes with asf_search
│   ├── hyp3.py           # order + download RTC products
│   ├── landmask.py       # OSM land polygons -> AOI land mask
│   ├── seaice.py         # HV pack-ice mask, distance to ice
│   ├── preprocess.py     # land mask, dB conversion, nodata handling
│   ├── cfar.py           # CFAR detector
│   ├── detections.py     # connected components -> GeoJSON with attributes
│   ├── groundtruth.py    # NAIS chart download, per-square count comparison
│   ├── iip.py            # IIP iceberg sightings (NSIDC G00807)
│   ├── sentinel2.py      # Sentinel-2 optical targets
│   ├── matching.py       # one-to-one matching, drift-offset estimation
│   ├── pointval.py       # point-level precision / recall
│   ├── export_czml.py    # detections -> CZML for Cesium
│   ├── chips.py          # Kaggle-matched chips (experimental classifier)
│   ├── classify/         # experimental iceberg vs. ship CNN
│   └── cli.py            # typer CLI tying it together
├── viewer/               # Vue 3 + Vite + CesiumJS front end
├── notebooks/            # exploration only; logic lives in src/
└── tests/
```

## Current status (as of 2026-10-05)
- **Milestones 0–3: done.** Search, HyP3 RTC (10 m), land + pack-ice masks, CA-CFAR on HV with strip rejection and an HH co-pol check, GeoJSON detections, and the Vue + Cesium viewer.
- **Scenes:** spring 2025 (2025-05-02, 05-08, 05-14) in the viewer, checked against NAIS charts. Spring 2019 (2019-04-15, 04-29) and 2020-06-10 for point-level validation.
- **Validation (beyond the plan):** 78% recall (209/268) against same-pass IIP analyst labels on 2019-04-29, median offset 128 m. Precision is not measurable without AIS. Large bergs (>120 m) are missed. See `docs/validation-findings.md`.
- **Experimental:** iceberg vs. ship CNN trained on Kaggle chips (`iceberg_prob`). Use it as a review hint only; see the README.
- **Milestone 4: not started.** README write-up, screenshots, and an end-to-end `run` command (there is no `cli.py run` yet).
- **Stretch not done:** OS-CFAR / K-distribution (`cfar.variant` has a slot for it).

## Milestones

### Milestone 0: Setup (short)
- Repo, virtual env, `pyproject.toml`, `.gitignore`, `config.yaml`, `aoi.geojson`.
- **Done when:** `python -m iceberg_sar.cli --help` runs.

### Milestone 1: First image
- Search Sentinel-1 scenes over the AOI for a spring date range. Order one HyP3 RTC job, download it, and open it with rasterio.
- Apply the land mask. Save a quicklook PNG in dB with a sensible stretch.
- **Done when:** I can see a masked SAR scene of the NL coast and spot bright targets by eye.

### Milestone 2: CFAR detection
- Implement **CA-CFAR** (cell-averaging) on linear intensity: target cell, guard window, background window, threshold factor from a chosen false-alarm rate. Use a vectorized or convolution-based implementation for speed.
- Post-process: connected components, min/max size filter, remove edge and nodata artifacts.
- Output GeoJSON points with attributes: id, scene id, timestamp, lat/lon, area (px and m²), peak and mean backscatter (dB) per band, incidence angle.
- Compare visually against Canadian Ice Service charts for the same date.
- **Done when:** a GeoJSON of candidate targets overlays sensibly on the quicklook, with few obvious false alarms at sea.
- Stretch: try a K-distribution or OS-CFAR and compare.

### Milestone 3: Cesium viewer
- Export detections to CZML with time intervals per scene, so the Cesium timeline steps through dates.
- Style points by detection attributes (e.g. size by `structure_m`, color by contrast or distance to pack ice). Show the scene footprint. Info box shows attributes.
- Optional: SAR quicklook as an imagery layer (static PNG with bounds, or COG tiles via TiTiler).
- **Done when:** I can open the viewer, scrub the timeline, and click icebergs.

### Milestone 4: Polish for the portfolio
- README with a short write-up: problem, data, method, results, limitations, screenshots or GIF.
- Reproducible run: `cli.py run --config config.yaml` goes end to end for a given date range.
- Optional: Dockerfile, deployed static viewer, short blog or LinkedIn post.
- **Done when:** someone else can clone the repo, follow the README, and reproduce the demo.

## Known pitfalls to watch for
- Speckle and sea state: rough seas raise clutter and false alarms. Log wind conditions where possible.
- Coastline and islands produce bright returns, so the land mask buffer matters.
- Near-range vs. far-range incidence angle changes backscatter. Keep incidence angle as an attribute of each detection.
- Sea ice (pack ice) looks very different from icebergs in open water. Early in the season, pack ice can cover parts of the AOI. Mask or exclude it rather than detecting it as icebergs.
- Keep dB vs. linear consistent: CFAR on linear intensity, dB for display and reported attributes.
- 10 m IW GRD is oversampled (~20 m true resolution). HV speckle grains span ~2×2 px and pass a 2 px minimum, so the HH co-pol check is needed.
- HyP3 removes the HV thermal noise floor, so open-water HV sits near −30 to −40 dB with ENL < 1. Estimate ENL from the scene instead of assuming it.
- Keep size parameters in metres in `config.yaml`. They are converted to pixels per scene, so tuning carries over between 20 m and 10 m.

## Decisions made
- First dates: spring 2025 (2025-05-02, 05-08, 05-14). Spring 2019 for IIP point validation, since that database ends in 2021.
- Viewer: Vue 3 + Vite + CesiumJS, Esri imagery basemap (no ion token needed).

## Open questions (decide with me as we go)
- Whether to add FastAPI + PostGIS for storing detections across many scenes, or keep flat files (flat files so far).
- Whether to keep the experimental classifier in the portfolio write-up, given that it can't be validated without AIS.
