# Iceberg Alley SAR: Iceberg Detection off Newfoundland

## What this project is
Detect icebergs off the coast of Newfoundland & Labrador in Sentinel-1 SAR imagery, classify each detection as **iceberg vs. ship**, and display the results on an interactive **CesiumJS** globe with a timeline.

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
- **ML**: PyTorch, scikit-learn (metrics, CV splits)
- **Viz**: CesiumJS (optionally in a small Vue app), CZML for time-dynamic detections
- **Optional**: FastAPI to serve results, Docker for reproducibility

## Data sources
1. **Sentinel-1 SAR imagery** via ASF Vertex (https://search.asf.alaska.edu). Requires a free NASA Earthdata login.
   - Product: GRD, processed with **HyP3 RTC** into analysis-ready GeoTIFFs (gamma0). Request **power (linear)** scale and convert to dB where needed. Note the HyP3 monthly processing credit limit.
   - **Check the polarization of each scene.** Depending on area and date, it may be IW (VV/VH) or EW (HH/HV). This matters for the classifier (see Milestone 3).
   - Fallback source: Copernicus Data Space Ecosystem.
2. **Area of interest**: "Iceberg Alley", the NE Newfoundland coast, roughly Bonavista to St. Anthony, extending offshore. Store the AOI as `aoi.geojson`.
3. **Season**: April–June has the most icebergs. Start with 2–3 scenes from one spring.
4. **Ground truth / validation**: Canadian Ice Service iceberg charts and the International Ice Patrol iceberg data for matching dates. These are for visual and approximate validation, not pixel-exact labels.
5. **Land mask**: OSM land polygons or GSHHG coastline, buffered ~500 m to remove coastal clutter.
6. **Training data for classification**: Kaggle **Statoil/C-CORE Iceberg Classifier Challenge**.
   - About 1,600 labeled 75×75 chips, two bands (HH, HV) in dB, plus incidence angle and label `is_iceberg`.
   - Requires a Kaggle account and accepting the competition rules. **Check the data terms before redistributing anything.** Never commit this data to the repo.

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
│   ├── kaggle/           # Statoil/C-CORE dataset
│   └── outputs/          # detections (GeoJSON), chips, CZML
├── src/iceberg_sar/
│   ├── search.py         # find scenes with asf_search
│   ├── hyp3.py           # order + download RTC products
│   ├── preprocess.py     # land mask, dB conversion, nodata handling
│   ├── cfar.py           # CFAR detector
│   ├── detections.py     # connected components -> GeoJSON with attributes
│   ├── chips.py          # extract 75x75 chips around detections
│   ├── classify/
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── predict.py
│   ├── export_czml.py    # detections -> CZML for Cesium
│   └── cli.py            # typer CLI tying it together
├── viewer/               # CesiumJS (optionally Vue) front end
├── notebooks/            # exploration only; logic lives in src/
└── tests/
```

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

### Milestone 3: Iceberg vs. ship classifier
- Load the Kaggle dataset. Build a small CNN (a few conv blocks) with 2 input bands (plus optional incidence angle as a scalar input).
- Augmentation: flips and 90° rotations. Stratified 5-fold CV. Report log loss and accuracy.
- Extract 75×75 chips around my own detections in dB, matching the training preprocessing exactly.
- **Domain shift warning:** Kaggle chips are HH/HV. If my scenes are VV/VH, flag this clearly and choose one of: (a) prefer HH/HV scenes, (b) train on hand-crafted features that transfer better, or (c) treat cross-polarization results as experimental. Discuss the choice with me before proceeding.
- **Done when:** each detection has an `iceberg_prob` attribute and I understand how much to trust it.

### Milestone 4: Cesium viewer
- Export detections to CZML with time intervals per scene, so the Cesium timeline steps through dates.
- Style points by class and probability (e.g. iceberg = cyan, ship = orange, opacity by confidence). Show the scene footprint. Info box shows attributes.
- Optional: SAR quicklook as an imagery layer (static PNG with bounds, or COG tiles via TiTiler).
- **Done when:** I can open the viewer, scrub the timeline, and click icebergs.

### Milestone 5: Polish for the portfolio
- README with a short write-up: problem, data, method, results, limitations, screenshots or GIF.
- Reproducible run: `cli.py run --config config.yaml` goes end to end for a given date range.
- Optional: Dockerfile, deployed static viewer, short blog or LinkedIn post.
- **Done when:** someone else can clone the repo, follow the README, and reproduce the demo.

## Known pitfalls to watch for
- Speckle and sea state: rough seas raise clutter and false alarms. Log wind conditions where possible.
- Coastline and islands produce bright returns, so the land mask buffer matters.
- Near-range vs. far-range incidence angle changes backscatter. Keep incidence angle as a feature.
- Sea ice (pack ice) looks very different from icebergs in open water. Early in the season, pack ice can cover parts of the AOI. Mask or exclude it rather than detecting it as icebergs.
- Keep dB vs. linear consistent: CFAR on linear, CNN on dB (to match Kaggle).

## Open questions (decide with me as we go)
- Which spring and which specific dates to use first.
- Whether to build the viewer as plain CesiumJS or a Vue app.
- Whether to add FastAPI + PostGIS for storing detections across many scenes, or keep flat files.
