"""End-to-end run for a date range: search -> HyP3 RTC -> masks -> CFAR -> CZML.

Every step skips work that is already on disk (downloaded products, masked rasters,
detections), so a re-run only does what is missing and costs no HyP3 credits for scenes
that were processed before.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import geopandas as gpd

from iceberg_sar.config import Config

# HyP3 RTC_GAMMA credit cost per scene by pixel spacing (ASF credit table, 2025).
RTC_CREDITS = {10: 60, 20: 15, 30: 5}


@dataclass(frozen=True)
class RunParams:
    pol: str | None = "HH+HV"       # None = any polarization
    min_aoi_overlap: float = 0.05   # fraction of the AOI a scene must cover
    max_scenes: int | None = 3      # cap on scenes per run (HyP3 credits); None = all
    band: str = "HV"                # detections exported to CZML


@dataclass(frozen=True)
class PlannedScene:
    granule: str
    start_time: str
    aoi_overlap: float
    product_dir: Path | None        # already downloaded HyP3 product, if any


def run_params(cfg: Config) -> RunParams:
    r = cfg.raw.get("run", {})
    d = RunParams()
    return RunParams(
        pol=r.get("pol", d.pol),
        min_aoi_overlap=float(r.get("min_aoi_overlap", d.min_aoi_overlap)),
        max_scenes=r.get("max_scenes", d.max_scenes),
        band=str(cfg.section("cfar").get("band", d.band)).upper(),
    )


def select_scenes(scenes: gpd.GeoDataFrame, p: RunParams) -> gpd.GeoDataFrame:
    """Filter by polarization and AOI overlap; keep the `max_scenes` best-covering, by time."""
    if scenes.empty:
        return scenes
    keep = scenes[scenes["aoi_overlap"] >= p.min_aoi_overlap]
    if p.pol:
        keep = keep[keep["polarization"] == p.pol]
    keep = keep.drop_duplicates("scene")
    if p.max_scenes is not None:
        keep = keep.sort_values("aoi_overlap", ascending=False, kind="stable").head(p.max_scenes)
    return keep.sort_values("start_time").reset_index(drop=True)


def find_product(raw_dir: Path, granule: str, resolution: int) -> Path | None:
    """Unzipped HyP3 RTC product for a granule, matched on platform, beam mode and start time.

    Granule `S1A_IW_GRDH_1SDH_20250508T094925_...`
    -> product `S1A_IW_20250508T094925_DHP_RTC10_...`.
    """
    parts = granule.split("_")
    start = next((s for s in parts if len(s) == 15 and s[8] == "T"), None)
    if start is None:
        raise ValueError(f"No start time in granule name '{granule}'")
    pattern = f"{parts[0]}_{parts[1]}_{start}_*_RTC{resolution}_*"
    matches = sorted(d for d in Path(raw_dir).glob(pattern) if d.is_dir())
    return matches[-1] if matches else None


def plan(scenes: gpd.GeoDataFrame, raw_dir: Path, resolution: int) -> list[PlannedScene]:
    return [
        PlannedScene(r.scene, str(r.start_time), float(r.aoi_overlap),
                     find_product(raw_dir, r.scene, resolution))
        for r in scenes.itertuples()
    ]


def order_and_download(granules: list[str], cfg: Config, log: Callable[[str], None]) -> None:
    """Submit RTC jobs (reusing existing ones), wait for them, download and unzip."""
    from iceberg_sar.hyp3 import download_jobs, get_client, submit_rtc

    client = get_client()
    log(f"HyP3 credits remaining: {client.check_credits()}")
    batches = [submit_rtc(client, g, cfg) for g in granules]
    batch = sum(batches[1:], batches[0])
    log(f"waiting for {len(batch)} HyP3 job(s); this often takes 30-60 min")
    batch = client.watch(batch)
    for d in download_jobs(batch, cfg.path("raw")):
        log(f"downloaded  {d.name}")


def process_product(product_dir: Path, land: gpd.GeoDataFrame, cfg: Config, band: str,
                    force: bool, log: Callable[[str], None]) -> Path:
    """Preprocess and detect one product unless its outputs exist. Returns the detections."""
    from iceberg_sar.detections import detect_product
    from iceberg_sar.preprocess import preprocess_product

    name = product_dir.name
    if force or not (cfg.path("interim") / name / f"{name}_preprocess.json").is_file():
        log(f"preprocess  {name}")
        preprocess_product(product_dir, land, cfg)
    detections = cfg.path("outputs") / "detections" / f"{name}_{band}_detections.geojson"
    if force or not detections.is_file():
        log(f"detect      {name}")
        detections = detect_product(product_dir, cfg, band=band)["detections"]
    return detections


def run_pipeline(cfg: Config, start: date, end: date, *, dry_run: bool = False,
                 force: bool = False, log: Callable[[str], None] = print) -> Path | None:
    """Run the whole pipeline for scenes acquired between `start` and `end` (inclusive).

    Returns the CZML path, or None for a dry run.
    """
    from iceberg_sar.export_czml import export_czml
    from iceberg_sar.landmask import build_land_mask, load_land
    from iceberg_sar.search import search_scenes

    p = run_params(cfg)
    res = int(cfg.section("hyp3")["resolution"])
    # The end date is inclusive: search up to the end of that day, not its first second.
    found = search_scenes(cfg, start=f"{start}T00:00:00Z", end=f"{end}T23:59:59Z")
    scenes = plan(select_scenes(found, p), cfg.path("raw"), res)
    if not scenes:
        raise RuntimeError(f"No {p.pol or ''} scenes over the AOI between {start} and {end}")
    for s in scenes:
        state = "downloaded" if s.product_dir else "to order"
        log(f"{s.start_time[:19]}  {s.granule}  aoi={s.aoi_overlap:.0%}  {state}")
    missing = [s.granule for s in scenes if s.product_dir is None]
    log(f"{len(scenes)} of {len(found)} scenes selected; {len(missing)} to order "
        f"(~{len(missing) * RTC_CREDITS.get(res, 0)} HyP3 credits at {res} m)")
    if dry_run:
        return None

    if missing:
        order_and_download(missing, cfg, log)
        scenes = plan(select_scenes(found, p), cfg.path("raw"), res)
    land = load_land(build_land_mask(cfg))
    for s in scenes:
        if s.product_dir is None:
            log(f"skipped     {s.granule} (HyP3 job did not succeed)")
            continue
        process_product(s.product_dir, land, cfg, p.band, force, log)

    out, packets = export_czml(cfg, p.band, start=start, end=end)
    n = sum(1 for k in packets if k["id"].startswith("scene/"))
    log(f"czml        {n} scene(s) -> {out}")
    return out
