"""Command-line entry point: `python -m iceberg_sar.cli --help`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from iceberg_sar import __version__
from iceberg_sar.config import DEFAULT_CONFIG, load_config

app = typer.Typer(
    help="Iceberg Alley SAR: detect icebergs in Sentinel-1 imagery.",
    no_args_is_help=True,
)

ConfigOption = Annotated[
    Path, typer.Option("--config", "-c", help="Path to config.yaml.")
]


@app.command()
def info(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Show version, config location, AOI, and data directories."""
    cfg = load_config(config)
    search = cfg.section("search")
    typer.echo(f"iceberg-sar {__version__}")
    typer.echo(f"config:     {cfg.root / Path(config).name}")
    aoi = cfg.path("aoi")
    typer.echo(f"aoi:        {aoi} ({'found' if aoi.is_file() else 'MISSING'})")
    typer.echo(f"dates:      {search['start_date']} to {search['end_date']}")
    for d in cfg.data_dirs:
        typer.echo(f"data dir:   {d} ({'exists' if d.is_dir() else 'not created'})")


@app.command("init-dirs")
def init_dirs(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Create the data/ directory tree from config.yaml."""
    cfg = load_config(config)
    for d in cfg.data_dirs:
        d.mkdir(parents=True, exist_ok=True)
        typer.echo(f"ok  {d}")


@app.command()
def search(
    config: ConfigOption = DEFAULT_CONFIG,
    pol: Annotated[str | None, typer.Option(help="Filter, e.g. 'HH+HV' or 'VV+VH'.")] = None,
) -> None:
    """List Sentinel-1 GRD scenes over the AOI and save them as GeoJSON footprints."""
    from iceberg_sar.search import save_scenes, search_scenes

    cfg = load_config(config)
    gdf = search_scenes(cfg)
    if pol:
        gdf = gdf[gdf.polarization == pol]
    if gdf.empty:
        typer.echo("No scenes found.")
        raise typer.Exit(1)
    s = cfg.section("search")
    out_name = f"scenes_{s['start_date']}_{s['end_date']}.geojson"
    out = save_scenes(gdf, cfg.path("outputs") / out_name)
    cols = ["scene", "beam_mode", "polarization", "flight_direction", "aoi_overlap"]
    typer.echo(gdf[cols].to_string())
    typer.echo(f"\n{len(gdf)} scenes -> {out}")


@app.command()
def order(granule: str, config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Submit a HyP3 RTC job for one Sentinel-1 granule (skips if already ordered)."""
    from iceberg_sar.hyp3 import get_client, submit_rtc

    cfg = load_config(config)
    client = get_client()
    typer.echo(f"Credits remaining: {client.check_credits()}")
    batch = submit_rtc(client, granule, cfg)
    for job in batch:
        typer.echo(f"job {job.job_id}  status={job.status_code}  granule={granule}")


@app.command()
def download(
    config: ConfigOption = DEFAULT_CONFIG,
    wait: Annotated[bool, typer.Option(help="Wait for running jobs to finish.")] = True,
) -> None:
    """Download finished HyP3 RTC jobs into data/raw and unzip them."""
    from iceberg_sar.hyp3 import JOB_NAME, download_jobs, get_client

    cfg = load_config(config)
    client = get_client()
    batch = client.find_jobs(name=JOB_NAME)
    if wait and not batch.complete():
        batch = client.watch(batch)
    for d in download_jobs(batch, cfg.path("raw")):
        typer.echo(f"ok  {d}")


@app.command("land-mask")
def land_mask(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Download OSM land polygons (once) and build the AOI land mask GeoPackage."""
    from iceberg_sar.landmask import build_land_mask

    cfg = load_config(config)
    typer.echo(f"ok  {build_land_mask(cfg)}")


@app.command()
def preprocess(
    product_dir: Annotated[Path, typer.Argument(help="Unzipped HyP3 RTC product directory.")],
    config: ConfigOption = DEFAULT_CONFIG,
) -> None:
    """Mask land and nodata, write masked linear rasters and dB quicklook PNGs."""
    from iceberg_sar.landmask import build_land_mask, load_land
    from iceberg_sar.preprocess import preprocess_product

    cfg = load_config(config)
    land = load_land(build_land_mask(cfg))
    for kind, path in preprocess_product(product_dir, land, cfg).items():
        typer.echo(f"{kind:<12} {path}")


@app.command()
def detect(
    product_dir: Annotated[Path, typer.Argument(help="HyP3 RTC product directory (in data/raw).")],
    config: ConfigOption = DEFAULT_CONFIG,
    band: Annotated[str | None, typer.Option(help="Override cfar.band, e.g. HH or HV.")] = None,
) -> None:
    """Run CA-CFAR on a preprocessed scene; write detections GeoJSON and an overlay PNG."""
    import json

    from iceberg_sar.detections import detect_product

    cfg = load_config(config)
    outputs = detect_product(product_dir, cfg, band=band)
    summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
    typer.echo(
        f"band={summary['cfar_band']}  ENL={summary['enl']}  "
        f"threshold={summary['alpha']}x background ({summary['alpha_db']} dB)  "
        f"detections={summary['n_detections']}"
    )
    for kind, path in outputs.items():
        typer.echo(f"{kind:<12} {path}")


@app.command("ground-truth")
def ground_truth(
    day: Annotated[str, typer.Argument(help="Date, YYYY-MM-DD (e.g. the scene date).")],
    days_around: Annotated[int, typer.Option(help="Also fetch this many days before/after.")] = 1,
    config: ConfigOption = DEFAULT_CONFIG,
) -> None:
    """Download NAIS iceberg chart GIFs (IIP + Canadian Ice Service) around a date."""
    from datetime import date, timedelta

    from iceberg_sar.groundtruth import download_nais_chart

    cfg = load_config(config)
    url = cfg.section("ground_truth")["nais_url"]
    d0 = date.fromisoformat(day)
    out_dir = cfg.path("ground_truth")
    for k in range(-days_around, days_around + 1):
        typer.echo(f"ok  {download_nais_chart(d0 + timedelta(days=k), out_dir, url)}")


@app.command()
def validate(
    product_dir: Annotated[Path, typer.Argument(help="HyP3 RTC product directory (in data/raw).")],
    counts: Annotated[Path, typer.Option(help="Transcribed NAIS counts CSV.")],
    band: Annotated[str, typer.Option(help="Which detections to validate.")] = "HV",
    config: ConfigOption = DEFAULT_CONFIG,
) -> None:
    """Compare SAR detections per 1-degree square with transcribed NAIS chart counts."""
    import geopandas as gpd

    from iceberg_sar.groundtruth import (
        compare_counts,
        load_degree_counts,
        valid_fraction_per_square,
    )

    cfg = load_config(config)
    name = Path(product_dir).name
    det = gpd.read_file(cfg.path("outputs") / "detections" / f"{name}_{band}_detections.geojson")
    masked = cfg.path("interim") / name / f"{name}_{band}_masked.tif"
    min_frac = cfg.section("ground_truth")["min_valid_fraction"]
    cmp = compare_counts(det, load_degree_counts(counts), valid_fraction_per_square(masked),
                         min_frac)
    shown = cmp[cmp.valid_fraction > 0].drop(columns="geometry")
    typer.echo(shown.to_string(index=False))
    used = cmp[cmp.compared]
    typer.echo()
    typer.echo(f"squares with >= {min_frac:.0%} coverage: SAR {used.sar_detections.sum()} "
               f"vs chart (coverage-scaled) {used.expected.sum():.1f}; "
               f"detections outside listed squares: {len(det) - cmp.sar_detections.sum()}")
    out = cfg.path("outputs") / "validation" / f"{name}_{band}_nais.geojson"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmp.to_file(out, driver="GeoJSON")
    typer.echo(f"ok  {out}")


if __name__ == "__main__":
    app()
