"""Command-line entry point: `python -m iceberg_sar.cli --help`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from iceberg_sar import __version__
from iceberg_sar.config import DEFAULT_CONFIG, load_config

app = typer.Typer(
    help="Iceberg Alley SAR: detect and classify icebergs in Sentinel-1 imagery.",
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


if __name__ == "__main__":
    app()
