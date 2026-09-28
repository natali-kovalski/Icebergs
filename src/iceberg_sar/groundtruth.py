"""Ground truth from the North American Ice Service (NAIS) daily Iceberg Analysis chart.

NAIS is the joint International Ice Patrol (US Coast Guard) + Canadian Ice Service product.
The chart gives an *estimate* of icebergs per 1-degree square (valid 00 UTC), built from
aircraft, ship and satellite reports plus drift models. It is not pixel-exact: counts
include bergy bits and growlers too small for 20 m SAR, and reconnaissance can be days old.
So we compare *counts per square*, normalised by how much of each square the scene sees.

Charts are only published as GIFs; counts are transcribed by hand into a CSV
(see `validation/`). The shapefiles IIP publishes hold only the iceberg/sea-ice limits.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from urllib.request import urlopen

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform as warp_transform
from shapely.geometry import box

NAIS_URL = "https://www.navcen.uscg.gov/sites/default/files/images/iip/data/{year}/{ymd}_NAIS65.gif"


def download_nais_chart(day: date, out_dir: Path, url_pattern: str = NAIS_URL) -> Path:
    """Download the NAIS iceberg chart GIF for one day (skips if already present)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ymd = day.strftime("%Y%m%d")
    out = out_dir / f"{ymd}_NAIS65.gif"
    if not out.is_file():
        with urlopen(url_pattern.format(year=day.year, ymd=ymd), timeout=60) as r:
            out.write_bytes(r.read())
    return out


def load_degree_counts(csv_path: Path) -> gpd.GeoDataFrame:
    """Transcribed counts (lat_min, lon_min, count) -> 1-degree square polygons in EPSG:4326."""
    df = pd.read_csv(csv_path, comment="#")
    geoms = [box(lon, lat, lon + 1, lat + 1) for lat, lon in zip(df.lat_min, df.lon_min,
                                                                  strict=True)]
    return gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:4326")


def valid_fraction_per_square(masked_tif: Path, decimate: int = 20) -> pd.DataFrame:
    """Fraction of each 1-degree square covered by valid (unmasked) scene pixels.

    Reads a decimated copy of the masked raster (every `decimate`-th pixel is enough for
    coverage), takes pixel-centre lat/lon, and bins by degree square.
    """
    with rasterio.open(masked_tif) as src:
        h, w = src.height // decimate, src.width // decimate
        data = src.read(1, out_shape=(h, w), resampling=Resampling.nearest)
        t = src.transform * rasterio.Affine.scale(src.width / w, src.height / h)
        rows, cols = np.nonzero(np.ones((h, w), dtype=bool))
        xs, ys = rasterio.transform.xy(t, rows, cols, offset="center")
        lons, lats = warp_transform(src.crs, "EPSG:4326", xs, ys)
        px_km2 = abs(t.a * t.e) / 1e6
    df = pd.DataFrame({
        "lat_min": np.floor(lats).astype(int),
        "lon_min": np.floor(lons).astype(int),
        "valid": np.isfinite(data.ravel()),
    })
    g = df.groupby(["lat_min", "lon_min"])["valid"].agg(["sum", "size"]).reset_index()
    # Area of a 1-degree square shrinks with cos(latitude).
    sq_km2 = 111.32 * 111.32 * np.cos(np.radians(g.lat_min + 0.5))
    g["valid_km2"] = g["sum"] * px_km2
    g["valid_fraction"] = (g["valid_km2"] / sq_km2).clip(upper=1.0)
    return g[["lat_min", "lon_min", "valid_km2", "valid_fraction"]]


def compare_counts(
    detections: gpd.GeoDataFrame,
    chart: gpd.GeoDataFrame,
    coverage: pd.DataFrame,
    min_valid_fraction: float = 0.2,
) -> gpd.GeoDataFrame:
    """Per degree square: chart count, SAR detections, and chart count scaled to coverage.

    `expected` = chart count x valid_fraction, assuming bergs are spread evenly within a
    square. That is crude (bergs hug the coast, and masking removes the coast), so treat
    the ratio as a sanity check, not an accuracy score.
    """
    det = detections.assign(lat_min=np.floor(detections.lat).astype(int),
                            lon_min=np.floor(detections.lon).astype(int))
    n_det = det.groupby(["lat_min", "lon_min"]).size().rename("sar_detections").reset_index()
    out = chart.merge(coverage, on=["lat_min", "lon_min"], how="left")
    out = out.merge(n_det, on=["lat_min", "lon_min"], how="left")
    out[["valid_km2", "valid_fraction"]] = out[["valid_km2", "valid_fraction"]].fillna(0.0)
    out["sar_detections"] = out["sar_detections"].fillna(0).astype(int)
    out["expected"] = (out["count"] * out["valid_fraction"]).round(1)
    out["compared"] = out["valid_fraction"] >= min_valid_fraction
    out["valid_fraction"] = out["valid_fraction"].round(2)
    out["valid_km2"] = out["valid_km2"].round(0)
    return gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:4326")
