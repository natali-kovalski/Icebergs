"""Build a land mask for the AOI from OpenStreetMap land polygons.

Source: https://osmdata.openstreetmap.de/data/land-polygons.html (ODbL, (c) OpenStreetMap
contributors). The "split" variant is tiled into small polygons, so a bbox read is fast.
The global zip (~900 MB) is downloaded once; only the AOI clip is used afterwards.
"""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

import geopandas as gpd

from iceberg_sar.config import Config
from iceberg_sar.search import load_aoi


def download(url: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / Path(url).name
    if not zip_path.is_file():
        tmp = zip_path.with_suffix(".part")
        with urllib.request.urlopen(url) as resp, tmp.open("wb") as f:
            shutil.copyfileobj(resp, f, length=1 << 20)
        tmp.rename(zip_path)
    return zip_path


def build_land_mask(cfg: Config, margin_deg: float = 0.5) -> Path:
    """Clip land polygons to the AOI bbox (plus margin), dissolve, save as GeoPackage."""
    out = cfg.path("land_mask")
    if out.is_file():
        return out
    p = cfg.section("preprocess")
    zip_path = download(p["land_polygons_url"], out.parent)
    minx, miny, maxx, maxy = load_aoi(cfg.path("aoi")).bounds
    bbox = (minx - margin_deg, miny - margin_deg, maxx + margin_deg, maxy + margin_deg)

    land = gpd.read_file(f"zip://{zip_path}!{p['land_polygons_member']}", bbox=bbox)
    land = land.to_crs("EPSG:4326").clip(bbox)
    # Dissolve the tile seams, then explode back into one row per landmass/island.
    land = gpd.GeoDataFrame(geometry=[land.union_all()], crs="EPSG:4326").explode(index_parts=False)
    land.reset_index(drop=True).to_file(out, driver="GPKG")
    return out


def load_land(path: Path) -> gpd.GeoDataFrame:
    return gpd.read_file(path)
