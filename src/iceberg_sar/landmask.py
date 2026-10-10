"""Build a land mask for the AOI from OpenStreetMap land polygons plus point-mapped rocks.

Source: https://osmdata.openstreetmap.de/data/land-polygons.html (ODbL, (c) OpenStreetMap
contributors). The "split" variant is tiled into small polygons, so a bbox read is fast.
The global zip (~900 MB) is downloaded once; only the AOI clip is used afterwards.

Land polygons are built only from coastline ways, so rocks, reefs and islets mapped as single
nodes (thousands off NL, many from the CanVec import) are missing. Those are bright, fixed
SAR targets, so they are fetched from the Overpass API and added as small discs.
"""

from __future__ import annotations

import json
import shutil
import time
import urllib.parse
import urllib.request
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point

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


def overpass_query(bbox: tuple[float, float, float, float], tags: dict[str, list[str]]) -> str:
    """Overpass QL for nodes with any of `tags` (key -> values) inside a lon/lat bbox."""
    minx, miny, maxx, maxy = bbox
    b = f"{miny},{minx},{maxy},{maxx}"   # Overpass wants south,west,north,east
    parts = "".join(f'node({b})["{k}"~"^({"|".join(v)})$"];' for k, v in tags.items())
    return f"[out:json][timeout:180];({parts});out;"


def point_features(elements: list[dict]) -> gpd.GeoDataFrame:
    """Overpass JSON nodes -> points with their tag (e.g. natural=rock), EPSG:4326."""
    keys = ("natural", "place", "man_made", "seamark:type")
    kind = [next((f"{k}={e['tags'][k]}" for k in keys if k in e.get("tags", {})), "")
            for e in elements]
    geom = [Point(e["lon"], e["lat"]) for e in elements]
    return gpd.GeoDataFrame({"osm_id": [e["id"] for e in elements], "kind": kind},
                            geometry=geom, crs="EPSG:4326")


def fetch_land_points(cfg: Config, bbox: tuple[float, float, float, float], out: Path,
                      rounds: int = 4) -> Path:
    """Download point-mapped rocks/islets in `bbox` once (tries each Overpass endpoint)."""
    if out.is_file():
        return out
    p = cfg.section("preprocess")
    data = urllib.parse.urlencode({"data": overpass_query(bbox, p["land_point_tags"])}).encode()
    errors = []
    for attempt in range(rounds):
        for url in p["overpass_urls"]:
            req = urllib.request.Request(url, data=data,
                                         headers={"User-Agent": "iceberg-alley-sar"})
            try:
                with urllib.request.urlopen(req, timeout=240) as resp:
                    elements = json.load(resp)["elements"]
                point_features(elements).to_file(out, driver="GPKG")
                return out
            except OSError as e:   # busy servers answer 429/500/504; try the next mirror
                errors.append(f"{url}: {e}")
        if attempt < rounds - 1:
            time.sleep(30 * (attempt + 1))
    raise RuntimeError("All Overpass endpoints failed:\n" + "\n".join(errors))


def points_to_discs(points: gpd.GeoDataFrame, radius_m: float, metric_crs: str) -> gpd.GeoSeries:
    """Small polygons around point features, so the land layer stays polygon-only."""
    return points.to_crs(metric_crs).buffer(radius_m).to_crs("EPSG:4326")


def build_land_mask(cfg: Config, margin_deg: float = 0.5, force: bool = False) -> Path:
    """Clip land polygons to the AOI bbox (plus margin), add point rocks/islets, save as GPKG."""
    out = cfg.path("land_mask")
    if out.is_file() and not force:
        return out
    p = cfg.section("preprocess")
    zip_path = download(p["land_polygons_url"], out.parent)
    minx, miny, maxx, maxy = load_aoi(cfg.path("aoi")).bounds
    bbox = (minx - margin_deg, miny - margin_deg, maxx + margin_deg, maxy + margin_deg)

    land = gpd.read_file(f"zip://{zip_path}!{p['land_polygons_member']}", bbox=bbox)
    geoms = list(land.to_crs("EPSG:4326").clip(bbox).geometry)
    if p.get("land_point_tags"):
        pts = gpd.read_file(fetch_land_points(cfg, bbox, out.parent / "osm_land_points_aoi.gpkg"))
        geoms += list(points_to_discs(pts, p["land_point_radius_m"], p["metric_crs"]))
    # Dissolve the tile seams, then explode back into one row per landmass/island/rock.
    union = gpd.GeoSeries(geoms, crs="EPSG:4326").union_all()
    land = gpd.GeoDataFrame(geometry=[union], crs="EPSG:4326").explode(index_parts=False)
    land.reset_index(drop=True).to_file(out, driver="GPKG")
    return out


def load_land(path: Path) -> gpd.GeoDataFrame:
    return gpd.read_file(path)
