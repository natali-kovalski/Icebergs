"""Find Sentinel-1 GRD scenes over the AOI with asf_search."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import asf_search as asf
import geopandas as gpd
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from iceberg_sar.config import Config

SCENE_COLUMNS = [
    "scene",
    "start_time",
    "beam_mode",
    "polarization",
    "processing_level",
    "flight_direction",
    "path",
    "frame",
    "aoi_overlap",
]


def load_aoi(path: str | Path) -> BaseGeometry:
    """AOI as a single EPSG:4326 geometry (union of all features)."""
    gdf = gpd.read_file(path).to_crs("EPSG:4326")
    return gdf.geometry.union_all()


def _record(product: asf.ASFProduct, aoi: BaseGeometry) -> dict[str, Any]:
    p = product.properties
    footprint = shape(product.geometry)
    overlap = footprint.intersection(aoi).area / aoi.area if aoi.area else 0.0
    return {
        "scene": p["sceneName"],
        "start_time": p["startTime"],
        "beam_mode": p.get("beamModeType"),
        "polarization": p.get("polarization"),
        "processing_level": p.get("processingLevel"),
        "flight_direction": p.get("flightDirection"),
        "path": p.get("pathNumber"),
        "frame": p.get("frameNumber"),
        "aoi_overlap": round(overlap, 3),
        "geometry": footprint,
    }


def search_scenes(cfg: Config) -> gpd.GeoDataFrame:
    """Search ASF for GRD scenes intersecting the AOI in the configured date range."""
    s = cfg.section("search")
    aoi = load_aoi(cfg.path("aoi"))
    results = asf.geo_search(
        intersectsWith=aoi.wkt,
        platform=asf.PLATFORM.SENTINEL1,
        processingLevel=s["processing_levels"],
        beamMode=s["beam_modes"],
        start=s["start_date"],
        end=s["end_date"],
        maxResults=s.get("max_results"),
    )
    records = [_record(r, aoi) for r in results]
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    if gdf.empty:
        return gdf
    return gdf.sort_values("start_time").reset_index(drop=True)


def save_scenes(gdf: gpd.GeoDataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GeoJSON")
    return out_path
