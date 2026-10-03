"""Sentinel-2 optical targets as independent validation for SAR detections.

Sentinel-2 L2A (10 m) from the Earth Search STAC catalogue (public COGs on AWS, no login).
Off Newfoundland it images the coastal strip around 14:40-15:00 UTC, about 5 h after the
morning Sentinel-1 pass, so bergs drift between the two (see `matching.estimate_offset`).

A target is a bright compact object in NIR (B08) whose surrounding ring is open water:
- icebergs are bright in NIR (snow/ice ~0.3-0.7 reflectance), open water is ~0-0.05;
- pack ice is bright everywhere, so objects inside it fail the ring test and are skipped;
- large cloud areas (SCL cloud classes) are removed with a buffer; small isolated objects
  that SCL calls cloud are kept, because SCL often labels icebergs as cloud.
Optical imagery cannot tell a berg from an isolated ice floe either, so the truth class is
"isolated ice target in open water", the same thing the SAR detector looks for.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from scipy import ndimage
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry

STAC_URL = "https://earth-search.aws.element84.com/v1/search"
COLLECTION = "sentinel-2-c1-l2a"
SCL_CLOUD = (3, 8, 9, 10)  # cloud shadow, cloud medium / high probability, thin cirrus


@dataclass(frozen=True)
class OpticalParams:
    target_reflectance: float = 0.15  # NIR above this = candidate ice pixel
    water_reflectance: float = 0.06  # NIR below this = open water
    min_area_px: int = 2
    max_area_m2: float = 800_000.0  # same cap as the SAR detector
    ring_m: float = 200.0  # width of the ring around a target that must be open water
    min_ring_water: float = 0.9  # fraction of the ring that is open water
    cloud_min_area_m2: float = 1_000_000.0  # SCL cloud patches smaller than this are ignored
    cloud_buffer_m: float = 500.0


def search_items(area: BaseGeometry, day: str, max_cloud: float = 100.0) -> list[dict]:
    """STAC items intersecting `area` (EPSG:4326) on `day` (YYYY-MM-DD)."""
    body = {
        "collections": [COLLECTION],
        "intersects": mapping(area),
        "datetime": f"{day}T00:00:00Z/{day}T23:59:59Z",
        "query": {"eo:cloud_cover": {"lte": max_cloud}},
        "limit": 200,
    }
    req = Request(STAC_URL, data=json.dumps(body).encode(),
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=120) as r:
        return json.load(r)["features"]


def _disk(radius_px: int) -> np.ndarray:
    y, x = np.ogrid[-radius_px : radius_px + 1, -radius_px : radius_px + 1]
    return x * x + y * y <= radius_px * radius_px


def cloud_mask(scl: np.ndarray, pixel_m: float, p: OpticalParams) -> np.ndarray:
    """Large SCL cloud / shadow patches, buffered. Small 'cloud' blobs (often bergs) are not."""
    cloud = np.isin(scl, SCL_CLOUD)
    labels, n = ndimage.label(cloud)
    if n == 0:
        return cloud
    area = ndimage.sum_labels(cloud, labels, np.arange(1, n + 1)) * pixel_m**2
    big = np.zeros(n + 1, dtype=bool)
    big[1:] = area >= p.cloud_min_area_m2
    mask = big[labels]
    r = max(1, round(p.cloud_buffer_m / pixel_m))
    return ndimage.binary_dilation(mask, structure=_disk(r))


def bright_targets(
    nir: np.ndarray, excluded: np.ndarray, pixel_m: float, p: OpticalParams
) -> pd.DataFrame:
    """Isolated bright objects in a NIR reflectance array. `excluded` = cloud / nodata."""
    valid = np.isfinite(nir) & ~excluded
    target = valid & (nir > p.target_reflectance)
    water = valid & (nir < p.water_reflectance)
    labels, n = ndimage.label(target, structure=np.ones((3, 3), bool))
    if n == 0:
        return pd.DataFrame(columns=["row", "col", "area_m2", "peak_reflectance"])
    idx = np.arange(1, n + 1)
    area_px = ndimage.sum_labels(target, labels, idx)
    keep = (area_px >= p.min_area_px) & (area_px * pixel_m**2 <= p.max_area_m2)
    # Ring test on each object's bounding box: dilate the object, look at the new pixels.
    r = max(1, round(p.ring_m / pixel_m))
    ring_ok = np.zeros(n, dtype=bool)
    for i, sl in enumerate(ndimage.find_objects(labels)):
        if not keep[i] or sl is None:
            continue
        r0, r1, c0, c1 = sl[0].start - r, sl[0].stop + r, sl[1].start - r, sl[1].stop + r
        if r0 < 0 or c0 < 0 or r1 > nir.shape[0] or c1 > nir.shape[1]:
            continue  # ring cut off by the tile edge: can't tell if it is isolated
        obj = labels[r0:r1, c0:c1] == i + 1
        ring = ndimage.binary_dilation(obj, structure=_disk(r)) & ~obj
        ring_ok[i] = ring.any() and water[r0:r1, c0:c1][ring].mean() >= p.min_ring_water
    keep &= ring_ok
    k = idx[keep]
    if len(k) == 0:
        return pd.DataFrame(columns=["row", "col", "area_m2", "peak_reflectance"])
    com = np.array(ndimage.center_of_mass(target, labels, k)).reshape(-1, 2)
    return pd.DataFrame({
        "row": com[:, 0],
        "col": com[:, 1],
        "area_m2": area_px[keep] * pixel_m**2,
        "peak_reflectance": ndimage.maximum(np.nan_to_num(nir), labels, k),
    })


def _reflectance(asset: dict, raw: np.ndarray) -> np.ndarray:
    band = asset.get("raster:bands", [{}])[0]
    out = raw.astype(np.float32) * band.get("scale", 1e-4) + band.get("offset", 0.0)
    out[raw == band.get("nodata", 0)] = np.nan
    return out


def item_targets(item: dict, p: OpticalParams) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Targets (points) and comparable area (clear, valid; polygons) for one tile, EPSG:4326."""
    nir_asset, scl_asset = item["assets"]["nir"], item["assets"]["scl"]
    with rasterio.open(nir_asset["href"]) as src:
        nir = _reflectance(nir_asset, src.read(1))
        transform, crs, pixel_m = src.transform, src.crs, abs(src.transform.a)
        with rasterio.open(scl_asset["href"]) as scl_src:
            scl = scl_src.read(1, out_shape=nir.shape, resampling=Resampling.nearest)
    excluded = cloud_mask(scl, pixel_m, p) | ~np.isfinite(nir)
    t = bright_targets(nir, excluded, pixel_m, p)
    xs, ys = rasterio.transform.xy(transform, t["row"].to_numpy(), t["col"].to_numpy())
    pts = gpd.GeoDataFrame(
        t.assign(item=item["id"],
                 time=pd.Timestamp(item["properties"]["datetime"]).tz_localize(None)),
        geometry=gpd.points_from_xy(xs, ys), crs=crs,
    ).to_crs("EPSG:4326")
    clear = _clear_area(~excluded, transform, crs)
    return pts, clear


def _clear_area(clear: np.ndarray, transform: rasterio.Affine, crs: object,
                factor: int = 10) -> gpd.GeoDataFrame:
    """Polygons of the clear area at `factor`x coarser resolution (100 m for 10 m pixels)."""
    from rasterio.features import shapes
    from shapely.geometry import shape

    h, w = clear.shape[0] // factor * factor, clear.shape[1] // factor * factor
    coarse = clear[:h, :w].reshape(h // factor, factor, w // factor, factor).all(axis=(1, 3))
    t = transform * rasterio.Affine.scale(factor)
    geoms = [shape(g) for g, v in shapes(coarse.astype(np.uint8), mask=coarse, transform=t) if v]
    return gpd.GeoDataFrame(geometry=geoms, crs=crs).to_crs("EPSG:4326")


def dedupe(points: gpd.GeoDataFrame, metric_crs: str, tol_m: float = 50.0) -> gpd.GeoDataFrame:
    """Drop repeats of the same target from overlapping tiles (keep the first)."""
    if points.empty:
        return points
    from scipy.spatial import cKDTree

    g = points.to_crs(metric_crs).geometry
    xy = np.column_stack([g.x, g.y])
    tree = cKDTree(xy)
    keep = np.ones(len(points), dtype=bool)
    for i, j in sorted(tree.query_pairs(tol_m)):
        if keep[i]:
            keep[j] = False
    return points[keep].reset_index(drop=True)


def extract_for_area(
    area: BaseGeometry, day: str, out_dir: Path, max_cloud: float, p: OpticalParams
) -> tuple[Path, Path, list[str]]:
    """Targets and clear area for all tiles over `area` on `day`; writes two GeoJSONs."""
    items = search_items(area, day, max_cloud)
    if not items:
        raise FileNotFoundError(f"No Sentinel-2 tiles over the area on {day}")
    pts, clear, log = [], [], []
    for it in items:
        t, c = item_targets(it, p)
        pts.append(t)
        clear.append(c)
        log.append(f"{it['id']}  cloud={it['properties']['eo:cloud_cover']:.0f}%  targets={len(t)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = dedupe(pd.concat(pts, ignore_index=True), "EPSG:32621")
    targets_path = out_dir / f"{day}_s2_targets.geojson"
    targets.to_file(targets_path, driver="GeoJSON")
    clear_path = out_dir / f"{day}_s2_clear.geojson"
    gpd.GeoDataFrame(geometry=[pd.concat(clear).union_all()], crs="EPSG:4326").to_file(
        clear_path, driver="GeoJSON")
    return targets_path, clear_path, log
