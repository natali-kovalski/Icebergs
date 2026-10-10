"""Fixed bright targets (unmapped rocks, islets, wrecks) found by repeat detection.

Icebergs drift hundreds of metres to kilometres between passes; a rock returns at the same
spot to within the RTC geolocation error (~10-30 m). A target detected within `radius_m`
on at least `min_dates` different days is treated as static and left out of the viewer.
Grounded bergs can also stay put for days, so `min_span_days` can demand a longer gap.
Detection files are kept as they are, so the list can be rebuilt as scenes are added.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from iceberg_sar.config import Config


@dataclass(frozen=True)
class StaticParams:
    radius_m: float = 50.0
    min_dates: int = 2
    min_span_days: float = 0.0
    metric_crs: str = "EPSG:32621"


def static_params(cfg: Config) -> StaticParams:
    s = cfg.raw.get("static_targets", {})
    d = StaticParams()
    return StaticParams(
        radius_m=float(s.get("radius_m", d.radius_m)),
        min_dates=int(s.get("min_dates", d.min_dates)),
        min_span_days=float(s.get("min_span_days", d.min_span_days)),
        metric_crs=str(cfg.raw.get("preprocess", {}).get("metric_crs", d.metric_crs)),
    )


def find_static(detections: list[tuple[date, gpd.GeoDataFrame]], p: StaticParams
                ) -> gpd.GeoDataFrame:
    """Clusters of detections (single linkage within `radius_m`) seen on enough dates.

    Returns one point per static target (EPSG:4326, the cluster centroid) with n_dates,
    first and last date, and spread_m (farthest member from the centroid).
    """
    parts = [g.to_crs(p.metric_crs).assign(day=pd.Timestamp(d)) for d, g in detections if len(g)]
    empty = gpd.GeoDataFrame({"n_dates": [], "first": [], "last": [], "spread_m": []},
                             geometry=[], crs="EPSG:4326")
    if not parts:
        return empty
    pts = pd.concat(parts, ignore_index=True)
    xy = np.c_[pts.geometry.x, pts.geometry.y]
    pairs = cKDTree(xy).query_pairs(p.radius_m, output_type="ndarray")
    n = len(xy)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    pts["cluster"] = labels

    rows = []
    for _, c in pts.groupby("cluster"):
        days = c["day"].dt.normalize().unique()
        span = (days.max() - days.min()).days
        if len(days) >= p.min_dates and span >= p.min_span_days:
            x, y = c.geometry.x.mean(), c.geometry.y.mean()
            spread = float(np.hypot(c.geometry.x - x, c.geometry.y - y).max())
            rows.append({"n_dates": len(days), "first": str(days.min().date()),
                         "last": str(days.max().date()), "spread_m": round(spread, 1),
                         "x": x, "y": y})
    if not rows:
        return empty
    df = pd.DataFrame(rows)
    out = gpd.GeoDataFrame(df.drop(columns=["x", "y"]),
                           geometry=gpd.points_from_xy(df.x, df.y), crs=p.metric_crs)
    return out.to_crs("EPSG:4326")


def is_static(gdf: gpd.GeoDataFrame, static: gpd.GeoDataFrame, p: StaticParams) -> np.ndarray:
    """True for detections within `radius_m` of a static target's cluster extent."""
    if gdf.empty or static.empty:
        return np.zeros(len(gdf), dtype=bool)
    a = gdf.to_crs(p.metric_crs)
    s = static.to_crs(p.metric_crs)
    dist, idx = cKDTree(np.c_[s.geometry.x, s.geometry.y]).query(np.c_[a.geometry.x, a.geometry.y])
    return dist <= p.radius_m + s["spread_m"].to_numpy()[idx]
