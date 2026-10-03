from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from iceberg_sar.pointval import PointValParams, validate_points

CRS = "EPSG:32621"
X0, Y1 = 600_000.0, 5_600_000.0  # 10 m pixels, 2000 x 2000 = 20 km square
SAR_TIME = pd.Timestamp("2019-04-15 09:57")


def _scene(tmp_path: Path) -> Path:
    a = np.ones((2000, 2000), dtype=np.float32)
    a[:, 1500:] = np.nan                                  # masked (land) on the east side
    path = tmp_path / "masked.tif"
    with rasterio.open(path, "w", driver="GTiff", height=2000, width=2000, count=1,
                       dtype="float32", crs=CRS, transform=from_origin(X0, Y1, 10, 10),
                       nodata=np.nan) as dst:
        dst.write(a, 1)
    return path


def _points(xy: np.ndarray, time: pd.Timestamp) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"time": [time] * len(xy)},
                            geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=CRS)


def test_drift_corrected_precision_and_recall(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    bergs = np.column_stack([rng.uniform(X0 + 1000, X0 + 14_000, 40),
                             rng.uniform(Y1 - 19_000, Y1 - 1000, 40)])
    det = _points(np.vstack([bergs[:30], [[X0 + 5000, Y1 - 5000]]]), SAR_TIME)  # 30 hits + 1 FA
    drift = np.array([2000.0, 1000.0])
    truth_xy = np.vstack([bergs + drift,                    # 40 bergs, 5 h later
                          [[X0 + 17_000 + 2000, Y1 - 9000]]])  # one on the masked side
    truth = _points(truth_xy, SAR_TIME + pd.Timedelta("5h"))
    area = gpd.GeoDataFrame(geometry=[box(X0, Y1 - 20_000, X0 + 20_000, Y1)], crs=CRS)

    s, t, d = validate_points(det, truth, _scene(tmp_path), SAR_TIME,
                              PointValParams(max_dist_m=300), truth_area=area)
    assert (s["offset"]["dx_m"], s["offset"]["dy_m"]) == pytest.approx((2000, 1000), abs=1)
    assert s["truth_compared"] == 40          # the berg over masked pixels is not compared
    assert s["matched"] == 30
    assert s["recall"] == 0.75
    assert s["precision"] == round(30 / 31, 3)
    assert t.matched.sum() == 30 and d.matched.sum() == 30
