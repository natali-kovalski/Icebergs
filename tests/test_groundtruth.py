"""Ground-truth comparison tests (no network)."""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from iceberg_sar.groundtruth import compare_counts, load_degree_counts


@pytest.fixture
def chart(tmp_path: Path) -> gpd.GeoDataFrame:
    p = tmp_path / "counts.csv"
    p.write_text("# comment\nlat_min,lon_min,count\n50,-55,10\n50,-54,4\n", encoding="utf-8")
    return load_degree_counts(p)


def test_load_degree_counts(chart: gpd.GeoDataFrame) -> None:
    assert chart.crs.to_epsg() == 4326
    assert chart.geometry.iloc[0].bounds == (-55, 50, -54, 51)


def test_compare_counts_scales_by_coverage(chart: gpd.GeoDataFrame) -> None:
    det = gpd.GeoDataFrame(
        {"lat": [50.5, 50.2, 50.9], "lon": [-54.5, -54.1, -53.5]},
        geometry=[Point(-54.5, 50.5), Point(-54.1, 50.2), Point(-53.5, 50.9)], crs="EPSG:4326",
    )
    cov = pd.DataFrame({"lat_min": [50, 50], "lon_min": [-55, -54],
                        "valid_km2": [3900.0, 780.0], "valid_fraction": [0.5, 0.1]})
    out = compare_counts(det, chart, cov, min_valid_fraction=0.2).set_index("lon_min")
    assert out.loc[-55, "sar_detections"] == 2 and out.loc[-55, "expected"] == 5.0
    assert out.loc[-54, "sar_detections"] == 1 and not out.loc[-54, "compared"]
