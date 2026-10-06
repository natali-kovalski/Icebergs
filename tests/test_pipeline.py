"""End-to-end run: scene selection, matching granules to downloaded products, dry runs."""

from datetime import date
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box

from iceberg_sar import pipeline
from iceberg_sar.config import Config
from iceberg_sar.pipeline import RunParams, find_product, run_pipeline, select_scenes

G_0502 = "S1C_IW_GRDH_1SDH_20250502T094802_20250502T094827_002150_00490B_014B"
G_0508 = "S1A_IW_GRDH_1SDH_20250508T094925_20250508T094950_059101_075526_EB40"
G_0514 = "S1C_IW_GRDH_1SDH_20250514T094802_20250514T094827_002325_004E8A_DEAF"


def _scenes() -> gpd.GeoDataFrame:
    rows = [
        (G_0514, "2025-05-14T09:48:02Z", "HH+HV", 0.41),
        (G_0502, "2025-05-02T09:48:02Z", "HH+HV", 0.41),
        ("S1A_IW_GRDH_1SDV_20250505T000000_x", "2025-05-05T00:00:00Z", "VV+VH", 0.60),
        ("S1A_IW_GRDH_1SDH_20250506T000000_x", "2025-05-06T00:00:00Z", "HH+HV", 0.01),
        (G_0508, "2025-05-08T09:49:25Z", "HH+HV", 0.42),
    ]
    return gpd.GeoDataFrame(
        [dict(zip(["scene", "start_time", "polarization", "aoi_overlap"], r, strict=True))
         for r in rows],
        geometry=[box(-55, 49, -53, 51)] * len(rows), crs="EPSG:4326")


def test_select_filters_pol_and_overlap_and_sorts_by_time() -> None:
    out = select_scenes(_scenes(), RunParams(max_scenes=None))
    assert list(out.scene) == [G_0502, G_0508, G_0514]


def test_select_keeps_best_covering_when_capped() -> None:
    out = select_scenes(_scenes(), RunParams(max_scenes=1))
    assert list(out.scene) == [G_0508]


def test_select_any_pol() -> None:
    out = select_scenes(_scenes(), RunParams(pol=None, max_scenes=None))
    assert len(out) == 4


def test_find_product_matches_platform_time_and_resolution(tmp_path: Path) -> None:
    (tmp_path / "S1A_IW_20250508T094925_DHP_RTC20_G_gpuned_FDE3").mkdir()
    (tmp_path / "S1A_IW_20250508T094925_DHP_RTC10_G_gpuned_02C1").mkdir()
    (tmp_path / "S1A_IW_20250508T094925_DHP_RTC10_G_gpuned_02C1.zip").touch()
    assert find_product(tmp_path, G_0508, 10).name.endswith("RTC10_G_gpuned_02C1")
    assert find_product(tmp_path, G_0508, 20).name.endswith("RTC20_G_gpuned_FDE3")
    assert find_product(tmp_path, G_0502, 10) is None


def _cfg(tmp_path: Path) -> Config:
    return Config(root=tmp_path, raw={
        "paths": {"raw": "raw", "interim": "interim", "outputs": "outputs"},
        "hyp3": {"resolution": 10},
        "cfar": {"band": "HV"},
        "run": {"max_scenes": 2},
    })


def test_dry_run_orders_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "raw" / "S1A_IW_20250508T094925_DHP_RTC10_G_gpuned_02C1").mkdir(parents=True)
    monkeypatch.setattr("iceberg_sar.search.search_scenes", lambda cfg, start, end: _scenes())
    monkeypatch.setattr(pipeline, "order_and_download",
                        lambda *a: pytest.fail("dry run must not order"))
    lines: list[str] = []
    out = run_pipeline(_cfg(tmp_path), date(2025, 5, 1), date(2025, 5, 15), dry_run=True,
                       log=lines.append)
    assert out is None
    assert "downloaded" in lines[0] and G_0508 in lines[0]
    assert "to order" in lines[1] and G_0514 in lines[1]
    assert "1 to order (~60 HyP3 credits at 10 m)" in lines[-1]


def test_no_scenes_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("iceberg_sar.search.search_scenes",
                        lambda cfg, start, end: _scenes().iloc[:0])
    with pytest.raises(RuntimeError, match="No HH\\+HV scenes"):
        run_pipeline(_cfg(tmp_path), date(2025, 5, 1), date(2025, 5, 15), dry_run=True)
