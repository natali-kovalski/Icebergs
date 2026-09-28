"""End-to-end detection on a small synthetic masked scene (no network)."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from iceberg_sar.cfar import CfarParams
from iceberg_sar.detections import (
    DetectionParams,
    detect_raster,
    scene_timestamp,
    to_geodataframe,
)

CRS = "EPSG:32621"
PIX = 20.0
X0, Y0 = 600_000.0, 5_600_000.0
H = W = 300
NAME = "S1A_IW_20250508T094925_DHP_RTC20_G_gpuned_ABCD"

COMPACT = (100, 100)          # 2x2 bright target
ON_BLOCK_EDGE = (49, 200)     # 2x2 target straddling rows 49/50 (block_rows=50)
NEAR_MASK = (244, 270)        # 6 px from the masked patch below it


def _write(path: Path, data: np.ndarray) -> Path:
    with rasterio.open(
        path, "w", driver="GTiff", height=H, width=W, count=1, dtype="float32",
        crs=CRS, transform=from_origin(X0, Y0, PIX, PIX), nodata=np.nan,
    ) as dst:
        dst.write(data.astype(np.float32), 1)
    return path


@pytest.fixture
def scene(tmp_path: Path) -> dict[str, Path]:
    rng = np.random.default_rng(0)
    hv = rng.gamma(4.0, 0.001 / 4.0, size=(H, W))
    for r, c in (COMPACT, ON_BLOCK_EDGE, NEAR_MASK):
        hv[r : r + 2, c : c + 2] = 0.1                # +20 dB
    hv[160, 30:110] = 0.03                            # 1.6 km ice strip, +15 dB
    hv[160, [50, 51, 80, 81]] = 0.1                   # bright knots on the strip
    hv[250:, 250:] = np.nan                           # masked (land / pack ice)
    return {
        "HV": _write(tmp_path / f"{NAME}_HV.tif", hv),
        "HH": _write(tmp_path / f"{NAME}_HH.tif", hv * 10),
    }


def _run(scene: dict[str, Path], block_rows: int) -> tuple:
    cfar = CfarParams(guard_px=3, background_px=10, pfa=1e-6, enl=4.0)
    p = DetectionParams(min_area_px=2, max_area_px=500, edge_buffer_px=10, max_extent_px=20,
                        max_structure_px=15, block_rows=block_rows)
    return detect_raster(scene, "HV", cfar, p)


@pytest.mark.parametrize("block_rows", [50, 1024])
def test_detects_compact_targets_only(scene: dict[str, Path], block_rows: int) -> None:
    table, _, _ = _run(scene, block_rows)
    rows, cols = table["row"] - 0.5, table["col"] - 0.5   # centroid of a 2x2 block
    found = {(round(r), round(c)) for r, c in zip(rows, cols, strict=True)}
    assert found == {COMPACT, ON_BLOCK_EDGE}   # strip and near-mask target rejected
    assert (table["area_px"] == 4).all()
    assert table["peak_db_HV"].to_numpy() == pytest.approx(-10.0, abs=0.01)
    assert table["peak_db_HH"].to_numpy() == pytest.approx(0.0, abs=0.01)
    assert (table["contrast_db"] > 15).all()


def test_geodataframe_attributes(scene: dict[str, Path]) -> None:
    table, transform, crs = _run(scene, 1024)
    gdf = to_geodataframe(table, transform, crs, NAME, inc_path=None)
    assert gdf.crs.to_epsg() == 4326
    assert list(gdf["id"]) == ["20250508T094925_00000", "20250508T094925_00001"]
    assert (gdf["timestamp"] == "2025-05-08T09:49:25Z").all()
    assert (gdf["area_m2"] == 4 * PIX * PIX).all()
    assert gdf["lon"].between(-58, -52).all() and gdf["lat"].between(49, 52).all()


def test_scene_timestamp() -> None:
    assert scene_timestamp(NAME) == "2025-05-08T09:49:25Z"
    with pytest.raises(ValueError):
        scene_timestamp("no_time_here")
