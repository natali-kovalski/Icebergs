"""Chip preprocessing: sigma0 conversion, HV noise floor, extraction from rasters."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from iceberg_sar.chips import (
    ChipParams,
    add_noise_floor,
    background_mask,
    chip_background_db,
    extract_chips,
    interp_curve,
    kaggle_background_curve,
    lag1_correlation,
    make_chip,
    speckled_noise,
)

CURVE = [(32.0, -24.0), (44.0, -29.0)]


def test_speckled_noise_matches_10m_grd_statistics() -> None:
    n = speckled_noise((400, 400), looks=4.5, rng=np.random.default_rng(0))
    assert n.mean() == pytest.approx(1.0, abs=0.02)
    assert n.mean() ** 2 / n.var() == pytest.approx(4.5, rel=0.1)  # ENL
    assert lag1_correlation(n) == pytest.approx(0.5, abs=0.05)  # Kaggle ~0.55


def test_noise_floor_reaches_target_median() -> None:
    rng = np.random.default_rng(1)
    lin = rng.gamma(1.0, 10 ** (-4.4), size=(75, 75))  # ~-44 dB noise-subtracted HV
    out = add_noise_floor(lin, -26.0, 4.5, rng)
    ring = background_mask(75)
    assert 10 * np.log10(np.median(out[ring])) == pytest.approx(-26.0, abs=0.05)


def test_noise_floor_leaves_bright_background_alone() -> None:
    lin = np.full((75, 75), 10 ** -2.0)  # -20 dB, above the -26 dB floor
    assert add_noise_floor(lin, -26.0, 4.5, np.random.default_rng(0)) is lin


def test_interp_curve_clamps() -> None:
    assert interp_curve(CURVE, 38.0) == pytest.approx(-26.5)
    assert interp_curve(CURVE, 20.0) == -24.0
    assert interp_curve(CURVE, 50.0) == -29.0


def test_kaggle_background_curve_bins_by_incidence() -> None:
    inc = np.concatenate([np.full(40, 33.0), np.full(40, 42.0), [np.nan]])
    hv = np.concatenate([np.full((40, 75, 75), -24.0), np.full((40, 75, 75), -28.0),
                         np.full((1, 75, 75), 0.0)])
    assert kaggle_background_curve(hv, inc) == [(33.0, -24.0), (42.0, -28.0)]


def test_make_chip_sigma0_and_order() -> None:
    hh = np.full((75, 75), 0.01)  # -20 dB gamma0
    hv = np.full((75, 75), 0.01)
    chip, valid = make_chip({"HH": hh, "HV": hv}, 60.0, ChipParams(hv_noise_floor=False), None,
                            np.random.default_rng(0))
    assert chip.shape == (2, 75, 75)
    assert valid == 1.0
    # cos(60 deg) = 0.5 -> -3 dB
    assert chip[0, 0, 0] == pytest.approx(-23.01, abs=0.01)


def test_make_chip_fills_nodata_with_median() -> None:
    hh = np.full((75, 75), 0.01)
    hh[:, :15] = 0.0
    chip, valid = make_chip({"HH": hh, "HV": hh.copy()}, 0.0,
                            ChipParams(to_sigma0=False, hv_noise_floor=False), None,
                            np.random.default_rng(0))
    assert valid == pytest.approx(60 / 75)
    assert np.allclose(chip, -20.0, atol=1e-4)


def _write(path: Path, data: np.ndarray) -> Path:
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
                       count=1, dtype="float32", crs="EPSG:32621",
                       transform=from_origin(600_000, 5_600_000, 10, 10), nodata=0) as dst:
        dst.write(data.astype(np.float32), 1)
    return path


def test_extract_chips_centred_and_reproducible(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    hh = rng.gamma(4.0, 0.01 / 4.0, size=(200, 200))
    hh[100:103, 120:123] = 1.0  # bright target, centre (101, 121)
    hv = rng.gamma(1.0, 10 ** -4.4, size=(200, 200))
    paths = {"HH": _write(tmp_path / "hh.tif", hh), "HV": _write(tmp_path / "hv.tif", hv)}
    args = (paths, np.array([101.0, 5.0]), np.array([121.0, 5.0]), np.array([38.0, 38.0]),
            ChipParams(), CURVE)
    chips, valid = extract_chips(*args)
    again, _ = extract_chips(*args)

    assert chips.shape == (2, 2, 75, 75)
    assert np.unravel_index(np.argmax(chips[0, 0]), (75, 75)) in {(36, 36), (36, 37), (37, 36),
                                                                  (37, 37), (38, 38), (36, 38),
                                                                  (38, 36), (37, 38), (38, 37)}
    assert valid[0] == 1.0
    assert valid[1] < 0.5  # chip hangs off the raster corner
    assert np.array_equal(chips, again)
    assert chip_background_db(chips)[0, 1] == pytest.approx(-26.5, abs=0.1)
