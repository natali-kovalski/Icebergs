"""75x75 dB chips around detections, preprocessed to look like the Kaggle Statoil/C-CORE chips.

Measured on this project's data (see README): Kaggle chips have ~10 m pixels like our 10 m RTC
(same speckle correlation and ENL), so chips are cut at native resolution. Two radiometric
differences remain and are corrected here:

- HyP3 RTC is gamma0; Kaggle is sigma0. Over flat sea, sigma0 = gamma0 * cos(incidence).
- HyP3 removes the thermal noise floor, so calm-sea HV is near zero (~-44 dB). Kaggle HV keeps
  it (~-24 to -29 dB, depending on incidence). We add speckled noise until the chip's HV
  background median matches the Kaggle background at the same incidence angle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

KAGGLE_SIZE = 75
RING_EXCLUDE = 31  # central box excluded when measuring a chip's background


@dataclass(frozen=True)
class ChipParams:
    size_px: int = KAGGLE_SIZE
    to_sigma0: bool = True
    hv_noise_floor: bool = True
    noise_looks: float = 4.5  # ENL of the added noise (Kaggle HV background ENL)
    seed: int = 42


def background_mask(size: int, exclude: int = RING_EXCLUDE) -> np.ndarray:
    """True outside the central `exclude` x `exclude` box."""
    m = np.ones((size, size), dtype=bool)
    c0 = (size - exclude) // 2
    m[c0:c0 + exclude, c0:c0 + exclude] = False
    return m


def kaggle_background_curve(
    hv_db: np.ndarray, inc_deg: np.ndarray, bin_deg: float = 3.0, min_count: int = 30
) -> list[tuple[float, float]]:
    """Median HV background (dB) of Kaggle chips per incidence-angle bin: [(inc, dB), ...].

    `hv_db` is (N, 75, 75); chips with NaN incidence are skipped.
    """
    ring = background_mask(hv_db.shape[-1])
    bg = np.median(hv_db[:, ring], axis=1)
    ok = np.isfinite(inc_deg)
    bg, inc = bg[ok], inc_deg[ok]
    if len(inc) == 0:
        raise ValueError("No chips with incidence angles to fit the HV background curve")
    knots = []
    for lo in np.arange(np.floor(inc.min()), inc.max() + bin_deg, bin_deg):
        sel = (inc >= lo) & (inc < lo + bin_deg)
        if sel.sum() >= min_count:
            knots.append((float(np.median(inc[sel])), float(np.median(bg[sel]))))
    # Too few chips per bin: one flat level.
    return knots or [(float(np.median(inc)), float(np.median(bg)))]


def interp_curve(curve: list[tuple[float, float]], inc_deg: float) -> float:
    """Piecewise-linear lookup, clamped at the ends."""
    x, y = np.array(curve, dtype=np.float64).T
    return float(np.interp(inc_deg, x, y))


def speckled_noise(shape: tuple[int, int], looks: float, rng: np.random.Generator) -> np.ndarray:
    """Unit-mean speckled noise with ENL ~`looks` and ~0.5 lag-1 correlation (like 10 m GRD).

    White gamma noise with looks/4 shape, then a 2x2 box mean (which multiplies ENL by 4).
    """
    k = max(looks / 4.0, 0.25)
    white = rng.gamma(k, 1.0 / k, size=(shape[0] + 1, shape[1] + 1))
    return 0.25 * (white[:-1, :-1] + white[1:, :-1] + white[:-1, 1:] + white[1:, 1:])


def add_noise_floor(
    lin: np.ndarray, target_db: float, looks: float, rng: np.random.Generator
) -> np.ndarray:
    """Add noise so the background median of `lin` reaches `target_db` (no-op if already above)."""
    ring = background_mask(lin.shape[0])
    target = 10.0 ** (target_db / 10.0)
    if np.median(lin[ring]) >= target:
        return lin
    noise = speckled_noise(lin.shape, looks, rng)
    lo, hi = 0.0, 4.0 * target
    for _ in range(30):  # bisection on the noise level
        mid = 0.5 * (lo + hi)
        if np.median((lin + mid * noise)[ring]) < target:
            lo = mid
        else:
            hi = mid
    return lin + 0.5 * (lo + hi) * noise


def to_db(lin: np.ndarray) -> np.ndarray:
    return (10.0 * np.log10(np.maximum(lin, 1e-10))).astype(np.float32)


def fill_invalid(lin: np.ndarray) -> tuple[np.ndarray, float]:
    """Replace nodata (0 / NaN) by the median of valid pixels; return the valid fraction."""
    valid = np.isfinite(lin) & (lin > 0)
    frac = float(valid.mean())
    if frac == 0:
        return np.full_like(lin, 1e-10), 0.0
    out = np.where(valid, lin, np.median(lin[valid]))
    return out, frac


def read_window(path: Path, row: float, col: float, size: int) -> np.ndarray:
    """size x size window centred on (row, col); outside the raster is 0 (nodata)."""
    half = size // 2
    r0, c0 = int(round(row)) - half, int(round(col)) - half
    with rasterio.open(path) as src:
        a = src.read(1, window=Window(c0, r0, size, size), boundless=True, fill_value=0)
    return a.astype(np.float64)


def make_chip(
    bands_lin: dict[str, np.ndarray],
    inc_deg: float,
    params: ChipParams,
    hv_curve: list[tuple[float, float]] | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Linear gamma0 HH/HV windows -> (2, size, size) dB chip in Kaggle order (HH, HV).

    Returns the chip and the smallest valid-pixel fraction of the two bands.
    """
    out, fracs = [], []
    for band in ("HH", "HV"):
        lin, frac = fill_invalid(bands_lin[band])
        fracs.append(frac)
        if params.to_sigma0 and np.isfinite(inc_deg):
            lin = lin * np.cos(np.radians(inc_deg))
        if band == "HV" and params.hv_noise_floor:
            if hv_curve is None:
                raise ValueError("hv_noise_floor needs the Kaggle HV background curve")
            lin = add_noise_floor(lin, interp_curve(hv_curve, inc_deg), params.noise_looks, rng)
        out.append(to_db(lin))
    return np.stack(out), min(fracs)


def extract_chips(
    band_paths: dict[str, Path],
    rows: np.ndarray,
    cols: np.ndarray,
    inc_deg: np.ndarray,
    params: ChipParams,
    hv_curve: list[tuple[float, float]] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Chips (N, 2, size, size) in dB and their valid fractions, for detections at (row, col).

    `band_paths` maps HH and HV to the raw HyP3 RTC GeoTIFFs (linear gamma0, 0 = nodata).
    The noise draw is seeded per detection index, so reruns give identical chips.
    """
    missing = {"HH", "HV"} - set(band_paths)
    if missing:
        raise ValueError(f"Chips need HH and HV bands; missing {sorted(missing)}")
    chips = np.empty((len(rows), 2, params.size_px, params.size_px), dtype=np.float32)
    valid = np.empty(len(rows), dtype=np.float32)
    for i, (r, c, inc) in enumerate(zip(rows, cols, inc_deg, strict=True)):
        wins = {b: read_window(band_paths[b], r, c, params.size_px) for b in ("HH", "HV")}
        rng = np.random.default_rng([params.seed, i])
        chips[i], valid[i] = make_chip(wins, float(inc), params, hv_curve, rng)
    return chips, valid


def chip_background_db(chips: np.ndarray) -> np.ndarray:
    """Median background (dB) per chip and band: (N, 2). Used to compare with Kaggle."""
    ring = background_mask(chips.shape[-1])
    return np.median(chips[..., ring], axis=-1)


def lag1_correlation(lin: np.ndarray) -> float:
    """Mean of row and column lag-1 correlation of a 2-D array (speckle oversampling check)."""
    rx = np.corrcoef(lin[:, :-1].ravel(), lin[:, 1:].ravel())[0, 1]
    ry = np.corrcoef(lin[:-1, :].ravel(), lin[1:, :].ravel())[0, 1]
    return float(0.5 * (rx + ry))

