"""Kaggle Statoil/C-CORE data: loading, normalization, augmentation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

BANDS = ("HH", "HV")  # Kaggle band_1, band_2


@dataclass
class KaggleData:
    chips: np.ndarray  # (N, 2, 75, 75) dB, HH then HV
    inc_deg: np.ndarray  # (N,), NaN where Kaggle has "na" (all ships: a label leak)
    labels: np.ndarray  # (N,) 1 = iceberg
    ids: list[str]


def load_kaggle(path: Path) -> KaggleData:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    chips = np.stack([
        np.stack([np.asarray(r["band_1"], np.float32), np.asarray(r["band_2"], np.float32)])
        for r in rows
    ]).reshape(len(rows), 2, 75, 75)
    inc = np.array([np.nan if r["inc_angle"] == "na" else float(r["inc_angle"]) for r in rows])
    labels = np.array([int(r["is_iceberg"]) for r in rows], dtype=np.int64)
    return KaggleData(chips, inc, labels, [r["id"] for r in rows])


def select_bands(chips: np.ndarray, bands: list[str]) -> np.ndarray:
    return chips[:, [BANDS.index(b) for b in bands]]


@dataclass
class Normalizer:
    """Per-band dB mean/std and incidence mean/std, fitted on training data only."""

    band_mean: list[float]
    band_std: list[float]
    inc_mean: float
    inc_std: float

    @classmethod
    def fit(cls, chips: np.ndarray, inc_deg: np.ndarray) -> Normalizer:
        inc = inc_deg[np.isfinite(inc_deg)]
        return cls(
            band_mean=chips.mean(axis=(0, 2, 3)).tolist(),
            band_std=chips.std(axis=(0, 2, 3)).tolist(),
            inc_mean=float(inc.mean()),
            inc_std=float(inc.std()),
        )

    def chips(self, chips: np.ndarray) -> np.ndarray:
        m = np.asarray(self.band_mean, np.float32)[None, :, None, None]
        s = np.asarray(self.band_std, np.float32)[None, :, None, None]
        return ((chips - m) / s).astype(np.float32)

    def inc(self, inc_deg: np.ndarray) -> np.ndarray:
        """Standardized incidence; missing values become 0 (the mean), hiding the 'na' leak."""
        z = (inc_deg - self.inc_mean) / self.inc_std
        return np.nan_to_num(z, nan=0.0).astype(np.float32)


def dihedral(x: torch.Tensor, k: int) -> torch.Tensor:
    """One of the 8 flips/90-degree rotations of a (..., H, W) batch, k in 0..7."""
    if k >= 4:
        x = torch.flip(x, dims=(-1,))
    return torch.rot90(x, k % 4, dims=(-2, -1))


def augment(x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Random dihedral transform per sample (SAR targets have no preferred orientation here)."""
    ks = torch.randint(0, 8, (x.shape[0],), generator=generator)
    return torch.stack([dihedral(xi, int(k)) for xi, k in zip(x, ks, strict=True)])
