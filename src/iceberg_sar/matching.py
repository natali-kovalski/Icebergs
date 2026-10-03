"""Match SAR detections to ground-truth points and score them.

Truth points come from other sensors or observers (IIP sightings, Sentinel-2 targets).
When they are not simultaneous with the SAR pass, bergs drift in between (typically
0.1-0.5 m/s, i.e. a few km in 5 h). Nearby bergs drift together with the current, so a
single offset per area is a good first model: `estimate_offset` finds the shift that lines
up the most points, and reports the match count a random shift would give, so a weak
peak (no real alignment) is visible.

All coordinates are metric (x, y) arrays of shape (n, 2) in one projected CRS.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class MatchResult:
    pairs: np.ndarray  # (k, 2) indices into (detections, truth)
    distances_m: np.ndarray  # (k,)
    n_detections: int
    n_truth: int

    @property
    def true_positives(self) -> int:
        return len(self.pairs)

    @property
    def precision(self) -> float:
        return self.true_positives / self.n_detections if self.n_detections else float("nan")

    @property
    def recall(self) -> float:
        return self.true_positives / self.n_truth if self.n_truth else float("nan")


@dataclass(frozen=True)
class OffsetResult:
    dx_m: float
    dy_m: float
    n_matched: int  # truth points with a detection within the radius at the best offset
    chance_n_matched: float  # median over all tried offsets: what alignment by luck gives


def match_points(det: np.ndarray, truth: np.ndarray, max_dist_m: float) -> MatchResult:
    """One-to-one matching minimising total distance; pairs beyond `max_dist_m` are dropped."""
    n_det, n_truth = len(det), len(truth)
    if n_det == 0 or n_truth == 0:
        return MatchResult(np.empty((0, 2), int), np.empty(0), n_det, n_truth)
    d = np.linalg.norm(det[:, None, :] - truth[None, :, :], axis=2)
    cost = np.where(d <= max_dist_m, d, max_dist_m * 1e3)  # unmatched pairs are very expensive
    rows, cols = linear_sum_assignment(cost)
    ok = d[rows, cols] <= max_dist_m
    pairs = np.column_stack([rows[ok], cols[ok]])
    return MatchResult(pairs, d[rows[ok], cols[ok]], n_det, n_truth)


def estimate_offset(
    det: np.ndarray,
    truth: np.ndarray,
    search_m: float,
    step_m: float,
    radius_m: float,
) -> OffsetResult:
    """Shift (dx, dy) with truth ~ det + shift that gives the most truth points near a detection."""
    if len(det) == 0 or len(truth) == 0:
        return OffsetResult(0.0, 0.0, 0, 0.0)
    tree = cKDTree(det)
    shifts = np.arange(-search_m, search_m + step_m / 2, step_m)
    counts = np.zeros((len(shifts), len(shifts)), dtype=int)
    for i, dy in enumerate(shifts):
        for j, dx in enumerate(shifts):
            dist, _ = tree.query(truth - [dx, dy], distance_upper_bound=radius_m)
            counts[i, j] = int(np.isfinite(dist).sum())
    i, j = np.unravel_index(np.argmax(counts), counts.shape)
    # Neighbouring grid shifts often tie; refine with the median displacement of the pairs
    # found at the peak, which is not limited to the grid step.
    dist, k = tree.query(truth - [shifts[j], shifts[i]], distance_upper_bound=radius_m)
    hit = np.isfinite(dist)
    dx, dy = np.median(truth[hit] - det[k[hit]], axis=0) if hit.any() else (0.0, 0.0)
    dist, _ = tree.query(truth - [dx, dy], distance_upper_bound=radius_m)
    return OffsetResult(round(float(dx), 1), round(float(dy), 1), int(np.isfinite(dist).sum()),
                        float(np.median(counts)))
