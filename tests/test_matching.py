import numpy as np
import pytest

from iceberg_sar.matching import estimate_offset, match_points


def test_match_is_one_to_one_and_respects_distance() -> None:
    det = np.array([[0.0, 0.0], [100.0, 0.0], [5000.0, 0.0]])
    truth = np.array([[10.0, 0.0], [20.0, 0.0], [9000.0, 0.0]])
    m = match_points(det, truth, max_dist_m=200)
    assert m.true_positives == 2  # both near truths matched, each to its own detection
    assert sorted(m.pairs[:, 1].tolist()) == [0, 1]
    assert m.precision == pytest.approx(2 / 3)
    assert m.recall == pytest.approx(2 / 3)


def test_match_empty_inputs() -> None:
    m = match_points(np.empty((0, 2)), np.array([[0.0, 0.0]]), 100)
    assert m.true_positives == 0
    assert m.recall == 0.0
    assert np.isnan(m.precision)


def test_offset_recovers_common_drift() -> None:
    rng = np.random.default_rng(1)
    det = rng.uniform(0, 50_000, size=(80, 2))
    truth = det[:60] + [3_000.0, -1_500.0] + rng.normal(0, 100, size=(60, 2))
    off = estimate_offset(det, truth, search_m=6_000, step_m=250, radius_m=500)
    assert (off.dx_m, off.dy_m) == pytest.approx((3_000, -1_500), abs=250)
    assert off.n_matched >= 55
    assert off.chance_n_matched < 15
