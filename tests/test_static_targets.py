"""Static targets: repeat detections at one spot on several dates (rocks, not bergs)."""

from datetime import date

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

from iceberg_sar.static_targets import StaticParams, find_static, is_static

UTM = "EPSG:32621"


def _pts(xy: list[tuple[float, float]]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in xy], crs=UTM).to_crs("EPSG:4326")


ROCK = (650_000.0, 5_600_000.0)


def _scenes() -> list[tuple[date, gpd.GeoDataFrame]]:
    # The rock repeats within 10 m; the berg drifts 2 km; the same-day pair is two frames.
    return [
        (date(2026, 9, 25), _pts([ROCK, (660_000, 5_610_000)])),
        (date(2026, 9, 25), _pts([(ROCK[0] + 5, ROCK[1])])),
        (date(2026, 10, 1), _pts([(ROCK[0] - 8, ROCK[1] + 6), (662_000, 5_610_000)])),
    ]


def test_find_static_needs_distinct_dates() -> None:
    static = find_static(_scenes(), StaticParams(radius_m=50, min_dates=2))
    assert len(static) == 1
    assert static.iloc[0]["n_dates"] == 2
    assert (static.iloc[0]["first"], static.iloc[0]["last"]) == ("2026-09-25", "2026-10-01")
    s = static.to_crs(UTM).geometry.iloc[0]
    assert abs(s.x - ROCK[0]) < 10 and abs(s.y - ROCK[1]) < 10
    # Same-day frames alone are not evidence of a fixed target.
    assert find_static(_scenes()[:2], StaticParams(min_dates=2)).empty


def test_find_static_min_span_keeps_short_lived_targets() -> None:
    assert find_static(_scenes(), StaticParams(min_dates=2, min_span_days=30)).empty


def test_is_static_flags_only_nearby_detections() -> None:
    p = StaticParams(radius_m=50)
    static = find_static(_scenes(), p)
    dets = _pts([(ROCK[0] + 30, ROCK[1]), (ROCK[0] + 500, ROCK[1])])
    np.testing.assert_array_equal(is_static(dets, static, p), [True, False])
    assert is_static(dets.iloc[:0], static, p).size == 0


def test_find_static_empty_input() -> None:
    assert find_static([], StaticParams()).empty
