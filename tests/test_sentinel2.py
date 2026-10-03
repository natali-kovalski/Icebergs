import numpy as np

from iceberg_sar.sentinel2 import OpticalParams, bright_targets, cloud_mask

P = OpticalParams()


def test_isolated_berg_kept_pack_ice_skipped() -> None:
    nir = np.full((200, 200), 0.02, dtype=np.float32)   # open water
    nir[50:53, 50:53] = 0.5                              # 30 m berg in open water
    nir[100:200, 100:200] = 0.4                          # pack ice ...
    nir[150:152, 150:152] = 0.7                          # ... with a bright floe inside
    t = bright_targets(nir, np.zeros_like(nir, bool), 10.0, P)
    assert len(t) == 1
    assert (round(t.row[0]), round(t.col[0])) == (51, 51)
    assert t.area_m2[0] == 900


def test_berg_next_to_ice_edge_fails_ring_test() -> None:
    nir = np.full((100, 100), 0.02, dtype=np.float32)
    nir[:, 60:] = 0.4                                    # ice edge
    nir[50:52, 52:54] = 0.5                              # 60 m from the edge, ring is ~half ice
    t = bright_targets(nir, np.zeros_like(nir, bool), 10.0, P)
    assert t.empty


def test_cloud_mask_ignores_small_cloud_blobs() -> None:
    scl = np.full((300, 300), 6, dtype=np.uint8)          # water
    scl[10:12, 10:12] = 9                                 # berg flagged as cloud by SCL
    scl[150:, 150:] = 9                                   # real cloud bank, 2.25 km2
    m = cloud_mask(scl, 10.0, P)
    assert not m[10:12, 10:12].any()
    assert m[150:, 150:].all()
    assert m[110, 200]                                    # buffered by 500 m


def test_swir_rejects_small_clouds() -> None:
    nir = np.full((100, 100), 0.02, dtype=np.float32)
    swir = np.full_like(nir, 0.01)
    nir[20:23, 20:23] = 0.5                              # berg: dark in SWIR
    swir[20:23, 20:23] = 0.05
    nir[70:73, 70:73] = 0.5                              # small cumulus: bright in SWIR
    swir[70:73, 70:73] = 0.35
    t = bright_targets(nir, np.zeros_like(nir, bool), 10.0, P, swir)
    assert [(round(r), round(c)) for r, c in zip(t.row, t.col, strict=True)] == [(21, 21)]
