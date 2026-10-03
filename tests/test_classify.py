"""Classifier: augmentation, model shapes, and a tiny train -> save -> load round trip."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from iceberg_sar.classify.dataset import KaggleData, Normalizer, dihedral  # noqa: E402
from iceberg_sar.classify.model import IcebergCNN  # noqa: E402
from iceberg_sar.classify.predict import load_ensemble  # noqa: E402
from iceberg_sar.classify.train import TrainParams, train_cv  # noqa: E402


def test_dihedral_gives_eight_distinct_views() -> None:
    x = torch.arange(9.0).view(1, 3, 3)
    views = {tuple(dihedral(x, k).flatten().tolist()) for k in range(8)}
    assert len(views) == 8


def test_normalizer_imputes_missing_incidence_to_mean() -> None:
    chips = np.random.default_rng(0).normal(-20, 3, size=(10, 2, 75, 75)).astype(np.float32)
    inc = np.array([30.0, 40.0] * 4 + [np.nan, np.nan])
    n = Normalizer.fit(chips, inc)
    z = n.chips(chips)
    assert abs(z.mean()) < 1e-3 and z.std() == pytest.approx(1.0, abs=1e-3)
    assert n.inc(inc)[-1] == 0.0


@pytest.mark.parametrize("in_ch,use_inc", [(2, True), (1, False)])
def test_model_output_shape(in_ch: int, use_inc: bool) -> None:
    m = IcebergCNN(in_ch, use_inc, width=4)
    out = m(torch.zeros(3, in_ch, 75, 75), torch.zeros(3) if use_inc else None)
    assert out.shape == (3,)


def _toy_data(n: int = 40) -> KaggleData:
    """Icebergs: bright compact blob. Ships: no blob. Learnable in a couple of epochs."""
    rng = np.random.default_rng(0)
    chips = rng.normal(-25, 2, size=(n, 2, 75, 75)).astype(np.float32)
    labels = np.array([0, 1] * (n // 2))
    chips[labels == 1, :, 33:42, 33:42] += 15
    inc = rng.uniform(30, 45, n)
    return KaggleData(chips, inc, labels, [f"id{i}" for i in range(n)])


def test_train_cv_round_trip(tmp_path: Path) -> None:
    data = _toy_data()
    params = TrainParams(bands=("HH",), folds=2, epochs=15, batch_size=8, width=4,
                         learning_rate=3e-3)
    metrics = train_cv(data, params, tmp_path)

    assert {"oof_log_loss", "oof_accuracy", "prior_log_loss"} <= metrics.keys()
    assert (tmp_path / "fold_0.pt").is_file() and (tmp_path / "fold_1.pt").is_file()
    ens = load_ensemble(tmp_path)
    assert ens.bands == ["HH"] and len(ens.members) == 2
    p = ens.predict(data.chips, data.inc_deg)
    assert p.shape == (40,) and ((p >= 0) & (p <= 1)).all()
    assert p[data.labels == 1].mean() > p[data.labels == 0].mean()
