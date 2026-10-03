"""Load a trained fold ensemble and score chips."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from iceberg_sar.classify.dataset import Normalizer, select_bands
from iceberg_sar.classify.model import IcebergCNN
from iceberg_sar.classify.train import device, predict_tta


@dataclass
class Ensemble:
    bands: list[str]
    use_inc: bool
    members: list[tuple[IcebergCNN, Normalizer]]
    hv_background_curve: list[tuple[float, float]]

    def predict(self, chips_db: np.ndarray, inc_deg: np.ndarray) -> np.ndarray:
        """Mean iceberg probability over folds (8-way TTA each). chips: (N, 2, 75, 75) HH, HV."""
        x = select_bands(chips_db.astype(np.float32), self.bands)
        probs = [predict_tta(m, n.chips(x), n.inc(inc_deg)) for m, n in self.members]
        return np.mean(probs, axis=0) if probs else np.empty(0)


def load_ensemble(model_dir: Path) -> Ensemble:
    model_dir = Path(model_dir)
    meta_path = model_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"No trained model in {model_dir}; run `train-classifier` first")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    p = meta["params"]
    dev = device()
    members = []
    for f in meta["folds"]:
        model = IcebergCNN(len(p["bands"]), p["use_inc"], p["width"], p["dropout"]).to(dev)
        state = torch.load(model_dir / f"fold_{f['fold']}.pt", map_location=dev, weights_only=True)
        model.load_state_dict(state)
        members.append((model, Normalizer(**f["normalizer"])))
    curve = [tuple(k) for k in meta["hv_background_curve"]]
    return Ensemble(list(p["bands"]), bool(p["use_inc"]), members, curve)
