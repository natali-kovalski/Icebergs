"""Stratified k-fold training on the Kaggle chips; saves one model per fold plus metadata."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from torch import nn

from iceberg_sar.chips import kaggle_background_curve
from iceberg_sar.classify.dataset import KaggleData, Normalizer, augment, dihedral, select_bands
from iceberg_sar.classify.model import IcebergCNN


@dataclass(frozen=True)
class TrainParams:
    bands: tuple[str, ...] = ("HH", "HV")
    use_inc: bool = True
    folds: int = 5
    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    width: int = 16
    dropout: float = 0.3
    seed: int = 42


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_tta(model: IcebergCNN, x: np.ndarray, inc: np.ndarray, batch: int = 256) -> np.ndarray:
    """Iceberg probability, averaged over the 8 flips/rotations (test-time augmentation)."""
    dev = next(model.parameters()).device
    model.eval()
    out = []
    for i in range(0, len(x), batch):
        xb = torch.from_numpy(x[i:i + batch]).to(dev)
        ib = torch.from_numpy(inc[i:i + batch]).to(dev)
        p = torch.stack([torch.sigmoid(model(dihedral(xb, k), ib)) for k in range(8)]).mean(0)
        out.append(p.cpu().numpy())
    return np.concatenate(out) if out else np.empty(0, np.float32)


def train_one(
    x: np.ndarray, inc: np.ndarray, y: np.ndarray, params: TrainParams, seed: int
) -> IcebergCNN:
    """Fixed-epoch training (no early stopping, so CV scores are not tuned on the held-out fold)."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    dev = device()
    model = IcebergCNN(x.shape[1], params.use_inc, params.width, params.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=params.learning_rate,
                            weight_decay=params.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=params.epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    xt, it, yt = torch.from_numpy(x), torch.from_numpy(inc), torch.from_numpy(y.astype(np.float32))
    for _ in range(params.epochs):
        model.train()
        perm = torch.randperm(len(xt), generator=gen)
        for i in range(0, len(perm), params.batch_size):
            idx = perm[i:i + params.batch_size]
            xb = augment(xt[idx], gen).to(dev)
            loss = loss_fn(model(xb, it[idx].to(dev)), yt[idx].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    return model


def train_cv(data: KaggleData, params: TrainParams, out_dir: Path) -> dict:
    """Stratified k-fold CV. Writes fold_<k>.pt, meta.json, cv_metrics.json, oof.csv to out_dir.

    The saved folds form the ensemble used for prediction; the out-of-fold (OOF) predictions
    give the reported log loss and accuracy.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x_all = select_bands(data.chips, list(params.bands))
    y = data.labels
    oof = np.zeros(len(y), dtype=np.float64)
    skf = StratifiedKFold(n_splits=params.folds, shuffle=True, random_state=params.seed)
    folds_meta, fold_scores = [], []
    t0 = time.time()
    for k, (tr, va) in enumerate(skf.split(x_all, y)):
        norm = Normalizer.fit(x_all[tr], data.inc_deg[tr])
        model = train_one(norm.chips(x_all[tr]), norm.inc(data.inc_deg[tr]), y[tr], params,
                          params.seed + k)
        oof[va] = predict_tta(model, norm.chips(x_all[va]), norm.inc(data.inc_deg[va]))
        torch.save(model.state_dict(), out_dir / f"fold_{k}.pt")
        folds_meta.append({"fold": k, "normalizer": asdict(norm)})
        fold_scores.append({
            "fold": k,
            "log_loss": round(float(log_loss(y[va], oof[va], labels=[0, 1])), 4),
            "accuracy": round(float(accuracy_score(y[va], oof[va] > 0.5)), 4),
        })
    metrics = {
        "bands": list(params.bands),
        "use_inc": params.use_inc,
        "n": int(len(y)),
        "iceberg_fraction": round(float(y.mean()), 4),
        "oof_log_loss": round(float(log_loss(y, oof)), 4),
        "oof_accuracy": round(float(accuracy_score(y, oof > 0.5)), 4),
        # Baseline: always predict the class prior.
        "prior_log_loss": round(float(log_loss(y, np.full(len(y), y.mean()))), 4),
        "folds": fold_scores,
        "train_seconds": round(time.time() - t0, 1),
    }
    meta = {
        "params": asdict(params),
        "folds": folds_meta,
        "hv_background_curve": kaggle_background_curve(data.chips[:, 1], data.inc_deg),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "cv_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    lines = ["id,is_iceberg,oof_prob"] + [
        f"{i},{t},{p:.5f}" for i, t, p in zip(data.ids, y, oof, strict=True)
    ]
    (out_dir / "oof.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics
