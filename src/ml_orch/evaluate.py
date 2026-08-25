"""Turning predictions into an `EvalReport` the gates can adjudicate.

The gates never see a model. They see an `EvalReport`: per-target metrics, the
per-molecule error vectors the significance test resamples, the per-target
standard deviations the normalisation policy divides by, and per-slice
summaries. That separation is why the entire gating layer is unit-testable
without fitting anything, and why swapping the model backend cannot change how
a promotion decision is reached.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .featurize import derive_slice_columns, slice_masks
from .gates import normalized_score
from .types import EvalReport, TargetMetrics


def _metrics(target: str, y_true: np.ndarray, y_pred: np.ndarray) -> TargetMetrics:
    errors = np.abs(y_true - y_pred)
    mae = float(errors.mean()) if errors.size else float("nan")
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if errors.size else float("nan")
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - float(np.sum((y_true - y_pred) ** 2)) / denom if denom else float("nan")
    return TargetMetrics(target=target, mae=mae, rmse=rmse, r2=r2, errors=errors)


def evaluate(predictor: Any, X: np.ndarray, frame: pd.DataFrame,
             targets: Sequence[str], *, model_version: str, eval_set: str,
             slice_spec: Mapping[str, Any] | None = None) -> EvalReport:
    """Score `predictor` on one evaluation set and package the result.

    `frame` supplies the ground truth and the columns slices are cut on. `X` is
    passed separately rather than derived here, because the featurizer that
    produced it must be the one bound to the model — deriving it inside this
    function is exactly the shortcut that reintroduces train/serve skew.
    """
    preds = predictor.predict(X)
    per_target: dict[str, TargetMetrics] = {}
    target_std: dict[str, float] = {}
    for t in targets:
        y_true = frame[t].to_numpy(dtype=float)
        per_target[t] = _metrics(t, y_true, np.asarray(preds[t], dtype=float))
        std = float(np.std(y_true))
        target_std[t] = std if std > 0 else 1.0

    report = EvalReport(model_version=model_version, eval_set=eval_set,
                        per_target=per_target, target_std=target_std)

    if slice_spec:
        enriched = derive_slice_columns(frame)
        masks = slice_masks(enriched, dict(slice_spec))
        slices: dict[str, dict[str, float]] = {}
        for name, mask in masks.items():
            if mask.sum() < 20:
                continue
            parts = [
                float(np.mean(per_target[t].errors[mask]) / target_std[t]) for t in targets
            ]
            slices[name] = {"normalized_mae": float(np.mean(parts)),
                            "n": int(mask.sum())}
        report = EvalReport(model_version=model_version, eval_set=eval_set,
                            per_target=per_target, target_std=target_std,
                            slices=slices)
    return report


def summarise(report: EvalReport, targets: Sequence[str] | None = None) -> dict[str, Any]:
    """A JSON-safe digest for the registry and the manifest.

    Error vectors are deliberately dropped: they are large, and they belong to
    the run rather than to the record of it.
    """
    names = list(targets or report.targets)
    return {
        "model_version": report.model_version,
        "eval_set": report.eval_set,
        "normalized_score": normalized_score(report, names),
        "per_target": {
            t: {"mae": report.per_target[t].mae,
                "rmse": report.per_target[t].rmse,
                "r2": report.per_target[t].r2,
                "n": report.per_target[t].n}
            for t in names
        },
        "slices": {k: dict(v) for k, v in report.slices.items()},
    }
