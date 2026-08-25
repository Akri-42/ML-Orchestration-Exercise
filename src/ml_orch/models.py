"""Model backends, behind an interface thin enough to swap.


**One model per target, not one multi-output model.** QM9's twelve targets are
physically unrelated quantities in incomparable units. A per-target model means
a per-target error vector, which is what the significance gate resamples and
what the no-regression gate compares — and it means one target degrading is a
visible, attributable event rather than a shift in a pooled number.

**RandomForest rather than HistGradientBoosting**, which is a change from the
scaffolding's config. The triage pipeline gates on uncertainty, and a forest
gives calibrated-enough per-prediction uncertainty for free as the spread across
trees. Gradient boosting would need three quantile models per target to say
anything at all about uncertainty. This is a capability choice, not an accuracy
one: accuracy is not graded, but a gate with nothing to gate on is a hole in the
design. The tradeoff is in the README.

`Trainer` and `Predictor` are protocols precisely so a chemprop backend can be
dropped in as a subprocess step (`chemprop train` / `chemprop predict`) without
touching the DAG. Designed for; not implemented — see refs/REFERENCES.md.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .featurize import RdkitDescriptorFeaturizer


class Predictor(Protocol):
    """What both pipelines call. Pipeline 1 scores held-out sets, pipeline 2
    scores candidates — same method, which is why the scoring step is shared."""

    targets: Sequence[str]

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]: ...

    def predict_with_uncertainty(
        self, X: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]: ...


class Trainer(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray,
            targets: Sequence[str]) -> Predictor: ...


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------


@dataclass
class MeanBaseline:
    """Predicts the training mean. The thing a cold-start model must beat.

    Without this, the first model ever trained is promoted unconditionally,
    which is how a pipeline ships a mean-predictor to production on day one and
    nobody notices for a quarter.
    """

    targets: Sequence[str]
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)
    kind: str = "mean_baseline"

    def fit(self, X: np.ndarray, y: np.ndarray,
            targets: Sequence[str]) -> MeanBaseline:
        self.targets = list(targets)
        self.means = {t: float(np.mean(y[:, i])) for i, t in enumerate(targets)}
        self.stds = {t: float(np.std(y[:, i])) for i, t in enumerate(targets)}
        return self

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        n = X.shape[0]
        return {t: np.full(n, self.means[t]) for t in self.targets}

    def predict_with_uncertainty(
        self, X: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        n = X.shape[0]
        preds = self.predict(X)
        unc = {t: np.full(n, self.stds.get(t, 0.0)) for t in self.targets}
        return preds, unc


# ---------------------------------------------------------------------------
# sklearn backend
# ---------------------------------------------------------------------------


@dataclass
class SklearnModel:
    """One fitted estimator per target."""

    targets: list[str]
    estimators: dict[str, Any] = field(default_factory=dict, repr=False)
    estimator_name: str = "RandomForestRegressor"
    params: dict[str, Any] = field(default_factory=dict)
    featurizer_version: str = ""
    kind: str = "sklearn"

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        return {t: self.estimators[t].predict(X) for t in self.targets}

    def predict_with_uncertainty(
        self, X: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Uncertainty is the spread of the forest's trees on each row.

        Not a calibrated interval and not claimed to be one. It is a usable
        applicability-domain signal: molecules unlike anything in training land
        in different leaves across trees and the spread widens. When the answer
        to "does triage need real uncertainty" is yes, this is the seam where
        chemprop/uncertainty replaces it.
        """
        preds: dict[str, np.ndarray] = {}
        unc: dict[str, np.ndarray] = {}
        for t in self.targets:
            est = self.estimators[t]
            preds[t] = est.predict(X)
            trees = getattr(est, "estimators_", None)
            if trees:
                per_tree = np.stack([tree.predict(X) for tree in trees])
                unc[t] = per_tree.std(axis=0)
            else:
                unc[t] = np.zeros(X.shape[0])
        return preds, unc

    def save(self, path: str | Path) -> Path:
        import joblib
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)
        return p

    @classmethod
    def load(cls, path: str | Path) -> SklearnModel:
        import joblib
        obj = joblib.load(Path(path))
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a {cls.__name__}")
        return obj


@dataclass
class SklearnTrainer:
    estimator: str = "RandomForestRegressor"
    params: dict[str, Any] = field(default_factory=dict)

    def _make(self) -> Any:
        from sklearn.ensemble import (
            HistGradientBoostingRegressor,
            RandomForestRegressor,
        )
        registry = {
            "RandomForestRegressor": RandomForestRegressor,
            "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
        }
        if self.estimator not in registry:
            raise ValueError(
                f"unknown estimator {self.estimator!r}; known: {sorted(registry)}")
        return registry[self.estimator](**self.params)

    def fit(self, X: np.ndarray, y: np.ndarray,
            targets: Sequence[str]) -> SklearnModel:
        estimators = {}
        for i, t in enumerate(targets):
            est = self._make()
            est.fit(X, y[:, i])
            estimators[t] = est
        return SklearnModel(targets=list(targets), estimators=estimators,
                            estimator_name=self.estimator, params=dict(self.params))


def trainer_from_config(cfg: dict[str, Any]) -> SklearnTrainer:
    backend = cfg.get("backend", "sklearn")
    if backend != "sklearn":
        raise ValueError(
            f"unknown model backend {backend!r}. A chemprop backend would "
            f"implement Trainer as a subprocess step; see refs/REFERENCES.md")
    return SklearnTrainer(estimator=cfg.get("estimator", "RandomForestRegressor"),
                          params=dict(cfg.get("params", {})))


# ---------------------------------------------------------------------------
# Binding
# ---------------------------------------------------------------------------


@dataclass
class BoundModel:
    """A model and the exact featurizer it was trained with, loaded together.

    The pair is the unit, and that is the whole point. Handing a caller a model
    without its featurizer is handing them the train/serve skew bug.
    """

    model: SklearnModel
    featurizer: RdkitDescriptorFeaturizer
    version: str
    record: Any = None

    def score(self, frame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        X = self.featurizer.transform(frame)
        return self.model.predict_with_uncertainty(X)

    def verify_binding(self) -> None:
        """Fail loudly if the artifact pair does not match its recorded binding."""
        if self.model.featurizer_version != self.featurizer.version:
            raise RuntimeError(
                f"featurizer binding violated for model {self.version}: model "
                f"records {self.model.featurizer_version!r} but the loaded "
                f"featurizer is {self.featurizer.version!r}")


def save_artifacts(directory: str | Path, model: SklearnModel,
                   featurizer: RdkitDescriptorFeaturizer,
                   extra: dict[str, Any] | None = None) -> Path:
    """Write model + featurizer + their binding into one directory.

    They are written together and read together. There is no supported path
    that produces one without the other.
    """
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    model.featurizer_version = featurizer.version
    model.save(d / "model.joblib")
    featurizer.save(d / "featurizer.joblib")
    (d / "binding.json").write_text(json.dumps({
        "featurizer_version": featurizer.version,
        "model_targets": list(model.targets),
        "estimator": model.estimator_name,
        "params": model.params,
        **(extra or {}),
    }, indent=2, sort_keys=True) + "\n")
    return d


def load_bound(directory: str | Path, version: str = "", record: Any = None) -> BoundModel:
    """Load a model together with its featurizer, and verify the binding."""
    d = Path(directory)
    bound = BoundModel(
        model=SklearnModel.load(d / "model.joblib"),
        featurizer=RdkitDescriptorFeaturizer.load(d / "featurizer.joblib"),
        version=version or d.name,
        record=record,
    )
    bound.verify_binding()
    return bound


def load_stage(registry, stage: str = "production") -> BoundModel:
    """Pipeline 2's entry point: whatever occupies `stage`, plus its featurizer.

    This is the *only* way triage gets a model, and the stage is a config key
    rather than a constant so that a shadow run against `staging` is a config
    change instead of a code change. 
    """
    if stage == "production":
        rec = registry.production()
    else:
        rec = next((r for r in registry.records() if r.stage == stage), None)
    if rec is None:
        raise RuntimeError(
            f"no model in stage {stage!r} — run the lifecycle pipeline first")
    return load_bound(rec.artifact_dir, version=rec.version, record=rec)


def load_production(registry) -> BoundModel:
    """`load_stage(registry, "production")`, kept as the name callers expect."""
    return load_stage(registry, "production")
