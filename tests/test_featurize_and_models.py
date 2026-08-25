"""Tests for the featurizer/model seam — mostly about binding, not accuracy.

The train/serve skew bug is the one that does real damage in this class of
system and the one that is hardest to notice, because nothing crashes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_orch.evaluate import evaluate, summarise
from ml_orch.featurize import (
    RdkitDescriptorFeaturizer,
    derive_slice_columns,
    featurize_targets,
    featurizer_from_config,
    slice_masks,
)
from ml_orch.gates import FeaturizerBindingGate, GateContext
from ml_orch.models import (
    MeanBaseline,
    SklearnTrainer,
    load_bound,
    save_artifacts,
)
from ml_orch.types import Versions

SMILES = ["C", "CC", "CCO", "CCN", "CCC", "CCCC", "c1ccccc1", "CC(=O)O",
          "CCF", "CCCl", "CO", "CN", "CCCO", "CCCN", "C1CC1", "C1CCC1",
          "CC(C)C", "CCOC", "CCOCC", "CC(N)=O", "FC(F)F", "CCCCO", "CCCCN",
          "c1ccncc1"]


def frame(n: int = 24, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    s = (SMILES * 3)[:n]
    return pd.DataFrame({
        "mol_id": [f"gdb_{i}" for i in range(n)],
        "smiles": s,
        "mu": rng.normal(2.0, 1.0, n),
        "gap": rng.normal(0.3, 0.05, n),
    })


TARGETS = ["mu", "gap"]


# ---------------------------------------------------------------------------
# Featurizer identity
# ---------------------------------------------------------------------------


def test_unfitted_featurizer_refuses_to_transform():
    with pytest.raises(RuntimeError, match="not fitted"):
        RdkitDescriptorFeaturizer().transform(frame())


def test_unfitted_version_can_never_match_a_bound_one():
    f = RdkitDescriptorFeaturizer()
    assert f.version.endswith("/unfitted")
    fitted = RdkitDescriptorFeaturizer().fit(frame())
    assert f.version != fitted.version


def test_version_changes_when_params_change():
    a = RdkitDescriptorFeaturizer(morgan_bits=1024).fit(frame())
    b = RdkitDescriptorFeaturizer(morgan_bits=512).fit(frame())
    assert a.version != b.version


def test_version_changes_when_the_fit_changes():
    """The point of a stateful featurizer: same params, different training data,
    different version — so binding is a real constraint, not a label."""
    a = RdkitDescriptorFeaturizer().fit(frame(seed=0))
    b = RdkitDescriptorFeaturizer().fit(frame(n=12, seed=1))
    assert a.params() == b.params()
    assert a.version != b.version


def test_version_is_stable_for_an_identical_fit():
    a = RdkitDescriptorFeaturizer().fit(frame())
    b = RdkitDescriptorFeaturizer().fit(frame())
    assert a.version == b.version, "version must be deterministic or it is useless"


def test_featurizer_roundtrips_through_disk(tmp_path):
    f = RdkitDescriptorFeaturizer().fit(frame())
    loaded = RdkitDescriptorFeaturizer.load(f.save(tmp_path / "f.joblib"))
    assert loaded.version == f.version
    np.testing.assert_allclose(loaded.transform(frame()), f.transform(frame()))


def test_config_refuses_to_build_a_featurizer_for_pipeline_two():
    """`from_promoted_model` must be resolved from the registry. Building one
    from config here is precisely the skew bug wearing a helpful face."""
    with pytest.raises(ValueError, match="registry"):
        featurizer_from_config({"kind": "from_promoted_model"})


def test_unparseable_molecule_yields_zeros_rather_than_exploding():
    f = RdkitDescriptorFeaturizer().fit(frame())
    df = frame(4)
    df.loc[1, "smiles"] = "not-a-molecule"
    X = f.transform(df)
    assert X.shape[0] == 4
    assert np.isfinite(X).all()


# ---------------------------------------------------------------------------
# Binding
# ---------------------------------------------------------------------------


def fit_everything(tmp_path, seed: int = 0):
    df = frame(seed=seed)
    f = RdkitDescriptorFeaturizer().fit(df)
    X, y = f.transform(df), featurize_targets(df, TARGETS)
    model = SklearnTrainer("RandomForestRegressor",
                           {"n_estimators": 5, "random_state": 0}).fit(X, y, TARGETS)
    d = save_artifacts(tmp_path / "artifacts", model, f)
    return df, f, model, d


def test_saved_artifacts_carry_their_binding(tmp_path):
    _, f, _, d = fit_everything(tmp_path)
    bound = load_bound(d, version="v1")
    assert bound.model.featurizer_version == f.version
    bound.verify_binding()


def test_a_mismatched_featurizer_is_caught_on_load(tmp_path):
    _, _, _, d = fit_everything(tmp_path)
    # Swap in a featurizer fitted on different data — the exact skew scenario.
    other = RdkitDescriptorFeaturizer().fit(frame(n=12, seed=9))
    other.save(d / "featurizer.joblib")
    with pytest.raises(RuntimeError, match="binding violated"):
        load_bound(d, version="v1")


def test_featurizer_binding_gate_blocks_the_skewed_run(tmp_path):
    _, f, _, _ = fit_everything(tmp_path)
    stale = RdkitDescriptorFeaturizer().fit(frame(n=12, seed=3))
    ctx = GateContext(
        versions=Versions(data="d", featurizer=stale.version, model="m",
                          code="c", config="cfg"),
        expected_featurizer=f.version,
    )
    r = FeaturizerBindingGate().evaluate(ctx)
    assert r.verdict is False and "mismatch" in r.reason


def test_binding_gate_fails_closed_when_it_cannot_check():
    """No version info is not permission to proceed."""
    assert FeaturizerBindingGate().evaluate(GateContext()).verdict is False


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def test_one_estimator_per_target(tmp_path):
    _, _, model, _ = fit_everything(tmp_path)
    assert set(model.estimators) == set(TARGETS)


def test_forest_uncertainty_is_positive_and_per_row(tmp_path):
    df, f, model, _ = fit_everything(tmp_path)
    _, unc = model.predict_with_uncertainty(f.transform(df))
    for t in TARGETS:
        assert unc[t].shape == (len(df),)
        assert np.all(unc[t] >= 0)
    assert unc["mu"].std() > 0, "uncertainty that is constant tells you nothing"


def test_mean_baseline_predicts_the_mean():
    df = frame()
    y = featurize_targets(df, TARGETS)
    base = MeanBaseline(targets=TARGETS).fit(np.zeros((len(df), 1)), y, TARGETS)
    preds = base.predict(np.zeros((len(df), 1)))
    assert preds["mu"] == pytest.approx(np.full(len(df), y[:, 0].mean()))


def test_unknown_estimator_is_rejected_by_name():
    with pytest.raises(ValueError, match="unknown estimator"):
        SklearnTrainer("MagicRegressor").fit(np.zeros((4, 2)), np.zeros((4, 1)), ["mu"])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def test_eval_report_carries_per_molecule_errors(tmp_path):
    df, f, model, _ = fit_everything(tmp_path)
    rep = evaluate(model, f.transform(df), df, TARGETS,
                   model_version="v1", eval_set="golden")
    for t in TARGETS:
        assert rep.per_target[t].errors.shape == (len(df),), \
            "the significance gate needs paired per-molecule errors, not a scalar"
        assert rep.target_std[t] > 0


def test_summarise_drops_error_vectors_but_keeps_the_score(tmp_path):
    df, f, model, _ = fit_everything(tmp_path)
    rep = evaluate(model, f.transform(df), df, TARGETS,
                   model_version="v1", eval_set="golden")
    s = summarise(rep)
    assert "normalized_score" in s and np.isfinite(s["normalized_score"])
    assert "errors" not in str(s["per_target"])


def test_slice_columns_are_derived_not_required_upstream():
    enriched = derive_slice_columns(frame())
    assert {"n_heavy_atoms", "has_F", "n_rings"} <= set(enriched.columns)
    assert enriched["has_F"].any(), "fixture should contain a fluorinated molecule"


def test_tiny_slices_are_dropped_rather_than_reported():
    """A MAE over three molecules is noise wearing a number's clothes."""
    enriched = derive_slice_columns(frame())
    masks = slice_masks(enriched, {"rings": {"column": "n_rings",
                                             "bins": [0, 1, 2, 99]}})
    assert all(m.sum() >= 20 for m in masks.values())


def test_evaluation_is_reproducible(tmp_path):
    df, f, model, _ = fit_everything(tmp_path)
    a = evaluate(model, f.transform(df), df, TARGETS, model_version="v1",
                 eval_set="golden")
    b = evaluate(model, f.transform(df), df, TARGETS, model_version="v1",
                 eval_set="golden")
    assert a.per_target["mu"].mae == b.per_target["mu"].mae
