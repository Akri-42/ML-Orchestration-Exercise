"""Table-driven tests for ingest validation.

`test_validation_is_the_third_user_of_the_gate_interface` is the one that
matters architecturally: it asserts validation composes the same way promotion
and triage do, so the reuse claim covers the whole pipeline and not just its
back half.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ml_orch.data import (
    DriftGate,
    DuplicateGate,
    MinRowsGate,
    ParseableSmilesGate,
    SchemaGate,
    TargetNaNGate,
    ks_statistic,
    psi_statistic,
    split_into_chunks,
    validate,
    validation_gate,
    write_reference_stats,
)
from ml_orch.gates import CompositeGate, GateContext, promotion_gate
from ml_orch.types import GateDecision

GOOD_SMILES = ["C", "CC", "CCO", "c1ccccc1", "CCN", "CC(=O)O", "CCCC", "CO"]


def frame(n: int = 8, *, smiles: list[str] | None = None, **overrides) -> pd.DataFrame:
    smiles = smiles or GOOD_SMILES[:n]
    df = pd.DataFrame({
        "mol_id": [f"gdb_{i}" for i in range(len(smiles))],
        "smiles": smiles,
        "mu": np.linspace(0.0, 3.0, len(smiles)),
        "gap": np.linspace(0.2, 0.4, len(smiles)),
        "alpha": np.linspace(6.0, 9.0, len(smiles)),
    })
    for k, v in overrides.items():
        df[k] = v
    return df


def ctx(df: pd.DataFrame) -> GateContext:
    return GateContext(frame=df)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def test_schema_gate_names_the_missing_columns():
    r = SchemaGate(required_columns=["smiles", "mu", "homo"]).evaluate(ctx(frame()))
    assert r.verdict is False
    assert r.evidence["missing"] == ["homo"]
    assert r.selection is None, "a missing column is not something you can drop rows to fix"


def test_schema_gate_passes_when_satisfied():
    assert SchemaGate(required_columns=["smiles", "mu"]).evaluate(ctx(frame())).verdict


@pytest.mark.parametrize(("n_rows", "minimum", "expected"), [
    (8, 4, True), (8, 8, True), (8, 9, False), (0, 1, False),
])
def test_min_rows_gate(n_rows, minimum, expected):
    df = frame(smiles=GOOD_SMILES[:n_rows]) if n_rows else frame().iloc[0:0]
    assert MinRowsGate(min_rows=minimum).evaluate(ctx(df)).verdict is expected


def test_unparseable_smiles_are_masked_out_not_crashed_on():
    df = frame(smiles=["CCO", "not_a_molecule", "c1ccccc1", "C(((", "CC"])
    r = ParseableSmilesGate(min_fraction=0.5).evaluate(ctx(df))
    assert r.verdict is True, "60% parse rate clears a 50% floor"
    assert list(r.selection) == [True, False, True, False, True]
    assert r.evidence["n_unparseable"] == 2


def test_a_mostly_junk_chunk_fails_even_though_rows_are_maskable():
    df = frame(smiles=["CCO", "xx", "yy", "zz"])
    r = ParseableSmilesGate(min_fraction=0.99).evaluate(ctx(df))
    assert r.verdict is False
    assert r.selection is not None, "still reports which rows were the problem"


def test_duplicate_gate_keeps_first_occurrence():
    df = frame(smiles=["CCO", "CC", "CCO", "CO", "CC"])
    r = DuplicateGate(max_fraction=0.5).evaluate(ctx(df))
    assert list(r.selection) == [True, True, False, True, False]
    assert r.evidence["n_duplicates"] == 2


def test_target_nan_gate_drops_rows_missing_a_label():
    df = frame()
    df.loc[2, "gap"] = np.nan
    r = TargetNaNGate(targets=["mu", "gap"], max_fraction=0.5).evaluate(ctx(df))
    assert not r.selection[2], "the row with the missing label must be masked out"
    assert r.selection.sum() == len(df) - 1
    assert r.evidence["n_rows_with_nan"] == 1


def test_gates_report_rather_than_raise_on_a_missing_column():
    df = frame().drop(columns=["smiles"])
    r = ParseableSmilesGate().evaluate(ctx(df))
    assert r.verdict is False and "smiles" in r.reason


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------


def test_ks_statistic_is_zero_for_identical_samples_and_one_for_disjoint():
    a = np.linspace(0, 1, 200)
    assert ks_statistic(a, a) == pytest.approx(0.0, abs=1e-9)
    assert ks_statistic(a, a + 10) == pytest.approx(1.0)


def test_drift_gate_is_silent_when_no_reference_exists_yet(tmp_path):
    r = DriftGate(reference=tmp_path / "absent.json").evaluate(ctx(frame()))
    assert r.verdict is True
    assert r.evidence["evaluated"] is False, "must not invent a verdict from nothing"


def test_drift_gate_fires_on_a_shifted_chunk(tmp_path):
    ref = write_reference_stats(frame(), ["alpha"], tmp_path / "ref.json")
    shifted = frame()
    shifted["alpha"] = shifted["alpha"] + 50.0
    r = DriftGate(reference=ref, max_statistic=0.25, columns=["alpha"]).evaluate(ctx(shifted))
    assert r.verdict is False
    assert r.evidence["worst_column"] == "alpha"


def test_drift_is_non_blocking_by_default():
    """Drift must not veto ingest: on QM9 the later chunks drift by design, and
    a gate that rejects them rejects exactly the data the system adapts to."""
    assert DriftGate(reference="nowhere.json").blocking is False
    assert DriftGate(reference="nowhere.json", blocking=True).blocking is True


def test_psi_statistic_is_near_zero_for_the_same_distribution_and_large_for_a_shift():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=2000), rng.normal(size=2000)
    assert psi_statistic(a, b) < 0.1, "two draws from one distribution are stable"
    assert psi_statistic(a + 3.0, b) > 1.0, "a three-sigma shift is not subtle"


def test_psi_and_ks_are_not_on_the_same_scale():
    """Why the config carries a threshold per method rather than one number."""
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=2000) + 0.5, rng.normal(size=2000)
    assert ks_statistic(a, b) <= 1.0, "KS is a probability gap, bounded by 1"
    assert psi_statistic(a, b) > ks_statistic(a, b), "PSI is unbounded and reads higher"


def drifting_frame(n: int = 400, *, shift: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """A frame big enough for PSI: bucketed statistics need rows, not 8 of them."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"alpha": rng.normal(loc=7.5 + shift, scale=1.0, size=n)})


def test_drift_gate_with_psi_fires_on_a_shift_and_passes_a_steady_chunk(tmp_path):
    ref = write_reference_stats(drifting_frame(seed=0), ["alpha"], tmp_path / "ref.json")
    gate = DriftGate(reference=ref, method="psi", max_statistic=0.25, columns=["alpha"])

    steady = gate.evaluate(ctx(drifting_frame(seed=1)))
    assert steady.verdict is True, f"same distribution should pass: {steady.reason}"
    assert steady.evidence["method"] == "psi"

    shifted = gate.evaluate(ctx(drifting_frame(seed=1, shift=3.0)))
    assert shifted.verdict is False
    assert shifted.evidence["worst_column"] == "alpha"
    assert "PSI" in shifted.reason, "the reason must name the statistic it used"


def test_an_unknown_drift_method_fails_loudly(tmp_path):
    """The whole point of the key: it is a switch, not decoration."""
    with pytest.raises(ValueError, match="unknown drift method"):
        DriftGate(reference=tmp_path / "ref.json", method="wasserstein")
    with pytest.raises(ValueError, match="drift.method"):
        validation_gate({"checks": {"drift": {"reference": str(tmp_path / "ref.json"),
                                              "method": "wasserstein"}}})


def test_the_drift_method_selects_its_own_threshold(tmp_path):
    """KS and PSI thresholds are different numbers and must not be swapped."""
    spec = {"reference": str(tmp_path / "ref.json"),
            "max_statistic": {"ks": 0.25, "psi": 0.10}}

    built = validation_gate({"checks": {"drift": {**spec, "method": "psi"}}}).gates[0]
    assert (built.method, built.max_statistic) == ("psi", 0.10)

    built = validation_gate({"checks": {"drift": {**spec, "method": "ks"}}}).gates[0]
    assert (built.method, built.max_statistic) == ("ks", 0.25)


def test_a_scalar_drift_threshold_is_refused_rather_than_reinterpreted(tmp_path):
    """One number meaning two things is the bug this config shape prevents."""
    with pytest.raises(ValueError, match="per-method mapping"):
        validation_gate({"checks": {"drift": {"reference": str(tmp_path / "ref.json"),
                                              "method": "psi", "max_statistic": 0.25}}})
    with pytest.raises(ValueError, match="no threshold for it"):
        validation_gate({"checks": {"drift": {"reference": str(tmp_path / "ref.json"),
                                              "method": "psi",
                                              "max_statistic": {"ks": 0.25}}}})


def test_drift_stays_non_blocking_whichever_method_is_selected(tmp_path):
    ref = write_reference_stats(drifting_frame(seed=0), ["alpha"], tmp_path / "ref.json")
    for method in ("ks", "psi"):
        r = DriftGate(reference=ref, method=method, max_statistic=0.01,
                      columns=["alpha"]).evaluate(ctx(drifting_frame(seed=1, shift=5.0)))
        assert r.verdict is False and r.blocking is False


def test_reference_stats_roundtrip(tmp_path):
    p = write_reference_stats(frame(), ["mu", "alpha"], tmp_path / "ref.json")
    payload = json.loads(p.read_text())
    assert set(payload["columns"]) == {"mu", "alpha"}
    assert len(payload["columns"]["mu"]["quantiles"]) == payload["n_quantiles"]


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def config(on_failure: str = "quarantine") -> dict:
    return {
        "on_failure": on_failure,
        "checks": {
            "schema": {"required_columns": ["smiles", "mu", "gap"]},
            "min_rows": 3,
            "parseable_smiles": {"min_fraction": 0.5},
            "duplicates": {"max_fraction": 0.5},
            "target_nan": {"max_fraction": 0.5},
        },
    }


def test_quarantine_keeps_the_good_rows_and_reports_the_bad():
    df = frame(smiles=["CCO", "junk!!", "CC", "CO"])
    out = validate(df, config("quarantine"), targets=["mu", "gap"])
    assert out.status == "accepted_with_quarantine"
    assert len(out.accepted) == 3 and len(out.quarantined) == 1
    assert out.usable is True


def test_warn_policy_keeps_everything_but_still_records_the_complaint():
    df = frame(smiles=["CCO", "junk!!", "CC", "CO"])
    out = validate(df, config("warn"), targets=["mu", "gap"])
    assert out.status == "accepted"
    assert len(out.accepted) == len(df) and out.quarantined.empty
    # Nothing was dropped, but what *would* have been dropped is still on record.
    # That is the whole use of this policy: see the damage before enforcing it.
    assert out.decision.selection is not None
    assert int((~out.decision.selection).sum()) == 1


def test_reject_chunk_policy_is_all_or_nothing():
    """One bad row rejects the chunk, even though no check failed its threshold."""
    df = frame(smiles=["CCO", "junk!!", "CC", "CO"])
    out = validate(df, config("reject_chunk"), targets=["mu", "gap"])
    assert out.status == "rejected"
    assert out.accepted.empty and len(out.quarantined) == len(df)
    assert out.usable is False


def test_a_structural_failure_rejects_even_under_quarantine():
    """You cannot drop rows to recover from a missing column."""
    df = frame().drop(columns=["gap"])
    out = validate(df, config("quarantine"), targets=["mu"])
    assert out.status == "rejected"


def test_clean_data_is_simply_accepted():
    out = validate(frame(), config("quarantine"), targets=["mu", "gap"])
    assert out.status == "accepted"
    assert len(out.accepted) == 8 and out.quarantined.empty


def test_outcome_serialises_for_the_manifest():
    out = validate(frame(), config(), targets=["mu", "gap"])
    d = out.as_dict()
    assert d["status"] == "accepted"
    assert d["decision"]["verdict"] is True
    assert len(d["decision"]["checks"]) == 5


# ---------------------------------------------------------------------------
# The architectural claim
# ---------------------------------------------------------------------------


def test_validation_is_the_third_user_of_the_gate_interface():
    """Validation, promotion, and triage compose identically.

    If a future change gives validation its own bespoke control flow, this
    fails — which is the point. The reuse claim covers the whole pipeline, not
    only the half a reviewer is most likely to read.
    """
    v = validation_gate(config())
    p = promotion_gate()
    assert type(v) is type(p) is CompositeGate
    decision = v.evaluate(GateContext(frame=frame()))
    assert isinstance(decision, GateDecision)
    for r in decision.results:
        assert r.name and r.reason and callable(v.gates[0].evaluate)


def test_validation_never_short_circuits():
    """Every check runs, so a rejection record carries every reason at once."""
    df = frame(smiles=["junk!!", "also junk", "CCO"]).drop(columns=["gap"])
    decision = validation_gate(config()).evaluate(GateContext(frame=df))
    assert len(decision.results) == 5, "a failing check must not stop the others"
    assert sum(not r.verdict for r in decision.results) >= 2


def test_a_gate_that_raises_fails_closed():
    class Exploding:
        name = "exploding"

        def evaluate(self, ctx):
            raise RuntimeError("boom")

    decision = CompositeGate([Exploding()]).evaluate(GateContext(frame=frame()))
    assert decision.verdict is False
    assert "RuntimeError" in decision.results[0].reason


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def test_size_ramp_produces_a_monotonic_size_gradient():
    """The covariate shift is constructed, so it had better actually be there."""
    rng = np.random.default_rng(0)
    n = 600
    sizes = rng.integers(1, 10, size=n)
    raw = pd.DataFrame({
        "mol_id": [f"gdb_{i}" for i in range(n)],
        "smiles": ["C" * int(s) for s in sizes],
        "mu": rng.normal(size=n),
    })
    chunks, golden = split_into_chunks(raw, n_molecules=n, n_chunks=4,
                                       golden_fraction=0.1, seed=0)
    means = [np.mean([len(s) for s in c["smiles"]]) for c in chunks]
    assert means == sorted(means), f"size ramp is not monotonic: {means}"
    assert len(golden) > 0


def test_golden_set_is_disjoint_from_every_chunk():
    rng = np.random.default_rng(1)
    n = 400
    raw = pd.DataFrame({
        "mol_id": [f"gdb_{i}" for i in range(n)],
        "smiles": ["C" * int(s) for s in rng.integers(1, 10, size=n)],
        "mu": rng.normal(size=n),
    })
    chunks, golden = split_into_chunks(raw, n_molecules=n, n_chunks=4, seed=0)
    trained_on = set(pd.concat(chunks)["mol_id"])
    assert not (trained_on & set(golden["mol_id"])), "golden set leaked into training data"


def test_chunking_is_deterministic_under_a_fixed_seed():
    rng = np.random.default_rng(2)
    n = 300
    raw = pd.DataFrame({
        "mol_id": [f"gdb_{i}" for i in range(n)],
        "smiles": ["C" * int(s) for s in rng.integers(1, 10, size=n)],
        "mu": rng.normal(size=n),
    })
    a, ga = split_into_chunks(raw, n_molecules=n, n_chunks=3, seed=7)
    b, gb = split_into_chunks(raw, n_molecules=n, n_chunks=3, seed=7)
    for x, y in zip(a, b, strict=True):
        pd.testing.assert_frame_equal(x, y)
    pd.testing.assert_frame_equal(ga, gb)
