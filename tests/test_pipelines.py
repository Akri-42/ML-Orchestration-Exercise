"""End-to-end tests for both pipelines, on synthetic data small enough to be fast.

These cover the behaviours the brief names explicitly and that unit tests
cannot reach: idempotency on re-arrival, a rejected promotion leaving the
incumbent serving, rollback, and — the load-bearing one —
`test_triage_adds_no_new_code_paths`, which asserts pipeline 2 is reached
entirely through the same functions pipeline 1 uses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_orch.data import source_from_config
from ml_orch.pipelines import run_lifecycle, run_triage
from ml_orch.registry import ModelRegistry

SMILES = ["C", "CC", "CCO", "CCN", "CCC", "CCCC", "c1ccccc1", "CC(=O)O", "CCF",
          "CCCl", "CO", "CN", "CCCO", "CCCN", "C1CC1", "C1CCC1", "CC(C)C",
          "CCOC", "CCOCC", "CC(N)=O", "FC(F)F", "CCCCO", "CCCCN", "c1ccncc1",
          "CCCCC", "CCCCCC", "CC(C)O", "CC(C)N", "COC", "CSC"]
TARGETS = ["mu", "gap"]


def make_frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    s = [SMILES[i % len(SMILES)] for i in range(n)]
    sizes = np.array([len(x) for x in s], dtype=float)
    return pd.DataFrame({
        "mol_id": [f"gdb_{seed}_{i}" for i in range(n)],
        "smiles": s,
        # Signal correlated with structure, so a model can beat the mean.
        "mu": sizes * 0.5 + rng.normal(0, 0.1, n),
        "gap": 0.3 - sizes * 0.01 + rng.normal(0, 0.005, n),
    })


@pytest.fixture
def workspace(tmp_path):
    """A complete miniature project: chunks, golden set, candidates, config."""
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    for i in range(3):
        make_frame(60, seed=i).to_csv(chunks / f"chunk_{i:02d}.csv", index=False)
    make_frame(40, seed=99).to_csv(tmp_path / "golden.csv", index=False)

    cand = tmp_path / "candidates"
    cand.mkdir()
    batch = make_frame(30, seed=7)[["smiles"]].copy()
    batch.insert(0, "candidate_id", [f"cand-{i:03d}" for i in range(len(batch))])
    batch.to_csv(cand / "batch_001.csv", index=False)

    lifecycle = {
        "pipeline": "lifecycle",
        "source": {"kind": "csv_chunks", "path": str(chunks), "glob": "chunk_*.csv"},
        "validation": {"on_failure": "quarantine", "checks": {
            "schema": {"required_columns": ["smiles", "mu", "gap"]},
            "parseable_smiles": {"min_fraction": 0.9},
        }},
        "featurizer": {"kind": "rdkit_descriptors",
                       "params": {"morgan_bits": 256, "morgan_radius": 2}},
        "model": {"backend": "sklearn", "estimator": "RandomForestRegressor",
                  "params": {"n_estimators": 6, "random_state": 0}},
        # Rolling is in the fixture deliberately: it had no end-to-end coverage
        # here, which is how a rolling set that was 100% training data shipped.
        "evaluation": {"targets": TARGETS,
                       "sets": {"golden": {"path": str(tmp_path / "golden.csv")},
                                "rolling": {"source": "last_n_chunks", "n": 2,
                                            "holdout_fraction": 0.2}}},
        "gate": {"cold_start_min_improvement": 0.05, "min_margin": 0.01,
                 "alpha": 0.05, "n_resamples": 200},
        "registry": {"root": str(tmp_path / "registry")},
    }
    triage = {
        "pipeline": "triage",
        "source": {"kind": "csv_batch", "path": str(cand / "batch_001.csv")},
        "validation": {"on_failure": "quarantine", "checks": {
            "schema": {"required_columns": ["candidate_id", "smiles"]},
            "parseable_smiles": {"min_fraction": 0.0},
        }},
        "featurizer": {"kind": "from_promoted_model"},
        "model": {"backend": "registry", "stage": "production"},
        "gate": [{"kind": "top_k", "target": "mu", "k": 5}],
        "output": {"shortlist": str(tmp_path / "runs/{run_id}/shortlist.csv"),
                   "rejected": str(tmp_path / "runs/{run_id}/rejected.csv")},
    }
    return {"dir": tmp_path, "lifecycle": lifecycle, "triage": triage,
            "runs": str(tmp_path / "runs")}


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_cold_start_promotes_when_it_beats_the_baseline(workspace):
    r = run_lifecycle(workspace["lifecycle"], runs_root=workspace["runs"])
    assert r.status == "promoted"
    assert ModelRegistry(workspace["lifecycle"]["registry"]["root"]).production() is not None


def test_the_same_chunk_arriving_twice_does_not_retrain(workspace):
    from ml_orch.data import source_from_config
    cfg = workspace["lifecycle"]
    chunk = source_from_config(cfg["source"]).chunks()[0]
    first = run_lifecycle(cfg, chunk=chunk, runs_root=workspace["runs"])
    second = run_lifecycle(cfg, chunk=chunk, runs_root=workspace["runs"])
    assert first.status == "promoted"
    assert second.status == "skipped"
    assert second.model_version == first.model_version


def test_idempotency_survives_the_model_being_superseded(workspace):
    """The bug this test exists for: once a model is archived, its chunk must
    still count as seen, or every arrival retrains forever."""
    from ml_orch.data import source_from_config
    cfg = workspace["lifecycle"]
    chunks = source_from_config(cfg["source"]).chunks()
    run_lifecycle(cfg, chunk=chunks[0], runs_root=workspace["runs"])
    run_lifecycle(cfg, chunk=chunks[1], runs_root=workspace["runs"])
    again = run_lifecycle(cfg, chunk=chunks[0], runs_root=workspace["runs"])
    assert again.status == "skipped"


def test_a_rejected_candidate_leaves_the_incumbent_serving(workspace):
    """The central failure-handling claim. A blocked promotion is a normal
    outcome: the model is recorded, the artifacts kept, nothing raises."""
    cfg = workspace["lifecycle"]
    cfg["gate"]["min_margin"] = 0.99          # nothing can clear a 99% margin
    from ml_orch.data import source_from_config
    chunks = source_from_config(cfg["source"]).chunks()

    first = run_lifecycle(cfg, chunk=chunks[0], runs_root=workspace["runs"])
    second = run_lifecycle(cfg, chunk=chunks[1], runs_root=workspace["runs"])

    assert first.status == "promoted"
    assert second.status == "rejected"
    reg = ModelRegistry(cfg["registry"]["root"])
    assert reg.production().version == first.model_version, "incumbent stopped serving"
    rejected = reg.get(second.model_version)
    assert rejected.stage == "rejected"
    assert rejected.gate_decision["checks"], "rejection recorded without its reasons"
    assert any(not c["verdict"] for c in rejected.gate_decision["checks"])


def test_rejection_keeps_artifacts_for_forensics(workspace):
    from pathlib import Path
    cfg = workspace["lifecycle"]
    cfg["gate"]["min_margin"] = 0.99
    from ml_orch.data import source_from_config
    chunks = source_from_config(cfg["source"]).chunks()
    run_lifecycle(cfg, chunk=chunks[0], runs_root=workspace["runs"])
    second = run_lifecycle(cfg, chunk=chunks[1], runs_root=workspace["runs"])
    rec = ModelRegistry(cfg["registry"]["root"]).get(second.model_version)
    assert (Path(rec.artifact_dir) / "model.joblib").exists()
    assert (Path(rec.artifact_dir) / "featurizer.joblib").exists()


def test_a_chunk_that_fails_validation_never_reaches_training(workspace):
    cfg = workspace["lifecycle"]
    cfg["validation"]["on_failure"] = "reject_chunk"
    broken = make_frame(60, seed=5).drop(columns=["gap"])
    path = workspace["dir"] / "chunks" / "chunk_00.csv"
    broken.to_csv(path, index=False)

    from ml_orch.data import source_from_config
    chunk = next(c for c in source_from_config(cfg["source"]).chunks()
                 if c.chunk_id == "chunk_00")
    r = run_lifecycle(cfg, chunk=chunk, runs_root=workspace["runs"])
    assert r.status == "data_rejected"
    assert ModelRegistry(cfg["registry"]["root"]).production() is None


def test_manifest_ties_all_five_versions_together(workspace):
    import json
    from pathlib import Path
    r = run_lifecycle(workspace["lifecycle"], runs_root=workspace["runs"])
    payload = json.loads(Path(r.manifest_path).read_text())
    assert set(payload["versions"]) == {"data", "featurizer", "model", "code", "config"}
    assert all(payload["versions"][k] for k in ("data", "featurizer", "model", "config"))


def test_rollback_restores_the_previous_model(workspace):
    from ml_orch.data import source_from_config
    cfg = workspace["lifecycle"]
    chunks = source_from_config(cfg["source"]).chunks()
    first = run_lifecycle(cfg, chunk=chunks[0], runs_root=workspace["runs"])
    second = run_lifecycle(cfg, chunk=chunks[1], runs_root=workspace["runs"])
    reg = ModelRegistry(cfg["registry"]["root"])
    if second.status != "promoted":
        pytest.skip("second chunk did not promote; rollback tested in test_registry")
    assert reg.production().version == second.model_version
    reg.rollback()
    assert reg.production().version == first.model_version


# ---------------------------------------------------------------------------
# Triage
# ---------------------------------------------------------------------------


def test_triage_scores_with_the_promoted_model(workspace):
    cfg = workspace["lifecycle"]
    promoted = run_lifecycle(cfg, runs_root=workspace["runs"])
    r = run_triage(workspace["triage"], registry_root=cfg["registry"]["root"],
                   runs_root=workspace["runs"])
    assert r.status == "completed"
    assert r.model_version == promoted.model_version
    assert 0 < r.n_shortlisted <= 5


def test_every_dropped_candidate_names_the_gate_that_dropped_it(workspace):
    cfg = workspace["lifecycle"]
    run_lifecycle(cfg, runs_root=workspace["runs"])
    r = run_triage(workspace["triage"], registry_root=cfg["registry"]["root"],
                   runs_root=workspace["runs"])
    rejected = pd.read_csv(r.shortlist_path.replace("shortlist", "rejected"))
    assert (rejected["dropped_by"] != "").all()
    assert set(rejected["dropped_by"]) <= {"top_k[mu]"}


def test_shortlist_rows_trace_to_a_model_and_a_run(workspace):
    cfg = workspace["lifecycle"]
    run_lifecycle(cfg, runs_root=workspace["runs"])
    r = run_triage(workspace["triage"], registry_root=cfg["registry"]["root"],
                   runs_root=workspace["runs"])
    shortlist = pd.read_csv(r.shortlist_path)
    assert (shortlist["model_version"] == r.model_version).all()
    assert (shortlist["run_id"] == r.run_id).all()


def test_triage_without_a_promoted_model_says_so(workspace):
    with pytest.raises(RuntimeError, match="no model in stage 'production'"):
        run_triage(workspace["triage"],
                   registry_root=workspace["lifecycle"]["registry"]["root"],
                   runs_root=workspace["runs"])


def test_triage_adds_no_new_code_paths():
    """The submission's central claim, asserted against the source.

    Pipeline 2 is reached through `source_from_config`, `validate`,
    the bound featurizer, `predict_with_uncertainty` and `shortlist_gate` —
    every one of them shared with pipeline 1 or with the promotion gate. If
    `run_triage` grows its own loading, featurizing, or thresholding, the names
    it calls will change and this fails.
    """
    import ast
    import inspect

    from ml_orch import pipelines

    tree = ast.parse(inspect.getsource(pipelines.run_triage))
    called = {
        node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        for node in ast.walk(tree) if isinstance(node, ast.Call)
    }
    shared = {"source_from_config", "validate", "shortlist_gate", "load_stage",
              "score", "config_hash", "code_version"}
    assert shared <= called, f"triage stopped using shared primitives: {shared - called}"

    forbidden = {"fit", "transform", "MolFromSmiles", "read_csv"}
    assert not (forbidden & called), (
        f"triage is re-implementing shared work: {forbidden & called}")


def test_the_shortlist_is_actually_ranked(workspace):
    """"Ranked shortlist" is the deliverable, so assert the file is ordered.

    The gate that truncates has to sort in order to truncate. Before this test
    existed it computed the ordering, applied it as a mask, and discarded the
    ordering -- so the output was filtered but in input order, and the word
    "ranked".
    """
    cfg = workspace["lifecycle"]
    run_lifecycle(cfg, runs_root=workspace["runs"])
    r = run_triage(workspace["triage"], registry_root=cfg["registry"]["root"],
                   runs_root=workspace["runs"])

    shortlist = pd.read_csv(r.shortlist_path)
    spec = next(g for g in workspace["triage"]["gate"] if g["kind"] == "top_k")
    ranked_by = f"pred_{spec['target']}"

    assert "rank" in shortlist.columns
    assert shortlist["rank"].is_monotonic_increasing
    values = shortlist[ranked_by].tolist()
    assert values == sorted(values, reverse=bool(spec.get("largest", False))), (
        f"shortlist is not ordered by {ranked_by}: {values}")


def test_rank_survives_the_filter(workspace):
    """Ranks are assigned over the whole batch, so the shortlist keeps gaps.

    A shortlist renumbered 0..k-1 would be indistinguishable from one that was
    never ranked against the candidates that lost.
    """
    cfg = workspace["lifecycle"]
    run_lifecycle(cfg, runs_root=workspace["runs"])
    r = run_triage(workspace["triage"], registry_root=cfg["registry"]["root"],
                   runs_root=workspace["runs"])
    shortlist = pd.read_csv(r.shortlist_path)
    rejected = pd.read_csv(r.shortlist_path.replace("shortlist", "rejected"))
    all_ranks = sorted(shortlist["rank"].tolist() + rejected["rank"].tolist())
    assert all_ranks == list(range(len(all_ranks))), "ranks are not a permutation"


def test_triage_reads_the_config_keys_it_documents(workspace, tmp_path):
    """The config keys are live, not decorative.

    An auditor found `featurizer`, `model` and `registry` present in the triage
    config and never read: deleting them changed nothing.
    """
    cfg = workspace["lifecycle"]
    run_lifecycle(cfg, runs_root=workspace["runs"])

    triage = dict(workspace["triage"])
    triage["registry"] = {"root": cfg["registry"]["root"]}
    triage["output"] = dict(triage.get("output", {}))
    triage["output"]["manifest"] = str(tmp_path / "elsewhere" / "manifest.json")

    # registry.root and output.manifest are honoured...
    r = run_triage(triage, registry_root="definitely-not-here",
                   runs_root=workspace["runs"])
    assert r.status == "completed"
    assert r.manifest_path == str(tmp_path / "elsewhere" / "manifest.json")
    assert (tmp_path / "elsewhere" / "manifest.json").exists()

    # ...model.stage selects the stage...
    staged = dict(triage, model={"stage": "archived"})
    with pytest.raises(RuntimeError, match="no model in stage 'archived'"):
        run_triage(staged, runs_root=workspace["runs"])

    # ...and featurizer.kind is enforced rather than obeyed.
    skewed = dict(triage, featurizer={"kind": "rdkit_descriptors"})
    with pytest.raises(ValueError, match="train/serve skew"):
        run_triage(skewed, runs_root=workspace["runs"])


def test_the_rolling_set_is_held_out_of_training(workspace):
    """The rolling set must not be the data the candidate just trained on.

    An auditor measured the original build at rolling-intersect-train = 100%:
    rolling was assembled from whole chunks, all of which were in the training
    set. 
    """
    from ml_orch.pipelines import _eval_frames, _holdout_split

    cfg = workspace["lifecycle"]
    chunks = source_from_config(cfg["source"]).chunks()
    fraction = cfg["evaluation"]["sets"]["rolling"]["holdout_fraction"]
    assert fraction > 0

    holdouts = {}
    for c in chunks:
        keep, held = _holdout_split(c.frame, fraction, seed=c.data_version)

        # The split must partition the chunk: every row goes to exactly one
        # side. Checking by value would be meaningless here -- the synthetic
        # chunks reuse the same SMILES -- so compare row multisets.
        assert len(keep) + len(held) == len(c.frame)
        recombined = pd.concat([keep, held], ignore_index=True)
        pd.testing.assert_frame_equal(
            recombined.sort_values(list(c.frame.columns)).reset_index(drop=True),
            c.frame.sort_values(list(c.frame.columns)).reset_index(drop=True))
        assert len(held) > 0, "nothing was held out"
        holdouts[c.chunk_id] = held

    # ...and rolling must be built from the held-out side, nothing else.
    rolling = _eval_frames(cfg, chunks, holdouts)["rolling"]
    assert not rolling.empty
    expected = pd.concat([holdouts[c.chunk_id] for c in chunks[-2:]], ignore_index=True)
    pd.testing.assert_frame_equal(rolling, expected)


def test_the_holdout_split_is_stable_across_runs(workspace):
    """Seeded on content, so consecutive scores stay comparable.

    A split that reshuffled every run would change what "rolling score" means
    between two runs while looking perfectly stable in the logs.
    """
    from ml_orch.pipelines import _holdout_split

    chunk = source_from_config(workspace["lifecycle"]["source"]).chunks()[0]
    a_keep, a_held = _holdout_split(chunk.frame, 0.2, seed=chunk.data_version)
    b_keep, b_held = _holdout_split(chunk.frame, 0.2, seed=chunk.data_version)
    pd.testing.assert_frame_equal(a_held, b_held)
    pd.testing.assert_frame_equal(a_keep, b_keep)

    # A different chunk must get a different split, or the "seeded on content"
    # claim is doing no work.
    _, other = _holdout_split(chunk.frame, 0.2, seed="a-different-content-hash")
    assert not other.equals(a_held)


def test_holdout_never_consumes_the_whole_chunk(workspace):
    """Degenerate fractions must still leave something to train on."""
    from ml_orch.pipelines import _holdout_split

    chunk = source_from_config(workspace["lifecycle"]["source"]).chunks()[0]
    keep, held = _holdout_split(chunk.frame, 1.0, seed=chunk.data_version)
    assert len(keep) >= 1
    assert len(keep) + len(held) == len(chunk.frame)

    keep, held = _holdout_split(chunk.frame, 0.0, seed=chunk.data_version)
    assert held.empty and len(keep) == len(chunk.frame)
