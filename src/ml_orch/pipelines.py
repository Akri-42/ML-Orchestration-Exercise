"""The two pipelines, as plain Python.

No Dagster import appears in this file, and that is structural rather than
stylistic. `definitions.py` wraps these functions as software-defined assets;
`cli.py` calls the same functions directly. So the on-demand trigger and the
data-arrival trigger genuinely share one entrypoint rather than two code paths
that are supposed to agree.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

from .data import Chunk, source_from_config, validate
from .evaluate import evaluate, summarise
from .featurize import featurize_targets, featurizer_from_config
from .gates import CompositeGate, GateContext, promotion_gate, shortlist_gate
from .manifest import RunManifest, code_version, config_hash
from .models import (
    MeanBaseline,
    load_bound,
    load_stage,
    save_artifacts,
    trainer_from_config,
)
from .registry import ModelRegistry
from .types import GateDecision, GateResult, Versions

log = logging.getLogger("ml_orch")


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def _run_id(prefix: str) -> str:
    """Sortable timestamp plus a random suffix.

    The suffix is not decoration. Second-resolution timestamps collide when two
    runs start inside the same second -- which a backfill over small chunks
    does routinely -- and the registry correctly refuses to register a
    duplicate version, turning a fast pipeline into a crashed one.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}-{uuid4().hex[:6]}"


def _merge_decisions(named: dict[str, GateDecision]) -> GateDecision:
    """Combine one decision per evaluation set into one, keeping every reason.

    Results are name-prefixed by set so a rejection record says *which* set objected, which is the first thing
    anyone reads.
    """
    results: list[GateResult] = []
    for set_name, decision in named.items():
        for r in decision.results:
            results.append(dataclasses.replace(r, name=f"{set_name}:{r.name}"))
    verdict = all(d.verdict for d in named.values())
    return GateDecision(verdict=verdict, results=results)


@dataclass
class LifecycleResult:
    run_id: str
    status: str                       # promoted | rejected | skipped | data_rejected
    data_version: str
    decision: GateDecision | None = None
    model_version: str = ""
    manifest_path: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class TriageResult:
    run_id: str
    status: str                       # completed | empty | failed
    model_version: str = ""
    n_candidates: int = 0
    n_shortlisted: int = 0
    decision: GateDecision | None = None
    shortlist_path: str = ""
    manifest_path: str = ""


# ---------------------------------------------------------------------------
# Pipeline 1 — lifecycle
# ---------------------------------------------------------------------------


def run_lifecycle(cfg: dict[str, Any], *, chunk: Chunk | None = None,
                  runs_root: str | Path = "runs") -> LifecycleResult:
    """ingest -> validate -> train -> benchmark -> gate -> register/promote.

    Returns rather than raises on every expected outcome, including rejection.
    A failed gate is a normal result: the incumbent keeps serving, the
    candidate is registered as `rejected` with its full decision record, the
    artifacts are kept for forensics, and the process exits 0.
    """
    targets = list(cfg["evaluation"]["targets"])
    registry = ModelRegistry(cfg["registry"]["root"])

    chunks = source_from_config(cfg["source"]).chunks() if chunk is None else None
    if chunk is None:
        if not chunks:
            raise RuntimeError("no chunks found — run scripts/prepare_data.py")
        chunk = chunks[-1]
    else:
        chunks = source_from_config(cfg["source"]).chunks()

    run_id = _run_id("lifecycle")
    run_dir = Path(runs_root) / run_id
    cfg_hash = config_hash(cfg)

    # -- idempotency: identity is content, not filename -------------------
    existing = registry.find_by_data_version(chunk.data_version)
    if existing is not None:
        log.info("skip: chunk %s already trained as %s", chunk.chunk_id, existing.version)
        manifest = RunManifest.start(
            run_id, "lifecycle",
            Versions(data=chunk.data_version, featurizer=existing.featurizer_version,
                     model=existing.version, code=code_version(), config=cfg_hash),
            chunk=str(chunk.path))
        manifest.event("idempotent_skip", existing_version=existing.version)
        path = manifest.finish("skipped").write(run_dir / "manifest.json")
        return LifecycleResult(run_id=run_id, status="skipped",
                               data_version=chunk.data_version,
                               model_version=existing.version, manifest_path=str(path))

    # -- validate ---------------------------------------------------------
    outcome = validate(chunk.frame, cfg["validation"], targets=targets)
    log.info("validation %s: %d accepted, %d quarantined",
             outcome.status, len(outcome.accepted), len(outcome.quarantined))

    if not outcome.usable:
        manifest = RunManifest.start(
            run_id, "lifecycle",
            Versions(data=chunk.data_version, featurizer="", model="",
                     code=code_version(), config=cfg_hash),
            chunk=str(chunk.path))
        manifest.event("validation_rejected", **outcome.as_dict())
        path = manifest.finish("failed", validation=outcome.as_dict()).write(
            run_dir / "manifest.json")
        run_dir.mkdir(parents=True, exist_ok=True)
        outcome.quarantined.to_csv(run_dir / "quarantined.csv", index=False)
        return LifecycleResult(run_id=run_id, status="data_rejected",
                               data_version=chunk.data_version,
                               decision=outcome.decision, manifest_path=str(path))

    # -- assemble training data: everything accepted up to and including this chunk
    upto = [c for c in chunks if c.chunk_id <= chunk.chunk_id]
    holdout_fraction = float(
        cfg["evaluation"]["sets"].get("rolling", {}).get("holdout_fraction", 0.0))
    frames: list[pd.DataFrame] = []
    holdouts: dict[str, pd.DataFrame] = {}
    for c in upto:
        o = validate(c.frame, cfg["validation"], targets=targets)
        if not o.usable:
            continue
        keep, held = _holdout_split(o.accepted, holdout_fraction, seed=c.data_version)
        frames.append(keep)
        holdouts[c.chunk_id] = held
    train = pd.concat(frames, ignore_index=True)
    n_held = sum(len(h) for h in holdouts.values())
    log.info("training on %d rows from %d chunk(s); %d rows held out for rolling",
             len(train), len(frames), n_held)

    # -- featurize + train -------------------------------------------------
    featurizer = featurizer_from_config(cfg["featurizer"]).fit(train)
    X = featurizer.transform(train)
    y = featurize_targets(train, targets)
    candidate = trainer_from_config(cfg["model"]).fit(X, y, targets)

    versions = Versions(data=chunk.data_version, featurizer=featurizer.version,
                        model=run_id, code=code_version(), config=cfg_hash)

    # -- evaluation sets ---------------------------------------------------
    eval_sets = _eval_frames(cfg, upto, holdouts)
    slice_spec = cfg["evaluation"].get("slices")

    baseline = MeanBaseline(targets=targets).fit(X, y, targets)
    incumbent_bound = None
    rec = registry.production()
    if rec is not None:
        incumbent_bound = load_bound(rec.artifact_dir, version=rec.version, record=rec)

    per_set: dict[str, GateDecision] = {}
    metrics: dict[str, Any] = {}
    for set_name, frame in eval_sets.items():
        Xe = featurizer.transform(frame)
        cand_report = evaluate(candidate, Xe, frame, targets, model_version=run_id,
                               eval_set=set_name, slice_spec=slice_spec)
        base_report = evaluate(baseline, Xe, frame, targets, model_version="baseline",
                               eval_set=set_name)
        inc_report = None
        if incumbent_bound is not None:
            # Scored with the INCUMBENT's own featurizer, not the candidate's.
            # Using one matrix for both models would quietly advantage whichever
            # featurizer produced it.
            Xi = incumbent_bound.featurizer.transform(frame)
            inc_report = evaluate(incumbent_bound.model, Xi, frame, targets,
                                  model_version=incumbent_bound.version,
                                  eval_set=set_name, slice_spec=slice_spec)

        ctx = GateContext(candidate=cand_report, incumbent=inc_report,
                          baseline=base_report, versions=versions,
                          expected_featurizer=featurizer.version)
        per_set[set_name] = promotion_gate(**cfg.get("gate", {})).evaluate(ctx)
        metrics[set_name] = summarise(cand_report, targets)
        if inc_report is not None:
            metrics[f"{set_name}_incumbent"] = summarise(inc_report, targets)

    decision = _merge_decisions(per_set)
    log.info("gate: %s", decision.summary())

    # -- register ----------------------------------------------------------
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = save_artifacts(run_dir / "artifacts", candidate, featurizer,
                                  extra={"targets": targets})

    manifest = RunManifest.start(run_id, "lifecycle", versions,
                                 chunk=str(chunk.path), n_train_rows=len(train))
    manifest.event("validated", **outcome.as_dict())
    manifest.gate_decision = decision.as_dict()

    stage = "staging" if decision.verdict else "rejected"
    registry.register(run_id, versions=versions.as_dict(), metrics=metrics,
                      gate_decision=decision.as_dict(), stage=stage,
                      artifact_src=artifact_dir,
                      notes="promoted by gate" if decision.verdict else "blocked by gate")

    if decision.verdict:
        registry.promote(run_id)
        status = "promoted"
        manifest.event("promoted", model_version=run_id)
    else:
        status = "rejected"
        manifest.event("rejected", reasons=[r.reason for r in decision.failed])

    path = manifest.finish(status, metrics=metrics).write(run_dir / "manifest.json")
    return LifecycleResult(run_id=run_id, status=status,
                           data_version=chunk.data_version, decision=decision,
                           model_version=run_id, manifest_path=str(path),
                           metrics=metrics)


def _eval_frames(cfg: dict[str, Any], chunks: list[Chunk],
                 holdouts: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """The golden and rolling evaluation sets.

    Golden is frozen and read from disk. Rolling is assembled from the slices
    held out of the most recent n chunks *before* training — not from the
    chunks themselves.

    That distinction is the whole value of the set. Built from whole chunks,
    rolling is 100% training data for the candidate and 0% for the incumbent,
    which does not measure adaptation to drift; it measures how well a forest
    memorises rows it was just fitted on, and it votes to promote essentially
    every time. Held out from both models, the comparison is fair and the set
    answers the question it exists to answer.
    """
    out: dict[str, pd.DataFrame] = {}
    spec = cfg["evaluation"]["sets"]
    if "golden" in spec:
        out["golden"] = pd.read_csv(spec["golden"]["path"])
    if "rolling" in spec and chunks:
        n = int(spec["rolling"].get("n", 2))
        parts = [holdouts[c.chunk_id] for c in chunks[-n:]
                 if not holdouts.get(c.chunk_id, pd.DataFrame()).empty]
        # An empty rolling set is omitted rather than passed on: gates fail
        # closed on empty error vectors, so an empty set would block every
        # promotion for a reason that has nothing to do with the model.
        if parts:
            out["rolling"] = pd.concat(parts, ignore_index=True)
    return out


def _holdout_split(frame: pd.DataFrame, fraction: float,
                   seed: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a validated chunk into (training rows, rolling-eval rows).

    Seeded on the chunk's *content hash*, so the same chunk always yields the
    same split. A split that moved between runs would make consecutive
    normalized scores incomparable while looking perfectly stable — the same
    class of silent bug as an unversioned golden set.
    """
    if fraction <= 0 or len(frame) < 2:
        return frame, frame.iloc[:0]
    rng = np.random.default_rng(
        int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16))
    n_held = min(len(frame) - 1, max(1, round(len(frame) * fraction)))
    order = rng.permutation(len(frame))
    held = frame.iloc[np.sort(order[:n_held])].reset_index(drop=True)
    keep = frame.iloc[np.sort(order[n_held:])].reset_index(drop=True)
    return keep, held


# ---------------------------------------------------------------------------
# Pipeline 2 — triage
# ---------------------------------------------------------------------------


def run_triage(cfg: dict[str, Any], *, registry_root: str | Path = "registry",
               runs_root: str | Path = "runs") -> TriageResult:
    """Score a candidate batch with the promoted model and shortlist it.

    Note what this function does *not* contain: no loading code, no
    featurization code, no thresholding code. It resolves a config into the
    same primitives pipeline 1 uses and reports what they decided.
    """
    registry = ModelRegistry(cfg.get("registry", {}).get("root", registry_root))

    # `featurizer.kind` is the one key here that is not policy. Triage must use
    # the featurizer bound to the promoted model; re-deriving it is the
    # train/serve skew bug this system exists to prevent. So the key is read and
    # *enforced* rather than obeyed — the only accepted value is the safe one,
    # and setting anything else fails loudly instead of silently doing something
    # else. A config key that can only hold one value is still worth reading:
    # it documents the constraint at the place someone would try to violate it.
    fkind = cfg.get("featurizer", {}).get("kind", "from_promoted_model")
    if fkind != "from_promoted_model":
        raise ValueError(
            f"featurizer.kind must be 'from_promoted_model' in a triage config, "
            f"got {fkind!r}. The featurizer is bound to the model it was trained "
            f"with; choosing one here would be train/serve skew by configuration.")

    stage = cfg.get("model", {}).get("stage", "production")
    bound = load_stage(registry, stage)        # model + its featurizer, together
    run_id = _run_id("triage")
    run_dir = Path(runs_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Every output path is config, resolved once here so that the failure path
    # writes its manifest to the same place the success path would.
    out = cfg.get("output", {})

    def _out(key: str, default: str) -> Path:
        path = Path(str(out.get(key, default)).replace("{run_id}", run_id))
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    sl_path = _out("shortlist", "runs/{run_id}/shortlist.csv")
    rj_path = _out("rejected", "runs/{run_id}/rejected.csv")
    mf_path = _out("manifest", "runs/{run_id}/manifest.json")

    batch = source_from_config(cfg["source"]).chunks()[0]
    outcome = validate(batch.frame, cfg["validation"], targets=[])
    log.info("validation %s: %d candidates, %d dropped",
             outcome.status, len(outcome.accepted), len(outcome.quarantined))

    versions = Versions(data=batch.data_version,
                        featurizer=bound.featurizer.version,
                        model=bound.version, code=code_version(),
                        config=config_hash(cfg))
    manifest = RunManifest.start(run_id, "triage", versions,
                                 batch=str(batch.path), n_candidates=len(batch.frame))
    manifest.event("validated", **outcome.as_dict())

    if not outcome.usable or outcome.accepted.empty:
        path = manifest.finish("failed", reason="no usable candidates").write(
            mf_path)
        return TriageResult(run_id=run_id, status="failed", model_version=bound.version,
                            n_candidates=len(batch.frame), decision=outcome.decision,
                            manifest_path=str(path))

    frame = outcome.accepted.reset_index(drop=True)
    scores, uncertainty = bound.score(frame)

    ids = (frame["candidate_id"].tolist() if "candidate_id" in frame.columns
           else [str(i) for i in range(len(frame))])
    ctx = GateContext(scores=scores, uncertainty=uncertainty, record_ids=ids,
                      versions=versions,
                      # The binding the promoted model recorded at training time.
                      expected_featurizer=bound.model.featurizer_version)

    decision: CompositeGate = shortlist_gate(cfg["gate"]).evaluate(ctx)
    log.info("triage gate: %s", decision.summary())

    mask = decision.selection if decision.selection is not None else np.ones(len(frame), bool)
    scored = frame.copy()
    for t, v in scores.items():
        scored[f"pred_{t}"] = v
    for t, v in uncertainty.items():
        scored[f"unc_{t}"] = v
    scored["run_id"] = run_id
    scored["model_version"] = bound.version

    # Carry the verdict and the attribution as columns before reordering, so
    # that sorting cannot desynchronise them from their rows. Positional masks
    # and a sorted frame are a bug waiting for the first reviewer who checks.
    scored["_selected"] = mask
    scored["dropped_by"] = _dropped_by(decision, len(frame))

    # "Ranked shortlist" is the deliverable, so the ordering has to reach the
    # file. The ranking gate already computed the sort; applying it here keeps
    # it in one place instead of asking every consumer to re-derive it.
    ranking = decision.ranking
    if ranking is not None:
        scored["rank"] = ranking
        scored = scored.sort_values("rank", kind="stable")

    shortlist = scored[scored["_selected"]].drop(columns=["_selected", "dropped_by"])
    rejected = scored[~scored["_selected"]].drop(columns=["_selected"])

    shortlist.to_csv(sl_path, index=False)
    rejected.to_csv(rj_path, index=False)
    # Quarantined candidates are reported too — they were never scored, but a
    # chemist asking "what happened to compound X" deserves an answer.
    outcome.quarantined.to_csv(run_dir / "quarantined.csv", index=False)

    manifest.gate_decision = decision.as_dict()
    status = "completed" if decision.verdict else "empty"
    path = manifest.finish(status, shortlist=str(sl_path), rejected=str(rj_path),
                           n_shortlisted=int(mask.sum())).write(mf_path)

    return TriageResult(run_id=run_id, status=status, model_version=bound.version,
                        n_candidates=len(frame), n_shortlisted=int(mask.sum()),
                        decision=decision, shortlist_path=str(sl_path),
                        manifest_path=str(path))


def _dropped_by(decision: GateDecision, n: int) -> np.ndarray:
    """Attribute each dropped candidate to the first gate that excluded it.

    "Rejected" without a reason is not a result, it is a shrug.
    """
    out = np.array([""] * n, dtype=object)
    for r in decision.results:
        if r.selection is None:
            continue
        newly = (~r.selection) & (out == "")
        out[newly] = r.name
    return out


def write_report(result: LifecycleResult | TriageResult, path: str | Path) -> Path:
    """Human-readable run summary next to the machine-readable manifest."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"run: {result.run_id}", f"status: {result.status}", ""]
    if result.decision is not None:
        lines.append(result.decision.summary())
    p.write_text("\n".join(lines) + "\n")
    return p
