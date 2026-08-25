"""Dagster definitions — the orchestrator layer, and the only file that knows
Dagster exists.

Everything here is a wrapper. `promoted_model` calls `run_lifecycle`;
`candidate_shortlist` calls `run_triage`; the sensor calls `run_lifecycle`.
The pipelines themselves are plain Python in `ml_orch.pipelines`, which is what
makes the tool choice a paragraph in the write-up rather than a rewrite.

Why Dagster, concretely:

* **Software-defined assets** mean the promoted model is an asset with real
  lineage, and `candidate_shortlist` declaring it as a dependency is the
  production pointer expressed in the graph rather than in prose.
* **`Config` schemas** make "pipeline 2 is new configuration" literally true —
  `TriageConfig` is a config object, not a second code path.
* **Sensors** handle data arrival natively, and `run_key` gives orchestrator-
  level deduplication on top of the content-hash idempotency the pipeline
  already enforces. Two independent guards against the same double-train,
  which is deliberate: the run_key is a convenience and the content hash is
  the correctness argument.
"""

# NOTE: deliberately NO `from __future__ import annotations` here. Dagster
# resolves Config classes from real annotation objects at runtime; with PEP 563
# they arrive as strings and every asset fails with "cannot be resolved". It is
# the one file in the repo that wants eager annotations.

from pathlib import Path
from typing import Optional

from dagster import (
    AssetExecutionContext,
    Config,
    Definitions,
    MetadataValue,
    RunRequest,
    SensorEvaluationContext,
    SensorResult,
    SkipReason,
    asset,
    define_asset_job,
    sensor,
)

from ml_orch.data import source_from_config
from ml_orch.manifest import content_hash
from ml_orch.pipelines import load_config, run_lifecycle, run_triage

LIFECYCLE_CONFIG = "configs/lifecycle.yaml"
TRIAGE_CONFIG = "configs/triage.yaml"


class LifecycleConfig(Config):
    config_path: str = LIFECYCLE_CONFIG
    chunk: Optional[str] = None       # chunk id; default is the newest  # noqa: UP045


class TriageConfig(Config):
    """Pipeline 2's entire novelty, as a config object rather than a module."""

    config_path: str = TRIAGE_CONFIG


@asset(
    description="The model currently serving, and the gate decision that put it there.",
    compute_kind="python",
)
def promoted_model(context: AssetExecutionContext, config: LifecycleConfig) -> dict:
    cfg = load_config(config.config_path)
    chunk = None
    if config.chunk:
        chunks = source_from_config(cfg["source"]).chunks()
        chunk = next((c for c in chunks if c.chunk_id == config.chunk), None)
        if chunk is None:
            raise ValueError(f"no chunk {config.chunk!r}")

    result = run_lifecycle(cfg, chunk=chunk)

    # A rejection is a normal materialisation with a different verdict, not a
    # failed asset. Failing here would page someone for the system working.
    context.add_output_metadata({
        "status": result.status,
        "model_version": result.model_version,
        "data_version": result.data_version,
        "manifest": MetadataValue.path(result.manifest_path),
        "gate": MetadataValue.md(
            f"```\n{result.decision.summary()}\n```" if result.decision
            else "_no gate decision (skipped or rejected at ingest)_"),
        "golden_normalized_score": MetadataValue.float(
            result.metrics.get("golden", {}).get("normalized_score", float("nan"))),
    })
    return {"status": result.status, "model_version": result.model_version}


@asset(
    description="Ranked candidate shortlist, scored by whatever is promoted.",
    compute_kind="python",
)
def candidate_shortlist(context: AssetExecutionContext, config: TriageConfig,
                        promoted_model: dict) -> dict:
    """Depends on `promoted_model` so the production pointer is in the graph.

    The dependency is what makes the lineage claim checkable: a shortlist row
    traces to a model version, which traces to a chunk hash, which traces to a
    gate decision.
    """
    result = run_triage(load_config(config.config_path))
    context.add_output_metadata({
        "status": result.status,
        "scored_by": result.model_version,
        "n_candidates": result.n_candidates,
        "n_shortlisted": result.n_shortlisted,
        "shortlist": MetadataValue.path(result.shortlist_path),
        "gate": MetadataValue.md(
            f"```\n{result.decision.summary()}\n```" if result.decision else "_none_"),
    })
    return {"status": result.status, "n_shortlisted": result.n_shortlisted}


lifecycle_job = define_asset_job("lifecycle_job", selection=[promoted_model])
triage_job = define_asset_job("triage_job", selection=[candidate_shortlist])


@sensor(job=lifecycle_job, minimum_interval_seconds=15,
        description="Fires the lifecycle when a new chunk lands.")
def chunk_arrival_sensor(context: SensorEvaluationContext):
    """Data-arrival trigger.

    `run_key` is the chunk's *content* hash, not its path or mtime. Dagster
    will not launch two runs with the same run_key, so a file rewritten with
    identical bytes does not retrain — matching the pipeline's own idempotency
    rule rather than merely coexisting with it.
    """
    cfg = load_config(LIFECYCLE_CONFIG)
    root = Path(cfg["source"]["path"])
    if not root.exists():
        return SkipReason(f"{root} does not exist yet")

    paths = sorted(root.glob(cfg["source"].get("glob", "chunk_*.csv")))
    if not paths:
        return SkipReason(f"no chunks in {root}")

    requests = []
    for p in paths:
        digest = content_hash(p)
        requests.append(RunRequest(
            run_key=digest,
            run_config={"ops": {"promoted_model": {"config": {
                "config_path": LIFECYCLE_CONFIG, "chunk": p.stem}}}},
            tags={"chunk": p.stem, "data_version": digest},
        ))
    if not requests:
        return SkipReason("nothing new")
    return SensorResult(run_requests=requests)


defs = Definitions(
    assets=[promoted_model, candidate_shortlist],
    jobs=[lifecycle_job, triage_job],
    sensors=[chunk_arrival_sensor],
)
