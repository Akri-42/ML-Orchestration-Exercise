#!/usr/bin/env python3
"""End-to-end demonstration, including the failure that matters.

    python scripts/demo.py


  1. cold start: no incumbent, so the first model must beat a mean predictor
  2. successive chunks: each retrains and is benchmarked against the incumbent
  3. a rejection: improvement lands inside the noise and the gate blocks it,
     the incumbent keeps serving, and the process exits 0
  4. rollback: one command
  5. triage: the promoted model shortlists a candidate batch
  6. skew: a deliberately mismatched featurizer, refused

Destructive: wipes `registry/` and `runs/` so the run is reproducible. Pass
--keep to leave them alone.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ml_orch.data import source_from_config
from ml_orch.pipelines import load_config, run_lifecycle, run_triage
from ml_orch.registry import ModelRegistry

RULE = "=" * 78


def head(n: int, title: str) -> None:
    print(f"\n{RULE}\n{n}. {title}\n{RULE}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", action="store_true", help="do not wipe registry/ and runs/")
    args = ap.parse_args()

    if not Path("data/chunks").exists():
        print("[fail] no chunks — run scripts/fetch_qm9.py then scripts/prepare_data.py",
              file=sys.stderr)
        return 1

    if not args.keep:
        for d in ("registry", "runs"):
            shutil.rmtree(d, ignore_errors=True)

    lifecycle = load_config("configs/lifecycle.yaml")
    triage_cfg = load_config("configs/triage.yaml")
    registry = ModelRegistry(lifecycle["registry"]["root"])
    chunks = source_from_config(lifecycle["source"]).chunks()

    head(1, "Lifecycle over every chunk, in arrival order")
    print(f"{'chunk':10} {'rows':>5} {'status':10} {'golden score':>13}  note")
    outcomes = []
    for c in chunks:
        r = run_lifecycle(lifecycle, chunk=c)
        outcomes.append(r)
        score = r.metrics.get("golden", {}).get("normalized_score")
        note = ""
        if r.decision and not r.decision.verdict:
            note = "; ".join(f"{x.name}" for x in r.decision.failed)
        print(f"{c.chunk_id:10} {c.n_rows:5d} {r.status:10} "
              f"{'-' if score is None else f'{score:13.4f}'}  {note}")

    rejected = [r for r in outcomes if r.status == "rejected"]
    head(2, "The interesting failure")
    if rejected:
        r = rejected[0]
        print(f"run {r.run_id} was NOT promoted. Every check, not just the first:\n")
        print(r.decision.summary())
        print(f"\nIncumbent still serving : {registry.production().version}")
        print(f"Candidate recorded as   : {registry.get(r.model_version).stage}")
        print(f"Artifacts kept at       : {registry.get(r.model_version).artifact_dir}")
        print("Process exit code       : 0  (a blocked promotion is a normal outcome)")
    else:
        print("No rejection this run — every candidate cleared the gate.")
        print("Re-run with a stricter `gate.min_margin` in configs/lifecycle.yaml to force one.")

    head(3, "Idempotency: the same chunk arriving again")
    again = run_lifecycle(lifecycle, chunk=chunks[0])
    print(f"chunk_00 re-arrival -> {again.status} "
          f"(matched existing model {again.model_version})")
    print("Identity is the chunk's content hash, not its filename or mtime.")

    head(4, "Rollback")
    before = registry.production().version
    restored = registry.rollback()
    print(f"production {before}\n        -> {restored.version}   (one command)")
    registry.promote(before)
    print(f"        -> {before}   (rolled forward again for the triage step)")

    head(5, "Triage: score a candidate batch with whatever is promoted")
    t = run_triage(triage_cfg)
    print(f"model      : {t.model_version}")
    print(f"candidates : {t.n_candidates}")
    print(f"shortlist  : {t.n_shortlisted} -> {t.shortlist_path}")
    print()
    print(t.decision.summary())

    head(6, "Train/serve skew is refused, not absorbed")
    from ml_orch.featurize import RdkitDescriptorFeaturizer
    from ml_orch.models import load_bound
    rec = registry.production()
    stale = RdkitDescriptorFeaturizer(morgan_bits=256).fit(chunks[0].frame)
    backup = Path(rec.artifact_dir) / "featurizer.joblib"
    keep = backup.read_bytes()
    try:
        stale.save(backup)
        load_bound(rec.artifact_dir, version=rec.version)
        print("!! skew was NOT caught — this is a bug")
    except RuntimeError as exc:
        print(f"load_bound refused the mismatched pair:\n  {exc}")
    finally:
        backup.write_bytes(keep)
        print("\n(featurizer restored)")

    head(7, "Everything traces to one manifest")
    import json
    payload = json.loads(Path(outcomes[-1].manifest_path).read_text())
    print(json.dumps(payload["versions"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
