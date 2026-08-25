"""RunManifest — the object that makes any output traceable to one run.

Every artifact either pipeline emits (a registered model, a candidate
shortlist) carries a manifest. If you can hold a row of a shortlist CSV and
cannot say which model, featurizer, data chunk, code sha, and config produced
it, the system has failed at the thing it exists to do.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .types import Versions


def content_hash(path: str | Path, *, algo: str = "sha256") -> str:
    """Content hash of a data chunk. This is the chunk's identity.

    Filenames and mtimes are not identity: the same chunk copied twice must
    produce the same version, and a chunk edited in place must produce a
    different one. Idempotent retraining depends on exactly this.
    """
    h = hashlib.new(algo)
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return f"{algo}:{h.hexdigest()[:16]}"


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()[:16]


def code_version() -> str:
    """Git sha, marked dirty when the tree has uncommitted changes."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True,
            stderr=subprocess.DEVNULL).strip()
        return f"dirty:{sha}" if dirty else sha
    except Exception:  # noqa: BLE001 — not a git checkout, or git absent
        return "unknown"


@dataclass
class RunManifest:
    run_id: str
    pipeline: str                       # "lifecycle" | "triage"
    started_at: str
    versions: Versions
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    gate_decision: dict[str, Any] = field(default_factory=dict)
    status: str = "running"             # running | promoted | rejected | failed | skipped
    finished_at: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def start(cls, run_id: str, pipeline: str, versions: Versions,
              **inputs: Any) -> RunManifest:
        return cls(
            run_id=run_id,
            pipeline=pipeline,
            started_at=datetime.now(UTC).isoformat(timespec="seconds"),
            versions=versions,
            inputs=inputs,
        )

    def event(self, kind: str, **payload: Any) -> None:
        self.events.append({
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "kind": kind, **payload,
        })

    def finish(self, status: str, **outputs: Any) -> RunManifest:
        self.status = status
        self.outputs.update(outputs)
        self.finished_at = datetime.now(UTC).isoformat(timespec="seconds")
        return self

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["versions"] = self.versions.as_dict()
        return d

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n")
        return p
