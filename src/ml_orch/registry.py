"""A file-backed model registry, deliberately ~200 legible lines.


* **Atomic writes.** Index updates are write-to-temp + `os.replace`, so a crash
  mid-promotion cannot leave a half-written index or an ambiguous production
  pointer.
* **Serialized updates.** Every mutation is a read-modify-write of one JSON
  file, so atomic *writes* alone are not enough: two concurrent writers each
  read, append in memory, and write back, and the loser's record disappears
  with no exception raised. `chunk_arrival_sensor` emits one run per chunk all
  at once, so this is the shipped fan-out, not a hypothetical. An advisory
  `flock` on `index.lock` makes the whole read-modify-write mutually exclusive
  across processes.
* **Rejected models are first-class.** A failed gate registers the model with
  `stage="rejected"` and the full decision record. Artifacts are kept. 
* **Idempotency by data version.** `find_by_data_version` lets a pipeline skip
  retraining when the same chunk arrives twice.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import tempfile
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

Stage = Literal["staging", "production", "archived", "rejected"]


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class ModelRecord:
    version: str
    stage: Stage
    created_at: str
    versions: dict[str, str]                     # data / featurizer / model / code / config
    metrics: dict[str, Any] = field(default_factory=dict)
    gate_decision: dict[str, Any] = field(default_factory=dict)
    artifact_dir: str = ""
    notes: str = ""

    @property
    def data_version(self) -> str:
        return self.versions.get("data", "")

    @property
    def featurizer_version(self) -> str:
        return self.versions.get("featurizer", "")


class ModelRegistry:
    """Directory layout::

        <root>/index.json
        <root>/models/<version>/{model.joblib, featurizer.joblib, manifest.json}
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.models_dir = self.root / "models"
        self.index_path = self.root / "index.json"
        self.lock_path = self.root / "index.lock"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._thread_lock = threading.RLock()
        self._lock_depth = 0
        # Creating the index is itself a read-modify-write: two processes
        # constructing a registry at the same moment would both find it absent
        # and both write an empty one, and the second would erase whatever the
        # first had already registered. So bootstrap under the lock too.
        with self._locked():
            if not self.index_path.exists():
                self._write_index({"records": [], "production": None})

    # -- mutual exclusion --------------------------------------------------

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Hold an exclusive advisory lock for one read-modify-write.

        `flock` and not a lockfile-we-create-and-delete, because the kernel
        drops a `flock` when the process dies -- a crashed run leaves no stale
        lock for a human to clean up at 3am. A sidecar lockfile rather than
        `index.json` itself, because `_write_index` replaces the inode and a
        lock held on the old inode would guard nothing.

        Two layers, and both are load-bearing. The `threading.RLock` gives
        mutual exclusion *within* a process, where a second `os.open` of the
        same path yields a second open file description that `flock` treats as
        an unrelated competitor. The `flock` gives it *across* processes, which
        is the case the Dagster fan-out actually produces.

        Re-entrant on purpose: `rollback()` holds the lock while it chooses a
        target and then calls `promote()`, which asks for it again. Without the
        depth count that call path would block on itself forever -- a deadlock
        reachable from a single-threaded `mlorch rollback`.
        """
        with self._thread_lock:
            if self._lock_depth:
                self._lock_depth += 1
                try:
                    yield
                finally:
                    self._lock_depth -= 1
                return
            fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                self._lock_depth = 1
                yield
            finally:
                self._lock_depth = 0
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)

    # -- persistence -------------------------------------------------------

    def _read_index(self) -> dict[str, Any]:
        return json.loads(self.index_path.read_text())

    def _write_index(self, payload: dict[str, Any]) -> None:
        """Write-then-rename. The rename is atomic on POSIX."""
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=".index-", suffix=".json")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.index_path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    # -- reads -------------------------------------------------------------

    def records(self) -> list[ModelRecord]:
        return [ModelRecord(**r) for r in self._read_index()["records"]]

    def get(self, version: str) -> ModelRecord | None:
        return next((r for r in self.records() if r.version == version), None)

    def production(self) -> ModelRecord | None:
        """One read, not two. Reads are unlocked -- `os.replace` means a reader
        always sees some whole index -- but reading twice can straddle a writer
        and answer from two different ones, which is how you get a pointer to a
        version whose record you already read as missing."""
        idx = self._read_index()
        prod = idx.get("production")
        rec = next((r for r in idx["records"] if r["version"] == prod), None) if prod else None
        return ModelRecord(**rec) if rec else None

    def find_by_data_version(self, data_version: str,
                             stages: Iterable[Stage] = (
                                 "production", "staging", "rejected", "archived"),
                             ) -> ModelRecord | None:
        """Idempotency hook: has this exact chunk already been trained on?

        `archived` must be in the default set. Omitting it  means a chunk 
        looks unseen again the moment its model is superseded,
        and every arrival of that chunk retrains from scratch forever.
        """
        wanted = set(stages)
        return next((r for r in self.records()
                     if r.data_version == data_version and r.stage in wanted), None)

    def artifact_dir(self, version: str) -> Path:
        return self.models_dir / version

    # -- writes ------------------------------------------------------------

    def register(
        self,
        version: str,
        *,
        versions: dict[str, str],
        metrics: dict[str, Any],
        gate_decision: dict[str, Any],
        stage: Stage = "staging",
        artifact_src: str | Path | None = None,
        notes: str = "",
    ) -> ModelRecord:
        # The artifact copy is inside the critical section, which serializes
        # concurrent registrations for as long as a copy takes. That is the
        # deliberate trade: the duplicate-version check and the append have to
        # see the same index, and artifacts here are a handful of joblib files.
        # If they ever became large, the fix is to stage them under a temp name
        # outside the lock and rename inside it -- not to shrink the section.
        with self._locked():
            idx = self._read_index()
            if any(r["version"] == version for r in idx["records"]):
                raise ValueError(f"version {version!r} already registered")

            dest = self.artifact_dir(version)
            dest.mkdir(parents=True, exist_ok=True)
            if artifact_src is not None:
                for item in Path(artifact_src).iterdir():
                    target = dest / item.name
                    if item.is_dir():
                        shutil.copytree(item, target, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, target)

            record = ModelRecord(
                version=version,
                stage=stage,
                created_at=_utcnow(),
                versions=versions,
                metrics=metrics,
                gate_decision=gate_decision,
                artifact_dir=str(dest),
                notes=notes,
            )
            (dest / "manifest.json").write_text(json.dumps(asdict(record), indent=2) + "\n")
            idx["records"].append(asdict(record))
            self._write_index(idx)
            return record

    def promote(self, version: str) -> ModelRecord:
        """Make `version` production; archive whoever held it. Atomic.

        Two promotions racing without the lock is the sharper version of the
        registry's failure mode: one of them wins the pointer and the other's
        stage change is lost, leaving a record that says `production` while the
        index points somewhere else.
        """
        with self._locked():
            idx = self._read_index()
            rec = next((r for r in idx["records"] if r["version"] == version), None)
            if rec is None:
                raise KeyError(f"unknown version {version!r}")
            if rec["stage"] == "rejected":
                raise ValueError(f"refusing to promote rejected model {version!r}")

            previous = idx.get("production")
            for r in idx["records"]:
                if r["version"] == previous and r["version"] != version:
                    r["stage"] = "archived"
            rec["stage"] = "production"
            idx["production"] = version
            self._write_index(idx)
            return ModelRecord(**rec)

    def rollback(self) -> ModelRecord:
        """Promote the most recently archived model. One command, as required.

        The lock spans the choice *and* the promotion. Choosing a target from
        an index another process is already rewriting is how a rollback lands
        on a model that stopped being the incumbent while you were reading.
        """
        with self._locked():
            idx = self._read_index()
            archived = [r for r in idx["records"] if r["stage"] == "archived"]
            if not archived:
                raise RuntimeError("nothing to roll back to: no archived models")
            target = max(archived, key=lambda r: r["created_at"])
            return self.promote(target["version"])
