"""Registry semantics: staging, promotion, rejection, rollback, idempotency.

Fast and infrastructure-free, so the behaviours the brief asks about by name
are covered unconditionally rather than as a side effect of a pipeline run.
"""

from __future__ import annotations

import fcntl
import json
import multiprocessing
import os

import pytest

from ml_orch.registry import ModelRegistry

VERSIONS = {"data": "sha256:aaa", "featurizer": "f@1/2", "model": "m",
            "code": "abc123", "config": "sha256:cfg"}


def reg(tmp_path) -> ModelRegistry:
    return ModelRegistry(tmp_path / "registry")


def add(r: ModelRegistry, version: str, *, data: str = "sha256:aaa",
        stage: str = "staging", verdict: bool = True):
    return r.register(version, versions={**VERSIONS, "data": data},
                      metrics={"golden": {"normalized_score": 0.5}},
                      gate_decision={"verdict": verdict, "checks": []}, stage=stage)


def test_a_new_registry_has_nothing_in_production(tmp_path):
    assert reg(tmp_path).production() is None


def test_promotion_archives_the_previous_holder(tmp_path):
    r = reg(tmp_path)
    add(r, "v1"); add(r, "v2", data="sha256:bbb")
    r.promote("v1")
    r.promote("v2")
    assert r.production().version == "v2"
    assert r.get("v1").stage == "archived"


def test_rollback_returns_to_the_most_recent_archived_model(tmp_path):
    r = reg(tmp_path)
    add(r, "v1"); add(r, "v2", data="sha256:bbb")
    r.promote("v1")
    r.promote("v2")
    r.rollback()
    assert r.production().version == "v1"
    assert r.get("v2").stage == "archived"


def test_rollback_with_nothing_to_roll_back_to_is_an_explicit_error(tmp_path):
    r = reg(tmp_path)
    add(r, "v1")
    r.promote("v1")
    with pytest.raises(RuntimeError, match="nothing to roll back"):
        r.rollback()


def test_a_rejected_model_can_never_be_promoted(tmp_path):
    r = reg(tmp_path)
    add(r, "bad", stage="rejected", verdict=False)
    with pytest.raises(ValueError, match="refusing to promote rejected"):
        r.promote("bad")


def test_rejected_models_keep_their_decision_record(tmp_path):
    r = reg(tmp_path)
    rec = add(r, "bad", stage="rejected", verdict=False)
    assert rec.gate_decision["verdict"] is False
    assert r.get("bad").stage == "rejected"


def test_duplicate_versions_are_refused(tmp_path):
    r = reg(tmp_path)
    add(r, "v1")
    with pytest.raises(ValueError, match="already registered"):
        add(r, "v1")


def test_find_by_data_version_includes_archived(tmp_path):
    """Archived is still 'we have done this work'. Omitting it means a chunk
    looks unseen the moment its model is superseded."""
    r = reg(tmp_path)
    add(r, "v1", data="sha256:chunk0")
    add(r, "v2", data="sha256:chunk1")
    r.promote("v1")
    r.promote("v2")
    assert r.get("v1").stage == "archived"
    assert r.find_by_data_version("sha256:chunk0").version == "v1"


def test_find_by_data_version_misses_an_unseen_chunk(tmp_path):
    r = reg(tmp_path)
    add(r, "v1", data="sha256:chunk0")
    assert r.find_by_data_version("sha256:never") is None


def test_the_index_is_valid_json_after_every_write(tmp_path):
    """Write-then-rename: a reader must never observe a half-written index."""
    r = reg(tmp_path)
    add(r, "v1"); r.promote("v1")
    add(r, "v2", data="sha256:bbb"); r.promote("v2")
    payload = json.loads((tmp_path / "registry" / "index.json").read_text())
    assert payload["production"] == "v2"
    assert len(payload["records"]) == 2


def test_each_model_carries_a_manifest_beside_its_artifacts(tmp_path):
    r = reg(tmp_path)
    rec = add(r, "v1")
    manifest = json.loads((tmp_path / "registry" / "models" / "v1" / "manifest.json").read_text())
    assert manifest["version"] == "v1"
    assert manifest["versions"]["data"] == rec.data_version


def test_promoting_an_unknown_version_is_a_key_error(tmp_path):
    with pytest.raises(KeyError):
        reg(tmp_path).promote("nope")


# -- concurrency ----------------------------------------------------------
#
# The shipped design fans out: `chunk_arrival_sensor` emits one RunRequest per
# chunk, all at once, with no Dagster concurrency limit. Several `register()`
# calls therefore land on one index.json at the same moment. Without a lock
# each of them reads the index, appends one record in memory, and writes the
# whole file back -- so the last writer wins and the other records vanish
# silently, with no exception and with the artifact directories still on disk.
# That is the worst shape a failure can take: a registry that looks healthy and
# has quietly forgotten models.

N_CONCURRENT = 8


def _register_in_child(root: str, version: str, barrier) -> None:
    """Runs in a separate process. The barrier makes the race deterministic:
    every child is parked immediately before the read-modify-write."""
    r = ModelRegistry(root)
    barrier.wait(timeout=30)
    r.register(version, versions={**VERSIONS, "data": f"sha256:{version}"},
               metrics={"golden": {"normalized_score": 0.5}},
               gate_decision={"verdict": True, "checks": []})


def test_concurrent_registrations_do_not_clobber_each_other(tmp_path):
    """N processes, N records. Separate processes, not threads, because the
    lock we rely on is an advisory file lock -- a thread-only test would pass
    against a `threading.Lock`."""
    ctx = multiprocessing.get_context("spawn")
    root = str(tmp_path / "registry")
    ModelRegistry(root)                       # create the index once, up front
    barrier = ctx.Barrier(N_CONCURRENT)

    procs = [ctx.Process(target=_register_in_child, args=(root, f"v{i}", barrier))
             for i in range(N_CONCURRENT)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)

    assert [p.exitcode for p in procs] == [0] * N_CONCURRENT

    payload = json.loads((tmp_path / "registry" / "index.json").read_text())
    survived = sorted(r["version"] for r in payload["records"])
    assert survived == sorted(f"v{i}" for i in range(N_CONCURRENT))


def test_a_failed_register_still_releases_the_lock(tmp_path):
    """A lock held past an exception is worse than the race it prevents: the
    next run blocks forever instead of losing a record. `LOCK_NB` on a second
    file description is the honest probe -- flock scopes to the open file, so
    this fails if the lock leaked even though it is the same process."""
    r = reg(tmp_path)
    add(r, "v1")
    with pytest.raises(ValueError, match="already registered"):
        add(r, "v1")

    fd = os.open(tmp_path / "registry" / "index.lock", os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)   # raises if still held
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def test_rollback_does_not_deadlock_against_its_own_promote(tmp_path):
    """`rollback()` holds the lock across its choice and then calls
    `promote()`, which asks for it again. Non-re-entrant locking makes a plain
    single-threaded `mlorch rollback` hang."""
    r = reg(tmp_path)
    add(r, "v1"); add(r, "v2", data="sha256:bbb")
    r.promote("v1"); r.promote("v2")
    assert r.rollback().version == "v1"
