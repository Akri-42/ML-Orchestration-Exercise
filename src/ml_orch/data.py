"""Ingest and validation.

Validation is the third user of the `Gate` interface, and that is the point.

A schema check, a promotion check, and a shortlist filter are the same shape: a
predicate over records that emits a verdict, a human-readable reason, a
machine-readable evidence dict, and — for row-level checks — a boolean mask.

So `validation_gate()` sits alongside `gates.promotion_gate()`: same factory
shape, same `CompositeGate`, same `GateDecision`. The two pipelines differ in
which checks the config names, not in what a check *is*.

Pipeline 1 is building a training set, so a bad row is a contaminant to quarantine. Pipeline
2 is triaging candidates, so a bad row is a candidate that gets reported as
undecidable and does not kill the batch.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd

from .gates import CompositeGate, Gate, GateContext
from .manifest import content_hash
from .types import GateDecision, GateResult

# The authoritative task list and column order, per
# deepchem/molnet/load_function/qm9_datasets.py. See refs/REFERENCES.md.
#
# NOTE: this is *not* the physical column order in the source CSV, where `cv`
# sits after `g298`. Always select by name. Positional indexing into QM9 is a
# bug waiting for a reader who assumes the docs describe the file.
QM9_TASKS: tuple[str, ...] = (
    "mu", "alpha", "homo", "lumo", "gap", "r2",
    "zpve", "cv", "u0", "u298", "h298", "g298",
)

SMILES_COLUMN = "smiles"
ID_COLUMN = "mol_id"

OnFailure = Literal["reject_chunk", "quarantine", "warn"]


def _ok(name: str, reason: str, *, selection: np.ndarray | None = None,
        **evidence: Any) -> GateResult:
    return GateResult(name=name, verdict=True, reason=reason,
                      evidence=evidence, selection=selection)


def _no(name: str, reason: str, *, selection: np.ndarray | None = None,
        blocking: bool = True, **evidence: Any) -> GateResult:
    return GateResult(name=name, verdict=False, reason=reason, evidence=evidence,
                      selection=selection, blocking=blocking)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    """One unit of arriving data, identified by content rather than by name.

    `data_version` is a hash of the file's bytes. That is the chunk's identity:
    the same chunk copied to a new filename is the same chunk and must not
    trigger a retrain, and a chunk edited in place is a different chunk even
    though the path did not change. The registry's idempotency check keys on
    exactly this.
    """

    chunk_id: str
    path: Path
    data_version: str
    frame: pd.DataFrame = field(repr=False)

    @property
    def n_rows(self) -> int:
        return len(self.frame)


class Source(Protocol):
    """Where records come from. Pipeline 1 gets training chunks, pipeline 2
    gets a candidate batch — same protocol, so downstream code cannot tell
    which pipeline it is running in."""

    kind: str

    def chunks(self) -> list[Chunk]: ...


@dataclass
class CsvChunkSource:
    """Pipeline 1's source: a directory of chunk files, in arrival order."""

    path: str | Path
    glob: str = "chunk_*.csv"
    kind: str = "csv_chunks"

    def chunks(self) -> list[Chunk]:
        root = Path(self.path)
        if not root.exists():
            raise FileNotFoundError(
                f"chunk directory {root} does not exist — run scripts/prepare_data.py")
        out = []
        for p in sorted(root.glob(self.glob)):
            out.append(Chunk(chunk_id=p.stem, path=p,
                             data_version=content_hash(p), frame=pd.read_csv(p)))
        return out


@dataclass
class CsvBatchSource:
    """Pipeline 2's source: one batch of candidate molecules."""

    path: str | Path
    kind: str = "csv_batch"

    def chunks(self) -> list[Chunk]:
        p = Path(self.path)
        if not p.exists():
            raise FileNotFoundError(f"candidate batch {p} does not exist")
        return [Chunk(chunk_id=p.stem, path=p,
                      data_version=content_hash(p), frame=pd.read_csv(p))]


def source_from_config(cfg: Mapping[str, Any]) -> Source:
    """`source:` block -> a Source. The only place `kind` strings are resolved."""
    kind = cfg.get("kind")
    if kind == "csv_chunks":
        return CsvChunkSource(path=cfg["path"], glob=cfg.get("glob", "chunk_*.csv"))
    if kind == "csv_batch":
        return CsvBatchSource(path=cfg["path"])
    raise ValueError(f"unknown source kind {kind!r} (expected csv_chunks | csv_batch)")


# ---------------------------------------------------------------------------
# Validation gates
# ---------------------------------------------------------------------------


@dataclass
class SchemaGate:
    """Batch-level. No mask: you cannot quarantine your way out of a missing column."""

    required_columns: Sequence[str]
    name: str = "schema"

    def evaluate(self, ctx: GateContext) -> GateResult:
        if ctx.frame is None:
            return _no(self.name, "no frame supplied")
        missing = [c for c in self.required_columns if c not in ctx.frame.columns]
        if missing:
            return _no(self.name, f"missing required column(s): {missing}",
                       missing=missing, present=list(ctx.frame.columns))
        return _ok(self.name, f"all {len(self.required_columns)} required columns present")


@dataclass
class MinRowsGate:
    """Batch-level. A chunk too small to learn from is a pipeline problem, not a
    data problem, and silently training on it produces a confidently bad model."""

    min_rows: int
    name: str = "min_rows"

    def evaluate(self, ctx: GateContext) -> GateResult:
        if ctx.frame is None:
            return _no(self.name, "no frame supplied")
        n = len(ctx.frame)
        if n < self.min_rows:
            return _no(self.name, f"{n} rows is below the {self.min_rows}-row minimum",
                       rows=n, minimum=self.min_rows)
        return _ok(self.name, f"{n} rows clears the {self.min_rows}-row minimum", rows=n)


@dataclass
class ParseableSmilesGate:
    """Row-level. Unparseable structures cannot be featurized, so they are
    dropped rather than allowed to explode inside the featurizer.

    `min_fraction` is the batch-level judgement on top of the row-level mask:
    a chunk where 40% of structures are junk is not a chunk with some bad rows,
    it is a broken upstream export, and the difference matters.
    """

    min_fraction: float = 0.99
    column: str = SMILES_COLUMN
    name: str = "parseable_smiles"

    def evaluate(self, ctx: GateContext) -> GateResult:
        if ctx.frame is None or self.column not in ctx.frame.columns:
            return _no(self.name, f"no {self.column!r} column to validate")
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")     # parse failures are data, not log noise

        values = ctx.frame[self.column].to_numpy()
        mask = np.array(
            [isinstance(s, str) and Chem.MolFromSmiles(s) is not None for s in values],
            dtype=bool)
        frac = float(mask.mean()) if mask.size else 0.0
        n_bad = int((~mask).sum())
        if frac < self.min_fraction:
            return _no(self.name,
                       f"only {frac:.2%} of structures parse, below the "
                       f"{self.min_fraction:.2%} floor ({n_bad} unparseable)",
                       selection=mask, fraction=frac, n_unparseable=n_bad,
                       minimum=self.min_fraction)
        return _ok(self.name, f"{frac:.2%} of structures parse ({n_bad} dropped)",
                   selection=mask, fraction=frac, n_unparseable=n_bad)


@dataclass
class DuplicateGate:
    """Row-level, keep-first. Duplicates inflate apparent performance by leaking
    the same molecule into both the training and evaluation halves."""

    max_fraction: float = 0.02
    subset: Sequence[str] = (SMILES_COLUMN,)
    name: str = "duplicates"

    def evaluate(self, ctx: GateContext) -> GateResult:
        if ctx.frame is None:
            return _no(self.name, "no frame supplied")
        cols = [c for c in self.subset if c in ctx.frame.columns]
        if not cols:
            return _no(self.name, f"none of the duplicate-key columns {list(self.subset)} present")
        dup = ctx.frame.duplicated(subset=cols, keep="first").to_numpy()
        mask = ~dup
        frac = float(dup.mean()) if dup.size else 0.0
        if frac > self.max_fraction:
            return _no(self.name,
                       f"{frac:.2%} duplicate rows exceeds the {self.max_fraction:.2%} ceiling",
                       selection=mask, fraction=frac, n_duplicates=int(dup.sum()),
                       maximum=self.max_fraction)
        return _ok(self.name, f"{frac:.2%} duplicates ({int(dup.sum())} dropped)",
                   selection=mask, fraction=frac, n_duplicates=int(dup.sum()))


@dataclass
class TargetNaNGate:
    """Row-level. A row missing the label cannot be trained on."""

    targets: Sequence[str]
    max_fraction: float = 0.01
    name: str = "target_nan"

    def evaluate(self, ctx: GateContext) -> GateResult:
        if ctx.frame is None:
            return _no(self.name, "no frame supplied")
        cols = [c for c in self.targets if c in ctx.frame.columns]
        if not cols:
            return _no(self.name, f"none of the target columns {list(self.targets)} present")
        bad = ctx.frame[cols].isna().any(axis=1).to_numpy()
        mask = ~bad
        frac = float(bad.mean()) if bad.size else 0.0
        if frac > self.max_fraction:
            return _no(self.name,
                       f"{frac:.2%} of rows have a missing target, above the "
                       f"{self.max_fraction:.2%} ceiling",
                       selection=mask, fraction=frac, n_rows_with_nan=int(bad.sum()),
                       maximum=self.max_fraction)
        return _ok(self.name, f"{frac:.2%} of rows have a missing target "
                              f"({int(bad.sum())} dropped)",
                   selection=mask, fraction=frac, n_rows_with_nan=int(bad.sum()))


def ks_statistic(sample: np.ndarray, reference: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic: max gap between the two ECDFs.

    Hand-rolled rather than pulled from scipy — it is six lines, scipy is only
    present transitively via scikit-learn, and an undeclared dependency is the
    kind of thing that breaks on someone else's machine.
    """
    a = np.sort(sample[np.isfinite(sample)])
    b = np.sort(reference[np.isfinite(reference)])
    if a.size == 0 or b.size == 0:
        return float("nan")
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / a.size
    cdf_b = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def psi_statistic(sample: np.ndarray, reference: np.ndarray, *,
                  buckets: int = 10, epsilon: float = 1e-6) -> float:
    """Population Stability Index: sum((actual - expected) * ln(actual/expected)).

    Bucket edges are the *reference's* quantiles, not an evenly spaced value
    grid. That choice is load-bearing on QM9, whose targets are heavy-tailed:
    equal-width bins would drop nearly all the reference mass into one bin and
    then report ~0 for any shift happening inside it. The frozen reference is
    already stored as a quantile grid (see `write_reference_stats`), so the
    edges fall straight out of it.

    Expected mass is *counted* from the reference rather than assumed to be
    1/buckets, because ties break that assumption: `mu` has a pile of exact
    zeros, so several quantiles coincide and the deduplicated edges no longer
    cut equal shares. `epsilon` floors an empty bucket so the log stays finite
    — the conventional dodge, and the reason a real-world PSI is large rather
    than infinite when the distributions stop overlapping.

    PSI wants meaningfully more rows than buckets — the floored empty buckets
    dominate the sum otherwise, and a 10-row chunk against 10 buckets reports a
    large "shift" that is really just sampling noise. The `min_rows` check in
    the same composite is what makes the default of 10 buckets safe here; drop
    `buckets` in config if a pipeline validates much smaller batches.

    Conventional reading is <0.1 stable, 0.1–0.25 moderate, >0.25 major, but
    that is a note for whoever writes the config: the threshold lives there,
    not here. Note also that PSI and KS are on different scales — KS is a
    probability gap bounded by 1, PSI is an unbounded divergence-like sum — so
    one threshold number cannot serve both.
    """
    if buckets < 2:
        raise ValueError(f"psi_statistic: buckets must be >= 2, got {buckets!r}")
    a = sample[np.isfinite(sample)]
    b = np.sort(reference[np.isfinite(reference)])
    if a.size == 0 or b.size == 0:
        return float("nan")
    edges = np.unique(np.quantile(b, np.linspace(0.0, 1.0, buckets + 1)[1:-1]))
    n_bins = edges.size + 1
    expected = np.bincount(np.searchsorted(edges, b, side="right"),
                           minlength=n_bins) / b.size
    actual = np.bincount(np.searchsorted(edges, a, side="right"),
                         minlength=n_bins) / a.size
    expected = np.maximum(expected, epsilon)
    actual = np.maximum(actual, epsilon)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


# The supported drift statistics. Named here so both the gate and the config
# factory reject an unknown one against the same list, and so adding a third
# is one entry plus one function rather than a hunt through `if` branches.
DRIFT_METHODS: tuple[str, ...] = ("ks", "psi")

# Fallback thresholds, used only when the config names no `max_statistic`. Per
# method, never one shared number: KS 0.25 is a quarter of the largest possible
# ECDF gap, PSI 0.25 is the conventional "major shift" line on an unbounded
# scale. The two are numerically equal and semantically unrelated.
DRIFT_DEFAULT_MAX: Mapping[str, float] = {"ks": 0.25, "psi": 0.25}


@dataclass
class DriftGate:
    """Batch-level covariate-shift check against a frozen reference.

    Non-blocking by default, and that is a deliberate policy choice rather than
    timidity. QM9 is ordered by molecule size, so sequential chunks drift *by
    construction* — that is the interesting property of this dataset, not a
    fault in it. Drift should therefore be recorded loudly and left to the
    promotion gate to adjudicate on held-out performance, where the question
    "did the model actually get worse" can be answered. Blocking ingest on
    drift would reject exactly the chunks the system exists to adapt to.

    Set `blocking: true` in config for a domain where novel inputs really are
    a stop-the-line event.

    `method` picks the statistic — `ks` or `psi`, see `DRIFT_METHODS`. It is a
    real switch: an unknown value raises here, at construction, rather than
    quietly falling back to KS. That matters more than it sounds, because
    `CompositeGate` deliberately converts an exception *inside* `evaluate` into
    a failed gate; a misconfiguration is not a verdict about the data, so it
    has to be caught while the gate is being built.

    `max_statistic` is the threshold for whichever method was chosen. The
    config carries one number per method (see `validation_gate`) because the
    two statistics are not on the same scale.
    """

    reference: str | Path
    max_statistic: float = 0.25
    columns: Sequence[str] = ()
    blocking: bool = False
    method: str = "ks"
    buckets: int = 10               # PSI resolution; ignored by KS
    name: str = "drift"

    def __post_init__(self) -> None:
        if self.method not in DRIFT_METHODS:
            raise ValueError(
                f"unknown drift method {self.method!r}; known: "
                f"{' | '.join(DRIFT_METHODS)}")

    def _statistic(self, sample: np.ndarray, reference: np.ndarray) -> float:
        if self.method == "psi":
            return psi_statistic(sample, reference, buckets=self.buckets)
        return ks_statistic(sample, reference)

    def evaluate(self, ctx: GateContext) -> GateResult:
        if ctx.frame is None:
            return _no(self.name, "no frame supplied", blocking=self.blocking)
        ref_path = Path(self.reference)
        if not ref_path.exists():
            # First chunk ever: there is nothing to drift from. Say so rather
            # than inventing a verdict.
            return _ok(self.name, f"no reference at {ref_path} yet; drift not evaluated",
                       reference=str(ref_path), method=self.method, evaluated=False)
        payload = json.loads(ref_path.read_text())
        ref_cols: dict[str, Any] = payload.get("columns", {})
        names = [c for c in (self.columns or ref_cols.keys()) if c in ctx.frame.columns]

        stats: dict[str, float] = {}
        for c in names:
            ref = ref_cols.get(c)
            if not ref:
                continue
            stats[c] = self._statistic(
                ctx.frame[c].to_numpy(dtype=float),
                np.asarray(ref["quantiles"], dtype=float),
            )
        if not stats:
            return _ok(self.name, "no comparable columns in the reference",
                       method=self.method, evaluated=False)

        label = self.method.upper()
        worst_col = max(stats, key=lambda k: (stats[k] if np.isfinite(stats[k]) else -1))
        worst = stats[worst_col]
        if worst > self.max_statistic:
            return _no(self.name,
                       f"distribution shift on {worst_col}: {label}={worst:.3f} exceeds "
                       f"{self.max_statistic:.3f}",
                       blocking=self.blocking, statistics=stats,
                       worst_column=worst_col, worst=worst,
                       method=self.method, maximum=self.max_statistic)
        return _ok(self.name, f"max {label}={worst:.3f} on {worst_col}, within "
                              f"{self.max_statistic:.3f}",
                   statistics=stats, worst_column=worst_col, worst=worst,
                   method=self.method)


def _drift_threshold(spec: Mapping[str, Any], method: str) -> float:
    """Resolve `max_statistic` for the selected drift method.

    The config carries a mapping — `max_statistic: {ks: 0.25, psi: 0.25}` —
    rather than a bare number, because KS and PSI answer to different scales
    and a single value would silently mean two different policies depending on
    `method`. A scalar is therefore refused rather than reinterpreted: it is
    the one shape whose meaning changes under the reader's feet, which is
    exactly the failure this gate is being fixed for.
    """
    if method not in DRIFT_METHODS:
        raise ValueError(
            f"unknown validation.checks.drift.method {method!r}; known: "
            f"{' | '.join(DRIFT_METHODS)}")
    raw = spec.get("max_statistic", DRIFT_DEFAULT_MAX)
    if not isinstance(raw, Mapping):
        # ValueError, not TypeError: this is a bad value in a config file, and
        # every other config complaint on this path raises ValueError. Keeping
        # one exception type is worth more than the lint's preference.
        raise ValueError(  # noqa: TRY004
            f"validation.checks.drift.max_statistic must be a per-method mapping "
            f"such as {{ks: 0.25, psi: 0.25}}, got {raw!r}. KS and PSI are on "
            f"different scales, so one number cannot serve both methods.")
    if method not in raw:
        raise ValueError(
            f"validation.checks.drift.method is {method!r} but max_statistic "
            f"has no threshold for it; configured: {sorted(raw)}")
    return float(raw[method])


def validation_gate(cfg: Mapping[str, Any], *,
                    targets: Sequence[str] = QM9_TASKS) -> CompositeGate:
    """Build the validation composite from a `validation:` config block.

    Deliberately mirrors `gates.promotion_gate()`: same factory shape, same
    return type. Only the checks the config names are built, so pipeline 2
    dropping `target_nan` and `min_rows` is a config edit, not a code path.
    """
    checks: Mapping[str, Any] = cfg.get("checks", {})
    built: list[Gate] = []

    if "schema" in checks:
        built.append(SchemaGate(required_columns=checks["schema"]["required_columns"]))
    if "min_rows" in checks:
        built.append(MinRowsGate(min_rows=int(checks["min_rows"])))
    if "parseable_smiles" in checks:
        built.append(ParseableSmilesGate(
            min_fraction=float(checks["parseable_smiles"].get("min_fraction", 0.99))))
    if "duplicates" in checks:
        built.append(DuplicateGate(
            max_fraction=float(checks["duplicates"].get("max_fraction", 0.02))))
    if "target_nan" in checks:
        built.append(TargetNaNGate(
            targets=list(targets),
            max_fraction=float(checks["target_nan"].get("max_fraction", 0.01))))
    if "drift" in checks:
        d = checks["drift"]
        # An unknown `method` raises here, at build time, rather than falling
        # back to KS or turning into a mysteriously failed gate at evaluate
        # time — see the note in DriftGate about CompositeGate failing closed.
        method = str(d.get("method", "ks"))
        built.append(DriftGate(reference=d["reference"],
                               method=method,
                               max_statistic=_drift_threshold(d, method),
                               buckets=int(d.get("buckets", 10)),
                               columns=list(d.get("columns", ())),
                               blocking=bool(d.get("blocking", False))))
    return CompositeGate(built, name="validation")


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass
class ValidationOutcome:
    """What validation decided, and the two frames that fall out of it."""

    decision: GateDecision
    accepted: pd.DataFrame = field(repr=False)
    quarantined: pd.DataFrame = field(repr=False)
    status: Literal["accepted", "accepted_with_quarantine", "rejected"]

    @property
    def usable(self) -> bool:
        return self.status != "rejected"

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "n_accepted": len(self.accepted),
            "n_quarantined": len(self.quarantined),
            "decision": self.decision.as_dict(),
        }


def apply_policy(frame: pd.DataFrame, decision: GateDecision,
                 on_failure: OnFailure = "quarantine") -> ValidationOutcome:
    """Turn a `GateDecision` into accepted/quarantined frames under a policy.

    
    * ``reject_chunk`` — any blocking failure rejects the whole chunk. For
      regulated inputs where a partially-bad batch is not a partially-good one.
    * ``quarantine`` — drop the offending rows, keep the rest, and keep the
      dropped rows with the gate that dropped them. The default.
    * ``warn`` — accept everything, record every complaint. For observability
      runs where you want to see what *would* have been dropped first.

    A batch-level blocking failure (schema, min_rows) rejects under every
    policy except ``warn``, because there are no offending *rows* to drop —
    the shape of the data is wrong, not its contents.
    """
    n = len(frame)
    empty = frame.iloc[0:0]

    if on_failure == "warn":
        return ValidationOutcome(decision=decision, accepted=frame, quarantined=empty,
                                 status="accepted")

    blocking_failures = [r for r in decision.results if not r.verdict and r.blocking]
    structural = [r for r in blocking_failures if r.selection is None]

    mask = decision.selection
    if mask is None:
        mask = np.ones(n, dtype=bool)
    n_masked = int((~mask).sum())

    # reject_chunk is all-or-nothing, so a *masked row* rejects too, not only a
    # failed check. A chunk can lose rows without any check failing its
    # batch-level threshold -- three good structures and one junk one clears a
    # 50% parse floor -- and silently accepting a partial chunk under a policy
    # named "reject_chunk" would be the wrong reading of the operator's intent.
    if structural or (on_failure == "reject_chunk" and (blocking_failures or n_masked)):
        return ValidationOutcome(decision=decision, accepted=empty, quarantined=frame,
                                 status="rejected")
    accepted = frame[mask]
    quarantined = frame[~mask]
    status = "accepted_with_quarantine" if len(quarantined) else "accepted"
    return ValidationOutcome(decision=decision, accepted=accepted,
                             quarantined=quarantined, status=status)


def validate(frame: pd.DataFrame, cfg: Mapping[str, Any], *,
             targets: Sequence[str] = QM9_TASKS) -> ValidationOutcome:
    """Validate one frame end to end: build the gate, run it, apply the policy."""
    gate = validation_gate(cfg, targets=targets)
    decision = gate.evaluate(GateContext(frame=frame))
    return apply_policy(frame, decision, cfg.get("on_failure", "quarantine"))


# ---------------------------------------------------------------------------
# Preparing the local dataset
# ---------------------------------------------------------------------------


def write_reference_stats(frame: pd.DataFrame, columns: Sequence[str],
                          path: str | Path, n_quantiles: int = 401) -> Path:
    """Freeze a reference distribution for `DriftGate`.

    Stored as a quantile grid rather than the raw sample: it is ~400 floats per
    column instead of thousands, and a grid uniformly spaced in probability is
    itself a fair sample of the reference distribution, so a two-sample KS
    against it is a sound approximation. 
    """
    qs = np.linspace(0.0, 1.0, n_quantiles)
    payload = {
        "n_quantiles": n_quantiles,
        "n_rows": len(frame),
        "columns": {
            c: {
                "quantiles": [float(v) for v in np.nanquantile(
                    frame[c].to_numpy(dtype=float), qs)],
                "mean": float(np.nanmean(frame[c].to_numpy(dtype=float))),
                "std": float(np.nanstd(frame[c].to_numpy(dtype=float))),
            }
            for c in columns if c in frame.columns
        },
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2) + "\n")
    return p


def heavy_atom_counts(smiles: Sequence[str]) -> np.ndarray:
    """Heavy-atom count per molecule. The size axis everything else hangs off."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    out = np.zeros(len(smiles), dtype=int)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        out[i] = mol.GetNumHeavyAtoms() if mol is not None else 0
    return out


def _size_stratified_pool(frame: pd.DataFrame, heavy: np.ndarray,
                          n_molecules: int, rng: np.random.Generator) -> pd.DataFrame:
    """Sample a pool that spans the size range, then order it by size.

    Allocation is two-pass. Every size first gets an equal share of the budget,
    capped at what exists — the small sizes are scarce (QM9 has three
    one-heavy-atom molecules and 111,594 nine-atom ones), so most of them
    exhaust immediately. The unspent remainder is then redistributed across the
    sizes that still have capacity, in proportion to what they have left.


    Within a size, original file order is preserved. Sampling decides *which*
    molecules arrive when; it never touches a value.
    """
    sizes = np.unique(heavy[heavy > 0])
    available = {int(sz): np.flatnonzero(heavy == sz) for sz in sizes}

    share = max(1, n_molecules // len(sizes))
    quota = {sz: min(share, len(idx)) for sz, idx in available.items()}

    leftover = n_molecules - sum(quota.values())
    spare = {sz: len(idx) - quota[sz] for sz, idx in available.items()}
    total_spare = sum(spare.values())
    if leftover > 0 and total_spare > 0:
        for sz, cap in spare.items():
            quota[sz] += min(cap, round(leftover * cap / total_spare))

    picks = [
        np.sort(rng.choice(idx, size=quota[sz], replace=False))
        if quota[sz] < len(idx) else idx
        for sz, idx in available.items() if quota[sz] > 0
    ]
    taken = np.concatenate(picks)
    order = np.lexsort((taken, heavy[taken]))     # by size, then file order
    return frame.iloc[taken[order]].reset_index(drop=True)


def split_into_chunks(raw: pd.DataFrame, *, n_molecules: int = 5000, n_chunks: int = 6,
                      golden_fraction: float = 0.15, seed: int = 0,
                      mode: Literal["size_ramp", "sequential"] = "size_ramp",
                      ) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    """Carve the raw table into arriving chunks plus a frozen golden set.

    **On the covariate shift, and why it is constructed rather than found.**

    The received wisdom is that QM9 is ordered by molecule size, so slicing it
    sequentially gives you drift for free. Measured, that turns out to be
    almost entirely false: QM9 is 83% nine-heavy-atom molecules, and every size
    below eight appears within the first 4,000 of 133,885 rows. Sequential
    chunks therefore saturate at nine heavy atoms after the first chunk and the
    remaining five are near-identical. The one target that does shift is
    `alpha`, because polarizability scales with size.

    So `mode="size_ramp"` (the default) constructs the shift on purpose: the
    sampled pool is stratified across sizes and then *ordered* by size, so
    chunk 0 is small molecules and chunk 5 is large ones. Arrival order is
    manipulated; no value is. 

    `mode="sequential"` preserves raw file order for comparison, and is what
    the received wisdom actually gets you. The EDA reports both.

    **The golden set is stratified across chunks, then frozen.** It is sampled
    from every chunk rather than held out from the tail, because under a size
    ramp a golden set drawn from early chunks would only ever ask "is the model
    still good at small molecules".
    """
    if n_chunks < 1:
        raise ValueError("n_chunks must be >= 1")
    rng = np.random.default_rng(seed)

    if mode == "size_ramp":
        heavy = heavy_atom_counts(raw[SMILES_COLUMN].tolist())
        sub = _size_stratified_pool(raw, heavy, n_molecules, rng)
    elif mode == "sequential":
        step = max(1, len(raw) // n_molecules)
        sub = raw.iloc[::step].head(n_molecules).reset_index(drop=True)
    else:
        raise ValueError(f"unknown mode {mode!r} (expected size_ramp | sequential)")

    # Slice by explicit position rather than np.array_split: under pandas 3 that
    # returns ndarrays and silently loses the columns.
    bounds = np.linspace(0, len(sub), n_chunks + 1).astype(int)

    chunks: list[pd.DataFrame] = []
    golden_parts: list[pd.DataFrame] = []
    for lo, hi in pairwise(bounds):
        part = sub.iloc[lo:hi].reset_index(drop=True)
        n_golden = round(len(part) * golden_fraction)
        idx = (rng.choice(len(part), size=n_golden, replace=False)
               if n_golden else np.array([], dtype=int))
        held = np.zeros(len(part), dtype=bool)
        held[idx] = True
        golden_parts.append(part[held])
        chunks.append(part[~held].reset_index(drop=True))

    golden = pd.concat(golden_parts, ignore_index=True) if golden_parts else sub.iloc[0:0]
    return chunks, golden
