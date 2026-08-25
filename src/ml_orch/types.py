"""Core value types shared by both pipelines.

Everything here is frozen/plain data. The point is that a `GateResult` from the
promotion gate and a `GateResult` from the candidate-shortlist filter are the
same object, so the reporting, logging, and manifest code is written once.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Versions:
    """The five things that must be pinned for a run to be reproducible.

    Any prediction the system emits must be traceable to exactly one of these.
    """

    data: str          # content hash of the input chunk
    featurizer: str    # featurizer id + params hash
    model: str         # model version in the registry
    code: str          # git sha (or "dirty:<sha>")
    config: str        # hash of the resolved config

    def as_dict(self) -> dict[str, str]:
        return {
            "data": self.data,
            "featurizer": self.featurizer,
            "model": self.model,
            "code": self.code,
            "config": self.config,
        }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetMetrics:
    """Per-target metrics for one model on one evaluation set.

    `errors` is the per-molecule absolute error vector, kept because the
    significance gate needs paired samples, not point estimates.
    """

    target: str
    mae: float
    rmse: float
    r2: float
    errors: np.ndarray = field(repr=False)

    @property
    def n(self) -> int:
        return int(self.errors.shape[0])


@dataclass(frozen=True)
class EvalReport:
    """A model's performance on one named evaluation set.

    QM9's twelve targets are in mutually incomparable units (Debye, Hartree,
    Bohr^2, cal/mol-K), so `per_target` is the only honest primitive. Any scalar
    summary is a *policy* applied on top of it — see `gates.normalized_score`.
    """

    model_version: str
    eval_set: str                       # "golden" | "rolling" | ...
    per_target: Mapping[str, TargetMetrics]
    target_std: Mapping[str, float]     # std of each target on this eval set
    slices: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def targets(self) -> list[str]:
        return list(self.per_target)


# ---------------------------------------------------------------------------
# Gate results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateResult:
    """The outcome of a single check.

    `verdict` answers the aggregate question ("promote?" / "does this batch
    yield anything?"). `selection` is set only by gates that filter records,
    and is a boolean mask aligned with the scored batch. A composite ANDs both.

    `ranking` is set only by gates that *order* records — currently `TopKGate`,
    which has to sort in order to truncate. It is a rank per record (0 = best),
    aligned with the batch like `selection`, so it survives masking: filter the
    rows and the surviving ranks still sort correctly. Emitting it is what makes
    "ranked shortlist" a property of the output rather than a claim in the
    README — a gate that sorts and then discards the sort has done the work and
    thrown away the answer.

    `reason` is human-readable and goes in the log and the report. `evidence`
    is machine-readable and goes in the manifest. Both are mandatory: a gate
    that cannot explain itself is not a gate, it is a coin flip.
    """

    name: str
    verdict: bool
    reason: str
    evidence: Mapping[str, Any] = field(default_factory=dict)
    selection: np.ndarray | None = field(default=None, repr=False)
    ranking: np.ndarray | None = field(default=None, repr=False)
    blocking: bool = True

    def __post_init__(self) -> None:
        if self.selection is not None and self.selection.dtype != np.bool_:
            raise TypeError(
                f"gate {self.name!r}: selection must be a boolean mask, "
                f"got dtype {self.selection.dtype}"
            )
        if self.ranking is not None:
            if not np.issubdtype(self.ranking.dtype, np.integer):
                raise TypeError(
                    f"gate {self.name!r}: ranking must be integer ranks, "
                    f"got dtype {self.ranking.dtype}"
                )
            if self.selection is not None and self.ranking.shape != self.selection.shape:
                raise ValueError(
                    f"gate {self.name!r}: ranking and selection must be aligned, "
                    f"got {self.ranking.shape} and {self.selection.shape}"
                )


@dataclass(frozen=True)
class GateDecision:
    """The composite outcome: one verdict, every reason retained."""

    verdict: bool
    results: Sequence[GateResult]
    selection: np.ndarray | None = field(default=None, repr=False)

    @property
    def ranking(self) -> np.ndarray | None:
        """The ordering the composite arrived at, if any gate supplied one.

        Last writer wins: gates are composed in the order the config lists them
        and the ranking gate is applied last by convention, so the most recent
        ordering is the operative one.
        """
        for r in reversed(list(self.results)):
            if r.ranking is not None:
                return r.ranking
        return None

    @property
    def failed(self) -> list[GateResult]:
        return [r for r in self.results if not r.verdict]

    @property
    def warnings(self) -> list[GateResult]:
        return [r for r in self.results if not r.verdict and not r.blocking]

    def summary(self) -> str:
        head = "PASS" if self.verdict else "FAIL"
        lines = [f"{head} ({len(self.results)} checks)"]
        for r in self.results:
            mark = "ok  " if r.verdict else ("warn" if not r.blocking else "FAIL")
            lines.append(f"  [{mark}] {r.name}: {r.reason}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "n_selected": None if self.selection is None else int(self.selection.sum()),
            "checks": [
                {
                    "name": r.name,
                    "verdict": r.verdict,
                    "blocking": r.blocking,
                    "reason": r.reason,
                    "evidence": dict(r.evidence),
                }
                for r in self.results
            ],
        }
