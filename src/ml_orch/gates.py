"""Gates — the load-bearing abstraction.

It composes a different list of `Gate` objects over the same `GateContext`.

No orchestrator import, no I/O, no global state — so it is trivially unit-testable, which is where the tests are.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .types import EvalReport, GateDecision, GateResult, Versions

# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@dataclass
class GateContext:
    """Everything any gate is allowed to look at.

    Promotion gates read `candidate` / `incumbent`. Shortlist gates read
    `scores` / `uncertainty`. A gate that needs a field the context lacks must
    say so and fail *closed* — never silently pass.
    """

    # -- promotion-shaped inputs
    candidate: EvalReport | None = None
    incumbent: EvalReport | None = None
    baseline: EvalReport | None = None       # dummy/mean predictor, for cold start

    # -- shortlist-shaped inputs
    scores: Mapping[str, np.ndarray] | None = None      # target -> (n_records,)
    uncertainty: Mapping[str, np.ndarray] | None = None
    record_ids: Sequence[str] | None = None

    # -- ingest-shaped inputs
    #
    # Validation is the third user of this interface. A schema check and a
    # shortlist filter are the same shape: a predicate over records emitting a
    # verdict, a reason, and (for row-level checks) a mask. Typed loosely as
    # Any so this module keeps its numpy-only surface and stays free of pandas.
    frame: Any = None

    # -- provenance
    versions: Versions | None = None
    expected_featurizer: str | None = None

    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def n_records(self) -> int:
        if self.scores is not None:
            return int(next(iter(self.scores.values())).shape[0])
        if self.frame is not None:
            return len(self.frame)
        return 0

    def all_pass_mask(self) -> np.ndarray:
        return np.ones(self.n_records, dtype=bool)


class Gate(Protocol):
    """A named predicate over a `GateContext`."""

    name: str

    def evaluate(self, ctx: GateContext) -> GateResult: ...


def _fail(name: str, reason: str, *, blocking: bool = True, **evidence: Any) -> GateResult:
    return GateResult(name=name, verdict=False, reason=reason,
                      evidence=evidence, blocking=blocking)


def _pass(name: str, reason: str, **evidence: Any) -> GateResult:
    return GateResult(name=name, verdict=True, reason=reason, evidence=evidence)


# ---------------------------------------------------------------------------
# Metric aggregation policy
# ---------------------------------------------------------------------------


def normalized_score(report: EvalReport, targets: Sequence[str] | None = None) -> float:
    """Aggregate a multi-target report into one comparable number.

    QM9's twelve targets are in incomparable units, so a mean of raw MAEs is
    dominated by whichever target happens to have the largest magnitude and is
    therefore meaningless. We normalise each target's MAE by that target's
    standard deviation on the evaluation set — i.e. MAE in units of "how much
    this property varies" — and then average. Lower is better; 1.0 is roughly
    the score of predicting the mean.

    This is a *policy*, and it is stated out loud rather than buried: swapping
    it for per-target MAE ratio against the incumbent is a config change.
    """
    names = list(targets or report.targets)
    if not names:
        raise ValueError("normalized_score: no targets")
    parts = []
    for t in names:
        std = report.target_std.get(t, 0.0)
        if not std or not math.isfinite(std):
            raise ValueError(f"normalized_score: target {t!r} has degenerate std={std!r}")
        parts.append(report.per_target[t].mae / std)
    return float(np.mean(parts))


def paired_bootstrap_pvalue(
    candidate_errors: np.ndarray,
    incumbent_errors: np.ndarray,
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """One-sided paired bootstrap: is the candidate's mean error lower?

    Returns `(observed_delta, p_value)` where `delta = incumbent_mae -
    candidate_mae` (positive means the candidate is better) and `p_value` is
    the fraction of resamples in which the candidate was *not* better.

    Paired, because the two models are evaluated on the same molecules and the
    per-molecule difficulty is enormous shared variance. Unpaired tests here
    would be underpowered to the point of uselessness.
    """
    if candidate_errors.shape != incumbent_errors.shape:
        raise ValueError(
            "paired_bootstrap_pvalue: error vectors must be aligned "
            f"({candidate_errors.shape} vs {incumbent_errors.shape}) — this "
            "usually means the two models were scored on different eval sets"
        )
    diff = incumbent_errors - candidate_errors          # >0 favours candidate
    observed = float(diff.mean())
    rng = np.random.default_rng(seed)
    n = diff.shape[0]
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = diff[idx].mean(axis=1)
    p = float((means <= 0).mean())
    return observed, p


# ---------------------------------------------------------------------------
# Promotion gates
# ---------------------------------------------------------------------------


@dataclass
class SanityGate:
    """Non-metric preconditions. Cheap, and catches the embarrassing failures."""

    name: str = "sanity"

    def evaluate(self, ctx: GateContext) -> GateResult:
        if ctx.candidate is None:
            return _fail(self.name, "no candidate report supplied")
        bad = [
            t for t, m in ctx.candidate.per_target.items()
            if not math.isfinite(m.mae) or not math.isfinite(m.rmse)
        ]
        if bad:
            return _fail(self.name, f"non-finite metrics for {bad}", targets=bad)
        empty = [t for t, m in ctx.candidate.per_target.items() if m.n == 0]
        if empty:
            return _fail(self.name, f"empty error vectors for {empty}", targets=empty)
        return _pass(self.name, f"{len(ctx.candidate.per_target)} targets finite and non-empty")


@dataclass
class FeaturizerBindingGate:
    """Train/serve skew guard.

    The classic way this whole class of system goes quietly wrong: inference
    re-derives the featurizer instead of loading the one bound to the promoted
    model. This makes that failure loud.
    """

    name: str = "featurizer_binding"

    def evaluate(self, ctx: GateContext) -> GateResult:
        if ctx.versions is None or ctx.expected_featurizer is None:
            return _fail(self.name, "cannot verify featurizer binding: version info missing")
        actual = ctx.versions.featurizer
        if actual != ctx.expected_featurizer:
            return _fail(
                self.name,
                f"featurizer mismatch: model expects {ctx.expected_featurizer!r}, "
                f"run used {actual!r}",
                expected=ctx.expected_featurizer, actual=actual,
            )
        return _pass(self.name, f"featurizer {actual!r} matches model binding")


@dataclass
class ColdStartGate:
    """No incumbent: the candidate must still beat a trivial baseline.

    Otherwise the first model ever trained gets promoted unconditionally, which
    is how a broken pipeline ships a mean-predictor to production on day one.
    """

    min_improvement: float = 0.20      # must be >=20% better than the baseline
    name: str = "cold_start"

    def evaluate(self, ctx: GateContext) -> GateResult:
        if ctx.incumbent is not None:
            return _pass(self.name, "incumbent exists; cold-start check not applicable")
        if ctx.candidate is None:
            return _fail(self.name, "no candidate report supplied")
        if ctx.baseline is None:
            return _fail(self.name, "no incumbent and no baseline: refusing to promote blind")
        cand = normalized_score(ctx.candidate)
        base = normalized_score(ctx.baseline)
        rel = (base - cand) / base if base else 0.0
        if rel < self.min_improvement:
            return _fail(
                self.name,
                f"cold start: candidate {cand:.4f} vs baseline {base:.4f} "
                f"= {rel:+.1%}, below the required {self.min_improvement:.0%}",
                candidate=cand, baseline=base, relative=rel,
            )
        return _pass(self.name, f"cold start: {rel:+.1%} vs baseline",
                     candidate=cand, baseline=base, relative=rel)


@dataclass
class MinimumMarginGate:
    """Require a real margin, not a coin flip.

    Without this, a pipeline that retrains nightly promotes roughly half the
    time on noise alone and the production model thrashes.
    """

    min_relative_improvement: float = 0.01
    name: str = "minimum_margin"

    def evaluate(self, ctx: GateContext) -> GateResult:
        if ctx.candidate is None:
            return _fail(self.name, "no candidate report supplied")
        if ctx.incumbent is None:
            return _pass(self.name, "no incumbent; margin check delegated to cold_start")
        cand = normalized_score(ctx.candidate)
        inc = normalized_score(ctx.incumbent)
        rel = (inc - cand) / inc if inc else 0.0
        if rel < self.min_relative_improvement:
            return _fail(
                self.name,
                f"improvement {rel:+.2%} is below the {self.min_relative_improvement:.2%} "
                f"margin (candidate {cand:.4f} vs incumbent {inc:.4f})",
                candidate=cand, incumbent=inc, relative=rel,
            )
        return _pass(self.name, f"improvement {rel:+.2%} clears the margin",
                     candidate=cand, incumbent=inc, relative=rel)


@dataclass
class SignificanceGate:
    """The improvement must exceed noise, judged by paired bootstrap."""

    alpha: float = 0.05
    n_resamples: int = 2000
    seed: int = 0
    name: str = "significance"

    def evaluate(self, ctx: GateContext) -> GateResult:
        if ctx.candidate is None:
            return _fail(self.name, "no candidate report supplied")
        if ctx.incumbent is None:
            return _pass(self.name, "no incumbent; significance not applicable")

        per_target: dict[str, dict[str, float]] = {}
        deltas = []
        for t, m in ctx.candidate.per_target.items():
            if t not in ctx.incumbent.per_target:
                return _fail(self.name, f"incumbent has no metrics for target {t!r}")
            std = ctx.candidate.target_std.get(t) or 1.0
            obs, p = paired_bootstrap_pvalue(
                m.errors / std,
                ctx.incumbent.per_target[t].errors / std,
                n_resamples=self.n_resamples, seed=self.seed,
            )
            per_target[t] = {"delta": obs, "p": p}
            deltas.append(obs)

        pooled_p = float(np.mean([v["p"] for v in per_target.values()]))
        if pooled_p > self.alpha:
            return _fail(
                self.name,
                f"improvement is within noise (pooled p={pooled_p:.3f} > alpha={self.alpha})",
                pooled_p=pooled_p, per_target=per_target,
            )
        return _pass(self.name, f"improvement is significant (pooled p={pooled_p:.3f})",
                     pooled_p=pooled_p, per_target=per_target)


@dataclass
class NoTargetRegressionGate:
    """Block a net-better model if any single target falls apart.

    An aggregate win can hide one property regressing badly. For candidate
    triage that is not a rounding error
    """

    max_relative_regression: float = 0.05
    name: str = "no_target_regression"

    def evaluate(self, ctx: GateContext) -> GateResult:
        if ctx.candidate is None:
            return _fail(self.name, "no candidate report supplied")
        if ctx.incumbent is None:
            return _pass(self.name, "no incumbent; per-target regression not applicable")
        offenders: dict[str, float] = {}
        for t, m in ctx.candidate.per_target.items():
            inc = ctx.incumbent.per_target.get(t)
            if inc is None or not inc.mae:
                continue
            rel = (m.mae - inc.mae) / inc.mae      # >0 means worse
            if rel > self.max_relative_regression:
                offenders[t] = rel
        if offenders:
            worst = max(offenders, key=offenders.get)
            return _fail(
                self.name,
                f"{len(offenders)} target(s) regressed beyond "
                f"{self.max_relative_regression:.0%}; worst is {worst} at {offenders[worst]:+.1%}",
                offenders=offenders,
            )
        return _pass(self.name, "no target regressed beyond threshold")


@dataclass
class SliceGate:
    """Subgroup guardrail — e.g. by heavy-atom count or rare-element presence.

    QM9 is roughly ordered by molecule size, so sequential chunks produce
    genuine covariate shift. A model can improve on average while getting worse
    on exactly the large molecules a later chunk introduced.
    """

    max_relative_regression: float = 0.10
    metric: str = "normalized_mae"
    name: str = "slices"

    def evaluate(self, ctx: GateContext) -> GateResult:
        if ctx.candidate is None:
            return _fail(self.name, "no candidate report supplied")
        if ctx.incumbent is None or not ctx.candidate.slices:
            return _pass(self.name, "no slice metrics to compare")
        offenders: dict[str, float] = {}
        for slice_name, vals in ctx.candidate.slices.items():
            inc_vals = ctx.incumbent.slices.get(slice_name)
            if not inc_vals or self.metric not in vals or self.metric not in inc_vals:
                continue
            base = inc_vals[self.metric]
            if not base:
                continue
            rel = (vals[self.metric] - base) / base
            if rel > self.max_relative_regression:
                offenders[slice_name] = rel
        if offenders:
            return _fail(self.name, f"slice regression on {sorted(offenders)}",
                         offenders=offenders)
        return _pass(self.name, f"{len(ctx.candidate.slices)} slices within tolerance")


# ---------------------------------------------------------------------------
# Shortlist gates — same interface, record-level selection
# ---------------------------------------------------------------------------


@dataclass
class PropertyThresholdGate:
    """Keep records whose predicted property falls in an acceptable window."""

    target: str
    minimum: float | None = None
    maximum: float | None = None
    name: str = ""

    def __post_init__(self) -> None:
        self.name = self.name or f"threshold[{self.target}]"

    def evaluate(self, ctx: GateContext) -> GateResult:
        if ctx.scores is None or self.target not in ctx.scores:
            return _fail(self.name, f"no predictions for target {self.target!r}")
        vals = ctx.scores[self.target]
        mask = np.ones(vals.shape[0], dtype=bool)
        if self.minimum is not None:
            mask &= vals >= self.minimum
        if self.maximum is not None:
            mask &= vals <= self.maximum
        kept = int(mask.sum())
        return GateResult(
            name=self.name,
            verdict=kept > 0,
            reason=f"{kept}/{mask.shape[0]} candidates within "
                   f"[{self.minimum}, {self.maximum}] on {self.target}",
            evidence={"kept": kept, "total": int(mask.shape[0]),
                      "min": self.minimum, "max": self.maximum},
            selection=mask,
        )


@dataclass
class UncertaintyGate:
    """Drop candidates the model is not entitled to have an opinion about.

    This is the applicability-domain check. It is also the gate that, inverted,
    becomes the acquisition function for an active-learning pipeline — the same
    class, a different config, no new primitives.
    """

    target: str
    max_uncertainty: float
    invert: bool = False       # invert=True selects the *most* uncertain (active learning)
    name: str = ""

    def __post_init__(self) -> None:
        kind = "acquisition" if self.invert else "uncertainty"
        self.name = self.name or f"{kind}[{self.target}]"

    def evaluate(self, ctx: GateContext) -> GateResult:
        if ctx.uncertainty is None or self.target not in ctx.uncertainty:
            return _fail(self.name, f"no uncertainty estimates for target {self.target!r}")
        u = ctx.uncertainty[self.target]
        mask = u > self.max_uncertainty if self.invert else u <= self.max_uncertainty
        kept = int(mask.sum())
        return GateResult(
            name=self.name,
            verdict=kept > 0,
            reason=f"{kept}/{mask.shape[0]} candidates "
                   f"{'above' if self.invert else 'within'} "
                   f"uncertainty {self.max_uncertainty:g} on {self.target}",
            evidence={"kept": kept, "total": int(mask.shape[0]),
                      "max_uncertainty": self.max_uncertainty, "invert": self.invert},
            selection=mask,
        )


@dataclass
class TopKGate:
    """Rank-and-truncate. Applied last, after the filters have had their say."""

    target: str
    k: int
    largest: bool = False       # QM9 targets are mostly "lower is better"
    name: str = ""

    def __post_init__(self) -> None:
        self.name = self.name or f"top_k[{self.target}]"

    def evaluate(self, ctx: GateContext) -> GateResult:
        if ctx.scores is None or self.target not in ctx.scores:
            return _fail(self.name, f"no predictions for target {self.target!r}")
        vals = ctx.scores[self.target]
        order = np.argsort(-vals if self.largest else vals, kind="stable")
        mask = np.zeros(vals.shape[0], dtype=bool)
        mask[order[: self.k]] = True
        # The sort is the expensive half of "rank-and-truncate" and it has
        # already been paid for. Invert the permutation into a rank per record
        # so the caller can order the shortlist without re-deriving it — and so
        # the rank survives the mask.
        ranking = np.empty(vals.shape[0], dtype=np.int64)
        ranking[order] = np.arange(vals.shape[0], dtype=np.int64)
        kept = int(mask.sum())
        return GateResult(
            name=self.name,
            verdict=kept > 0,
            reason=f"kept top {kept} of {mask.shape[0]} by {self.target} "
                   f"({'desc' if self.largest else 'asc'})",
            evidence={"k": self.k, "kept": kept, "total": int(mask.shape[0]),
                      "ranked_by": self.target,
                      "direction": "desc" if self.largest else "asc"},
            selection=mask,
            ranking=ranking,
        )


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


@dataclass
class CompositeGate:
    """Run every gate, always. Combine the verdicts; never short-circuit.

    Short-circuiting would be faster and much less useful: when a promotion is
    rejected you want *all* the reasons in the manifest, not just the first
    one, because the next run is going to be triaged by a human reading that
    record.

    Non-blocking gates report and are recorded but cannot veto.
    """

    gates: Sequence[Gate]
    name: str = "composite"

    def evaluate(self, ctx: GateContext) -> GateDecision:
        results: list[GateResult] = []
        for g in self.gates:
            try:
                results.append(g.evaluate(ctx))
            except Exception as exc:  # noqa: BLE001 — a broken gate must fail closed
                results.append(_fail(
                    getattr(g, "name", g.__class__.__name__),
                    f"gate raised {type(exc).__name__}: {exc}",
                    error=type(exc).__name__,
                ))

        verdict = all(r.verdict for r in results if r.blocking)

        selection: np.ndarray | None = None
        for r in results:
            if r.selection is None:
                continue
            selection = r.selection.copy() if selection is None else (selection & r.selection)

        if selection is not None:
            # For record-filtering pipelines the aggregate verdict means
            # "did this batch yield anything at all", not "did every gate pass".
            verdict = verdict and bool(selection.any())

        return GateDecision(verdict=verdict, results=results, selection=selection)


def shortlist_gate(specs: Sequence[Mapping[str, Any]]) -> CompositeGate:
    """Build the triage composite from the `gate:` list in a triage config.

    The counterpart to `promotion_gate` and `data.validation_gate`, and the
    literal answer to "how much of pipeline 2 is configuration": this function
    is the whole of it.

    `FeaturizerBindingGate` is prepended unconditionally and cannot be
    configured away. Every other check here is policy — which property window,
    which uncertainty ceiling, how many candidates — and policy belongs in
    config. Scoring candidates with a featurizer that is not the one bound to
    the promoted model is not a policy, it is a defect, and a defect must not
    be switchable off from a YAML file.
    """
    built: list[Gate] = [FeaturizerBindingGate()]
    for spec in specs:
        kind = spec.get("kind")
        if kind == "property_threshold":
            built.append(PropertyThresholdGate(
                target=spec["target"], minimum=spec.get("minimum"),
                maximum=spec.get("maximum")))
        elif kind == "uncertainty":
            built.append(UncertaintyGate(
                target=spec["target"], max_uncertainty=float(spec["max_uncertainty"]),
                invert=bool(spec.get("invert", False))))
        elif kind == "top_k":
            built.append(TopKGate(target=spec["target"], k=int(spec["k"]),
                                  largest=bool(spec.get("largest", False))))
        else:
            raise ValueError(
                f"unknown shortlist gate kind {kind!r}; known: "
                f"property_threshold | uncertainty | top_k")
    return CompositeGate(built, name="triage")


def promotion_gate(**overrides: Any) -> CompositeGate:
    """The default promotion policy. Config overrides thresholds, not code."""
    return CompositeGate([
        SanityGate(),
        FeaturizerBindingGate(),
        ColdStartGate(min_improvement=overrides.get("cold_start_min_improvement", 0.20)),
        MinimumMarginGate(min_relative_improvement=overrides.get("min_margin", 0.01)),
        SignificanceGate(alpha=overrides.get("alpha", 0.05),
                         n_resamples=overrides.get("n_resamples", 2000)),
        NoTargetRegressionGate(
            max_relative_regression=overrides.get("max_target_regression", 0.05)),
        SliceGate(max_relative_regression=overrides.get("max_slice_regression", 0.10)),
    ], name="promotion")
