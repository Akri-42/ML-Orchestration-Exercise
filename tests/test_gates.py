"""Table-driven tests for the promotion gate.

These are the tests that matter. 

Each case names the situation, builds a candidate/incumbent pair, and asserts
both the verdict *and* which specific check fired. Asserting the verdict alone
would let a gate pass for the wrong reason, which is the failure mode that
matters.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml_orch.gates import GateContext, promotion_gate
from ml_orch.types import EvalReport, TargetMetrics, Versions

TARGETS = ["mu", "alpha", "homo"]
N = 400
FEATURIZER = "rdkit-desc-v3:9f1c2a"


def make_report(
    scale: dict[str, float] | float = 1.0,
    *,
    model_version: str = "m",
    seed: int = 0,
    targets: list[str] | None = None,
    nan_target: str | None = None,
    n: int = N,
) -> EvalReport:
    """Build a report whose per-molecule errors are a scaled common draw.

    Scaling one shared draw keeps the pairs *paired*: two reports differ by a
    multiplicative factor on the same molecules, which is what lets the
    bootstrap see a real signal at realistic sample sizes.
    """
    names = targets or TARGETS
    rng = np.random.default_rng(seed)
    base = np.abs(rng.normal(size=n))
    per_target, stds = {}, {}
    for i, t in enumerate(names):
        s = scale if isinstance(scale, float) else scale.get(t, 1.0)
        errs = base * s * (1.0 + 0.1 * i)
        if nan_target == t:
            errs = errs.copy()
            errs[0] = np.nan
        mae = float(np.mean(errs))
        per_target[t] = TargetMetrics(
            target=t, mae=mae,
            rmse=float(np.sqrt(np.mean(errs ** 2))),
            r2=0.9, errors=errs,
        )
        stds[t] = 1.0 + 0.1 * i
    return EvalReport(model_version=model_version, eval_set="golden",
                      per_target=per_target, target_std=stds)


def ctx_for(candidate, incumbent=None, baseline=None, *,
            featurizer: str = FEATURIZER) -> GateContext:
    return GateContext(
        candidate=candidate,
        incumbent=incumbent,
        baseline=baseline,
        versions=Versions(data="sha256:aaaa", featurizer=featurizer,
                          model="cand", code="abc123", config="sha256:bbbb"),
        expected_featurizer=FEATURIZER,
    )


# ---------------------------------------------------------------------------
# name, context builder, expected verdict, check that must fail
# ---------------------------------------------------------------------------

CASES = [
    (
        "clearly better -> promote",
        lambda: ctx_for(make_report(0.70), make_report(1.00)),
        True, None,
    ),
    (
        "clearly worse -> blocked on margin",
        lambda: ctx_for(make_report(1.30), make_report(1.00)),
        False, "minimum_margin",
    ),
    (
        "exact tie -> blocked on margin, not promoted by accident",
        lambda: ctx_for(make_report(1.00), make_report(1.00)),
        False, "minimum_margin",
    ),
    (
        "within noise -> blocked; a 0.2% edge is not an improvement",
        lambda: ctx_for(make_report(0.998), make_report(1.00)),
        False, "minimum_margin",
    ),
    (
        "net better but one target regresses -> blocked",
        lambda: ctx_for(
            make_report({"mu": 0.50, "alpha": 0.50, "homo": 1.40}),
            make_report(1.00),
        ),
        False, "no_target_regression",
    ),
    (
        "no incumbent, beats baseline -> promote",
        lambda: ctx_for(make_report(0.50), None, make_report(1.00)),
        True, None,
    ),
    (
        "no incumbent, no baseline -> refuse to promote blind",
        lambda: ctx_for(make_report(0.50), None, None),
        False, "cold_start",
    ),
    (
        "no incumbent, barely beats baseline -> blocked on cold-start margin",
        lambda: ctx_for(make_report(0.95), None, make_report(1.00)),
        False, "cold_start",
    ),
    (
        "NaN metrics -> blocked on sanity",
        lambda: ctx_for(make_report(0.50, nan_target="mu"), make_report(1.00)),
        False, "sanity",
    ),
    (
        "stale featurizer -> blocked even though the model is better",
        lambda: ctx_for(make_report(0.50), make_report(1.00),
                        featurizer="rdkit-desc-v2:deadbe"),
        False, "featurizer_binding",
    ),
]


@pytest.mark.parametrize("name,build,expected,failing_check",
                         CASES, ids=[c[0] for c in CASES])
def test_promotion_gate(name, build, expected, failing_check):
    decision = promotion_gate().evaluate(build())
    assert decision.verdict is expected, decision.summary()
    if failing_check is not None:
        failed = {r.name for r in decision.failed}
        assert failing_check in failed, (
            f"expected {failing_check!r} to fire; failed checks were {sorted(failed)}\n"
            f"{decision.summary()}"
        )


def test_every_check_reports_even_when_one_fails():
    """No short-circuiting: a rejection record must carry all the evidence."""
    decision = promotion_gate().evaluate(
        ctx_for(make_report(1.30), make_report(1.00)))
    assert len(decision.results) == 7
    assert all(r.reason for r in decision.results)
    assert all(isinstance(r.evidence, dict) for r in decision.results)


def test_a_raising_gate_fails_closed():
    """A gate that explodes must block promotion, not be skipped."""
    class Exploding:
        name = "exploding"

        def evaluate(self, ctx):
            raise RuntimeError("boom")

    from ml_orch.gates import CompositeGate, SanityGate

    decision = CompositeGate([SanityGate(), Exploding()]).evaluate(
        ctx_for(make_report(0.5), make_report(1.0)))
    assert decision.verdict is False
    assert "exploding" in {r.name for r in decision.failed}
    assert "RuntimeError" in decision.failed[0].reason


def test_misaligned_error_vectors_are_rejected_not_silently_compared():
    """Comparing models scored on different eval sets is a bug, not a result."""
    decision = promotion_gate().evaluate(
        ctx_for(make_report(0.5, n=400), make_report(1.0, n=380)))
    assert decision.verdict is False
    assert "significance" in {r.name for r in decision.failed}


def test_decision_is_serialisable_for_the_manifest():
    d = promotion_gate().evaluate(ctx_for(make_report(0.7), make_report(1.0)))
    import json
    payload = json.loads(json.dumps(d.as_dict(), default=float))
    assert payload["verdict"] is True
    assert {c["name"] for c in payload["checks"]} >= {"sanity", "significance"}
