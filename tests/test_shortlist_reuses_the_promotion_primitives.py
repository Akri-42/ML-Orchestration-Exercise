"""

The triage shortlist is a `CompositeGate` built
from `Gate` implementations, evaluated against the same `GateContext`, yielding
the same `GateDecision` type that the promotion path produces.

"""

from __future__ import annotations

import numpy as np

from ml_orch.gates import (
    CompositeGate,
    GateContext,
    PropertyThresholdGate,
    TopKGate,
    UncertaintyGate,
    promotion_gate,
)
from ml_orch.types import GateDecision

N = 50


def batch_ctx(seed: int = 0) -> GateContext:
    rng = np.random.default_rng(seed)
    return GateContext(
        scores={"gap": rng.uniform(0.1, 0.4, size=N),
                "mu": rng.uniform(0.0, 5.0, size=N)},
        uncertainty={"gap": rng.uniform(0.0, 0.1, size=N)},
        record_ids=[f"cand-{i:03d}" for i in range(N)],
    )


def triage_gate() -> CompositeGate:
    """Exactly what a YAML config would build. No new classes."""
    return CompositeGate([
        PropertyThresholdGate(target="gap", minimum=0.20, maximum=0.35),
        UncertaintyGate(target="gap", max_uncertainty=0.05),
        TopKGate(target="mu", k=10),
    ], name="triage")


def test_shortlist_and_promotion_share_the_interface():
    promo = promotion_gate()
    triage = triage_gate()
    assert type(promo) is type(triage) is CompositeGate
    for g in list(promo.gates) + list(triage.gates):
        assert hasattr(g, "name") and callable(g.evaluate)


def test_shortlist_returns_the_same_decision_type():
    decision = triage_gate().evaluate(batch_ctx())
    assert isinstance(decision, GateDecision)
    assert decision.selection is not None
    assert decision.selection.dtype == np.bool_


def test_filters_intersect_rather_than_last_one_winning():
    ctx = batch_ctx()
    decision = triage_gate().evaluate(ctx)
    mask = decision.selection

    within_window = (ctx.scores["gap"] >= 0.20) & (ctx.scores["gap"] <= 0.35)
    confident = ctx.uncertainty["gap"] <= 0.05
    assert np.all(mask <= within_window), "kept a candidate outside the property window"
    assert np.all(mask <= confident), "kept a candidate the model is not confident about"
    assert mask.sum() <= 10, "top-k truncation did not apply"


def test_every_survivor_carries_a_reason_trail():
    decision = triage_gate().evaluate(batch_ctx())
    assert len(decision.results) == 3
    assert all(r.reason and r.evidence for r in decision.results)


def test_empty_shortlist_is_a_false_verdict_not_a_crash():
    ctx = batch_ctx()
    gate = CompositeGate([PropertyThresholdGate(target="gap", minimum=99.0)])
    decision = gate.evaluate(ctx)
    assert decision.verdict is False
    assert decision.selection is not None and decision.selection.sum() == 0
    assert "0/" in decision.results[0].reason


def test_active_learning_is_one_config_flip():
    """The P3 story, asserted rather than asserted-in-prose.

    Inverting the uncertainty gate turns the triage pipeline into an
    acquisition step that selects the molecules worth sending to an assay.
    Same class, same context, same decision object — no new primitive.
    """
    ctx = batch_ctx()
    confident = CompositeGate([UncertaintyGate(target="gap", max_uncertainty=0.05)])
    acquire = CompositeGate([
        UncertaintyGate(target="gap", max_uncertainty=0.05, invert=True)])

    a = confident.evaluate(ctx).selection
    b = acquire.evaluate(ctx).selection
    assert not np.any(a & b), "confident and acquisition sets must be disjoint"
    assert np.all(a | b), "every candidate belongs to exactly one of the two"
