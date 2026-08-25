"""ml_orch — orchestration primitives shared by both pipelines.

Pipeline 1 (lifecycle) and pipeline 2 (triage) are configurations over these
primitives, not separate programs. See README.md for the argument.
"""

from .gates import (
    ColdStartGate,
    CompositeGate,
    FeaturizerBindingGate,
    Gate,
    GateContext,
    MinimumMarginGate,
    NoTargetRegressionGate,
    PropertyThresholdGate,
    SanityGate,
    SignificanceGate,
    SliceGate,
    TopKGate,
    UncertaintyGate,
    normalized_score,
    paired_bootstrap_pvalue,
    promotion_gate,
    shortlist_gate,
)
from .manifest import RunManifest, code_version, config_hash, content_hash
from .registry import ModelRecord, ModelRegistry
from .types import EvalReport, GateDecision, GateResult, TargetMetrics, Versions

__all__ = [
    "ColdStartGate",
    "CompositeGate",
    "EvalReport",
    "FeaturizerBindingGate",
    "Gate",
    "GateContext",
    "GateDecision",
    "GateResult",
    "MinimumMarginGate",
    "ModelRecord",
    "ModelRegistry",
    "NoTargetRegressionGate",
    "PropertyThresholdGate",
    "RunManifest",
    "SanityGate",
    "SignificanceGate",
    "SliceGate",
    "TargetMetrics",
    "TopKGate",
    "UncertaintyGate",
    "Versions",
    "code_version",
    "config_hash",
    "content_hash",
    "normalized_score",
    "paired_bootstrap_pvalue",
    "promotion_gate",
    "shortlist_gate",
]

__version__ = "0.1.0"
