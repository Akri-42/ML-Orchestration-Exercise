"""Exploratory analysis, as functions rather than a notebook.



* `size_dominance` is why four targets are excluded from the gate.
* `scale_disparity` is why the aggregation policy normalises.
* `chunk_drift` is why the drift gate is non-blocking.
* `target_redundancy` is why the twelve targets are not twelve independent
  pieces of evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import QM9_TASKS, SMILES_COLUMN, heavy_atom_counts, ks_statistic

# Units, from refs/REFERENCES.md. Carried here because the whole normalisation
# argument rests on them being mutually incomparable.
UNITS: dict[str, str] = {
    "mu": "Debye", "alpha": "Bohr³", "homo": "Hartree", "lumo": "Hartree",
    "gap": "Hartree", "r2": "Bohr²", "zpve": "Hartree", "cv": "cal/(mol·K)",
    "u0": "Hartree", "u298": "Hartree", "h298": "Hartree", "g298": "Hartree",
}


@dataclass
class SizeProfile:
    counts: pd.Series               # heavy atoms -> n molecules
    fraction_largest: float
    first_row_of_size: dict[int, int]
    n_molecules: int


def size_profile(raw: pd.DataFrame, sample: int | None = None) -> SizeProfile:
    """How QM9 is actually distributed over molecule size.

    The headline number — that one size accounts for most of the dataset — is
    the reason the "sequential chunks give you drift for free" 
    """
    frame = raw if sample is None else raw.head(sample)
    heavy = heavy_atom_counts(frame[SMILES_COLUMN].tolist())
    counts = pd.Series(heavy).value_counts().sort_index()
    first = {int(s): int(np.argmax(heavy == s)) for s in counts.index}
    return SizeProfile(counts=counts,
                       fraction_largest=float(counts.max() / counts.sum()),
                       first_row_of_size=first,
                       n_molecules=len(frame))


def scale_disparity(frame: pd.DataFrame,
                    targets: list[str] | None = None) -> pd.DataFrame:
    """Magnitude of each target, which is the case for normalising.

    A mean of raw MAEs across these is a number with no meaning: it is
    dominated by whichever target happens to be measured in the largest units.
    """
    names = targets or [t for t in QM9_TASKS if t in frame.columns]
    rows = []
    for t in names:
        v = frame[t].to_numpy(dtype=float)
        rows.append({"target": t, "unit": UNITS.get(t, "?"),
                     "mean": float(np.nanmean(v)), "std": float(np.nanstd(v)),
                     "abs_mean": float(np.nanmean(np.abs(v))),
                     "range": float(np.nanmax(v) - np.nanmin(v))})
    out = pd.DataFrame(rows).set_index("target")
    out["orders_of_magnitude_vs_smallest"] = np.log10(
        out["std"] / out["std"].min()).round(2)
    return out


def size_dominance(frame: pd.DataFrame,
                   targets: list[str] | None = None) -> pd.DataFrame:
    """How much of each target is explained by heavy-atom count *alone*.

    A target with R² near 1.0 against a single integer is not a modelling
    achievement waiting to happen.
    """
    names = targets or [t for t in QM9_TASKS if t in frame.columns]
    heavy = heavy_atom_counts(frame[SMILES_COLUMN].tolist()).astype(float)
    rows = []
    for t in names:
        y = frame[t].to_numpy(dtype=float)
        ok = np.isfinite(y)
        slope, intercept = np.polyfit(heavy[ok], y[ok], 1)
        pred = slope * heavy[ok] + intercept
        ss_res = float(np.sum((y[ok] - pred) ** 2))
        ss_tot = float(np.sum((y[ok] - y[ok].mean()) ** 2))
        rows.append({"target": t, "r2_vs_heavy_atoms": 1.0 - ss_res / ss_tot if ss_tot else np.nan})
    return pd.DataFrame(rows).set_index("target").sort_values(
        "r2_vs_heavy_atoms", ascending=False)


def element_count_dominance(frame: pd.DataFrame,
                           targets: list[str] | None = None) -> pd.DataFrame:
    """R² of a linear model on **element counts alone** (nC, nN, nO, nF, nH).

   A single-variable fit against
    heavy-atom count puts `u0` at R²=0.41, which looks like a real modelling
    problem. Against element counts it is R²=1.00000 — the quantity is a sum of
    per-atom contributions and a linear model recovers it exactly.

    
    """
    names = targets or [t for t in QM9_TASKS if t in frame.columns]
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    elements = ("C", "N", "O", "F", "H")
    mols = [Chem.AddHs(Chem.MolFromSmiles(s)) for s in frame[SMILES_COLUMN]]
    counts = np.array([[sum(1 for a in m.GetAtoms() if a.GetSymbol() == e)
                        for e in elements] for m in mols], dtype=float)
    design = np.hstack([counts, np.ones((len(counts), 1))])

    rows = []
    for t in names:
        y = frame[t].to_numpy(dtype=float)
        ok = np.isfinite(y)
        coef, *_ = np.linalg.lstsq(design[ok], y[ok], rcond=None)
        pred = design[ok] @ coef
        ss_tot = float(np.sum((y[ok] - y[ok].mean()) ** 2))
        r2 = 1.0 - float(np.sum((y[ok] - pred) ** 2)) / ss_tot if ss_tot else np.nan
        rows.append({"target": t, "r2_element_counts": r2,
                     "verdict": "atom counting" if r2 > 0.99
                     else ("mostly size" if r2 > 0.9 else "real chemistry")})
    return pd.DataFrame(rows).set_index("target").sort_values(
        "r2_element_counts", ascending=False)


def target_redundancy(frame: pd.DataFrame,
                      targets: list[str] | None = None) -> pd.DataFrame:
    """Correlation between targets.

    Twelve targets are not twelve independent pieces of evidence. If four of
    them are near-perfectly correlated, a per-target no-regression gate counts
    that one physical quantity four times, which quietly overweights it.
    """
    names = targets or [t for t in QM9_TASKS if t in frame.columns]
    return frame[names].corr()


def chunk_drift(chunks: dict[str, pd.DataFrame], reference: pd.DataFrame,
                targets: list[str] | None = None) -> pd.DataFrame:
    """KS statistic per chunk per target, against a fixed reference chunk."""
    names = targets or [t for t in QM9_TASKS if t in reference.columns]
    rows = {}
    for chunk_id, frame in chunks.items():
        rows[chunk_id] = {
            t: ks_statistic(frame[t].to_numpy(dtype=float),
                            reference[t].to_numpy(dtype=float))
            for t in names if t in frame.columns
        }
    return pd.DataFrame(rows).T


def chunk_sizes(chunks: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Mean/min/max heavy atoms per chunk — the constructed ramp, measured."""
    rows = {}
    for chunk_id, frame in chunks.items():
        heavy = heavy_atom_counts(frame[SMILES_COLUMN].tolist())
        rows[chunk_id] = {"n": len(frame), "mean_heavy": float(heavy.mean()),
                          "min_heavy": int(heavy.min()), "max_heavy": int(heavy.max())}
    return pd.DataFrame(rows).T


def model_trajectory(registry) -> pd.DataFrame:
    """Every model the registry has seen, in order, with its gate outcome.
    """
    rows = []
    for rec in sorted(registry.records(), key=lambda r: r.created_at):
        gate = rec.gate_decision or {}
        failed = [c["name"] for c in gate.get("checks", []) if not c["verdict"]]
        rows.append({
            "version": rec.version,
            "stage": rec.stage,
            "golden_score": rec.metrics.get("golden", {}).get("normalized_score"),
            "rolling_score": rec.metrics.get("rolling", {}).get("normalized_score"),
            "verdict": gate.get("verdict"),
            "blocked_by": ", ".join(failed),
        })
    return pd.DataFrame(rows)


def per_target_metrics(registry, eval_set: str = "golden") -> pd.DataFrame:
    """Per-target MAE for every registered model — where regressions hide."""
    rows = {}
    for rec in sorted(registry.records(), key=lambda r: r.created_at):
        per = rec.metrics.get(eval_set, {}).get("per_target", {})
        if per:
            rows[rec.version[-13:]] = {t: v["mae"] for t, v in per.items()}
    return pd.DataFrame(rows).T


def slice_table(registry, eval_set: str = "golden") -> pd.DataFrame:
    """Per-slice normalized MAE for the production model."""
    rec = registry.production()
    if rec is None:
        return pd.DataFrame()
    slices = rec.metrics.get(eval_set, {}).get("slices", {})
    return pd.DataFrame(slices).T


def triage_funnel(run_dir: str | Path) -> pd.DataFrame:
    """How a candidate batch narrowed, gate by gate."""
    import json
    manifest = json.loads((Path(run_dir) / "manifest.json").read_text())
    checks = manifest.get("gate_decision", {}).get("checks", [])
    rows = []
    for c in checks:
        ev = c.get("evidence", {})
        if "kept" in ev:
            rows.append({"gate": c["name"], "kept": ev["kept"],
                         "total": ev["total"], "dropped": ev["total"] - ev["kept"]})
    rows.append({"gate": "final (intersection)",
                 "kept": manifest.get("outputs", {}).get("n_shortlisted"),
                 "total": manifest.get("inputs", {}).get("n_candidates"),
                 "dropped": None})
    return pd.DataFrame(rows)


def load_chunks(path: str | Path = "data/chunks") -> dict[str, pd.DataFrame]:
    return {p.stem: pd.read_csv(p) for p in sorted(Path(path).glob("chunk_*.csv"))}


def findings(raw: pd.DataFrame, chunks: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """The numbers the write-up quotes, computed once so they cannot drift."""
    prof = size_profile(raw)
    dom = size_dominance(raw.head(20000))
    elem = element_count_dominance(raw.head(20000))
    scale = scale_disparity(raw)
    ids = sorted(chunks)
    drift = chunk_drift(chunks, chunks[ids[0]])
    return {
        "n_molecules": prof.n_molecules,
        "fraction_largest_size": prof.fraction_largest,
        "largest_size": int(prof.counts.idxmax()),
        "first_row_of_largest": prof.first_row_of_size[int(prof.counts.idxmax())],
        "size_dominated_targets_naive_test": dom[
            dom["r2_vs_heavy_atoms"] > 0.95].index.tolist(),
        "atom_counting_targets": elem[
            elem["r2_element_counts"] > 0.99].index.tolist(),
        "scale_spread_orders": float(scale["orders_of_magnitude_vs_smallest"].max()),
        "max_drift": float(drift.max().max()),
        "drift_by_chunk": drift.max(axis=1).to_dict(),
    }
