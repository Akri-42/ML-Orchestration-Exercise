#!/usr/bin/env python3
"""Exploratory analysis: figures and a written report.

    python scripts/eda.py

Writes PNGs to `reports/figures/` and a summary to `reports/EDA.md`.

Run the lifecycle first (`scripts/demo.py` or `ml-orch backfill`) so the
model-outcome figures have a registry to read.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ml_orch import eda
from ml_orch.registry import ModelRegistry

INK = "#1b1b1f"
ACCENT = "#2f6f9f"
WARN = "#c2492d"
MUTED = "#9aa0a6"
GOOD = "#3f7d5a"

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 140, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "figure.facecolor": "white",
})


def save(fig, out: Path, name: str) -> str:
    path = out / name
    fig.savefig(path)
    plt.close(fig)
    print(f"[fig ] {path}")
    return name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=Path("data/raw/qm9.csv"))
    ap.add_argument("--out", type=Path, default=Path("reports"))
    ap.add_argument("--sample", type=int, default=20000,
                    help="rows used for the expensive RDKit passes")
    args = ap.parse_args()

    figures = args.out / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.raw)
    sample = raw.head(args.sample)
    chunks = eda.load_chunks()
    ids = sorted(chunks)
    print(f"[read] {len(raw)} molecules, {len(chunks)} chunks")

    # -- 1. the shape of QM9 ------------------------------------------------
    prof = eda.size_profile(raw)
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    bars = ax.bar(prof.counts.index, prof.counts.to_numpy(), color=ACCENT, width=0.72)
    bars[list(prof.counts.index).index(prof.counts.idxmax())].set_color(WARN)
    ax.set_yscale("log")
    ax.set_xlabel("heavy atoms per molecule")
    ax.set_ylabel("molecules (log)")
    ax.set_title("QM9 is one molecule size wearing a distribution")
    ax.annotate(f"{prof.fraction_largest:.1%} of the dataset\nis exactly "
                f"{prof.counts.idxmax()} heavy atoms",
                xy=(prof.counts.idxmax(), prof.counts.max()),
                xytext=(-135, -18), textcoords="offset points", color=WARN,
                fontsize=8.5, fontweight="bold")
    f1 = save(fig, figures, "01_size_distribution.png")

    # -- 2. why naive mean-MAE is meaningless -------------------------------
    scale = eda.scale_disparity(raw)
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    order = scale.sort_values("std")
    ax.barh(range(len(order)), order["std"], color=ACCENT)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{t}  ({eda.UNITS.get(t,'?')})" for t in order.index])
    ax.set_xscale("log")
    ax.set_xlabel("standard deviation (log scale, native units)")
    ax.set_title("Targets differ by four orders of magnitude — so we normalise")
    f2 = save(fig, figures, "02_scale_disparity.png")

    # -- 3. the atom-counting trap -----------------------------------------
    elem = eda.element_count_dominance(sample)
    naive = eda.size_dominance(sample)
    joined = elem.join(naive).sort_values("r2_element_counts")
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    y = np.arange(len(joined))
    ax.barh(y + 0.19, joined["r2_element_counts"], height=0.38, color=WARN,
            label="linear model on element counts")
    ax.barh(y - 0.19, joined["r2_vs_heavy_atoms"], height=0.38, color=MUTED,
            label="single fit on heavy-atom count")
    ax.axvline(0.99, color=INK, ls=":", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(joined.index)
    ax.set_xlabel("R² — variance explained without any chemistry")
    ax.set_title("Which targets are just atom counting")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.text(0.985, len(joined) - 0.4, "R²=0.99", ha="right", fontsize=7.5, color=INK)
    f3 = save(fig, figures, "03_atom_counting_trap.png")

    # -- 4. the constructed size ramp --------------------------------------
    sizes = eda.chunk_sizes(chunks)
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.plot(range(len(ids)), sizes["mean_heavy"], "o-", color=ACCENT, lw=2)
    ax.fill_between(range(len(ids)), sizes["min_heavy"], sizes["max_heavy"],
                    color=ACCENT, alpha=0.15)
    ax.set_xticks(range(len(ids))); ax.set_xticklabels(ids, rotation=20)
    ax.set_ylabel("heavy atoms")
    ax.set_title("Arrival order is constructed: chunks get larger, then hit QM9's ceiling")
    f4 = save(fig, figures, "04_size_ramp.png")

    # -- 5. drift ----------------------------------------------------------
    drift = eda.chunk_drift(chunks, chunks[ids[0]])
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    im = ax.imshow(drift.to_numpy(), aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(drift.columns))); ax.set_xticklabels(drift.columns, rotation=45)
    ax.set_yticks(range(len(drift.index))); ax.set_yticklabels(drift.index)
    ax.set_title("Covariate shift by chunk (KS vs chunk_00) — the drift gate's input")
    for i in range(drift.shape[0]):
        for j in range(drift.shape[1]):
            v = drift.to_numpy()[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if v > 0.6 else INK)
    fig.colorbar(im, ax=ax, label="KS statistic", shrink=0.85)
    f5 = save(fig, figures, "05_drift_heatmap.png")

    # -- 6/7/8. model outcomes --------------------------------------------
    figs_models: list[str] = []
    registry = ModelRegistry("registry")
    traj = eda.model_trajectory(registry)
    if not traj.empty and traj["golden_score"].notna().any():
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        x = range(len(traj))
        ax.plot(x, traj["golden_score"], "o-", color=ACCENT, lw=2, label="golden (frozen)")
        if traj["rolling_score"].notna().any():
            ax.plot(x, traj["rolling_score"], "s--", color=MUTED, lw=1.5,
                    label="rolling (recent chunks)")
        for i, row in traj.iterrows():
            if row["verdict"] is False:
                ax.scatter([i], [row["golden_score"]], s=190, facecolors="none",
                           edgecolors=WARN, lw=2, zorder=5)
                ax.annotate(f"REJECTED\n{row['blocked_by'].split(':')[-1]}",
                            xy=(i, row["golden_score"]), xytext=(-16, 26),
                            textcoords="offset points", color=WARN, fontsize=8,
                            fontweight="bold", ha="center")
        ax.set_xticks(list(x)); ax.set_xticklabels([f"m{i}" for i in x])
        ax.set_ylabel("normalized score (lower is better)")
        ax.set_title("Every candidate the gate saw, and the one it blocked")
        ax.legend(frameon=False, fontsize=8)
        figs_models.append(save(fig, figures, "06_model_trajectory.png"))

        per = eda.per_target_metrics(registry)
        if not per.empty:
            norm = per / per.iloc[0]
            fig, ax = plt.subplots(figsize=(6.6, 3.4))
            for t in norm.columns:
                ax.plot(range(len(norm)), norm[t], "o-", lw=1.4, label=t, alpha=0.85)
            ax.axhline(1.0, color=MUTED, ls=":", lw=1)
            ax.set_xticks(range(len(norm)))
            ax.set_xticklabels([f"m{i}" for i in range(len(norm))])
            ax.set_ylabel("MAE relative to the first model")
            ax.set_title("Per-target MAE — where an aggregate win hides a regression")
            ax.legend(ncol=4, fontsize=7.5, frameon=False)
            figs_models.append(save(fig, figures, "07_per_target.png"))

        sl = eda.slice_table(registry)
        if not sl.empty:
            fig, ax = plt.subplots(figsize=(6.2, 3.0))
            sl = sl.sort_values("normalized_mae")
            colors = [WARN if v > sl["normalized_mae"].min() * 1.5 else GOOD
                      for v in sl["normalized_mae"]]
            ax.barh(range(len(sl)), sl["normalized_mae"], color=colors)
            ax.set_yticks(range(len(sl))); ax.set_yticklabels(sl.index)
            ax.set_xlabel("normalized MAE (lower is better)")
            ax.set_title("Production model by slice — worse where molecules are bigger")
            for i, (v, n) in enumerate(zip(sl["normalized_mae"], sl["n"], strict=True)):
                ax.text(v, i, f"  n={int(n)}", va="center", fontsize=7.5, color=MUTED)
            figs_models.append(save(fig, figures, "08_slice_performance.png"))

    # -- 9. triage funnel --------------------------------------------------
    triage_runs = sorted(Path("runs").glob("triage-*")) if Path("runs").exists() else []
    if triage_runs:
        funnel = eda.triage_funnel(triage_runs[-1])
        fig, ax = plt.subplots(figsize=(6.2, 3.0))
        ax.barh(range(len(funnel)), funnel["kept"], color=[ACCENT] * (len(funnel) - 1) + [GOOD])
        ax.set_yticks(range(len(funnel))); ax.set_yticklabels(funnel["gate"])
        ax.invert_yaxis()
        ax.set_xlabel("candidates surviving")
        ax.set_title("Triage funnel — each gate's own verdict, then the intersection")
        for i, v in enumerate(funnel["kept"]):
            if pd.notna(v):
                ax.text(v, i, f"  {int(v)}", va="center", fontsize=8, color=INK)
        figs_models.append(save(fig, figures, "09_triage_funnel.png"))

    # -- report ------------------------------------------------------------
    found = eda.findings(raw, chunks)
    red = eda.target_redundancy(sample)
    energy_block = [t for t in ("u0", "u298", "h298", "g298") if t in red.columns]
    redundancy_note = ""
    if len(energy_block) > 1:
        sub = red.loc[energy_block, energy_block].to_numpy()
        off = sub[~np.eye(len(energy_block), dtype=bool)]
        redundancy_note = (f"The four energy targets correlate at "
                           f"{off.min():.5f}–{off.max():.5f} with each other — they are "
                           f"one quantity counted four times.")

    lines = [
        "# EDA — what the data actually says",
        "",
        "Generated by `scripts/eda.py`. Every figure here changed a design decision;",
        "none of them are included because a chart was available.",
        "",
        "## 1. QM9 is one molecule size wearing a distribution",
        "",
        (f"- {found['n_molecules']:,} molecules, of which "
         f"**{found['fraction_largest_size']:.1%} have exactly "
         f"{found['largest_size']} heavy atoms**."),
        (f"- The first {found['largest_size']}-atom molecule appears at row "
         f"{found['first_row_of_largest']:,} — so every size below it lives in the "
         "first 3% of the file."),
        "",
        "**Consequence.** The folklore that sequential chunks of QM9 give you covariate",
        "shift for free is very nearly false: slice it in order and every chunk after the",
        "first is saturated. The shift in this project is therefore constructed on purpose",
        "(`split_into_chunks(mode=\"size_ramp\")`), which is stated rather than smuggled.",
        "",
        f"![size distribution](figures/{f1})",
        f"![size ramp](figures/{f4})",
        "",
        "## 2. The targets are not comparable, so the aggregation policy is not optional",
        "",
        (f"- Target standard deviations span **{found['scale_spread_orders']:.1f} "
         "orders of magnitude**, in Debye, Hartree, Bohr², Bohr³ and cal/(mol·K)."),
        "- A mean of raw MAEs across them is dominated by whichever target has the largest",
        "  units — here `r2`, whose MAE is ~91 against `zpve`'s ~0.008.",
        "",
        f"![scale disparity](figures/{f2})",
        "",
        "## 3. Some targets are atom counting, not chemistry",
        "",
        "A linear model on element counts alone (nC, nN, nO, nF, nH) explains:",
        "",
        "| target | R² from element counts | verdict |",
        "|---|---|---|",
    ]
    for t, row in elem.iterrows():
        lines.append(f"| `{t}` | {row['r2_element_counts']:.5f} | {row['verdict']} |")
    lines += [
        "",
        "**Two corrections to the received wisdom here.**",
        "",
        "First, the test has to be the right one. A single-variable fit against heavy-atom",
        "count puts `u0` at R²=0.41, which looks like an ordinary modelling problem. Against",
        "element counts it is R²=1.00000. Testing the easy version would have confirmed the",
        "wrong conclusion.",
        "",
        (f"Second, `zpve` is at {elem.loc['zpve', 'r2_element_counts']:.5f} — "
         "essentially atom"),
        "counting too, and it was in the default gate targets. It has been removed, because",
        "the rule that excludes the four energies has to apply to it as well.",
        "",
        "And the usual remedy is weaker than advertised: the atomization energies",
        "(`u0_atom` and friends) are still ~0.99 on the same test. They are an improvement",
        "on 1.00000, not a solution.",
        "",
        redundancy_note,
        "",
        f"![atom counting](figures/{f3})",
        "",
        "## 4. The drift is real, measurable, and deliberately non-blocking",
        "",
        f"- Maximum KS across all chunks and targets: **{found['max_drift']:.3f}**.",
        ("- By chunk (max across targets): "
         + ", ".join(f"`{k}`={v:.2f}" for k, v in found["drift_by_chunk"].items()) + "."),
        "",
        "The drift gate records this loudly and does **not** block ingest. These chunks",
        "drift by construction, and rejecting them would reject exactly the data the system",
        "exists to adapt to. Whether the drift actually hurt is a question about held-out",
        "performance, and that is the promotion gate's job.",
        "",
        f"![drift](figures/{f5})",
        "",
    ]
    if figs_models:
        lines += [
            "## 5. What the gate did",
            "",
            traj.to_markdown(index=False) if not traj.empty else "_no models registered_",
            "",
            "The interesting row is the rejection. The candidate was genuinely better and",
            "was blocked anyway, because the improvement could not be separated from noise",
            "by a paired bootstrap — and a model promoted on noise thrashes.",
            "",
        ] + [f"![{n}](figures/{n})" for n in figs_models]

    report = args.out / "EDA.md"
    report.write_text("\n".join(lines) + "\n")
    print(f"[ok  ] {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
