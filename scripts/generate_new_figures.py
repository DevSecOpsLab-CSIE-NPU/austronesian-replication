#!/usr/bin/env python3
"""
Generate 3 new publication-quality figures for the JQL paper (Figures 10-12).
Reads data from existing JSON result files.
"""

import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "paper" / "jql" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── JQL Publication Style (Match existing) ─────────────────────────────────────
COL_W = 4.80  # single-column width (inches)
COL_2 = 6.90  # double-column / full-page width (inches)

JQL_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.title_fontsize": 8,
    "legend.frameon": False,
    "legend.borderpad": 0.3,
    "lines.linewidth": 1.0,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "patch.linewidth": 0.6,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.35,
    "figure.constrained_layout.use": True,
}
plt.rcParams.update(JQL_RC)


def save_figure(out: Path):
    """Save current matplotlib figure as both PNG and PDF."""
    plt.savefig(out)
    plt.savefig(out.with_suffix(".pdf"))

# ── Color Palettes (Match existing) ────────────────────────────────────────────
MET_COLORS = {
    "levenshtein": "#332288",  # indigo
    "sound_class": "#882255",  # wine
    "weighted": "#44AA99",  # teal
    "geographic": "#E69F00",  # amber
    "random": "#888888",  # grey
    "dcor": "#CC6677",  # rose
    "mantel": "#44AA99",  # teal
}


# ── Figure 10: Scale Sensitivity ───────────────────────────────────────────────
def fig10_scale_sensitivity():
    print("  [1/3] Figure 10: Scale sensitivity...")
    with open(RESULTS / "dataset_scale_sensitivity.json") as f:
        data = json.load(f)

    scales = data["scales"]
    # Ensure scales are sorted integers
    scales = sorted([int(s) for s in scales])

    mantel_r = [data["results"][str(s)]["mantel_r"] for s in scales]
    cophenetic_r = [data["results"][str(s)]["cophenetic_r"] for s in scales]

    fig, ax1 = plt.subplots(figsize=(COL_W, COL_W * 0.75))

    # Plot Mantel r on left axis
    color1 = MET_COLORS["weighted"]
    l1 = ax1.plot(
        scales, mantel_r, marker="o", color=color1, label="Mantel $r$", linewidth=1.5
    )
    ax1.set_xlabel("Dataset Size ($n$ languages)")
    ax1.set_ylabel("Mantel $r$", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0.15, 0.35)
    ax1.set_xticks(scales)

    # Plot Cophenetic r on right axis
    ax2 = ax1.twinx()
    color2 = MET_COLORS["levenshtein"]
    l2 = ax2.plot(
        scales,
        cophenetic_r,
        marker="s",
        color=color2,
        label="Cophenetic $r$",
        linewidth=1.5,
        linestyle="--",
    )
    ax2.set_ylabel("Cophenetic $r$", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0.85, 0.96)

    # Clean up right spine for ax2 (it's needed for ticks but we can style it)
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax1.spines["top"].set_visible(False)

    # Combined legend
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right")

    plt.title(
        "Stability of Genealogical Signal Across Dataset Scales\n(Mantel correlation vs Cophenetic correlation)",
        fontsize=9,
        fontweight="bold",
    )

    out = FIGURES / "fig10_scale_sensitivity.png"
    save_figure(out)
    plt.close()
    print(f"     Saved: {out}")


# ── Figure 11: Geographic Baseline ─────────────────────────────────────────────
def fig11_geographic_baseline():
    print("  [2/3] Figure 11: Geographic baseline...")
    with open(RESULTS / "geographic_tree_baseline.json") as f:
        data = json.load(f)

    top_methods = ["Lexical", "Geographic", "Random null\n(label permutation)"]
    top_nrf = [
        data["lexical_levenshtein_nj"]["nRF_vs_glottolog"],
        data["geographic_nj"]["nRF_vs_glottolog"],
        data["random_null"]["nRF_vs_glottolog"],
    ]
    top_vals = [1.0 - v for v in top_nrf]

    depth_methods = ["Lexical", "Geographic"]
    depth_vals = [
        data["lexical_levenshtein_nj"]["cophenetic_r"],
        data["geographic_nj"]["cophenetic_r"],
    ]

    delta_nrf = data["geographic_nj"]["nRF_vs_glottolog"] - data["lexical_levenshtein_nj"]["nRF_vs_glottolog"]
    delta_rf_similarity = top_vals[0] - top_vals[1]

    fig, axes = plt.subplots(
        1, 2, figsize=(COL_W * 1.05, COL_W * 0.58),
        gridspec_kw={"width_ratios": [1.18, 0.82]}
    )

    # Panel A: RF similarity
    ax = axes[0]
    x = np.arange(len(top_methods))
    colors = ["#3366AA", "#228833", "#999999"]
    bars = ax.bar(x, top_vals, width=0.58, color=colors, alpha=0.90, edgecolor="white")
    for bar, val in zip(bars, top_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.012,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=7.4, fontweight="bold"
        )
    ax.set_xticks(x)
    ax.set_xticklabels(top_methods)
    ax.set_ylim(0, 0.46)
    ax.set_ylabel("RF similarity (1 − nRF)")
    ax.set_title("A. Topological Agreement", fontsize=8.2, fontweight="bold")
    ax.axhline(0.0, color="#888888", linewidth=1.0, linestyle="--", alpha=0.8, zorder=0)
    ax.text(
        0.96, 0.94,
        f"Lexical − Geographic\nΔRF similarity = {delta_rf_similarity:.3f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=7.1,
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#cccccc", alpha=0.88),
    )

    # Panel B: cophenetic correlation
    ax = axes[1]
    x = np.arange(len(depth_methods))
    colors = ["#3366AA", "#228833"]
    bars = ax.bar(x, depth_vals, width=0.58, color=colors, alpha=0.90, edgecolor="white")
    for bar, val in zip(bars, depth_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.008,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=7.4, fontweight="bold"
        )
    ax.set_xticks(x)
    ax.set_xticklabels(depth_methods)
    ax.set_ylim(0.85, 1.00)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
    ax.axhline(0.90, color="#888888", linewidth=1.0, linestyle="--", alpha=0.8, zorder=0)
    ax.set_ylabel("Cophenetic $r$")
    ax.set_title("B. Depth Preservation", fontsize=8.4, fontweight="bold")

    fig.suptitle(
        "Phylogenetic Signal Comparison: Lexical vs Geographic Distance",
        fontsize=8.8, fontweight="bold"
    )

    out = FIGURES / "fig11_geographic_baseline.png"
    save_figure(out)
    plt.close()
    print(f"     Saved: {out}")


# ── Figure 12: dCor vs Mantel ──────────────────────────────────────────────────
def fig12_dcor_vs_mantel():
    print("  [3/3] Figure 12: dCor vs Mantel...")
    with open(RESULTS / "dcor_results.json") as f:
        data = json.load(f)

    res = data["results"]

    # Define pairs to plot
    pairs = [
        ("levenshtein_vs_genealogical", "Lev\nvs Gen"),
        ("sound_class_vs_genealogical", "SC\nvs Gen"),
        ("weighted_vs_genealogical", "Wt\nvs Gen"),
        ("levenshtein_vs_geographic", "Lev\nvs Geo"),
        ("sound_class_vs_geographic", "SC\nvs Geo"),
        ("weighted_vs_geographic", "Wt\nvs Geo"),
    ]

    labels = [p[1] for p in pairs]
    dcor_vals = [res[p[0]]["dcor"] for p in pairs]
    mantel_vals = [res[p[0]]["mantel_r_for_comparison"] for p in pairs]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(COL_2, COL_2 * 0.5))

    rects1 = ax.bar(
        x - width / 2,
        dcor_vals,
        width,
        label="Distance Correlation (dCor)",
        color=MET_COLORS["dcor"],
        alpha=0.85,
    )
    rects2 = ax.bar(
        x + width / 2,
        mantel_vals,
        width,
        label="Mantel $r$ (Pearson)",
        color=MET_COLORS["mantel"],
        alpha=0.85,
    )

    ax.set_ylabel("Correlation Coefficient")
    ax.set_title("Non-linear (dCor) vs. Linear (Mantel) Dependence", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 0.4)

    # Add value labels
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    # Add divider line between Genealogical and Geographic comparisons
    ax.axvline(2.5, color="gray", linestyle="--", alpha=0.5)
    ax.text(
        1.0,
        0.37,
        "Genealogical",
        ha="center",
        fontsize=9,
        fontweight="bold",
        color="#555",
    )
    ax.text(
        4.0,
        0.37,
        "Geographic",
        ha="center",
        fontsize=9,
        fontweight="bold",
        color="#555",
    )

    out = FIGURES / "fig12_dcor_vs_mantel.png"
    save_figure(out)
    plt.close()
    print(f"     Saved: {out}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fig10_scale_sensitivity()
    fig11_geographic_baseline()
    fig12_dcor_vs_mantel()
    print("Done.")
