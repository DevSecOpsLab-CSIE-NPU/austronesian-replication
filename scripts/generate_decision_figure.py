#!/usr/bin/env python3
"""Generate diagnostic decision-rule flowchart figure for JQL paper.

Reads empirical thresholds from results/ JSON files and produces a
publication-quality flowchart (600 dpi PNG) illustrating the four-step
diagnostic framework for assessing when annotation-free lexical distance
is a reliable genealogical proxy.

Usage:
    .venv/bin/python3 scripts/generate_decision_figure.py
"""

import sys
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "paper" / "jql" / "figures"


def save_figure(fig, out: Path):
    """Save figure as both PNG and PDF."""
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white", pad_inches=0.18)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white", pad_inches=0.18)

# ---------------------------------------------------------------------------
# Load empirical values from results/
# ---------------------------------------------------------------------------


def load_results():
    """Load all relevant statistics from JSON result files."""
    with open(RESULTS_DIR / "mantel_test_results.json") as f:
        mantel = json.load(f)

    with open(RESULTS_DIR / "ie_cross_family.json") as f:
        ie = json.load(f)

    with open(RESULTS_DIR / "st_cross_family.json") as f:
        st = json.load(f)

    with open(RESULTS_DIR / "separation_ratio_permtest.json") as f:
        sep = json.load(f)

    with open(RESULTS_DIR / "tree_evaluation_results.json") as f:
        tree_eval = json.load(f)

    with open(RESULTS_DIR / "mrm_results.json") as f:
        mrm = json.load(f)

    # Extract key values
    # Mantel r values for three families
    # AN mantel: find the levenshtein entry
    an_mantel_r = None
    if isinstance(mantel, dict):
        # Try different structures
        for key in ["levenshtein", "lev", "Levenshtein"]:
            if key in mantel:
                entry = mantel[key]
                if isinstance(entry, dict) and "mantel_r" in entry:
                    an_mantel_r = entry["mantel_r"]
                    break
        if an_mantel_r is None:
            # Try top-level
            if "mantel_r" in mantel:
                an_mantel_r = mantel["mantel_r"]
            elif "results" in mantel:
                for entry in (
                    mantel["results"]
                    if isinstance(mantel["results"], list)
                    else [mantel["results"]]
                ):
                    if (
                        isinstance(entry, dict)
                        and "levenshtein" in str(entry.get("metric", "")).lower()
                    ):
                        an_mantel_r = entry.get("mantel_r", entry.get("r_observed"))
                        break
    if an_mantel_r is None:
        an_mantel_r = 0.255  # fallback from paper

    ie_mantel_r = ie.get("mantel_r_lev_vs_gen", 0.256)
    st_mantel_r = st.get("mantel_r_lev_vs_gen", 0.226)

    # Cophenetic r
    an_coph_r = 0.923
    if isinstance(tree_eval, dict):
        for key in ["levenshtein", "lev"]:
            if key in tree_eval:
                entry = tree_eval[key]
                if isinstance(entry, dict):
                    an_coph_r = entry.get(
                        "cophenetic_r", entry.get("upgma_cophenetic_r", an_coph_r)
                    )
                    break
        if "cophenetic_r" in tree_eval:
            an_coph_r = tree_eval["cophenetic_r"]

    # Separation ratios
    phil_ratio = sep.get("Philippine", {}).get("observed_ratio", 1.182)
    form_ratio = sep.get("Formosan", {}).get("observed_ratio", 1.037)
    mp_ratio = sep.get("Malayo-Polynesian", {}).get("observed_ratio", 1.013)

    # MRM betas
    mrm_beta_gen = None
    mrm_beta_geo = None
    if isinstance(mrm, dict):
        if "levenshtein" in mrm:
            entry = mrm["levenshtein"]
        elif "results" in mrm:
            entry = mrm["results"] if isinstance(mrm["results"], dict) else mrm
        else:
            entry = mrm
        if isinstance(entry, dict):
            mrm_beta_gen = entry.get("beta_genealogical", entry.get("beta_gen"))
            mrm_beta_geo = entry.get("beta_geographic", entry.get("beta_geo"))

    return {
        "an_mantel_r": an_mantel_r,
        "ie_mantel_r": ie_mantel_r,
        "st_mantel_r": st_mantel_r,
        "an_coph_r": an_coph_r,
        "phil_ratio": phil_ratio,
        "form_ratio": form_ratio,
        "mp_ratio": mp_ratio,
        "mrm_beta_gen": mrm_beta_gen,
        "mrm_beta_geo": mrm_beta_geo,
    }


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

# Colours
C_GREEN = "#6aa84f"
C_YELLOW = "#d4a72c"
C_RED = "#d62728"
C_BLUE = "#6d9ec1"
C_LIGHTGREEN = "#d9ead3"
C_LIGHTYELLOW = "#fff3cd"
C_LIGHTRED = "#f8d7da"
C_LIGHTBLUE = "#c7dbe6"
C_LIGHTGREY = "#f7f7f7"
C_WHITE = "#ffffff"


def draw_diamond(
    ax,
    cx,
    cy,
    w,
    h,
    text,
    fontsize=9,
    facecolor=C_LIGHTBLUE,
    edgecolor=C_BLUE,
    linewidth=1.5,
):
    """Draw a decision diamond centred at (cx, cy)."""
    verts = [
        (cx, cy + h / 2),  # top
        (cx + w / 2, cy),  # right
        (cx, cy - h / 2),  # bottom
        (cx - w / 2, cy),  # left
        (cx, cy + h / 2),  # close
    ]
    poly = plt.Polygon(
        verts,
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=3,
    )
    ax.add_patch(poly)
    ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        wrap=True,
        zorder=4,
        bbox=dict(boxstyle="square,pad=0", facecolor="none", edgecolor="none"),
    )


def draw_box(
    ax,
    cx,
    cy,
    w,
    h,
    text,
    fontsize=8,
    facecolor=C_LIGHTGREY,
    edgecolor="#666666",
    linewidth=1.2,
    rounded=True,
    fontstyle="normal",
    fontweight="normal",
    text_color="black",
):
    """Draw a rectangular box (rounded or sharp) centred at (cx, cy)."""
    if rounded:
        box = FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle="round,pad=0.015",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=3,
        )
    else:
        box = FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle="square,pad=0.008",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=3,
        )
    ax.add_patch(box)
    ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontstyle=fontstyle,
        fontweight=fontweight,
        color=text_color,
        zorder=4,
    )


def draw_arrow(
    ax,
    x1,
    y1,
    x2,
    y2,
    label=None,
    label_side="right",
    color="black",
    linewidth=1.2,
    fontsize=8,
    label_frac=0.5,
):
    """Draw an arrow from (x1,y1) to (x2,y2) with optional label.

    label_frac: position along arrow for label (0=start, 1=end, 0.5=mid).
    """
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=linewidth,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=2,
    )
    if label:
        lx = x1 + (x2 - x1) * label_frac
        ly = y1 + (y2 - y1) * label_frac
        offset = (8, 0) if label_side == "right" else (-8, 0)
        if abs(x1 - x2) < 0.01:  # vertical arrow
            offset = (12, 0) if label_side == "right" else (-12, 0)
        if label == "NO":
            offset = (-12, 0)
        ax.annotate(
            label,
            (lx, ly),
            fontsize=fontsize,
            fontweight="bold",
            color=color,
            ha="center",
            va="center",
            xytext=offset,
            textcoords="offset points",
        )

# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------


def main():
    vals = load_results()

    fig, ax = plt.subplots(figsize=(11.6, 6.9))
    ax.set_xlim(-5.00, 5.46)
    ax.set_ylim(-0.12, 10.58)
    ax.axis("off")

    # Geometry
    step_w, step_h = 4.0, 1.0
    outcome_w, outcome_h = 2.3, 0.74

    def protocol_box(
        cx,
        cy,
        title,
        body,
        threshold,
        facecolor=C_LIGHTBLUE,
        edgecolor=C_BLUE,
        height=step_h,
    ):
        draw_box(
            ax,
            cx,
            cy,
            step_w,
            height,
            f"{title}\n{body}\n{threshold}",
            fontsize=9.8,
            facecolor=facecolor,
            edgecolor=edgecolor,
            fontweight="bold",
            rounded=True,
        )

    # Input
    draw_box(
        ax, 0, 10.02, 4.45, 0.78,
        "Input data\nWordlist with phonetic forms + subgroup metadata",
        fontsize=8.4, facecolor=C_LIGHTGREY, edgecolor="#8a8a8a",
        fontweight="bold", rounded=True
    )

    # Step 1
    protocol_box(0, 8.75, "STEP 1", "Cross-metric agreement", "Mantel  $r \\geq 0.20$")
    draw_box(
        ax, 3.92, 8.75, outcome_w, outcome_h,
        "Failure\nNo detectable\ngenealogical signal",
        fontsize=8.5, facecolor=C_LIGHTRED, edgecolor=C_RED,
        fontweight="bold", rounded=True
    )
    draw_arrow(ax, 0, 9.72, 0, 9.22, color="#4a4a4a")
    draw_arrow(ax, 2.02, 8.75, 2.72, 8.75, label="NO", label_side="right", color=C_RED, fontsize=7.8, label_frac=0.48)

    # Step 2
    protocol_box(0, 6.95, "STEP 2", "Genealogical depth preservation", "Cophenetic correlation  $r \\geq 0.85$")
    draw_box(
        ax, 3.92, 6.95, outcome_w, outcome_h,
        "Failure\nDepth ordering\nnot preserved",
        fontsize=8.5, facecolor=C_LIGHTRED, edgecolor=C_RED,
        fontweight="bold", rounded=True
    )
    draw_arrow(ax, 0, 8.05, 0, 7.44, label="YES", label_side="left", color=C_GREEN, fontsize=7.8, label_frac=0.35)
    draw_arrow(ax, 2.02, 6.95, 2.72, 6.95, label="NO", label_side="right", color=C_RED, fontsize=7.8, label_frac=0.48)

    # Split to parallel checks
    split_y = 5.28
    ax.plot([0, 0], [6.34, split_y], color="#5a5a5a", lw=1.4, zorder=2)
    ax.plot([-2.75, 2.75], [split_y, split_y], color="#5a5a5a", lw=1.4, zorder=2)
    ax.text(
        0,
        split_y + 0.18,
        "Parallel validation checks (both must be satisfied)",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#5a5a5a",
        fontstyle="italic",
    )

    # Step 3 and Step 4
    step3_x, step4_x = -2.65, 2.65
    protocol_box(
        step3_x,
        4.18,
        "STEP 3",
        "Subgroup discrimination strength",
        "Separation ratio  (cross / within)",
        height=1.26,
    )
    protocol_box(
        step4_x,
        4.18,
        "STEP 4",
        "Geographic confound control\nGenealogical vs geographic signal",
        r"MRM regression  $\beta_{\mathrm{gen}} > \beta_{\mathrm{geo}}$",
        height=1.26,
    )
    ax.plot([step3_x, step3_x], [split_y, 4.79], color="#5a5a5a", lw=1.4, zorder=2)
    ax.plot([step4_x, step4_x], [split_y, 4.79], color="#5a5a5a", lw=1.4, zorder=2)

    # Step 3 outcomes
    draw_box(
        ax, -4.05, 2.80, 1.48, 0.68,
        r"$\leq 1.00$" "\n" r"Unreliable",
        fontsize=8.0, facecolor=C_LIGHTRED, edgecolor=C_RED,
        fontweight="bold", rounded=True
    )
    draw_box(
        ax, -2.38, 2.80, 1.48, 0.68,
        "$1.00$--$1.10$\nMarginal",
        fontsize=8.0, facecolor=C_LIGHTYELLOW, edgecolor=C_YELLOW,
        fontweight="bold", rounded=True
    )
    draw_box(
        ax, -0.71, 2.80, 1.48, 0.68,
        "$> 1.10$\nReliable",
        fontsize=8.0, facecolor=C_LIGHTGREEN, edgecolor=C_GREEN,
        fontweight="bold", rounded=True
    )
    draw_arrow(ax, step3_x - 0.92, 3.72, -4.05, 3.15, color=C_RED, linewidth=1.0)
    draw_arrow(ax, step3_x, 3.72, -2.38, 3.17, color=C_YELLOW, linewidth=1.0)
    draw_arrow(ax, step3_x + 0.92, 3.72, -0.71, 3.15, color=C_GREEN, linewidth=1.0)

    # Step 4 outcomes
    draw_box(
        ax, 1.83, 2.80, 1.68, 0.68,
        "Pass\nGenealogical signal\ndominates",
        fontsize=8.0, facecolor=C_LIGHTGREEN, edgecolor=C_GREEN,
        fontweight="bold", rounded=True
    )
    draw_box(
        ax, 3.78, 2.80, 1.68, 0.68,
        "Caution\nGeographic confound\npossible",
        fontsize=8.0, facecolor=C_LIGHTYELLOW, edgecolor=C_YELLOW,
        fontweight="bold", rounded=True
    )
    draw_arrow(ax, step4_x - 0.46, 3.72, 1.83, 3.15, color=C_GREEN, linewidth=1.0)
    draw_arrow(ax, step4_x + 0.46, 3.72, 3.78, 3.15, color=C_YELLOW, linewidth=1.0)

    # Final decision
    draw_box(
        ax, -0.05, 1.62, 6.95, 1.18,
        "FINAL DIAGNOSTIC OUTCOME\n\nAnnotation-free lexical distance\nis a reliable genealogical proxy.",
        fontsize=8.0, facecolor=C_LIGHTGREEN, edgecolor=C_GREEN, linewidth=2.1,
        fontweight="bold", rounded=True
    )
    draw_arrow(ax, -4.0, 2.46, -1.95, 2.10, color=C_GREEN, linewidth=0.95)
    draw_arrow(ax, 1.7, 2.46, 1.05, 2.10, color=C_GREEN, linewidth=0.95)

    # Empirical dataset box
    empirical_text = (
        "Empirical\nreference values\n\n"
        f"Austronesian\nMantel $r$ = {vals['an_mantel_r']:.3f}\n"
        f"Cophenetic $r$ = {vals['an_coph_r']:.3f}\n\n"
        f"Indo-European\nMantel $r$ = {vals['ie_mantel_r']:.3f}\n\n"
        f"Sino-Tibetan\nMantel $r$ = {vals['st_mantel_r']:.3f}"
    )
    draw_box(
        ax, 4.22, 1.08, 1.50, 2.30,
        empirical_text,
        fontsize=6.0, facecolor="#fdfdfd", edgecolor="#d0d0d0",
        fontweight="normal", rounded=True
    )

    # Save
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "fig_decision_rule.png"
    save_figure(fig, out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"  Size: {out_path.stat().st_size / 1024:.0f} KB")

if __name__ == "__main__":
    main()
