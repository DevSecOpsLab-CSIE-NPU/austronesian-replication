#!/usr/bin/env python3
"""Draw STEDT transformation-layer diagram for DSH Figure 3."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

FIG = Path(__file__).parent.parent / "paper" / "DSH" / "figures" / "fig_stedt_pipeline.png"

fig, ax = plt.subplots(figsize=(11.2, 6.7))
ax.set_xlim(0, 11.5)
ax.set_ylim(0, 12)
ax.axis("off")

def box(x, y, w, h, text, color="#e8f0ff", edge="#4a6fa5", fs=8.6, weight="normal"):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
                                 boxstyle="round,pad=0.08,rounding_size=0.1",
                                 linewidth=1.4, edgecolor=edge, facecolor=color))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            wrap=True, weight=weight)

def arrow(x1, y1, x2, y2, label=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.3, color="#333"))
    if label:
        ax.text((x1+x2)/2 + 0.15, (y1+y2)/2, label,
                fontsize=7.5, color="#555", style="italic")

# Layer bands: the figure is about transformation, not only data flow.
def layer(y0, y1, title, subtitle, color):
    ax.axhspan(y0, y1, color=color, alpha=0.35, zorder=0)
    ax.text(0.35, (y0+y1)/2, title, ha="left", va="center",
            fontsize=9.5, weight="bold", color="#333")

layer(9.35, 11.25, "Interface layer:\nmachine-readable access",
      "", "#fff0e0")
layer(6.15, 9.15, "Interoperability layer:\nidentifier standardisation",
      "", "#e8f0ff")
layer(2.35, 5.95, "Analytical layer:\ncross-family comparability",
      "", "#f2e8f5")

# Nodes (top to bottom)
box(4.15, 10.25, 3.0, 0.75, "Single-query UI\n(STEDT web interface)",
    color="#fff8ee", edge="#c07040")
box(8.25, 10.25, 3.15, 0.75, "AJAX endpoint\n/search/ajax?tbl=lexicon&s=<gloss>",
    color="#fff8ee", edge="#c07040")

box(6.25, 8.25, 4.9, 0.9, "JSON response\n16 fields per record\nreflex, gloss, language, analysis, ...",
    color="#eef5ff", edge="#4a6fa5")
box(6.25, 6.85, 7.05, 1.05,
    "$\\mathbf{428}$-entry alias mapping $\\rightarrow$ Glottocode resolution\n"
    "resolves inconsistent STEDT language names into\n"
    "CLDF/Glottolog identifiers",
    color="#eef8e8", edge="#4f8a3a", fs=8.9, weight="bold")

box(3.75, 4.75, 3.35, 0.95,
    "Method A\nexpert etyma tags\n(preferred)",
    color="#f7ecfb", edge="#7a4a8a")
box(8.75, 4.75, 3.35, 0.95,
    "Method B\nNLD clustering fallback\n(reproducibility check)",
    color="#f7ecfb", edge="#7a4a8a")
box(6.25, 3.25, 5.8, 0.95,
    "Per-concept interoperable table\nGlottocode, form, cognate class,\n"
    "homeland distance, retention",
    color="#fff9d8", edge="#b09030")
box(6.25, 1.45, 6.8, 1.15,
    "Reusable cross-family workflow output\n"
    "Spearman $\\rho$ + bootstrap CI → CI-disjoint classification\n"
    "anchor / founder-effect / borrowing / ceiling / null",
    color="#ffe8e8", edge="#b04040", fs=8.4, weight="bold")

# Arrows
arrow(5.65, 10.25, 6.65, 10.25, "")
ax.text(6.15, 10.48, "HTTP GET; rate 0.5 s",
        fontsize=7.5, color="#555", style="italic", ha="center")
arrow(8.25, 9.88, 6.55, 8.70, "JSON parse")
arrow(6.25, 7.80, 6.25, 7.38, "normalise names")
arrow(6.25, 6.32, 4.00, 5.28, "expert-tag")
arrow(6.25, 6.32, 8.50, 5.28, "fallback")
arrow(3.75, 4.28, 5.80, 3.62, "")
arrow(8.75, 4.28, 6.70, 3.62, "")
arrow(6.25, 2.78, 6.25, 2.05, "same rule set as other families")

# Title + side note
ax.text(6.1, 11.65, "STEDT as reusable Digital-Humanities infrastructure",
        ha="center", fontsize=13, weight="bold")
ax.text(0.15, 0.35,
        "Outputs per query: 41--107 unique Glottocodes  |  101--513 forms  |  "
        "9 cultural concepts  |  script: scripts/xfam_st_stedt_full_extraction.py",
        fontsize=7.2, color="#555")
ax.text(0.15, 0.10,
        "Method B keeps the route usable when expert etyma tags are absent; all routes feed the same cross-family classifier.",
        fontsize=7.2, color="#555")

FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG, dpi=220, bbox_inches="tight")
print(f"wrote {FIG}")
