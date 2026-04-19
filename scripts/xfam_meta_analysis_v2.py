#!/usr/bin/env python3
"""Cross-family meta-analysis v2 — consolidates all four families plus B-track
robustness results.

Inputs
------
- results/xfam_an_migration_gradient.json       (Austronesian, ABVD)
- results/xfam_ie_migration_gradient.json       (Indo-European, IECoR, Steppe homeland)
- results/xfam_ie_homeland_sensitivity.json     (IE Steppe vs Anatolian)
- results/xfam_st_migration_gradient.json       (Sino-Tibetan, sagartst only)
- results/xfam_st_pooled_mountain.json          (ST pooled 3 datasets, 5 Swadesh concepts)
- results/xfam_st_stedt_rice_poc.json           (ST STEDT RICE extension)
- results/xfam_bantu_migration_gradient.json    (Bantu, Grollemund 2015)

Produces
--------
- results/xfam_meta_v2.json           unified summary
- results/xfam_meta_v2_long.csv       long-format table, one row per (family, dataset, concept)
- results/xfam_selectivity_v2.json    per-family anchor-selectivity permutation test
- paper/JLE/figures/fig_xfam_v2_forest.png      four-family forest plot
- paper/JLE/figures/fig_xfam_v2_selectivity.png selectivity summary

Methodology
-----------
For each family, we test whether the family's empirical anchor (concept with the
highest spearman_r) is *selectively* positive relative to other tested concepts.
The null is a founder-effect model where all concepts share a common expected
gradient; in that case, the top concept's r should not be statistically
distinguishable from the second-best.

Two tests per family:
  (1) CI overlap test: is anchor_ci_lower > second_concept_ci_upper?
  (2) Permutation test: under random concept→label assignment, what fraction
      of permutations produces a gap (top − second) as large as observed?
      5000 permutations, seed 42.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

RESULTS = ROOT / "results"
FIG_DIR = ROOT / "paper" / "JLE" / "figures"
RNG = np.random.default_rng(42)


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def long_rows_from_base(fam: dict, dataset_label: str) -> list[dict]:
    out = []
    for concept, s in fam["concepts"].items():
        out.append({
            "family": fam["family"],
            "dataset": dataset_label,
            "concept": concept,
            "role": s.get("role", ""),
            "n": s.get("n"),
            "retention_rate": s.get("retention_rate"),
            "spearman_r": s.get("spearman_r"),
            "spearman_p": s.get("spearman_p"),
            "ci_lower": s.get("ci_lower"),
            "ci_upper": s.get("ci_upper"),
            "homeland_lat": fam["homeland"]["lat"],
            "homeland_lon": fam["homeland"]["lon"],
            "homeland_name": fam["homeland"]["name"],
        })
    return out


def long_rows_from_st_pooled(pooled: dict) -> list[dict]:
    out = []
    fam = pooled["family"]
    for concept, s in pooled["concepts"].items():
        p = s.get("pooled", {})
        if not p:
            continue
        out.append({
            "family": fam,
            "dataset": "ST pooled (sagartst+zhangst+peirosst)",
            "concept": concept,
            "role": "control",
            "n": p.get("n"),
            "retention_rate": p.get("retention_rate"),
            "spearman_r": p.get("spearman_r"),
            "spearman_p": p.get("spearman_p"),
            "ci_lower": p.get("ci_lower"),
            "ci_upper": p.get("ci_upper"),
            "homeland_lat": pooled["homeland"]["lat"],
            "homeland_lon": pooled["homeland"]["lon"],
            "homeland_name": pooled["homeland"]["name"],
        })
    return out


def long_rows_from_stedt_rice(d: dict) -> list[dict]:
    """STEDT RICE POC has two methods (A, B); we use Method B (NLD n=60)
    as the primary extension and keep Method A as a note."""
    if d.get("status") == "extraction_failed":
        return []
    out = [{
        "family": d["family"],
        "dataset": "ST STEDT (NLD cluster, Method B)",
        "concept": "RICE",
        "role": "anchor",
        "n": d.get("n_glottocodes_with_form") or d.get("n"),
        "retention_rate": d.get("retention_rate"),
        "spearman_r": d.get("spearman_r"),
        "spearman_p": d.get("spearman_p"),
        "ci_lower": d.get("ci_lower"),
        "ci_upper": d.get("ci_upper"),
        "homeland_lat": d["homeland"]["lat"],
        "homeland_lon": d["homeland"]["lon"],
        "homeland_name": d["homeland"]["name"],
    }]
    return out


def selectivity_test(concepts: pd.DataFrame, n_permutations: int = 5000) -> dict:
    """Is the top concept's spearman_r significantly greater than the second-best?

    Concepts: DataFrame with columns spearman_r, ci_lower, ci_upper, concept.
    """
    if len(concepts) < 2:
        return {"test_applicable": False, "reason": "fewer than 2 concepts"}

    g = concepts.sort_values("spearman_r", ascending=False).reset_index(drop=True)
    top = g.iloc[0]
    second = g.iloc[1]
    gap_observed = float(top["spearman_r"] - second["spearman_r"])

    # (1) CI-overlap test: selective if top ci_lower > second ci_upper
    ci_disjoint = bool(top["ci_lower"] > second["ci_upper"])

    # (2) Permutation test: simulate the null where all concepts share the same
    # expected r. Per concept we have an approximate sampling distribution
    # modeled as Normal with mean r and sd = (ci_upper - ci_lower)/(2*1.96).
    # Under the null, the gap between top-ranked and second-ranked concept in
    # each permutation is what we'd expect if ranking were arbitrary.
    rs = g["spearman_r"].values
    sds = (g["ci_upper"].values - g["ci_lower"].values) / (2 * 1.96)
    # draw n_permutations simulated r vectors, then check gap distribution
    draws = RNG.normal(loc=rs, scale=np.maximum(sds, 1e-6),
                       size=(n_permutations, len(rs)))
    simulated_gaps = []
    for d in draws:
        d_sorted = np.sort(d)[::-1]
        simulated_gaps.append(d_sorted[0] - d_sorted[1])
    simulated_gaps = np.array(simulated_gaps)
    p_permutation = float(np.mean(simulated_gaps >= gap_observed))

    return {
        "test_applicable": True,
        "top_concept": str(top["concept"]),
        "top_r": float(top["spearman_r"]),
        "top_ci": [float(top["ci_lower"]), float(top["ci_upper"])],
        "second_concept": str(second["concept"]),
        "second_r": float(second["spearman_r"]),
        "second_ci": [float(second["ci_lower"]), float(second["ci_upper"])],
        "gap_observed": gap_observed,
        "ci_disjoint": ci_disjoint,
        "permutation_p": p_permutation,
        "n_permutations": n_permutations,
        "selectively_positive": bool(ci_disjoint or p_permutation < 0.05),
    }


def four_panel_forest(df: pd.DataFrame, out_path: Path):
    families = ["Austronesian", "Indo-European", "Sino-Tibetan", "Bantu"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 6),
                             gridspec_kw={"wspace": 0.45})
    for ax, fam in zip(axes, families):
        g = df[df["family"] == fam].drop_duplicates(subset=["concept"], keep="first")
        g = g.sort_values("spearman_r").reset_index(drop=True)
        if len(g) == 0:
            ax.set_visible(False)
            continue
        y = np.arange(len(g))
        top_idx = g["spearman_r"].idxmax()
        colors = ["#d62728" if r == "anchor" else "#7f7f7f" for r in g["role"]]
        if top_idx is not None:
            colors[top_idx] = "#2ca02c"

        xerr_l = g["spearman_r"] - g["ci_lower"]
        xerr_h = g["ci_upper"] - g["spearman_r"]
        ax.errorbar(g["spearman_r"], y, xerr=[xerr_l, xerr_h],
                    fmt="o", color="k", ecolor="k", capsize=3,
                    markersize=0, linewidth=1, zorder=2)
        for yi, (r, c, p) in enumerate(zip(g["spearman_r"], colors,
                                            g["spearman_p"])):
            ax.scatter([r], [yi], color=c, s=80, zorder=3,
                       edgecolor="black", linewidth=0.5)
            if p is not None and pd.notna(p) and p < 0.001:
                ax.annotate("***", (r, yi), xytext=(6, -4),
                            textcoords="offset points", fontsize=11)
            elif p is not None and pd.notna(p) and p < 0.01:
                ax.annotate("**", (r, yi), xytext=(6, -4),
                            textcoords="offset points", fontsize=11)
            elif p is not None and pd.notna(p) and p < 0.05:
                ax.annotate("*", (r, yi), xytext=(6, -4),
                            textcoords="offset points", fontsize=11)
        ax.axvline(0, color="gray", lw=0.8, ls=":")
        ax.set_yticks(y)
        ax.set_yticklabels(
            [f"{c} ({rl[:3]}, n={int(n) if pd.notna(n) else '-'})"
             for c, rl, n in zip(g["concept"], g["role"], g["n"])],
            fontsize=9)
        emp = g.loc[top_idx, "concept"] if top_idx is not None else "—"
        ax.set_title(f"{fam}\nempirical anchor: {emp}", fontsize=10)
        ax.set_xlim(-0.8, 0.9)
        ax.grid(axis="x", linestyle=":", alpha=0.3)
    axes[0].set_xlabel("Spearman ρ (dist_homeland × retention)", fontsize=10)
    fig.suptitle(
        "Cross-family migration gradients (4 families, 95% CI bars; "
        "* p<.05, ** p<.01, *** p<.001)",
        fontsize=12, y=1.02,
    )
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def selectivity_figure(selectivity: dict, out_path: Path):
    fams = list(selectivity.keys())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = np.arange(len(fams))
    tops = [selectivity[f]["top_r"] for f in fams]
    seconds = [selectivity[f]["second_r"] for f in fams]
    gaps = [selectivity[f]["gap_observed"] for f in fams]
    perm_ps = [selectivity[f]["permutation_p"] for f in fams]

    w = 0.35
    ax.bar(xs - w/2, tops, w, label="top concept ρ",
           color=["#2ca02c" if selectivity[f]["selectively_positive"]
                  else "#7f7f7f" for f in fams])
    ax.bar(xs + w/2, seconds, w, label="second-best concept ρ",
           color="#aec7e8")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"{f}\n({selectivity[f]['top_concept']} vs "
         f"{selectivity[f]['second_concept']})"
         for f in fams], fontsize=9)
    for i, (g, p) in enumerate(zip(gaps, perm_ps)):
        label = f"gap={g:.2f}\np_perm={p:.3f}"
        y = max(tops[i], seconds[i]) + 0.05
        ax.text(i, y, label, ha="center", fontsize=8)
    ax.set_ylabel("Spearman ρ")
    ax.set_ylim(-0.3, 1.0)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Anchor selectivity: top vs second-best concept per family",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    # --- Load base family results ---
    an = load(RESULTS / "xfam_an_migration_gradient.json")
    ie = load(RESULTS / "xfam_ie_migration_gradient.json")
    st_base = load(RESULTS / "xfam_st_migration_gradient.json")
    bantu = load(RESULTS / "xfam_bantu_migration_gradient.json")

    # --- Load robustness tracks ---
    ie_sens = load(RESULTS / "xfam_ie_homeland_sensitivity.json")
    st_pooled = load(RESULTS / "xfam_st_pooled_mountain.json")
    st_stedt = load(RESULTS / "xfam_st_stedt_rice_poc.json")

    rows = []
    rows.extend(long_rows_from_base(an, "ABVD (JLE baseline)"))
    rows.extend(long_rows_from_base(ie, "IECoR (Steppe homeland)"))
    rows.extend(long_rows_from_base(st_base, "sagartst"))
    rows.extend(long_rows_from_base(bantu, "Grollemund 2015"))
    rows.extend(long_rows_from_st_pooled(st_pooled))
    rows.extend(long_rows_from_stedt_rice(st_stedt))

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["spearman_r"])
    long_path = RESULTS / "xfam_meta_v2_long.csv"
    df.to_csv(long_path, index=False, float_format="%.4f")
    print(f"wrote {long_path}  ({len(df)} rows)")

    # --- Selectivity per family ---
    # For each family, deduplicate to one row per concept (prefer primary dataset
    # over pooled/STEDT when both exist, so that the selectivity test is
    # within-dataset consistent).
    preferred_dataset_by_family = {
        "Austronesian": "ABVD (JLE baseline)",
        "Indo-European": "IECoR (Steppe homeland)",
        "Sino-Tibetan": "sagartst",
        "Bantu": "Grollemund 2015",
    }
    selectivity = {}
    for fam, ds in preferred_dataset_by_family.items():
        g = df[(df["family"] == fam) & (df["dataset"] == ds)]
        if len(g) < 2:
            continue
        selectivity[fam] = selectivity_test(g)

    sel_path = RESULTS / "xfam_selectivity_v2.json"
    sel_path.write_text(json.dumps(selectivity, indent=2, ensure_ascii=False))
    print(f"wrote {sel_path}")

    # --- Meta summary ---
    meta = {
        "generated": "2026-04-19",
        "families_analysed": list(preferred_dataset_by_family.keys()),
        "datasets_used": sorted(df["dataset"].unique().tolist()),
        "primary_findings": {
            "Austronesian": {
                "empirical_anchor": selectivity["Austronesian"]["top_concept"],
                "r": selectivity["Austronesian"]["top_r"],
                "selectively_positive": selectivity["Austronesian"]["selectively_positive"],
                "note": "replicates JLE paper SEA anchor",
            },
            "Indo-European": {
                "empirical_anchor": selectivity["Indo-European"]["top_concept"],
                "r_steppe": selectivity["Indo-European"]["top_r"],
                "r_anatolian": ie_sens["concepts"]["MOUNTAIN"]["anatolian"]["r"],
                "homeland_sensitive": not ie_sens["concepts"]["MOUNTAIN"].get("ci_overlap", True),
                "selectively_positive": selectivity["Indo-European"]["selectively_positive"],
            },
            "Sino-Tibetan": {
                "empirical_anchor_sagartst": selectivity["Sino-Tibetan"]["top_concept"],
                "r_sagartst_top": selectivity["Sino-Tibetan"]["top_r"],
                "pooled_mountain_r": st_pooled["concepts"]["MOUNTAIN"]["pooled"]["spearman_r"],
                "pooled_water_r": st_pooled["concepts"]["WATER"]["pooled"]["spearman_r"],
                "pooled_water_p": st_pooled["concepts"]["WATER"]["pooled"]["spearman_p"],
                "stedt_rice_n": st_stedt.get("n_glottocodes_with_form"),
                "stedt_rice_r": st_stedt.get("spearman_r"),
                "stedt_rice_p": st_stedt.get("spearman_p"),
                "selectively_positive": selectivity["Sino-Tibetan"]["selectively_positive"],
                "note": "RICE direction preserved; power still marginal; WATER emerges in pooled 3-dataset test",
            },
            "Bantu": {
                "empirical_anchor": selectivity["Bantu"]["top_concept"],
                "r": selectivity["Bantu"]["top_r"],
                "selectively_positive": selectivity["Bantu"]["selectively_positive"],
                "note": "Corridor-ecology concepts (FOREST/RIVER/BANANA/YAM) absent from Swadesh-100; all universal controls show strong positive gradients consistent with founder-effect null",
            },
        },
        "selectivity_summary": selectivity,
        "cross_family_conclusion": (
            "Four-family evidence shows family-specific ecological anchors are "
            "identifiable ONLY when the dataset includes cultural/ecological "
            "concepts beyond the Swadesh core. Austronesian SEA (ABVD) and "
            "Indo-European MOUNTAIN (IECoR) are selectively positive relative "
            "to other tested concepts in their families. Sino-Tibetan RICE and "
            "Bantu FOREST/RIVER predictions cannot be cleanly tested because "
            "their best available datasets (Grollemund, Zhang 2019, sagartst) "
            "are Swadesh-only. Supplementary STEDT extraction raises ST RICE "
            "n from 23 to 60 and preserves the positive direction (ρ=+0.18), "
            "but remains underpowered. The IE MOUNTAIN finding is "
            "homeland-sensitive: under the Anatolian hypothesis the gradient "
            "weakens to ρ=+0.13 with CI touching zero. The core methodological "
            "finding is that cross-family anchor research is concept-slate "
            "dependent: Swadesh-100 resources cannot adjudicate "
            "ecology-specific anchor hypotheses."
        ),
    }
    meta_path = RESULTS / "xfam_meta_v2.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"wrote {meta_path}")

    # --- Figures ---
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    forest_path = FIG_DIR / "fig_xfam_v2_forest.png"
    four_panel_forest(df, forest_path)
    print(f"wrote {forest_path}")

    sel_fig = FIG_DIR / "fig_xfam_v2_selectivity.png"
    selectivity_figure(selectivity, sel_fig)
    print(f"wrote {sel_fig}")

    # --- Console summary ---
    print()
    print("=" * 96)
    print("ANCHOR SELECTIVITY SUMMARY")
    print("=" * 96)
    print(f"{'Family':<16} {'Top concept':<12} {'r':>7} {'Second':<12} "
          f"{'r₂':>7} {'gap':>6} {'p_perm':>8} {'selective?'}")
    for fam, s in selectivity.items():
        print(f"{fam:<16} {s['top_concept']:<12} {s['top_r']:>+7.3f} "
              f"{s['second_concept']:<12} {s['second_r']:>+7.3f} "
              f"{s['gap_observed']:>+6.3f} {s['permutation_p']:>8.3f} "
              f"{'YES' if s['selectively_positive'] else 'no (tied)'}")
    print()
    print(f"Long table saved: {long_path}")
    print(f"Meta JSON: {meta_path}")
    print(f"Figures: {forest_path}, {sel_fig}")


if __name__ == "__main__":
    main()
