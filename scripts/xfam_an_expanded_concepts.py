#!/usr/bin/env python3
"""Austronesian maritime-cultural-package concept gradient extension.

Paper 1 (JLE JOLE-2026-016) established that SEA is a uniquely anchored
migration-gradient signal within the Austronesian family (ρ = +0.422,
n = 612, p < 1e-27, 95% CI disjoint from all four body-relative controls:
LEFT, RIGHT, ABOVE, BELOW).

GitHub issue #79 asks whether the anchoring pattern generalises within
Austronesian to the maritime cultural package, i.e. concepts widely
reconstructed for Proto-Malayo-Polynesian material culture and ecology:
CANOE, COCONUT, STAR, FISH, CLOUD, TARO, BANANA, PIG, DOG, CHICKEN.

This script re-runs the identical pipeline used by
``scripts/xfam_an_migration_gradient.py`` for every concept above that
exists in ABVD's 210-concept Swadesh-based parameter list, plus SEA as
a reproduction baseline:

  1. Deduplicate by Glottocode (majority-vote cognate class).
  2. Proto-form retention R_j ∈ {0, 1}: 1 iff j's cognate class equals
     the global mode for that concept.
  3. Haversine km from Taiwan (23.5°N, 121.0°E).
  4. Spearman ρ(dist_km, retention), two-tailed p.
  5. 500-bootstrap percentile 95% CI, seed 42.

Concepts that are not in ABVD's parameter list are recorded under
``concepts_not_found`` in the JSON output rather than silently dropped.

Outputs:
  results/xfam_an_expanded_concepts.json
  paper/JLE/figures/fig_an_expanded_forest.png
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RAW_CLDF = ROOT / "data" / "raw" / "cldf"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
FIG_DIR = ROOT / "paper" / "JLE" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

HOMELAND_NAME = "Taiwan"
HOMELAND_LAT = 23.5
HOMELAND_LON = 121.0

N_BOOTSTRAP = 500
SEED = 42

# Concepts requested by issue #79. SEA is retained as the anchor
# reproduction baseline against Paper 1.
TARGET_CONCEPTS = [
    "SEA",
    "CANOE",
    "COCONUT",
    "STAR",
    "FISH",
    "CLOUD",
    "TARO",
    "BANANA",
    "PIG",
    "DOG",
    "CHICKEN",
]

# Role tagging for forest-plot colouring and interpretation.
#   "anchor_baseline"  → SEA (replicates Paper 1 result)
#   "anchor_hypothesis" → the ten maritime-package candidates
CONCEPT_ROLES = {
    "SEA": "anchor_baseline",
    "CANOE": "anchor_hypothesis",
    "COCONUT": "anchor_hypothesis",
    "STAR": "anchor_hypothesis",
    "FISH": "anchor_hypothesis",
    "CLOUD": "anchor_hypothesis",
    "TARO": "anchor_hypothesis",
    "BANANA": "anchor_hypothesis",
    "PIG": "anchor_hypothesis",
    "DOG": "anchor_hypothesis",
    "CHICKEN": "anchor_hypothesis",
}

# Aliases to try when a concept's primary name is absent — we search
# both ABVD's ``name`` field and the ``concepticon_gloss`` field.
CONCEPT_ALIASES = {
    "CANOE": ["CANOE", "canoe", "BOAT"],
    "COCONUT": ["COCONUT", "coconut", "COCONUT PALM", "COCONUT TREE",
                "COCONUT MEAT", "coconut meat", "coconut_meat"],
    "STAR": ["STAR", "star"],
    "FISH": ["FISH", "fish"],
    "CLOUD": ["CLOUD", "cloud"],
    "TARO": ["TARO", "taro"],
    "BANANA": ["BANANA", "banana", "PLANTAIN"],
    "PIG": ["PIG", "pig", "BOAR", "swine"],
    "DOG": ["DOG", "dog"],
    "CHICKEN": ["CHICKEN", "chicken", "FOWL", "HEN"],
    "SEA": ["SEA", "sea", "OCEAN"],
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r_earth = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    )
    return r_earth * 2 * atan2(sqrt(a), sqrt(1 - a))


def find_meaning_id(
    concept: str, meanings: pd.DataFrame
) -> tuple[str | None, str | None, list[str]]:
    """Look up an ABVD meaning_id for a concept, trying aliases.

    Returns (meaning_id, matched_name, aliases_tried).
    """
    aliases = CONCEPT_ALIASES.get(concept, [concept, concept.lower()])
    tried: list[str] = []
    for alias in aliases:
        tried.append(alias)
        # Exact match on concepticon_gloss (uppercase).
        hit = meanings[meanings["concepticon_gloss"] == alias.upper()]
        if not hit.empty:
            return str(hit.iloc[0]["id"]), alias, tried
        # Exact match on lowercase name.
        hit = meanings[meanings["name"].str.lower() == alias.lower()]
        if not hit.empty:
            return str(hit.iloc[0]["id"]), alias, tried
    return None, None, tried


def explode_cognate_classes(sub: pd.DataFrame) -> pd.DataFrame:
    """Explode comma-separated cognate_class strings, one class per row.

    This matches the pre-processing used by ``xfam_an_migration_gradient.py``.
    """
    sub = sub.copy()
    sub["cognate_class"] = sub["cognate_class"].astype(str)
    sub = sub.assign(
        cognate_class=sub["cognate_class"].str.split(r",\s*")
    ).explode("cognate_class")
    sub["cognate_class"] = sub["cognate_class"].str.strip()
    sub = sub[
        sub["cognate_class"].notna()
        & (sub["cognate_class"] != "")
        & (sub["cognate_class"].str.lower() != "nan")
    ]
    return sub


def build_concept_table(
    wl: pd.DataFrame,
    lang: pd.DataFrame,
    meaning_id: str,
) -> tuple[pd.DataFrame, str]:
    """Build the per-Glottocode retention table for one concept.

    Methodology identical to ``xfam_an_migration_gradient.py``: loans are
    NOT filtered out here — the ABVD cognate-class encoding plus the
    per-Glottocode majority vote absorb borrowed-form noise into the same
    mode-vs-not-mode retention comparison used in Paper 1. Reproducing
    Paper 1's SEA result (ρ≈+0.422, n=612) requires this pipeline.
    Returns (table_with_retention, proto_form_cognate_id).
    """
    sub = wl[wl["meaning_id"] == meaning_id].copy()

    merged = sub.merge(
        lang[["id", "glottocode", "latitude", "longitude"]].dropna(
            subset=["glottocode", "latitude", "longitude"]
        ),
        left_on="language_id",
        right_on="id",
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(), ""

    merged = explode_cognate_classes(merged)
    if merged.empty:
        return pd.DataFrame(), ""

    # Majority-vote dedup per Glottocode.
    per_glotto: dict[str, dict] = {}
    for gc, grp in merged.groupby("glottocode"):
        cls_counts = Counter(grp["cognate_class"])
        top_class = sorted(
            cls_counts.items(), key=lambda kv: (-kv[1], kv[0])
        )[0][0]
        per_glotto[gc] = {
            "glottocode": gc,
            "cognate_class": top_class,
            "latitude": float(grp["latitude"].iloc[0]),
            "longitude": float(grp["longitude"].iloc[0]),
        }

    if not per_glotto:
        return pd.DataFrame(), ""

    global_counts = Counter(r["cognate_class"] for r in per_glotto.values())
    proto_form = sorted(
        global_counts.items(), key=lambda kv: (-kv[1], kv[0])
    )[0][0]

    rows = []
    for r in per_glotto.values():
        rows.append({
            **r,
            "retention": int(r["cognate_class"] == proto_form),
        })
    tbl = pd.DataFrame(rows)
    tbl["dist_homeland_km"] = tbl.apply(
        lambda row: haversine_km(
            HOMELAND_LAT, HOMELAND_LON, row["latitude"], row["longitude"]
        ),
        axis=1,
    )
    return tbl, proto_form


def bootstrap_spearman_ci(
    dist_km: np.ndarray,
    retention: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(dist_km)
    if n < 3:
        return float("nan"), float("nan")
    rs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if (len(np.unique(retention[idx])) < 2
                or len(np.unique(dist_km[idx])) < 2):
            rs[b] = np.nan
            continue
        r_b, _ = spearmanr(dist_km[idx], retention[idx])
        rs[b] = r_b
    rs = rs[~np.isnan(rs)]
    if rs.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def analyze() -> dict:
    wl_path = DATA_PROCESSED / "clean_wordlist.csv"
    lang_path = DATA_PROCESSED / "languages.csv"
    meanings_path = DATA_PROCESSED / "meanings.csv"
    for p in (wl_path, lang_path, meanings_path):
        if not p.exists():
            raise FileNotFoundError(f"Expected {p}")

    print("Loading ABVD tables...")
    wl = pd.read_csv(wl_path, dtype={"language_id": str}, low_memory=False)
    lang = pd.read_csv(lang_path, dtype={"id": str}, low_memory=False)
    meanings = pd.read_csv(meanings_path)
    print(f"  wordlist rows:   {len(wl):,}")
    print(f"  language rows:   {len(lang):,}")
    print(f"  parameter rows:  {len(meanings):,}")

    concepts_out: dict[str, dict] = {}
    concepts_found: list[str] = []
    concepts_not_found: list[dict] = []
    all_glottocodes: set[str] = set()

    for concept in TARGET_CONCEPTS:
        meaning_id, matched_alias, aliases_tried = find_meaning_id(
            concept, meanings
        )
        if meaning_id is None:
            print(f"  [MISS] {concept}: no ABVD parameter matches; "
                  f"tried={aliases_tried}")
            concepts_not_found.append({
                "concept": concept,
                "aliases_tried": aliases_tried,
                "reason": "not present in ABVD 210-concept parameter list",
            })
            continue

        tbl, proto_form = build_concept_table(wl, lang, meaning_id)
        if tbl.empty:
            print(f"  [EMPTY] {concept} ({meaning_id}): zero usable rows")
            concepts_out[concept] = {
                "role": CONCEPT_ROLES.get(concept, "unknown"),
                "parameter_id": meaning_id,
                "parameter_name": matched_alias,
                "n": 0,
                "notes": "no ABVD rows after loan/dedup filtering",
            }
            concepts_found.append(concept)
            continue

        n = int(len(tbl))
        retention = tbl["retention"].to_numpy(dtype=int)
        dist = tbl["dist_homeland_km"].to_numpy(dtype=float)
        r, p = spearmanr(dist, retention)
        ci_lo, ci_hi = bootstrap_spearman_ci(dist, retention, N_BOOTSTRAP, SEED)

        all_glottocodes.update(tbl["glottocode"].astype(str).tolist())

        concepts_out[concept] = {
            "role": CONCEPT_ROLES.get(concept, "unknown"),
            "parameter_id": meaning_id,
            "parameter_name": matched_alias,
            "n": n,
            "n_retention_1": int(retention.sum()),
            "retention_rate": round(float(retention.mean()), 4),
            "spearman_r": round(float(r), 4),
            "spearman_p": float(p),
            "ci_lower": round(ci_lo, 4),
            "ci_upper": round(ci_hi, 4),
            "proto_form_cognate_id": proto_form,
        }
        concepts_found.append(concept)
        print(f"  [OK]   {concept:<8s} id={meaning_id:<12s} n={n:>4d} "
              f"ρ={float(r):+.4f} p={float(p):.2e} "
              f"CI=[{ci_lo:+.3f},{ci_hi:+.3f}]")

    # JLE SEA reproduction sanity check.
    sea = concepts_out.get("SEA", {})
    reproduced_n = int(sea.get("n", 0))
    reproduced_r = float(sea.get("spearman_r", float("nan")))
    match = (
        reproduced_n == 612
        and np.isfinite(reproduced_r)
        and abs(reproduced_r - 0.422) < 0.005
    )

    # Maritime-package-wide verdict.
    hyp = [
        (c, v) for c, v in concepts_out.items()
        if v.get("role") == "anchor_hypothesis" and v.get("n", 0) >= 3
    ]
    n_positive = sum(1 for _, v in hyp if v.get("spearman_r", 0) > 0)
    n_ci_above_zero = sum(
        1 for _, v in hyp
        if v.get("ci_lower") is not None
        and np.isfinite(v.get("ci_lower", float("nan")))
        and v["ci_lower"] > 0
    )
    n_ci_strong = sum(
        1 for _, v in hyp
        if v.get("ci_lower", 0) > 0 and v.get("spearman_r", 0) >= 0.3
    )
    total_hyp = len(hyp)
    if total_hyp == 0:
        interpretation = (
            "No maritime-package candidates available in ABVD; "
            "SEA remains the sole tested Austronesian anchor."
        )
    elif n_ci_strong >= max(3, int(0.5 * total_hyp)):
        interpretation = (
            f"Maritime package collectively anchored: "
            f"{n_ci_strong}/{total_hyp} candidates ρ ≥ +0.3 with CI > 0."
        )
    elif n_ci_above_zero == 0:
        interpretation = (
            f"SEA uniquely anchored: 0/{total_hyp} maritime-package "
            f"candidates clear the CI > 0 threshold."
        )
    else:
        interpretation = (
            f"Mixed: {n_ci_above_zero}/{total_hyp} maritime-package "
            f"candidates have CI > 0; {n_ci_strong} reach ρ ≥ +0.3."
        )

    result = {
        "family": "Austronesian",
        "dataset": "ABVD (Lexibank CLDF)",
        "homeland": {
            "name": HOMELAND_NAME,
            "lat": HOMELAND_LAT,
            "lon": HOMELAND_LON,
        },
        "n_languages_total": len(all_glottocodes),
        "concepts_found": concepts_found,
        "concepts_not_found": concepts_not_found,
        "concepts": concepts_out,
        "comparison_to_JLE_SEA": {
            "JLE_n": 612,
            "JLE_r": 0.422,
            "reproduced_n": reproduced_n,
            "reproduced_r": reproduced_r,
            "match": bool(match),
        },
        "maritime_package_finding": {
            "n_concepts_with_positive_r": int(n_positive),
            "n_concepts_with_CI_above_zero": int(n_ci_above_zero),
            "n_concepts_with_CI_above_zero_and_r_ge_0.3": int(n_ci_strong),
            "n_maritime_candidates_tested": int(total_hyp),
            "interpretation": interpretation,
        },
        "method": {
            "loan_exclusion": ("none at this stage — identical to Paper 1 "
                               "pipeline; cognate-class majority vote "
                               "absorbs loan-form noise"),
            "dedup": "by Glottocode, majority-vote cognate class",
            "retention_definition": ("cognate class equals global mode "
                                     "for concept"),
            "distance": "haversine km from Taiwan",
            "bootstrap": {
                "n": N_BOOTSTRAP,
                "seed": SEED,
                "method": "percentile",
            },
        },
    }
    return result


def print_summary_table(result: dict) -> None:
    print("\n" + "=" * 82)
    print(f"SUMMARY  family={result['family']}  "
          f"homeland={result['homeland']['name']} "
          f"({result['homeland']['lat']}°N, "
          f"{result['homeland']['lon']}°E)")
    print("=" * 82)
    header = (
        f"  {'concept':<9s} {'role':<18s} {'n':>5s} {'ρ':>8s} "
        f"{'p':>10s} {'95% CI':>22s}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    # Sort by descending ρ; missing n=0 rows sort to the end.
    def sort_key(item):
        c, v = item
        if v.get("n", 0) == 0:
            return (1, -1.0, c)
        return (0, -float(v.get("spearman_r", 0.0)), c)

    for concept, v in sorted(result["concepts"].items(), key=sort_key):
        if v.get("n", 0) == 0:
            print(f"  {concept:<9s} {v.get('role', '?'):<18s}  (no data)")
            continue
        ci = f"[{v['ci_lower']:+.3f}, {v['ci_upper']:+.3f}]"
        print(
            f"  {concept:<9s} {v['role']:<18s} {v['n']:>5d} "
            f"{v['spearman_r']:>+8.4f} {v['spearman_p']:>10.2e} {ci:>22s}"
        )
    print("=" * 82)

    nf = result["concepts_not_found"]
    if nf:
        print(f"\n  concepts NOT found in ABVD ({len(nf)}):")
        for entry in nf:
            print(f"    {entry['concept']:<9s} aliases tried: "
                  f"{entry['aliases_tried']}")

    cmp_ = result["comparison_to_JLE_SEA"]
    mark = "PASS" if cmp_["match"] else "MISMATCH"
    print(
        f"\n  SEA reproduction: n={cmp_['reproduced_n']} "
        f"(JLE=612), ρ={cmp_['reproduced_r']:+.4f} "
        f"(JLE=+0.422) -> {mark}"
    )

    mpf = result["maritime_package_finding"]
    print(f"\n  Maritime-package verdict: {mpf['interpretation']}")


def render_forest(result: dict, out_path: Path) -> None:
    """Single-panel forest plot of ρ per concept with bootstrap 95% CIs.

    Colouring:
        anchor_baseline (SEA)       → red
        anchor_hypothesis, CI > 0   → blue (maritime-package candidate)
        anchor_hypothesis, else     → grey (control-like)
    """
    import matplotlib.pyplot as plt

    items = [
        (c, v) for c, v in result["concepts"].items()
        if v.get("n", 0) > 0 and np.isfinite(v.get("spearman_r", float("nan")))
    ]
    # Sort ascending so the largest ρ appears at top.
    items.sort(key=lambda kv: kv[1]["spearman_r"])

    concepts = [c for c, _ in items]
    rs = np.array([v["spearman_r"] for _, v in items])
    los = np.array([v["ci_lower"] for _, v in items])
    his = np.array([v["ci_upper"] for _, v in items])
    roles = [v["role"] for _, v in items]

    colors = []
    for c, v, role in zip(concepts, [v for _, v in items], roles):
        if role == "anchor_baseline":
            colors.append("#d62728")  # red
        elif role == "anchor_hypothesis" and v.get("ci_lower", 0) > 0:
            colors.append("#1f77b4")  # blue: maritime-package candidate
        else:
            colors.append("#7f7f7f")  # grey: control-like

    fig, ax = plt.subplots(figsize=(8.0, 0.45 * len(concepts) + 1.6))
    y = np.arange(len(concepts))

    ax.hlines(y, los, his, color=colors, linewidth=2.2)
    ax.scatter(rs, y, color=colors, s=58, zorder=3,
               edgecolors="black", linewidths=0.6)

    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{c}  (n={dict(items)[c]['n']})" for c in concepts],
        fontsize=10,
    )
    ax.set_xlabel(
        "Spearman ρ(distance from Taiwan, retention)  [95% CI]",
        fontsize=11,
    )
    ax.set_title(
        "Austronesian maritime-cultural-package concepts:\n"
        "migration-gradient extension of the SEA anchor (Paper 1)",
        fontsize=12,
    )

    # Legend.
    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
               markeredgecolor="black", markersize=8,
               label="SEA (Paper 1 anchor)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4",
               markeredgecolor="black", markersize=8,
               label="Maritime-package candidate (CI > 0)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#7f7f7f",
               markeredgecolor="black", markersize=8,
               label="Control-like (CI includes 0)"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=9,
              frameon=True, framealpha=0.9)

    ax.set_xlim(-1.0, 1.0)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


def main() -> None:
    result = analyze()
    print_summary_table(result)

    out_json = RESULTS / "xfam_an_expanded_concepts.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_json}")

    fig_path = FIG_DIR / "fig_an_expanded_forest.png"
    render_forest(result, fig_path)


if __name__ == "__main__":
    main()
