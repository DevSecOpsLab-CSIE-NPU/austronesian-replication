#!/usr/bin/env python3
"""Sino-Tibetan migration gradient analysis (parallel to Austronesian JLE pipeline).

Tests whether proto-form retention varies with distance from the
Proto-Sino-Tibetan homeland (Upper Yellow River, 35.0 N, 104.0 E;
Sagart 2019 PNAS). Uses the Sagart / Jacques / Lai / List 2019 ST
Database of Lexical Cognates (sagartst, CLDF).

For each target concept C:
  1. Resolve the sagartst parameter ID from data/external/anchor_concepts.csv.
  2. Collect forms (Parameter_ID == C), exclude loans.
  3. Join cognates.csv on Form_ID to attach Cognateset_ID.
  4. Majority-vote one Cognateset_ID per (Glottocode, C).
  5. Retention R_j = 1 iff language j's class == global mode for C.
  6. Haversine distance of each language from the homeland.
  7. Spearman rho(distance, retention), n, two-tailed p.
  8. 500-resample percentile bootstrap 95% CI, seed 42.

Output: results/xfam_st_migration_gradient.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "data" / "raw" / "sagartst" / "cldf"
EXT = ROOT / "data" / "external"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# Proto-Sino-Tibetan homeland (Sagart 2019 PNAS): Upper Yellow River basin
HOMELAND_LAT = 35.0
HOMELAND_LON = 104.0
HOMELAND_NAME = "Upper Yellow River basin"

SEED = 42
N_BOOTSTRAP = 500


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    return float(R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


def majority_vote(classes: list) -> str:
    """Majority-vote cognate class; tie-break by lexicographic order for determinism."""
    counts = Counter(classes)
    top_count = max(counts.values())
    tied = sorted(c for c, v in counts.items() if v == top_count)
    return tied[0]


def analyze_concept(
    label: str,
    role: str,
    param_id: str,
    forms: pd.DataFrame,
    cognates: pd.DataFrame,
    langs: pd.DataFrame,
    params: pd.DataFrame,
    rng: np.random.Generator,
) -> dict:
    """Run migration-gradient analysis for one concept."""
    if param_id not in set(params["ID"]):
        raise RuntimeError(
            f"Concept {label}: parameter ID {param_id!r} not found in sagartst parameters.csv"
        )

    # Step 2: collect forms for this concept, drop loans
    sub_forms = forms[forms["Parameter_ID"] == param_id].copy()
    n_raw = len(sub_forms)
    sub_forms = sub_forms[sub_forms["Loan"] != True]  # noqa: E712 (bool column)
    n_nonloan = len(sub_forms)

    # Step 3: join with cognates on Form_ID
    merged = sub_forms.merge(
        cognates[["Form_ID", "Cognateset_ID"]],
        left_on="ID",
        right_on="Form_ID",
        how="left",
    )
    missing_cog = merged["Cognateset_ID"].isna().sum()
    if n_nonloan > 0 and missing_cog / n_nonloan > 0.05:
        raise RuntimeError(
            f"Concept {label}: forms->cognates join lost "
            f"{missing_cog}/{n_nonloan} ({missing_cog/n_nonloan:.1%}) of forms (>5% threshold)"
        )
    merged = merged.dropna(subset=["Cognateset_ID"])

    # Attach language metadata (Glottocode, Latitude, Longitude)
    merged = merged.merge(
        langs[["ID", "Glottocode", "Latitude", "Longitude"]].rename(
            columns={"ID": "Language_ID_lang"}
        ),
        left_on="Language_ID",
        right_on="Language_ID_lang",
        how="left",
    )
    merged = merged.dropna(subset=["Glottocode", "Latitude", "Longitude"])

    # Step 4: deduplicate per (Glottocode, C) via majority vote
    per_lang = (
        merged.groupby("Glottocode")
        .agg(
            cognate_class=("Cognateset_ID", lambda s: majority_vote(list(s))),
            lat=("Latitude", "mean"),
            lon=("Longitude", "mean"),
        )
        .reset_index()
    )

    n = len(per_lang)
    if n < 5:
        return {
            "role": role,
            "param_id": param_id,
            "n": n,
            "notes": f"insufficient data (n={n}, raw_forms={n_raw}, nonloan={n_nonloan})",
        }

    # Step 5: global-mode proto-form; retention flag
    class_counts = Counter(per_lang["cognate_class"])
    top_count = max(class_counts.values())
    tied = sorted(c for c, v in class_counts.items() if v == top_count)
    proto_class = tied[0]
    per_lang["retained"] = (per_lang["cognate_class"] == proto_class).astype(int)

    # Step 6: homeland distance
    per_lang["dist_km"] = per_lang.apply(
        lambda row: haversine(HOMELAND_LAT, HOMELAND_LON, row["lat"], row["lon"]),
        axis=1,
    )

    dist = per_lang["dist_km"].to_numpy(dtype=float)
    retain = per_lang["retained"].to_numpy(dtype=float)

    # Step 7: Spearman rho (guard against degenerate retention)
    if retain.std() == 0.0 or dist.std() == 0.0:
        r_obs, p_obs = float("nan"), float("nan")
    else:
        r_obs, p_obs = spearmanr(dist, retain)
        r_obs = float(r_obs)
        p_obs = float(p_obs)

    # Step 8: percentile bootstrap 95% CI
    boot = np.empty(N_BOOTSTRAP, dtype=float)
    idx_arr = np.arange(n)
    for b in range(N_BOOTSTRAP):
        samp = rng.choice(idx_arr, size=n, replace=True)
        d_b = dist[samp]
        r_b = retain[samp]
        if r_b.std() == 0.0 or d_b.std() == 0.0:
            boot[b] = np.nan
            continue
        rr, _ = spearmanr(d_b, r_b)
        boot[b] = rr
    boot = boot[~np.isnan(boot)]
    if boot.size >= 10:
        ci_lo = float(np.percentile(boot, 2.5))
        ci_hi = float(np.percentile(boot, 97.5))
    else:
        ci_lo, ci_hi = float("nan"), float("nan")

    result = {
        "role": role,
        "param_id": param_id,
        "n": int(n),
        "n_retention_1": int(per_lang["retained"].sum()),
        "retention_rate": float(per_lang["retained"].mean()),
        "spearman_r": None if np.isnan(r_obs) else round(r_obs, 4),
        "spearman_p": None if np.isnan(p_obs) else round(p_obs, 6),
        "ci_lower": None if np.isnan(ci_lo) else round(ci_lo, 4),
        "ci_upper": None if np.isnan(ci_hi) else round(ci_hi, 4),
        "proto_form_cognate_id": str(proto_class),
    }
    if len(tied) > 1:
        result["notes"] = f"tie among {len(tied)} classes at top; lex-min chosen"
    return result


def main() -> None:
    print("=" * 78)
    print("Sino-Tibetan migration gradient (sagartst, Sagart et al. 2019 PNAS)")
    print(f"Homeland: {HOMELAND_NAME} ({HOMELAND_LAT} N, {HOMELAND_LON} E)")
    print("=" * 78)

    # Load data
    langs = pd.read_csv(DATA / "languages.csv")
    params = pd.read_csv(DATA / "parameters.csv")
    forms = pd.read_csv(DATA / "forms.csv")
    cognates = pd.read_csv(DATA / "cognates.csv")
    anchors = pd.read_csv(EXT / "anchor_concepts.csv")

    print(
        f"  languages: {len(langs)} | parameters: {len(params)} | "
        f"forms: {len(forms)} | cognates: {len(cognates)}"
    )

    # Filter anchor slate to Sino-Tibetan, non-excluded
    st = anchors[
        (anchors["family"] == "Sino-Tibetan") & (anchors["role"] != "excluded")
    ].copy()
    print(f"  ST target concepts: {len(st)}")

    rng = np.random.default_rng(SEED)

    concepts: dict[str, dict] = {}
    rows_for_table: list[tuple] = []
    for _, row in st.iterrows():
        label = row["concept"]
        role = row["role"]
        param_id = row["iecor_meaning"]
        if not isinstance(param_id, str) or not param_id.strip():
            print(f"  [{label}] SKIP: no parameter ID")
            continue
        res = analyze_concept(
            label=label,
            role=role,
            param_id=param_id,
            forms=forms,
            cognates=cognates,
            langs=langs,
            params=params,
            rng=rng,
        )
        concepts[label] = res
        r_str = (
            f"{res['spearman_r']:+.4f}"
            if res.get("spearman_r") is not None
            else "   NA  "
        )
        p_str = (
            f"{res['spearman_p']:.4f}"
            if res.get("spearman_p") is not None
            else "  NA  "
        )
        ci_str = (
            f"[{res['ci_lower']:+.3f},{res['ci_upper']:+.3f}]"
            if res.get("ci_lower") is not None
            else "       NA        "
        )
        rr = res.get("retention_rate")
        rr_str = f"{rr:.3f}" if rr is not None else " NA "
        rows_for_table.append(
            (label, role, res.get("n", 0), rr_str, r_str, ci_str, p_str)
        )

    # Emit table
    print()
    print("-" * 90)
    print(
        f"{'concept':<10} {'role':<8} {'n':>3}  {'retain':>6}  "
        f"{'rho':>8}  {'95% CI':<19} {'p':>7}"
    )
    print("-" * 90)
    for label, role, n, rr, r_str, ci_str, p_str in rows_for_table:
        print(f"{label:<10} {role:<8} {n:>3}  {rr:>6}  {r_str:>8}  {ci_str:<19} {p_str:>7}")
    print("-" * 90)

    # Sanity priors (weak flags only)
    print()
    sea = concepts.get("SEA", {})
    sea_r = sea.get("spearman_r")
    if sea_r is not None:
        tag = "OK (near zero/negative as expected)" if sea_r <= 0.1 else "FLAG (positive, unexpected for inland homeland)"
        print(f"  SEA gradient: r = {sea_r:+.3f}  -- {tag}")
    anchor_hits = [
        (k, concepts[k].get("spearman_r"))
        for k in ("RICE", "MOUNTAIN", "SALT", "BARLEY")
        if k in concepts and concepts[k].get("spearman_r") is not None
    ]
    positive_anchors = [k for k, r in anchor_hits if r is not None and r > 0]
    if positive_anchors:
        print(
            f"  Anchors with positive gradient: {', '.join(positive_anchors)} "
            f"(of {len(anchor_hits)} tested)"
        )
    else:
        print(
            "  FLAG: no candidate anchor (RICE/MOUNTAIN/SALT/BARLEY) "
            "shows a positive gradient"
        )

    output = {
        "family": "Sino-Tibetan",
        "homeland": {
            "name": HOMELAND_NAME,
            "lat": HOMELAND_LAT,
            "lon": HOMELAND_LON,
        },
        "dataset": "sagartst (Sagart et al. 2019, PNAS)",
        "n_languages_total": int(len(langs)),
        "concepts": concepts,
        "method": {
            "dedup": "by Glottocode, majority-vote cognate class",
            "retention_definition": "cognate class equals global mode for concept",
            "distance": "haversine km from homeland",
            "bootstrap": {
                "n": N_BOOTSTRAP,
                "seed": SEED,
                "method": "percentile",
            },
            "loans_excluded": True,
        },
    }

    out_path = RESULTS / "xfam_st_migration_gradient.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
