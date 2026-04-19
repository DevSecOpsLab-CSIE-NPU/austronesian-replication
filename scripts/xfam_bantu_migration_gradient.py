#!/usr/bin/env python3
"""Bantu migration gradient analysis (fourth cross-family test).

Tests whether proto-form retention varies with distance from the
Proto-Bantu homeland (Grassfields region, Cameroon/Nigeria border;
Greenberg 1972; Nurse & Philippson 2003). Uses the Grollemund et al.
2015 PNAS Bantu expansion dataset (``lexibank/grollemundbantu``, CLDF).

Pre-registered prediction (expansion-corridor landscape hypothesis):
the empirical anchor for Bantu should be FOREST (Central African
tropical forest zone) or RIVER (Congo/Zambezi riverine dispersal
routes). Neither concept is present in the Grollemund basic-vocabulary
slate, so we test the 4 target concepts that *are* present (FIRE,
WATER, STONE, TREE) plus any other candidates on the pre-registered
slate; coverage gaps are reported verbatim.

For each concept C with sufficient coverage:
  1. Collect forms (Parameter_ID == C), drop loans.
  2. Join cognates.csv on Form_ID to attach Cognateset_ID.
  3. Attach language Glottocode / lat / lon.
  4. Majority-vote one Cognateset_ID per (Glottocode, C).
  5. Retention R_j = 1 iff language j's class == global mode for C.
  6. Haversine distance of each language from the homeland.
  7. Spearman rho(distance, retention), n, two-tailed p.
  8. 500-resample percentile bootstrap 95% CI, seed 42.

Output: results/xfam_bantu_migration_gradient.json
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

DATA = ROOT / "data" / "raw" / "grollemundbantu" / "cldf"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# Proto-Bantu homeland (Grassfields; Greenberg 1972; Nurse & Philippson 2003)
HOMELAND_LAT = 6.0
HOMELAND_LON = 11.0
HOMELAND_NAME = "Grassfields (Cameroon/Nigeria border)"

SEED = 42
N_BOOTSTRAP = 500

# Pre-registered concept slate.  Keys are display labels; values are the
# Parameter_ID in the Grollemund dataset (lowercase), or None if the concept
# is absent from the dataset.  Presence is verified at run time.
CONCEPT_SLATE: list[tuple[str, str, str | None]] = [
    # (label, role, grollemund parameter_id)
    ("FOREST", "anchor", None),          # not in dataset
    ("RIVER", "anchor", None),           # not in dataset
    ("BANANA", "anchor", None),          # not in dataset
    ("YAM", "anchor", None),             # not in dataset
    ("MOUNTAIN", "corridor", None),      # not in dataset
    ("SEA", "corridor", None),           # not in dataset
    ("TREE", "control", "tree"),
    ("FIRE", "control", "fire"),
    ("WATER", "control", "water"),
    ("STONE", "control", "stone"),
    ("LEFT", "body-rel", None),          # not in dataset
    ("RIGHT", "body-rel", None),         # not in dataset
]


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
    """Majority-vote cognate class; tie-break by lexicographic order."""
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
    rng: np.random.Generator,
) -> dict:
    """Run migration-gradient analysis for one concept."""
    sub_forms = forms[forms["Parameter_ID"] == param_id].copy()
    n_raw = len(sub_forms)
    # Loan column in Grollemund is entirely NaN, but treat any truthy value as loan.
    loan_raw = sub_forms["Loan"].to_numpy()
    loan_mask = np.array(
        [bool(x) if x == x else False for x in loan_raw],  # x == x filters NaN
        dtype=bool,
    )
    sub_forms = sub_forms[~loan_mask]
    n_nonloan = len(sub_forms)
    n_loans_excluded = n_raw - n_nonloan

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
            f"{missing_cog}/{n_nonloan} ({missing_cog/n_nonloan:.1%}) of forms"
        )
    merged = merged.dropna(subset=["Cognateset_ID"])

    merged = merged.merge(
        langs[["ID", "Glottocode", "Latitude", "Longitude"]].rename(
            columns={"ID": "Language_ID_lang"}
        ),
        left_on="Language_ID",
        right_on="Language_ID_lang",
        how="left",
    )
    merged = merged.dropna(subset=["Glottocode", "Latitude", "Longitude"])

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
            "n": int(n),
            "notes": (
                f"insufficient data (n={n}, raw_forms={n_raw}, nonloan={n_nonloan})"
            ),
        }

    class_counts = Counter(per_lang["cognate_class"])
    top_count = max(class_counts.values())
    tied = sorted(c for c, v in class_counts.items() if v == top_count)
    proto_class = tied[0]
    per_lang["retained"] = (per_lang["cognate_class"] == proto_class).astype(int)

    per_lang["dist_km"] = per_lang.apply(
        lambda row: haversine(HOMELAND_LAT, HOMELAND_LON, row["lat"], row["lon"]),
        axis=1,
    )

    dist = per_lang["dist_km"].to_numpy(dtype=float)
    retain = per_lang["retained"].to_numpy(dtype=float)

    if retain.std() == 0.0 or dist.std() == 0.0:
        r_obs, p_obs = float("nan"), float("nan")
    else:
        r_obs, p_obs = spearmanr(dist, retain)
        r_obs = float(r_obs)
        p_obs = float(p_obs)

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
        "retention_rate": round(float(per_lang["retained"].mean()), 4),
        "spearman_r": None if np.isnan(r_obs) else round(r_obs, 4),
        "spearman_p": None if np.isnan(p_obs) else round(p_obs, 6),
        "ci_lower": None if np.isnan(ci_lo) else round(ci_lo, 4),
        "ci_upper": None if np.isnan(ci_hi) else round(ci_hi, 4),
        "proto_form_cognate_id": str(proto_class),
        "n_loans_excluded": int(n_loans_excluded),
        "n_bootstrap_usable": int(boot.size),
    }
    if len(tied) > 1:
        result["notes"] = f"tie among {len(tied)} top classes; lex-min chosen"
    return result


def main() -> None:
    print("=" * 78)
    print("Bantu migration gradient (grollemundbantu, Grollemund et al. 2015 PNAS)")
    print(f"Homeland: {HOMELAND_NAME} ({HOMELAND_LAT} N, {HOMELAND_LON} E)")
    print("=" * 78)

    langs = pd.read_csv(DATA / "languages.csv")
    params = pd.read_csv(DATA / "parameters.csv")
    forms = pd.read_csv(DATA / "forms.csv")
    cognates = pd.read_csv(DATA / "cognates.csv")

    print(
        f"  languages: {len(langs)} | parameters: {len(params)} | "
        f"forms: {len(forms)} | cognates: {len(cognates)}"
    )

    # Build availability map from parameters.csv (match on lowercase gloss).
    available_ids = {str(x).strip().lower() for x in params["ID"].dropna().tolist()}
    available_glosses = {
        str(x).strip().upper() for x in params["Concepticon_Gloss"].dropna().tolist()
    }
    concept_availability: dict[str, bool] = {}
    for label, _role, pid in CONCEPT_SLATE:
        present = False
        if pid is not None and pid.lower() in available_ids:
            present = True
        elif label.upper() in available_glosses:
            present = True
        concept_availability[label] = present

    # Print coverage summary
    print("\n  concept availability:")
    for label, _role, _pid in CONCEPT_SLATE:
        mark = "YES" if concept_availability[label] else " no"
        print(f"    [{mark}] {label}")

    rng = np.random.default_rng(SEED)

    concepts: dict[str, dict] = {}
    rows_for_table: list[tuple] = []
    for label, role, pid in CONCEPT_SLATE:
        if not concept_availability[label] or pid is None:
            continue
        res = analyze_concept(
            label=label,
            role=role,
            param_id=pid,
            forms=forms,
            cognates=cognates,
            langs=langs,
            rng=rng,
        )
        concepts[label] = res
        r_val = res.get("spearman_r")
        p_val = res.get("spearman_p")
        r_str = f"{r_val:+.4f}" if r_val is not None else "   NA  "
        p_str = f"{p_val:.4f}" if p_val is not None else "  NA  "
        lo, hi = res.get("ci_lower"), res.get("ci_upper")
        ci_str = f"[{lo:+.3f},{hi:+.3f}]" if lo is not None else "       NA        "
        rr = res.get("retention_rate")
        rr_str = f"{rr:.3f}" if rr is not None else " NA "
        rows_for_table.append(
            (label, role, res.get("n", 0), rr_str, r_str, ci_str, p_str)
        )

    print()
    print("-" * 92)
    print(
        f"{'concept':<10} {'role':<9} {'n':>3}  {'retain':>6}  "
        f"{'rho':>8}  {'95% CI':<19} {'p':>7}"
    )
    print("-" * 92)
    for label, role, n, rr, r_str, ci_str, p_str in rows_for_table:
        print(f"{label:<10} {role:<9} {n:>3}  {rr:>6}  {r_str:>8}  {ci_str:<19} {p_str:>7}")
    print("-" * 92)

    # Empirical anchor = concept with strongest positive ρ (with ρ defined).
    anchor_label = None
    anchor_r = None
    anchor_p = None
    for label, res in concepts.items():
        r = res.get("spearman_r")
        if r is None:
            continue
        if anchor_r is None or r > anchor_r:
            anchor_r = r
            anchor_label = label
            anchor_p = res.get("spearman_p")

    # Prediction: FOREST or RIVER.  Neither is in this dataset, so the
    # pre-registered test cannot be fulfilled; we note the gap.
    predicted_set = {"FOREST", "RIVER"}
    prediction_matched = anchor_label in predicted_set

    print()
    if anchor_label is not None:
        print(
            f"Empirical anchor: {anchor_label}  rho={anchor_r:+.4f}  p={anchor_p:.4f}"
        )
        print(
            f"Prediction match (FOREST/RIVER): {'YES' if prediction_matched else 'NO'}"
        )
    else:
        print("Empirical anchor: (no concept with defined rho)")

    notes_parts: list[str] = []
    missing_preregistered = [
        lbl for lbl in ("FOREST", "RIVER", "BANANA", "YAM", "MOUNTAIN", "SEA",
                        "LEFT", "RIGHT")
        if not concept_availability.get(lbl, False)
    ]
    if missing_preregistered:
        notes_parts.append(
            "pre-registered concepts absent from Grollemund basic-vocabulary "
            "slate: " + ", ".join(missing_preregistered)
        )
    notes_parts.append(
        "Grollemund Loan column is uniformly NaN; no forms were flagged as "
        "loans in the source dataset."
    )
    if anchor_label is not None and not prediction_matched:
        notes_parts.append(
            f"empirical anchor {anchor_label} is not in the pre-registered "
            "anchor set {FOREST, RIVER}; the corridor-ecology prediction "
            "cannot be evaluated against Grollemund's 100-item Swadesh slate."
        )

    output = {
        "family": "Bantu",
        "homeland": {
            "name": HOMELAND_NAME,
            "lat": HOMELAND_LAT,
            "lon": HOMELAND_LON,
        },
        "dataset": (
            "grollemundbantu (Grollemund et al. 2015, PNAS; lexibank CLDF)"
        ),
        "n_languages_total": int(len(langs)),
        "concept_availability": concept_availability,
        "concepts": concepts,
        "empirical_anchor": anchor_label,
        "empirical_anchor_r": None if anchor_r is None else round(float(anchor_r), 4),
        "empirical_anchor_p": None if anchor_p is None else round(float(anchor_p), 6),
        "prediction_matched": bool(prediction_matched),
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
        "notes": " | ".join(notes_parts) if notes_parts else "",
    }

    out_path = RESULTS / "xfam_bantu_migration_gradient.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
