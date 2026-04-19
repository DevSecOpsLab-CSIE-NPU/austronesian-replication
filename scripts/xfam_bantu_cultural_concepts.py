#!/usr/bin/env python3
"""Bantu cultural-concept gradient analysis (issue #82).

Context
-------
Paper 2 ("xfam_bantu_migration_gradient.py") reported Bantu as
"untestable" because the Grollemund 2015 Swadesh-100 slate was assumed
to lack the cultural/subsistence vocabulary needed for corridor-ecology
anchor testing (FOREST, RIVER, BANANA, YAM, CATTLE, IRON).

Re-inspection of the Grollemund parameter list (issue #82) shows that
the slate does in fact contain IRON (``Parameter_ID=iron``,
Concepticon_Gloss=IRON, Concepticon_ID=621), with 286 single-language
attestations.  It also contains several other material-culture and
animal concepts relevant to the Bantu-expansion debate:
ELEPHANT, GOAT, SPEAR, KNIFE, HOUSE, VILLAGE, ROAD, BED, SALT, WAR.

FOREST, RIVER, BANANA, YAM, CATTLE remain absent from Grollemund;
``lexibank/bantubvd`` (Greenhill & Gray 2015) does contain those
glosses, but ships (i) only 10 languages and (ii) an empty Cognacy
column, which rules it out for a retention-based Spearman gradient
test.  See ``data/external/bantu_cultural_dataset_survey.md``.

This script therefore runs the existing migration-gradient pipeline
against the Grollemund slate extended to the cultural concepts that
*are* present, and reports whether any of them is the empirical anchor
for Bantu.

Method (identical to ``xfam_bantu_migration_gradient.py``):
    1. Collect forms for each concept, drop loans (Loan column in
       Grollemund is uniformly NaN, so nothing is actually dropped).
    2. Attach Cognateset_ID via cognates.csv.
    3. Dedup by Glottocode via majority vote.
    4. Global-mode retention: R_j = 1 iff language j's class == global
       mode for that concept.
    5. Haversine distance from Grassfields (6.0 N, 11.0 E).
    6. Spearman rho(distance, retention), n, two-tailed p.
    7. 500-resample percentile bootstrap 95% CI, seed 42.

Output: ``results/xfam_bantu_cultural.json``
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

# Extended concept slate.  Parameter_ID is the lowercase ID used by
# Grollemund; None means the concept is absent from the slate and is
# carried through only so the output documents the gap verbatim.
CONCEPT_SLATE: list[tuple[str, str, str | None]] = [
    # --- pre-registered ecological anchors ---
    ("FOREST", "anchor-eco", None),       # absent
    ("RIVER", "anchor-eco", None),        # absent
    # --- pre-registered subsistence anchors ---
    ("BANANA", "anchor-subs", None),      # absent
    ("YAM", "anchor-subs", None),         # absent
    ("CATTLE", "anchor-subs", None),      # absent
    # --- cultural/material items PRESENT in Grollemund (issue #82) ---
    ("IRON", "anchor-culture", "iron"),
    ("ELEPHANT", "anchor-fauna", "elephant"),
    ("GOAT", "anchor-subs", "goat"),
    ("SPEAR", "culture", "spear"),
    ("KNIFE", "culture", "knife"),
    ("HOUSE", "culture", "house"),
    ("VILLAGE", "culture", "village"),
    ("ROAD", "culture", "road"),
    ("BED", "culture", "bed"),
    ("SALT", "culture", "salt"),
    ("WAR", "culture", "war"),
    # --- Swadesh controls (same as the original script) ---
    ("TREE", "control", "tree"),
    ("FIRE", "control", "fire"),
    ("WATER", "control", "water"),
    ("STONE", "control", "stone"),
]


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    return float(R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


def majority_vote(classes: list) -> str:
    counts = Counter(classes)
    top = max(counts.values())
    tied = sorted(c for c, v in counts.items() if v == top)
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
    sub_forms = forms[forms["Parameter_ID"] == param_id].copy()
    n_raw = len(sub_forms)
    loan_raw = sub_forms["Loan"].to_numpy()
    loan_mask = np.array(
        [bool(x) if x == x else False for x in loan_raw], dtype=bool
    )
    sub_forms = sub_forms[~loan_mask]
    n_nonloan = len(sub_forms)

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
            f"{missing_cog}/{n_nonloan} of forms"
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
            "notes": f"insufficient data (n={n})",
        }

    class_counts = Counter(per_lang["cognate_class"])
    top = max(class_counts.values())
    tied = sorted(c for c, v in class_counts.items() if v == top)
    proto_class = tied[0]
    per_lang["retained"] = (per_lang["cognate_class"] == proto_class).astype(int)

    per_lang["dist_km"] = per_lang.apply(
        lambda r: haversine(HOMELAND_LAT, HOMELAND_LON, r["lat"], r["lon"]),
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
        "n_bootstrap_usable": int(boot.size),
    }
    if len(tied) > 1:
        result["notes"] = f"tie among {len(tied)} top classes; lex-min chosen"
    return result


def main() -> None:
    print("=" * 80)
    print("Bantu cultural-concept gradient (issue #82)")
    print(f"Homeland: {HOMELAND_NAME} ({HOMELAND_LAT} N, {HOMELAND_LON} E)")
    print("=" * 80)

    langs = pd.read_csv(DATA / "languages.csv")
    params = pd.read_csv(DATA / "parameters.csv")
    forms = pd.read_csv(DATA / "forms.csv")
    cognates = pd.read_csv(DATA / "cognates.csv")
    print(
        f"  langs: {len(langs)} | params: {len(params)} | "
        f"forms: {len(forms)} | cognates: {len(cognates)}"
    )

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

    print("\n  concept availability (Grollemund 2015 slate):")
    for label, role, _pid in CONCEPT_SLATE:
        mark = "YES" if concept_availability[label] else " no"
        print(f"    [{mark}] {label:<10} ({role})")

    rng = np.random.default_rng(SEED)

    concepts: dict[str, dict] = {}
    table: list[tuple] = []
    for label, role, pid in CONCEPT_SLATE:
        if not concept_availability[label] or pid is None:
            continue
        res = analyze_concept(label, role, pid, forms, cognates, langs, rng)
        concepts[label] = res
        r = res.get("spearman_r")
        p = res.get("spearman_p")
        lo, hi = res.get("ci_lower"), res.get("ci_upper")
        rr = res.get("retention_rate")
        table.append(
            (
                label,
                role,
                res.get("n", 0),
                f"{rr:.3f}" if rr is not None else " NA  ",
                f"{r:+.4f}" if r is not None else "  NA   ",
                f"[{lo:+.3f},{hi:+.3f}]" if lo is not None else "       NA        ",
                f"{p:.4f}" if p is not None else "  NA  ",
            )
        )

    print()
    print("-" * 96)
    print(
        f"{'concept':<10} {'role':<14} {'n':>3}  {'retain':>6}  "
        f"{'rho':>8}  {'95% CI':<19} {'p':>7}"
    )
    print("-" * 96)
    for row in table:
        print(f"{row[0]:<10} {row[1]:<14} {row[2]:>3}  "
              f"{row[3]:>6}  {row[4]:>8}  {row[5]:<19} {row[6]:>7}")
    print("-" * 96)

    # Empirical anchor = largest positive rho among concepts with defined rho.
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

    predicted_set = {"FOREST", "RIVER", "BANANA", "YAM", "CATTLE", "IRON"}
    prediction_matched = anchor_label in predicted_set

    print()
    if anchor_label is not None:
        print(
            f"Empirical anchor: {anchor_label}  rho={anchor_r:+.4f}  p={anchor_p:.4f}"
        )
        print(
            f"Prediction match (corridor-eco set): "
            f"{'YES' if prediction_matched else 'NO'}"
        )
    else:
        print("Empirical anchor: (no concept with defined rho)")

    notes_parts: list[str] = []
    missing = [lbl for lbl in ("FOREST", "RIVER", "BANANA", "YAM", "CATTLE")
               if not concept_availability[lbl]]
    if missing:
        notes_parts.append(
            "ecological/subsistence anchors absent from Grollemund slate: "
            + ", ".join(missing)
            + " (see data/external/bantu_cultural_dataset_survey.md)"
        )
    notes_parts.append(
        "Grollemund Loan column is uniformly NaN; no forms dropped as loans."
    )
    notes_parts.append(
        "bantubvd (Greenhill & Gray 2015) contains FOREST/RIVER/BANANA/CATTLE "
        "but only 10 languages and an empty Cognacy column, so retention "
        "analysis is not possible."
    )

    output = {
        "family": "Bantu",
        "analysis": "cultural-concept extension of xfam_bantu_migration_gradient",
        "issue": "#82",
        "homeland": {
            "name": HOMELAND_NAME,
            "lat": HOMELAND_LAT,
            "lon": HOMELAND_LON,
        },
        "dataset": "grollemundbantu (Grollemund et al. 2015 PNAS; lexibank CLDF)",
        "datasets_surveyed": [
            "lexibank/grollemundbantu (used here)",
            "lexibank/bantubvd (rejected: 10 langs, empty Cognacy)",
            "lexibank/polyglottaafricana (rejected: no cognate coding)",
            "CBOLD (rejected: no bulk download / no CLDF)",
        ],
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
            "bootstrap": {"n": N_BOOTSTRAP, "seed": SEED, "method": "percentile"},
            "loans_excluded": True,
        },
        "notes": " | ".join(notes_parts),
    }

    out_path = RESULTS / "xfam_bantu_cultural.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
