#!/usr/bin/env python3
"""Cross-family migration gradient — Indo-European, cultural-vocabulary slate.

Supplement to ``xfam_ie_migration_gradient.py``. The IECoR v1.2 concept list
(170 items, Swadesh-oriented) excludes the culturally diagnostic vocabulary
central to Anthony (2007)'s steppe-pastoralist hypothesis: HORSE, WHEEL, YOKE,
PLOUGH, AXLE. This script fills that gap using the DIACL CLDF derivation
(Carling 2017; lexibank/diacl), which is the only publicly available
cognate-coded IE lexical dataset we found that includes all five concepts.

DIACL's lexibank CLDF ``forms.csv`` has an empty ``Cognacy`` column. Cognacy
information is stored in ``raw/etymology.json.gz`` as a parent/child edge list
over lexeme IDs. We reconstruct cognate classes per concept by:

  1. Collecting every etymology entry that mentions any lexeme of the target
     concept.
  2. Building a parent->child map from its ``etymologies`` edges.
  3. Walking each lexeme to its root within the entry. Forms sharing a root
     form one cognate class.

This yields 100% cognate-class coverage for HORSE (16 classes / 68 IE langs),
WHEEL (13 / 68), YOKE (9 / 59), PLOUGH[_V] (10 / 61), PLOUGH[_N] (16 / 67),
AXLE (7 / 54).

Inputs
------
- data/raw/diacl/cldf/{languages,parameters,forms}.csv
- data/raw/diacl/raw/etymology.json.gz

Output
------
- results/xfam_ie_cultural.json

Usage
-----
    .venv/bin/python3 scripts/xfam_ie_cultural_concepts.py
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Path setup — follow repo convention
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

DIACL_CLDF = ROOT / "data" / "raw" / "diacl" / "cldf"
DIACL_RAW = ROOT / "data" / "raw" / "diacl" / "raw"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUT_JSON = RESULTS_DIR / "xfam_ie_cultural.json"

HOMELAND_NAME = "Pontic-Caspian steppe"
HOMELAND_LAT = 48.0
HOMELAND_LON = 45.0

N_BOOTSTRAP = 500
SEED = 42

# DIACL Parameter_ID -> reporting gloss. DIACL encodes two PLOUGH entries: a
# verb ("to plough") and a noun ("plough (instrument)"); we keep both as
# separate concepts for transparency.
TARGET_CONCEPTS = {
    "HORSE": "38_horse",
    "WHEEL": "14_wheel",
    "YOKE": "15_yoke",
    "PLOUGH_V": "1_toplough",
    "PLOUGH_N": "8_plough",
    "AXLE": "7_axle",
}


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def haversine_km(lat1: float, lon1: float, lat2, lon2) -> np.ndarray:
    """Great-circle distance in km. Vectorised over lat2/lon2."""
    R = 6371.0088
    lat1r = np.radians(lat1)
    lon1r = np.radians(lon1)
    lat2r = np.radians(np.asarray(lat2, dtype=float))
    lon2r = np.radians(np.asarray(lon2, dtype=float))
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# DIACL loading & cognate-class reconstruction
# ---------------------------------------------------------------------------


def load_diacl() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    langs = pd.read_csv(DIACL_CLDF / "languages.csv")
    params = pd.read_csv(DIACL_CLDF / "parameters.csv")
    forms = pd.read_csv(DIACL_CLDF / "forms.csv", low_memory=False)
    forms["diacl_id"] = pd.to_numeric(forms["diacl_id"], errors="coerce")
    with gzip.open(DIACL_RAW / "etymology.json.gz") as fh:
        etym = json.load(fh)
    return langs, params, forms, etym


def _root_of(node: int, parent_map: dict[int, int]) -> int:
    """Walk parent_map until a node with no parent is reached. Cycle-safe."""
    seen: set[int] = set()
    while node in parent_map:
        if node in seen:
            return node
        seen.add(node)
        node = parent_map[node]
    return node


def build_cognate_index(etym: dict, concept_lexeme_ids: set[int]) -> dict[int, tuple[str, int]]:
    """For every lexeme in ``concept_lexeme_ids``, find which etymology entry
    references it and which root (within that entry's child/parent graph) it
    descends from. Return ``lexeme_id -> (entry_id, root_lexeme_id)``.

    Lexemes that belong to the same (entry_id, root_lexeme_id) pair are
    reflexes of the same proto-ancestor and are thus cognate.
    """
    out: dict[int, tuple[str, int]] = {}
    for entry_id, entry in etym.items():
        connected = set(entry.get("connectedLexemesById", []))
        hit = connected & concept_lexeme_ids
        if not hit:
            continue
        edges = entry.get("etymologies", {}) or {}
        parent_map = {e["FkChildId"]: e["FkParentId"] for e in edges.values()}
        for lid in hit:
            root = _root_of(lid, parent_map)
            # First-seen wins; etymology entries for IE target concepts are
            # disjoint in practice (verified for HORSE, WHEEL, YOKE, PLOUGH,
            # AXLE — no lexeme appears in two entries).
            out.setdefault(lid, (entry_id, root))
    return out


# ---------------------------------------------------------------------------
# Per-concept analysis
# ---------------------------------------------------------------------------


def analyze_concept(
    concept_label: str,
    param_id: str,
    langs: pd.DataFrame,
    forms: pd.DataFrame,
    etym: dict,
    ie_lang_ids: set,
    rng: np.random.Generator,
) -> dict:
    ie_forms = forms[
        (forms["Parameter_ID"] == param_id)
        & (forms["Language_ID"].isin(ie_lang_ids))
    ].copy()
    n_forms_total = len(ie_forms)
    if n_forms_total == 0:
        raise RuntimeError(
            f"[{concept_label}] No IE forms for Parameter_ID={param_id} in DIACL."
        )

    # Map each form to a cognate class via etymology tree
    lex_ids = set(ie_forms["diacl_id"].dropna().astype(int).tolist())
    cog_index = build_cognate_index(etym, lex_ids)

    def _cog(lid):
        if pd.isna(lid):
            return None
        key = cog_index.get(int(lid))
        if key is None:
            return None
        return f"{key[0]}:{key[1]}"  # stringified (entry_id, root) pair

    ie_forms["cognate"] = ie_forms["diacl_id"].map(_cog)

    n_no_cog = int(ie_forms["cognate"].isna().sum())
    # Hard guard: cultural-vocabulary slate should be fully cognate-coded in
    # DIACL. Abort if >5% forms lack a class (matches the IECoR script's
    # integrity guard).
    if n_no_cog / n_forms_total > 0.05:
        raise RuntimeError(
            f"[{concept_label}] {n_no_cog}/{n_forms_total} IE forms lack a "
            f"cognate class after etymology reconstruction (>5%) — aborting."
        )
    merged = ie_forms.dropna(subset=["cognate"]).copy()

    # Attach language metadata
    merged = merged.merge(
        langs[["ID", "Glottocode", "Latitude", "Longitude"]],
        left_on="Language_ID",
        right_on="ID",
        suffixes=("", "_lang"),
        how="left",
    )
    merged = merged.dropna(subset=["Glottocode", "Latitude", "Longitude"]).copy()

    # Deduplicate per Glottocode (DIACL has many historical stages per lineage;
    # we collapse to one entry per Glottocode using majority vote, ties
    # broken by lexicographic cognate-class key — deterministic).
    def majority_cognate(series: pd.Series) -> str:
        vc = series.value_counts()
        top = vc.iloc[0]
        tied = sorted(vc[vc == top].index.tolist())
        return tied[0]

    by_gc = (
        merged.groupby("Glottocode")
        .agg(
            cognate=("cognate", majority_cognate),
            latitude=("Latitude", "first"),
            longitude=("Longitude", "first"),
        )
        .reset_index()
    )

    n = len(by_gc)
    if n < 20:
        print(f"  [{concept_label}] WARNING: only {n} languages after dedup.")

    mode_cognate = by_gc["cognate"].value_counts().index[0]
    retention = (by_gc["cognate"].values == mode_cognate).astype(int)
    n_retention_1 = int(retention.sum())
    retention_rate = float(n_retention_1 / n) if n else float("nan")

    dist = haversine_km(
        HOMELAND_LAT, HOMELAND_LON,
        by_gc["latitude"].values, by_gc["longitude"].values,
    )

    if n >= 3 and 0 < retention.sum() < n:
        rho, pval = spearmanr(dist, retention)
        rho = float(rho)
        pval = float(pval)
    else:
        rho = float("nan")
        pval = float("nan")

    rhos: list[float] = []
    if np.isfinite(rho):
        idx_all = np.arange(n)
        for _ in range(N_BOOTSTRAP):
            idx_b = rng.choice(idx_all, size=n, replace=True)
            ret_b = retention[idx_b]
            dist_b = dist[idx_b]
            if ret_b.sum() == 0 or ret_b.sum() == n:
                continue
            r_b, _ = spearmanr(dist_b, ret_b)
            if np.isfinite(r_b):
                rhos.append(r_b)
    if len(rhos) >= 50:
        ci_lower = float(np.percentile(rhos, 2.5))
        ci_upper = float(np.percentile(rhos, 97.5))
    else:
        ci_lower = float("nan")
        ci_upper = float("nan")

    n_classes = int(merged["cognate"].nunique())

    notes_bits = []
    if n_no_cog:
        notes_bits.append(f"{n_no_cog} forms lacked cognate reconstruction (dropped)")
    notes = "; ".join(notes_bits) if notes_bits else ""

    return {
        "role": "cultural_anchor",
        "parameter_id": param_id,
        "diacl_meaning": param_id,
        "n": int(n),
        "n_retention_1": n_retention_1,
        "n_cognate_classes": n_classes,
        "retention_rate": round(retention_rate, 4) if np.isfinite(retention_rate) else None,
        "spearman_r": round(rho, 4) if np.isfinite(rho) else None,
        "spearman_p": round(pval, 6) if np.isfinite(pval) else None,
        "ci_lower": round(ci_lower, 4) if np.isfinite(ci_lower) else None,
        "ci_upper": round(ci_upper, 4) if np.isfinite(ci_upper) else None,
        "proto_form_cognate_id": mode_cognate,
        "n_bootstrap_usable": len(rhos),
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 72)
    print("IE cultural-vocabulary migration gradient (DIACL)")
    print("=" * 72)

    rng = np.random.default_rng(SEED)

    langs, params, forms, etym = load_diacl()
    ie_lang_ids = set(langs[langs["Family"] == "Indo-European"]["ID"])
    n_ie_total = len(ie_lang_ids)
    n_ie_geo = int(
        langs[langs["Family"] == "Indo-European"][["Latitude", "Longitude", "Glottocode"]]
        .notna()
        .all(axis=1)
        .sum()
    )
    print(
        f"DIACL loaded: languages_total={len(langs)} IE={n_ie_total} "
        f"IE_with_geo={n_ie_geo} forms={len(forms)} etymology_entries={len(etym)}"
    )

    concepts_out: dict[str, dict] = {}
    print()
    print(
        f"{'Concept':<10} {'n':>4} {'classes':>7} {'ret%':>6} "
        f"{'rho':>8} {'p':>10} {'CI95':>22}"
    )
    print("-" * 72)
    for label, pid in TARGET_CONCEPTS.items():
        res = analyze_concept(
            concept_label=label,
            param_id=pid,
            langs=langs,
            forms=forms,
            etym=etym,
            ie_lang_ids=ie_lang_ids,
            rng=rng,
        )
        concepts_out[label] = res
        ret_pct = (res["retention_rate"] * 100) if res["retention_rate"] is not None else float("nan")
        rho = res["spearman_r"] if res["spearman_r"] is not None else float("nan")
        pval = res["spearman_p"] if res["spearman_p"] is not None else float("nan")
        ci_lo = res["ci_lower"] if res["ci_lower"] is not None else float("nan")
        ci_hi = res["ci_upper"] if res["ci_upper"] is not None else float("nan")
        print(
            f"{label:<10} {res['n']:>4} {res['n_cognate_classes']:>7} "
            f"{ret_pct:>6.1f} {rho:>+8.3f} {pval:>10.4g} "
            f"[{ci_lo:>+6.3f},{ci_hi:>+6.3f}]"
        )

    payload = {
        "family": "Indo-European",
        "homeland": {
            "name": HOMELAND_NAME,
            "lat": HOMELAND_LAT,
            "lon": HOMELAND_LON,
        },
        "dataset": "DIACL CLDF (lexibank/diacl, derived from Carling 2017)",
        "cognacy_source": (
            "raw/etymology.json.gz parent->child lexeme tree; cognate classes "
            "reconstructed by walking each lexeme to its root within the "
            "etymology entry that references it."
        ),
        "n_ie_languages_in_dataset": n_ie_total,
        "concepts": concepts_out,
        "method": {
            "dedup": "by Glottocode, majority-vote cognate class",
            "retention_definition": "cognate class equals global mode for concept",
            "distance": "haversine km from homeland",
            "bootstrap": {"n": N_BOOTSTRAP, "seed": SEED, "method": "percentile"},
            "loans_excluded": False,
            "loans_note": (
                "DIACL forms.csv Loan column is empty; loan flags are not "
                "available in the CLDF derivation. Reported values therefore "
                "include any loan-influenced forms."
            ),
        },
    }

    with OUT_JSON.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print()
    print(f"Wrote {OUT_JSON}")

    # Prediction reminder (not a pass/fail gate): Anthony (2007) predicts that
    # HORSE/WHEEL/YOKE/PLOUGH/AXLE should show a positive rho with distance
    # from the Pontic-Caspian homeland (peripheral lineages drift away from
    # the steppe-pastoralist proto-form faster than core ones), consistent
    # with the RIVER/FOREST/DOG anchors in xfam_ie_migration_gradient.py.
    print()
    print("Prediction (Anthony 2007, steppe hypothesis): rho > 0 for each.")


if __name__ == "__main__":
    main()
