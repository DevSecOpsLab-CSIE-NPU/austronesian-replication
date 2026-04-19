#!/usr/bin/env python3
"""Cross-family migration gradient — Uralic (UraLex), 5th-family test.

Pre-registered fifth-family replication of Paper 2's expansion-corridor
anchor hypothesis. The prediction is written BEFORE looking at the
gradient data; see ``paper-2-5th-family-preregistration.md`` at the repo
root for the ex-ante hypothesis.

For each concept we compute the Spearman correlation between

    retention(language) = 1 iff the language's majority-vote cognate class
                          equals the global modal cognate class
                          (proto-form proxy) for this concept,
    dist_from_homeland  = Haversine km from the Volga-Kama homeland
                          (58.0 N, 55.0 E).

A 500-sample bootstrap percentile 95 percent CI is computed with
``np.random.default_rng(42)``.

Pre-registered anchor concepts (positive gradient predicted):
    FOREST (primary), SNOW, ICE, TREE.

Universal controls (weak / inconsistent gradient expected):
    FIRE, WATER, STONE, RIVER, SEA, MOUNTAIN, COLD.

Dataset: ``data/raw/uralex/cldf/`` (lexibank/uralex, UraLex basic
vocabulary). Forms whose ``ID`` appears in ``borrowings.csv::Target_Form_ID``
are excluded as loans before cognate assignment.

Output: ``results/xfam_5th_uralic_migration_gradient.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Path setup (follow CLAUDE.md convention)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

URALEX_DIR = ROOT / "data" / "raw" / "uralex" / "cldf"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUT_JSON = RESULTS_DIR / "xfam_5th_uralic_migration_gradient.json"

# Pre-registered Uralic homeland: middle Volga / Kama region
# (Grunthal et al. 2022; Honkola et al. 2013).
HOMELAND_NAME = "Middle Volga / Kama (Uralic Urheimat)"
HOMELAND_LAT = 58.0
HOMELAND_LON = 55.0

# Pre-registered concept slate (written BEFORE running; see preregistration
# markdown). Anchors predicted positive; universal controls predicted weak.
PRE_REGISTERED_ANCHORS = ["FOREST", "SNOW", "ICE", "TREE"]
UNIVERSAL_CONTROLS = ["FIRE", "WATER", "STONE", "RIVER", "SEA", "MOUNTAIN", "COLD"]
# Note: TREE sits in both lists conceptually (boreal anchor + universal
# control). We classify it as "anchor" here per the pre-registration and
# report the overlap explicitly. The decision rule only cares whether the
# empirical top-rho concept is in PRE_REGISTERED_ANCHORS.

# Concepticon_Gloss -> desired column label mapping for this run.
# Confirmed present in uralex/cldf/parameters.csv.
CONCEPT_MAP = {
    "FOREST": "FOREST",   # 'woods'
    "SNOW": "SNOW",
    "ICE": "ICE",
    "TREE": "TREE",
    "FIRE": "FIRE",
    "WATER": "WATER",
    "STONE": "STONE",
    "RIVER": "RIVER",
    "SEA": "SEA",
    "MOUNTAIN": "MOUNTAIN",
    "COLD": "COLD",
}

N_BOOTSTRAP = 500
SEED = 42


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def haversine_km(lat1: float, lon1: float, lat2, lon2) -> np.ndarray:
    """Great-circle distance in km. Vectorised over lat2/lon2."""
    R = 6371.0088  # mean Earth radius
    lat1r = np.radians(lat1)
    lon1r = np.radians(lon1)
    lat2r = np.radians(np.asarray(lat2, dtype=float))
    lon2r = np.radians(np.asarray(lon2, dtype=float))
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_uralex() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, set]:
    """Load languages, parameters, forms, cognates; return loan Form_IDs."""
    langs = pd.read_csv(URALEX_DIR / "languages.csv")
    params = pd.read_csv(URALEX_DIR / "parameters.csv")
    forms = pd.read_csv(URALEX_DIR / "forms.csv", low_memory=False)
    cogs = pd.read_csv(URALEX_DIR / "cognates.csv", low_memory=False)
    borrowings = pd.read_csv(URALEX_DIR / "borrowings.csv")
    loan_form_ids = set(borrowings["Target_Form_ID"].dropna().astype(str).tolist())
    return langs, params, forms, cogs, loan_form_ids


# ---------------------------------------------------------------------------
# Core analysis for one concept
# ---------------------------------------------------------------------------


def analyze_concept(
    concept_label: str,
    concepticon_gloss: str,
    role: str,
    langs: pd.DataFrame,
    params: pd.DataFrame,
    forms: pd.DataFrame,
    cogs: pd.DataFrame,
    loan_form_ids: set,
    rng: np.random.Generator,
) -> dict:
    """Run the migration-gradient pipeline for one concept. Returns a dict."""

    # 1. Resolve parameter ID via Concepticon_Gloss (exact, upper-case match)
    pmatch = params[
        params["Concepticon_Gloss"].astype(str).str.upper() == concepticon_gloss.upper()
    ]
    if pmatch.empty:
        return {
            "role": role,
            "parameter_id": None,
            "concepticon_gloss": concepticon_gloss,
            "n": 0,
            "available": False,
            "n_retention_1": 0,
            "retention_rate": None,
            "spearman_r": None,
            "spearman_p": None,
            "ci_lower": None,
            "ci_upper": None,
            "proto_form_cognate_id": None,
            "n_bootstrap_usable": 0,
            "notes": "concept absent in uralex parameters.csv",
        }
    # parameters.csv ID column is int in uralex
    param_id = pmatch.iloc[0]["ID"]

    # 2. Collect forms for this parameter
    cforms = forms[forms["Parameter_ID"] == param_id].copy()
    n_forms_total = len(cforms)

    # 3. Exclude loan forms (form-level borrowings.csv -> Target_Form_ID)
    n_before_loan = len(cforms)
    cforms = cforms[~cforms["ID"].astype(str).isin(loan_form_ids)].copy()
    n_loans_excluded = n_before_loan - len(cforms)

    # 4. Join with cognates on Form_ID (left-join to detect orphan forms)
    cog_slim = cogs[["Form_ID", "Cognateset_ID", "Doubt"]].drop_duplicates(
        subset=["Form_ID"]
    )
    merged = cforms.merge(cog_slim, left_on="ID", right_on="Form_ID", how="left")
    n_no_cog = int(merged["Cognateset_ID"].isna().sum())
    if n_forms_total > 0 and n_no_cog / n_forms_total > 0.05:
        # Document rather than abort: uralex has expert cognate coverage
        # but Proto-Uralic reconstructions are frequently unassigned.
        pass
    merged = merged.dropna(subset=["Cognateset_ID"]).copy()
    # Optional: drop doubtful cognate assignments if any (uralex has none,
    # but keep the guard in case of future releases).
    if "Doubt" in merged.columns:
        merged = merged[merged["Doubt"].astype(str).str.lower() != "true"].copy()

    # 5. Attach language metadata (Glottocode, lat, lon).
    #    uralex languages.csv: Language_ID in forms.csv corresponds to
    #    languages.csv::ID (integer ID like 101, 102, ...).
    merged = merged.merge(
        langs[["ID", "Glottocode", "Latitude", "Longitude", "Name"]],
        left_on="Language_ID",
        right_on="ID",
        suffixes=("", "_lang"),
        how="left",
    )
    merged = merged.dropna(subset=["Glottocode", "Latitude", "Longitude"]).copy()

    # 6. Deduplicate per Glottocode: majority vote of Cognateset_ID (string)
    def majority_cognate(series: pd.Series) -> str:
        vc = series.value_counts()
        top_count = vc.iloc[0]
        tied = sorted(vc[vc == top_count].index.tolist())
        return tied[0]  # deterministic tie-break

    by_gc = (
        merged.groupby("Glottocode")
        .agg(
            cognate=("Cognateset_ID", majority_cognate),
            latitude=("Latitude", "first"),
            longitude=("Longitude", "first"),
        )
        .reset_index()
    )

    n = len(by_gc)

    if n == 0:
        return {
            "role": role,
            "parameter_id": int(param_id) if pd.notna(param_id) else None,
            "concepticon_gloss": concepticon_gloss,
            "n": 0,
            "available": True,
            "n_retention_1": 0,
            "retention_rate": None,
            "spearman_r": None,
            "spearman_p": None,
            "ci_lower": None,
            "ci_upper": None,
            "proto_form_cognate_id": None,
            "n_bootstrap_usable": 0,
            "notes": "zero languages after filtering",
        }

    # 7. Global mode cognate = proto-form proxy
    mode_cognate = by_gc["cognate"].value_counts().index[0]
    retention = (by_gc["cognate"].values == mode_cognate).astype(int)
    n_retention_1 = int(retention.sum())
    retention_rate = float(n_retention_1 / n)

    # 8. Homeland distance
    dist = haversine_km(
        HOMELAND_LAT, HOMELAND_LON,
        by_gc["latitude"].values, by_gc["longitude"].values,
    )

    # 9. Spearman rho
    if n >= 3 and 0 < retention.sum() < n:
        rho, pval = spearmanr(dist, retention)
        rho = float(rho)
        pval = float(pval)
    else:
        rho = float("nan")
        pval = float("nan")

    # 10. Bootstrap 95 percent CI (percentile)
    rhos = []
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

    notes_bits = []
    if n_loans_excluded:
        notes_bits.append(f"excluded {n_loans_excluded} loan forms")
    if n_no_cog:
        notes_bits.append(f"{n_no_cog} forms lacked cognate assignment (dropped)")
    notes = "; ".join(notes_bits) if notes_bits else ""

    return {
        "role": role,
        "parameter_id": int(param_id) if pd.notna(param_id) else None,
        "concepticon_gloss": concepticon_gloss,
        "available": True,
        "n": int(n),
        "n_retention_1": n_retention_1,
        "retention_rate": round(retention_rate, 4),
        "spearman_r": round(rho, 4) if np.isfinite(rho) else None,
        "spearman_p": round(pval, 6) if np.isfinite(pval) else None,
        "ci_lower": round(ci_lower, 4) if np.isfinite(ci_lower) else None,
        "ci_upper": round(ci_upper, 4) if np.isfinite(ci_upper) else None,
        "proto_form_cognate_id": str(mode_cognate),
        "n_bootstrap_usable": len(rhos),
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 72)
    print("5th-Family Migration Gradient Test — Uralic (UraLex)")
    print("Pre-registered (see paper-2-5th-family-preregistration.md)")
    print("=" * 72)

    rng = np.random.default_rng(SEED)

    langs, params, forms, cogs, loan_form_ids = load_uralex()
    print(
        f"uralex loaded: languages={len(langs)} parameters={len(params)} "
        f"forms={len(forms)} cognates={len(cogs)} "
        f"loan_form_ids={len(loan_form_ids)}"
    )
    n_languages_total = int(
        langs.dropna(subset=["Glottocode", "Latitude", "Longitude"]).shape[0]
    )
    print(f"Languages with Glottocode + geo: {n_languages_total}")
    if n_languages_total < 25:
        raise RuntimeError(
            f"Sanity check failed: only {n_languages_total} languages with "
            f"Glottocode + geo; pre-registration required n>=25."
        )

    # Concept availability map (report even for absent ones)
    concept_availability = {}
    for c in PRE_REGISTERED_ANCHORS + UNIVERSAL_CONTROLS:
        hit = params[params["Concepticon_Gloss"].astype(str).str.upper() == c.upper()]
        concept_availability[c] = bool(not hit.empty)
    print()
    print("Concept availability vs pre-registration:")
    for c, ok in concept_availability.items():
        status = "PRESENT" if ok else "ABSENT"
        role = "anchor" if c in PRE_REGISTERED_ANCHORS else "control"
        print(f"  [{role:7}] {c:10} {status}")
    print()

    print(
        f"{'Concept':<10} {'Role':<8} {'n':>4} {'ret%':>6} {'rho':>8} "
        f"{'p':>10} {'CI95':>22}"
    )
    print("-" * 72)

    concepts_out: dict[str, dict] = {}

    # Run anchors first, then controls (fixed order)
    ordered = [(c, "anchor") for c in PRE_REGISTERED_ANCHORS] + \
              [(c, "control") for c in UNIVERSAL_CONTROLS]

    for label, role in ordered:
        res = analyze_concept(
            concept_label=label,
            concepticon_gloss=CONCEPT_MAP[label],
            role=role,
            langs=langs,
            params=params,
            forms=forms,
            cogs=cogs,
            loan_form_ids=loan_form_ids,
            rng=rng,
        )
        concepts_out[label] = res
        ret = res["retention_rate"]
        rho = res["spearman_r"]
        pval = res["spearman_p"]
        ci_lo = res["ci_lower"]
        ci_hi = res["ci_upper"]
        ret_pct = (ret * 100) if ret is not None else float("nan")
        rho_f = rho if rho is not None else float("nan")
        pval_f = pval if pval is not None else float("nan")
        ci_lo_f = ci_lo if ci_lo is not None else float("nan")
        ci_hi_f = ci_hi if ci_hi is not None else float("nan")
        print(
            f"{label:<10} {role:<8} {res['n']:>4} {ret_pct:>6.1f} "
            f"{rho_f:>+8.3f} {pval_f:>10.4g} "
            f"[{ci_lo_f:>+6.3f},{ci_hi_f:>+6.3f}]"
        )
    print()

    # Pre-registration verdict
    # Empirical top anchor = concept with max positive spearman_r over all
    # concepts that have a defined rho.
    scored = [
        (lab, r["spearman_r"]) for lab, r in concepts_out.items()
        if r["spearman_r"] is not None and np.isfinite(r["spearman_r"])
    ]
    scored.sort(key=lambda kv: kv[1], reverse=True)
    empirical_anchor = scored[0][0] if scored else None
    prediction_matched = bool(
        empirical_anchor is not None
        and empirical_anchor in PRE_REGISTERED_ANCHORS
    )

    # Interpretation sentence
    if empirical_anchor is None:
        interpretation = (
            "No concept produced a defined Spearman rho; test uninformative."
        )
    else:
        top_rho = concepts_out[empirical_anchor]["spearman_r"]
        top_ci_lo = concepts_out[empirical_anchor]["ci_lower"]
        top_ci_hi = concepts_out[empirical_anchor]["ci_upper"]
        ci_txt = (
            f"CI=[{top_ci_lo:+.3f},{top_ci_hi:+.3f}]"
            if top_ci_lo is not None and top_ci_hi is not None
            else "CI=n/a"
        )
        if prediction_matched:
            interpretation = (
                f"Pre-registered anchor '{empirical_anchor}' is the top "
                f"positive gradient (rho={top_rho:+.3f}, {ci_txt}). "
                f"Fifth-family prediction confirmed without post-hoc "
                f"concept selection."
            )
        else:
            interpretation = (
                f"Top positive gradient is universal control "
                f"'{empirical_anchor}' (rho={top_rho:+.3f}, {ci_txt}), "
                f"not a pre-registered anchor. Prediction NOT confirmed — "
                f"report as a null result that tempers the cross-family "
                f"anchor claim."
            )

    print(f"Top positive-rho concept: {empirical_anchor}")
    print(f"Prediction matched (top rho in pre-registered anchors): "
          f"{prediction_matched}")
    print(f"Interpretation: {interpretation}")
    print()

    payload = {
        "family": "Uralic",
        "dataset": "lexibank/uralex (UraLex basic vocabulary)",
        "homeland": {
            "name": HOMELAND_NAME,
            "lat": HOMELAND_LAT,
            "lon": HOMELAND_LON,
        },
        "n_languages_total": n_languages_total,
        "pre_registered_anchors": PRE_REGISTERED_ANCHORS,
        "universal_controls": UNIVERSAL_CONTROLS,
        "concept_availability": concept_availability,
        "concepts": concepts_out,
        "empirical_anchor": empirical_anchor,
        "prediction_matched": prediction_matched,
        "interpretation": interpretation,
        "method": {
            "dedup": "by Glottocode, majority-vote cognate class",
            "retention_definition": (
                "cognate class equals global mode for concept (proto-form "
                "proxy)"
            ),
            "distance": "Haversine km from Volga-Kama homeland (58 N, 55 E)",
            "bootstrap": {"n": N_BOOTSTRAP, "seed": SEED, "method": "percentile"},
            "loans_excluded": True,
            "loans_exclusion_note": (
                "uralex forms.csv Loan column is empty; loans are flagged at "
                "the form level in borrowings.csv via Target_Form_ID. Forms "
                "whose ID appears there are excluded before cognate counting."
            ),
        },
    }

    with OUT_JSON.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
