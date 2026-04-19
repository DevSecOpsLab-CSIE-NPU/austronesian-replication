#!/usr/bin/env python3
"""Cross-family migration gradient — Turkic (Savelyev & Robbeets 2020), 6th-family test.

Pre-registered sixth-family replication of Paper 2's expansion-corridor
anchor hypothesis (issue #86). The prediction is written BEFORE looking
at the gradient data; see ``paper-2-6th-family-preregistration.md`` at
the repo root for the ex-ante hypothesis.

For each concept we compute the Spearman correlation between

    retention(language) = 1 iff the language's majority-vote cognate class
                          equals the global modal cognate class
                          (proto-form proxy) for this concept,
    dist_from_homeland  = Haversine km from the Altai / north-Mongolian
                          Proto-Turkic homeland (48.0 N, 100.0 E).

A 500-sample bootstrap percentile 95 percent CI is computed with
``np.random.default_rng(42)``.

Pre-registered primary anchors (Turkic pastoralist package):
    HORSE, SHEEP, STEPPE, FELT.
All four are absent from the savelyevturkic concept slate; see
preregistration for the gap statement.

Pre-registered fallback anchors (best available proxies, registered
BEFORE running):
    DOG, MEAT, MOUNTAIN OR HILL, HORN (ANATOMY).

Universal controls (weak / inconsistent gradient expected):
    TREE, FIRE, WATER, STONE, LEFT, RIGHT, RIVER, SEA, SNOW, COLD.

Dataset: ``data/raw/savelyevturkic/cldf/`` (lexibank/savelyevturkic,
Savelyev and Robbeets 2020, 32 Turkic varieties, 254 Swadesh-style
parameters, expert cognate classification). The Loan column in forms.csv
is fully NaN for this dataset; the authors' expert Cognateset_ID
classification implicitly separates borrowings from inherited cognates
(see README). No separate borrowings.csv is provided. We document this
and do not synthesise a loan filter post-hoc.

Output: ``results/xfam_6th_turkic_migration_gradient.json``.
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

TURKIC_DIR = ROOT / "data" / "raw" / "savelyevturkic" / "cldf"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUT_JSON = RESULTS_DIR / "xfam_6th_turkic_migration_gradient.json"

# Pre-registered Proto-Turkic homeland: Altai / north-Mongolian steppe
# (Savelyev & Robbeets 2020; Dybo 2007; Janhunen 1996).
HOMELAND_NAME = "Altai / north-Mongolian steppe (Proto-Turkic Urheimat)"
HOMELAND_LAT = 48.0
HOMELAND_LON = 100.0

# Pre-registered concept slates — written BEFORE running. Primary anchors
# are the theoretical pastoralist package; all four are absent from this
# dataset's concept slate, so fallback anchors are used for the decision
# rule. Both lists were fixed in paper-2-6th-family-preregistration.md.
PRIMARY_ANCHORS = ["HORSE", "SHEEP", "STEPPE", "FELT"]
FALLBACK_ANCHORS = ["DOG", "MEAT", "MOUNTAIN OR HILL", "HORN (ANATOMY)"]
UNIVERSAL_CONTROLS = [
    "TREE", "FIRE", "WATER", "STONE",
    "LEFT", "RIGHT", "RIVER", "SEA", "SNOW", "COLD",
]

# Column labels (display) -> Concepticon_Gloss lookup. All concepts
# are queried by exact upper-cased match against parameters.csv's
# Concepticon_Gloss column.
CONCEPT_MAP: dict[str, str] = {
    # primary anchors (all expected absent)
    "HORSE": "HORSE",
    "SHEEP": "SHEEP",
    "STEPPE": "STEPPE",
    "FELT": "FELT",
    # fallback anchors (all confirmed present before running)
    "DOG": "DOG",
    "MEAT": "MEAT",
    "MOUNTAIN OR HILL": "MOUNTAIN OR HILL",
    "HORN (ANATOMY)": "HORN (ANATOMY)",
    # universal controls
    "TREE": "TREE",
    "FIRE": "FIRE",
    "WATER": "WATER",
    "STONE": "STONE",
    "LEFT": "LEFT",
    "RIGHT": "RIGHT",
    "RIVER": "RIVER",
    "SEA": "SEA",
    "SNOW": "SNOW",
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


def load_turkic() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, set]:
    """Load languages, parameters, forms, cognates; return loan Form_IDs."""
    langs = pd.read_csv(TURKIC_DIR / "languages.csv")
    params = pd.read_csv(TURKIC_DIR / "parameters.csv")
    forms = pd.read_csv(TURKIC_DIR / "forms.csv", low_memory=False)
    cogs = pd.read_csv(TURKIC_DIR / "cognates.csv", low_memory=False)
    # savelyevturkic has no borrowings.csv. forms.csv Loan column is
    # documented as fully NaN for this dataset; the authors' expert
    # Cognateset_ID classification is the loan-handling channel. We
    # collect any form IDs that are positively flagged as loans (in
    # case of future dataset revisions) but expect this to be empty.
    loan_form_ids: set = set()
    if "Loan" in forms.columns:
        loan_mask = forms["Loan"].fillna(False).astype(bool)
        loan_form_ids = set(forms.loc[loan_mask, "ID"].astype(str).tolist())
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
            "notes": "concept absent in savelyevturkic parameters.csv",
        }
    param_id = pmatch.iloc[0]["ID"]

    # 2. Collect forms for this parameter
    cforms = forms[forms["Parameter_ID"] == param_id].copy()
    n_forms_total = len(cforms)

    # 3. Exclude loan forms (form-level Loan flag; empty for this dataset)
    n_before_loan = len(cforms)
    cforms = cforms[~cforms["ID"].astype(str).isin(loan_form_ids)].copy()
    n_loans_excluded = n_before_loan - len(cforms)

    # 4. Join with cognates on Form_ID
    cog_slim = cogs[["Form_ID", "Cognateset_ID", "Doubt"]].drop_duplicates(
        subset=["Form_ID"]
    )
    merged = cforms.merge(cog_slim, left_on="ID", right_on="Form_ID", how="left")
    n_no_cog = int(merged["Cognateset_ID"].isna().sum())
    merged = merged.dropna(subset=["Cognateset_ID"]).copy()
    if "Doubt" in merged.columns:
        merged = merged[merged["Doubt"].astype(str).str.lower() != "true"].copy()

    # 5. Attach language metadata (Glottocode, lat, lon).
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
            "parameter_id": str(param_id) if pd.notna(param_id) else None,
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
        "parameter_id": str(param_id) if pd.notna(param_id) else None,
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
    print("6th-Family Migration Gradient Test — Turkic (Savelyev & Robbeets 2020)")
    print("Pre-registered (see paper-2-6th-family-preregistration.md, issue #86)")
    print("=" * 72)

    rng = np.random.default_rng(SEED)

    langs, params, forms, cogs, loan_form_ids = load_turkic()
    print(
        f"savelyevturkic loaded: languages={len(langs)} parameters={len(params)} "
        f"forms={len(forms)} cognates={len(cogs)} "
        f"loan_form_ids={len(loan_form_ids)}"
    )
    n_languages_total = int(
        langs.dropna(subset=["Glottocode", "Latitude", "Longitude"]).shape[0]
    )
    print(f"Languages with Glottocode + geo: {n_languages_total}")
    if n_languages_total < 20:
        raise RuntimeError(
            f"Sanity check failed: only {n_languages_total} languages with "
            f"Glottocode + geo; below minimum viable floor (20)."
        )
    if n_languages_total < 25:
        print(
            f"NOTE: n={n_languages_total} is below the n>=25 Uralic-era "
            f"pre-registration floor. This gap is documented in the "
            f"sixth-family pre-registration and is not a post-hoc revision."
        )

    # Concept availability map (report even for absent ones)
    all_concepts = PRIMARY_ANCHORS + FALLBACK_ANCHORS + UNIVERSAL_CONTROLS
    concept_availability = {}
    for c in all_concepts:
        hit = params[params["Concepticon_Gloss"].astype(str).str.upper() == c.upper()]
        concept_availability[c] = bool(not hit.empty)
    print()
    print("Concept availability vs pre-registration:")
    for c in all_concepts:
        ok = concept_availability[c]
        status = "PRESENT" if ok else "ABSENT"
        if c in PRIMARY_ANCHORS:
            role = "anchor_primary"
        elif c in FALLBACK_ANCHORS:
            role = "anchor_fallback"
        else:
            role = "control"
        print(f"  [{role:16}] {c:20} {status}")
    print()

    print(
        f"{'Concept':<20} {'Role':<16} {'n':>4} {'ret%':>6} {'rho':>8} "
        f"{'p':>10} {'CI95':>22}"
    )
    print("-" * 96)

    concepts_out: dict[str, dict] = {}

    # Run primary anchors first, then fallback anchors, then controls
    ordered = (
        [(c, "anchor_primary") for c in PRIMARY_ANCHORS]
        + [(c, "anchor_fallback") for c in FALLBACK_ANCHORS]
        + [(c, "control") for c in UNIVERSAL_CONTROLS]
    )

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
            f"{label:<20} {role:<16} {res['n']:>4} {ret_pct:>6.1f} "
            f"{rho_f:>+8.3f} {pval_f:>10.4g} "
            f"[{ci_lo_f:>+6.3f},{ci_hi_f:>+6.3f}]"
        )
    print()

    # ---------------------------------------------------------------
    # Pre-registration decision rule:
    # supported    = top-rho in fallback/primary anchors AND CI disjoint
    #                from second-best concept's CI
    # partial      = top-rho in anchors but CI overlaps next concept's CI
    # registered_null = top-rho is a universal control
    # untestable   = fewer than 3 defined rho values
    # ---------------------------------------------------------------
    scored = [
        (lab, r["spearman_r"], r["ci_lower"], r["ci_upper"])
        for lab, r in concepts_out.items()
        if r["spearman_r"] is not None and np.isfinite(r["spearman_r"])
    ]
    scored.sort(key=lambda kv: kv[1], reverse=True)

    anchor_set = set(PRIMARY_ANCHORS + FALLBACK_ANCHORS)

    if len(scored) < 3:
        verdict = "untestable"
        empirical_anchor = scored[0][0] if scored else None
        interpretation = (
            f"Fewer than three concepts produced a defined Spearman rho "
            f"(got {len(scored)}); sixth-family test uninformative."
        )
    else:
        top_label, top_rho, top_lo, top_hi = scored[0]
        second_label, second_rho, second_lo, second_hi = scored[1]
        empirical_anchor = top_label

        ci_disjoint = (
            top_lo is not None and second_hi is not None
            and top_lo > second_hi
        ) or (
            top_hi is not None and second_lo is not None
            and top_hi < second_lo
        )

        if top_label in anchor_set:
            if ci_disjoint:
                verdict = "supported"
                interpretation = (
                    f"Pre-registered anchor '{top_label}' is the top positive "
                    f"gradient (rho={top_rho:+.3f}, CI=[{top_lo:+.3f},"
                    f"{top_hi:+.3f}]) and its 95% CI is disjoint from the "
                    f"second-best concept '{second_label}' "
                    f"(rho={second_rho:+.3f}, CI=[{second_lo:+.3f},"
                    f"{second_hi:+.3f}]). Sixth-family prediction "
                    f"confirmed without post-hoc concept selection."
                )
            else:
                verdict = "partial"
                interpretation = (
                    f"Pre-registered anchor '{top_label}' is the top positive "
                    f"gradient (rho={top_rho:+.3f}, CI=[{top_lo:+.3f},"
                    f"{top_hi:+.3f}]), but its 95% CI overlaps the "
                    f"second-best concept '{second_label}' "
                    f"(rho={second_rho:+.3f}, CI=[{second_lo:+.3f},"
                    f"{second_hi:+.3f}]). Direction consistent with the "
                    f"anchor prediction but CI-separation criterion not met; "
                    f"report as partial support."
                )
        else:
            verdict = "registered_null"
            interpretation = (
                f"Top positive gradient is universal control '{top_label}' "
                f"(rho={top_rho:+.3f}, CI=[{top_lo:+.3f},{top_hi:+.3f}]), "
                f"not a pre-registered anchor. Sixth-family prediction NOT "
                f"confirmed. Report as registered null that tempers the "
                f"cross-family anchor claim and adds to the Uralic null."
            )

    prediction_matched = verdict == "supported"

    print(f"Top positive-rho concept: {empirical_anchor}")
    print(f"Verdict: {verdict}")
    print(f"Interpretation: {interpretation}")
    print()

    payload = {
        "family": "Turkic",
        "dataset": "lexibank/savelyevturkic (Savelyev & Robbeets 2020)",
        "homeland": {
            "name": HOMELAND_NAME,
            "lat": HOMELAND_LAT,
            "lon": HOMELAND_LON,
        },
        "n_languages_total": n_languages_total,
        "primary_anchors_pre_registered": PRIMARY_ANCHORS,
        "fallback_anchors_pre_registered": FALLBACK_ANCHORS,
        "universal_controls": UNIVERSAL_CONTROLS,
        "concept_availability": concept_availability,
        "concepts": concepts_out,
        "empirical_anchor": empirical_anchor,
        "verdict": verdict,
        "prediction_matched": prediction_matched,
        "interpretation": interpretation,
        "method": {
            "dedup": "by Glottocode, majority-vote cognate class",
            "retention_definition": (
                "cognate class equals global mode for concept "
                "(proto-form proxy)"
            ),
            "distance": f"Haversine km from Altai homeland ({HOMELAND_LAT} N, {HOMELAND_LON} E)",
            "bootstrap": {"n": N_BOOTSTRAP, "seed": SEED, "method": "percentile"},
            "loans_excluded": False,
            "loans_exclusion_note": (
                "savelyevturkic has no borrowings.csv; forms.csv Loan "
                "column is fully NaN. The authors' expert Cognateset_ID "
                "classification is the loan-handling channel for this "
                "dataset. No post-hoc loan filter was applied."
            ),
            "ci_disjointness_criterion": (
                "pre-registered: top-rho CI must be disjoint from "
                "second-best concept's CI for 'supported' verdict"
            ),
        },
    }

    with OUT_JSON.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
