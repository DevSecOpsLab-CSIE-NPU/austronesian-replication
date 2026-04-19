#!/usr/bin/env python3
"""Cross-family migration gradient — Indo-European (IECoR).

Parallel pipeline to the Austronesian and Sino-Tibetan migration-gradient
analyses. For each concept we compute the Spearman correlation between

    retention(language) = 1 iff language's cognate class equals the global
                          mode for this concept,
    dist_from_homeland  = Haversine km from the Pontic-Caspian steppe
                          (48.0°N, 45.0°E),

and bootstrap a 95% percentile CI.

Anchors (predicted positive gradient): RIVER, FOREST, DOG, SALT.
Controls (predicted near-zero or negative): SEA, TREE, LEFT, RIGHT, MOUNTAIN.

Inputs
------
- data/raw/iecor/cldf/{languages,parameters,forms,cognates,loans}.csv
- data/external/anchor_concepts.csv

Output
------
- results/xfam_ie_migration_gradient.json

Usage
-----
    .venv/bin/python3 scripts/xfam_ie_migration_gradient.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Path setup — follow repo convention
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

IECOR_DIR = ROOT / "data" / "raw" / "iecor" / "cldf"
ANCHOR_CSV = ROOT / "data" / "external" / "anchor_concepts.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUT_JSON = RESULTS_DIR / "xfam_ie_migration_gradient.json"

# Pontic-Caspian steppe (Yamnaya core; Anthony 2007; Heggarty 2023 consensus)
HOMELAND_NAME = "Pontic-Caspian steppe"
HOMELAND_LAT = 48.0
HOMELAND_LON = 45.0

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


def load_iecor() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, set[int]]:
    """Load languages, parameters, forms, cognates; return loan cognateset IDs."""
    langs = pd.read_csv(IECOR_DIR / "languages.csv")
    params = pd.read_csv(IECOR_DIR / "parameters.csv")
    forms = pd.read_csv(IECOR_DIR / "forms.csv", low_memory=False)
    cogs = pd.read_csv(IECOR_DIR / "cognates.csv", low_memory=False)
    loans = pd.read_csv(IECOR_DIR / "loans.csv")
    loan_cogset_ids = set(loans["Cognateset_ID"].dropna().astype(int).tolist())
    return langs, params, forms, cogs, loan_cogset_ids


def load_anchor_slate() -> pd.DataFrame:
    """Rows of anchor_concepts.csv for Indo-European, non-excluded."""
    df = pd.read_csv(ANCHOR_CSV)
    df = df[(df["family"] == "Indo-European") & (df["role"] != "excluded")].copy()
    df = df.reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Core analysis for one concept
# ---------------------------------------------------------------------------


def analyze_concept(
    concept_label: str,
    iecor_meaning: str,
    role: str,
    langs: pd.DataFrame,
    params: pd.DataFrame,
    forms: pd.DataFrame,
    cogs: pd.DataFrame,
    loan_cogset_ids: set[int],
    rng: np.random.Generator,
) -> dict:
    """Run the migration-gradient pipeline for one concept. Never returns None;
    a fatal mismatch raises so the caller can STOP and report (per task spec)."""

    # 1. Resolve parameter ID
    pmatch = params[params["Name"].astype(str).str.lower() == iecor_meaning.lower()]
    if pmatch.empty:
        raise RuntimeError(
            f"[{concept_label}] No Parameter found with Name == '{iecor_meaning}' "
            f"in IECoR parameters.csv — aborting (do not silently skip)."
        )
    param_id = int(pmatch.iloc[0]["ID"])

    # 2. Collect forms
    cforms = forms[forms["Parameter_ID"] == param_id].copy()
    n_forms_total = len(cforms)

    # 3. Join with cognates on Form_ID (left-join to detect orphan forms)
    cog_slim = cogs[["Form_ID", "Cognateset_ID"]].drop_duplicates(subset=["Form_ID"])
    merged = cforms.merge(cog_slim, left_on="ID", right_on="Form_ID", how="left")
    n_no_cog = int(merged["Cognateset_ID"].isna().sum())
    if n_forms_total > 0 and n_no_cog / n_forms_total > 0.05:
        raise RuntimeError(
            f"[{concept_label}] {n_no_cog}/{n_forms_total} forms lack a "
            f"cognate assignment (>5%) — aborting per data-integrity guard."
        )
    merged = merged.dropna(subset=["Cognateset_ID"]).copy()
    merged["Cognateset_ID"] = merged["Cognateset_ID"].astype(int)

    # 4. Exclude loans.
    #    IECoR forms.csv Loan column is all NaN (not populated at the form
    #    level); loan information lives in loans.csv, which flags entire
    #    cognatesets as borrowed. We drop forms whose cognateset appears
    #    in loans.csv. This is the closest loan exclusion possible with
    #    IECoR's schema.
    n_before_loan = len(merged)
    merged = merged[~merged["Cognateset_ID"].isin(loan_cogset_ids)].copy()
    n_loans_excluded = n_before_loan - len(merged)

    # 5. Attach language metadata (Glottocode, lat, lon)
    merged = merged.merge(
        langs[["ID", "Glottocode", "Latitude", "Longitude"]],
        left_on="Language_ID",
        right_on="ID",
        suffixes=("", "_lang"),
        how="left",
    )
    merged = merged.dropna(subset=["Glottocode", "Latitude", "Longitude"]).copy()

    # 6. Deduplicate per Glottocode: majority vote of Cognateset_ID
    def majority_cognate(series: pd.Series) -> int:
        vc = series.value_counts()
        # ties → smallest Cognateset_ID (deterministic)
        top_count = vc.iloc[0]
        tied = sorted(vc[vc == top_count].index.tolist())
        return int(tied[0])

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
    if n < 20:
        # Unlikely for 170-concept IE list; flag loudly.
        print(f"  [{concept_label}] WARNING: only {n} languages after dedup.")

    # 7. Global mode cognate = proto-form
    mode_cognate = int(by_gc["cognate"].value_counts().index[0])
    retention = (by_gc["cognate"].values == mode_cognate).astype(int)
    n_retention_1 = int(retention.sum())
    retention_rate = float(n_retention_1 / n) if n else float("nan")

    # 8. Homeland distance
    dist = haversine_km(
        HOMELAND_LAT, HOMELAND_LON,
        by_gc["latitude"].values, by_gc["longitude"].values,
    )

    # 9. Spearman
    if n >= 3 and 0 < retention.sum() < n:
        rho, pval = spearmanr(dist, retention)
        rho = float(rho)
        pval = float(pval)
    else:
        rho = float("nan")
        pval = float("nan")

    # 10. Bootstrap 95% CI (percentile)
    rhos = []
    if np.isfinite(rho):
        idx_all = np.arange(n)
        for _ in range(N_BOOTSTRAP):
            idx_b = rng.choice(idx_all, size=n, replace=True)
            ret_b = retention[idx_b]
            dist_b = dist[idx_b]
            if ret_b.sum() == 0 or ret_b.sum() == n:
                continue  # degenerate resample
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
        notes_bits.append(f"excluded {n_loans_excluded} loan-cognateset forms")
    if n_no_cog:
        notes_bits.append(f"{n_no_cog} forms lacked cognate assignment (dropped)")
    notes = "; ".join(notes_bits) if notes_bits else ""

    return {
        "role": role,
        "parameter_id": param_id,
        "iecor_meaning": iecor_meaning,
        "n": int(n),
        "n_retention_1": n_retention_1,
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
    print("Cross-family Migration Gradient — Indo-European (IECoR)")
    print("=" * 72)

    rng = np.random.default_rng(SEED)

    langs, params, forms, cogs, loan_cogset_ids = load_iecor()
    print(
        f"IECoR loaded: languages={len(langs)} parameters={len(params)} "
        f"forms={len(forms)} cognates={len(cogs)} loan_cognatesets={len(loan_cogset_ids)}"
    )

    slate = load_anchor_slate()
    print(f"Anchor slate (IE, non-excluded): {len(slate)} concepts")
    print(slate[["role", "concept", "iecor_meaning"]].to_string(index=False))
    print()

    # Drive order: anchors first, then controls
    slate = slate.sort_values(
        by="role", key=lambda s: s.map({"anchor": 0, "control": 1}).fillna(2)
    ).reset_index(drop=True)

    n_languages_total = int(langs["Glottocode"].nunique())

    concepts_out: dict[str, dict] = {}
    print(f"{'Concept':<10} {'Role':<8} {'n':>4} {'ret%':>6} {'rho':>8} {'p':>10} "
          f"{'CI95':>22}")
    print("-" * 72)
    for _, row in slate.iterrows():
        label = row["concept"]
        meaning = row["iecor_meaning"]
        role = row["role"]
        res = analyze_concept(
            concept_label=label,
            iecor_meaning=meaning,
            role=role,
            langs=langs,
            params=params,
            forms=forms,
            cogs=cogs,
            loan_cogset_ids=loan_cogset_ids,
            rng=rng,
        )
        concepts_out[label] = res
        ret_pct = (res["retention_rate"] * 100) if res["retention_rate"] is not None else float("nan")
        rho = res["spearman_r"]
        pval = res["spearman_p"]
        ci_lo = res["ci_lower"]
        ci_hi = res["ci_upper"]
        print(
            f"{label:<10} {role:<8} {res['n']:>4} {ret_pct:>6.1f} "
            f"{rho if rho is not None else float('nan'):>+8.3f} "
            f"{pval if pval is not None else float('nan'):>10.4g} "
            f"[{ci_lo if ci_lo is not None else float('nan'):>+6.3f},"
            f"{ci_hi if ci_hi is not None else float('nan'):>+6.3f}]"
        )
    print()

    payload = {
        "family": "Indo-European",
        "homeland": {
            "name": HOMELAND_NAME,
            "lat": HOMELAND_LAT,
            "lon": HOMELAND_LON,
        },
        "dataset": "IECoR v1.2 (Heggarty et al. 2023, Science)",
        "n_languages_total": n_languages_total,
        "concepts": concepts_out,
        "method": {
            "dedup": "by Glottocode, majority-vote cognate class",
            "retention_definition": "cognate class equals global mode for concept",
            "distance": "haversine km from homeland",
            "bootstrap": {"n": N_BOOTSTRAP, "seed": SEED, "method": "percentile"},
            "loans_excluded": True,
            "loans_exclusion_note": (
                "IECoR forms.csv Loan column is empty; loans are flagged at the "
                "cognateset level in loans.csv. Forms whose Cognateset_ID appears "
                "in loans.csv are excluded."
            ),
        },
    }

    with OUT_JSON.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"Wrote {OUT_JSON}")

    # Sanity-check hint (weak priors; do not self-overrule)
    sea = concepts_out.get("SEA", {})
    river = concepts_out.get("RIVER", {})
    tree = concepts_out.get("TREE", {})
    print()
    print("Sanity checks (weak priors, report-only):")
    if sea and sea.get("spearman_r") is not None:
        msg = "near-zero/negative" if sea["spearman_r"] <= 0.05 else "POSITIVE (flags review)"
        print(f"  SEA control   rho={sea['spearman_r']:+.3f}  -> {msg}")
    if river and river.get("spearman_r") is not None:
        msg = "positive (as predicted)" if river["spearman_r"] > 0 else "non-positive (report as-is)"
        print(f"  RIVER anchor  rho={river['spearman_r']:+.3f}  -> {msg}")
    if tree and tree.get("retention_rate") is not None:
        msg = "OK (>=60%)" if tree["retention_rate"] >= 0.60 else "below 60%"
        print(f"  TREE control  retention={tree['retention_rate']*100:.1f}%  -> {msg}")


if __name__ == "__main__":
    main()
