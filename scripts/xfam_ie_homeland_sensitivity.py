#!/usr/bin/env python3
"""IE migration gradient — homeland sensitivity (Steppe vs Anatolian).

Rerun the Indo-European migration-gradient analysis under the Anatolian
homeland hypothesis (Renfrew 1987; Gray & Atkinson 2003) and compare
concept-by-concept against the Steppe-anchored results. The critical
question: does the MOUNTAIN finding (rho = +0.410) survive a ~5,000 km
southwest relocation of the putative homeland? If yes, the gradient is
structural (driven by the geography of IE languages relative to mountain
ranges); if no, it is homeland-contingent.

Inputs
------
- data/raw/iecor/cldf/{languages,parameters,forms,cognates,loans}.csv
- data/external/anchor_concepts.csv
- results/xfam_ie_migration_gradient.json  (existing Steppe result)

Output
------
- results/xfam_ie_homeland_sensitivity.json

Usage
-----
    .venv/bin/python3 scripts/xfam_ie_homeland_sensitivity.py
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
sys.path.insert(0, str(ROOT / "scripts"))

# Reuse loaders and haversine from the primary IE script. We re-implement
# analyze_concept locally so the homeland can be parameterised (the original
# reads module-level HOMELAND_LAT/LON).
from xfam_ie_migration_gradient import (  # noqa: E402
    haversine_km,
    load_iecor,
    load_anchor_slate,
)

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

STEPPE_JSON = RESULTS_DIR / "xfam_ie_migration_gradient.json"
OUT_JSON = RESULTS_DIR / "xfam_ie_homeland_sensitivity.json"

# Homelands
HOMELANDS = {
    "steppe": {
        "name": "Pontic-Caspian steppe",
        "lat": 48.0,
        "lon": 45.0,
        "source": "Anthony 2007; Heggarty 2023 consensus",
    },
    "anatolian": {
        "name": "Central Anatolian plateau",
        "lat": 38.9,
        "lon": 34.5,
        "source": "Renfrew 1987; Gray & Atkinson 2003",
    },
}

N_BOOTSTRAP = 500
SEED = 42


# ---------------------------------------------------------------------------
# Homeland-parameterised analysis — mirrors xfam_ie_migration_gradient exactly
# ---------------------------------------------------------------------------


def analyze_concept_homeland(
    concept_label: str,
    iecor_meaning: str,
    role: str,
    homeland_lat: float,
    homeland_lon: float,
    langs: pd.DataFrame,
    params: pd.DataFrame,
    forms: pd.DataFrame,
    cogs: pd.DataFrame,
    loan_cogset_ids: set[int],
    rng: np.random.Generator,
) -> dict:
    """Replicates analyze_concept from xfam_ie_migration_gradient.py but
    takes the homeland as an argument rather than reading a module global."""

    # 1. Resolve parameter ID
    pmatch = params[params["Name"].astype(str).str.lower() == iecor_meaning.lower()]
    if pmatch.empty:
        raise RuntimeError(
            f"[{concept_label}] No Parameter found with Name == '{iecor_meaning}'"
        )
    param_id = int(pmatch.iloc[0]["ID"])

    # 2. Collect forms
    cforms = forms[forms["Parameter_ID"] == param_id].copy()
    n_forms_total = len(cforms)

    # 3. Join with cognates on Form_ID
    cog_slim = cogs[["Form_ID", "Cognateset_ID"]].drop_duplicates(subset=["Form_ID"])
    merged = cforms.merge(cog_slim, left_on="ID", right_on="Form_ID", how="left")
    n_no_cog = int(merged["Cognateset_ID"].isna().sum())
    if n_forms_total > 0 and n_no_cog / n_forms_total > 0.05:
        raise RuntimeError(
            f"[{concept_label}] {n_no_cog}/{n_forms_total} forms lack cognate "
            f"assignment (>5%)"
        )
    merged = merged.dropna(subset=["Cognateset_ID"]).copy()
    merged["Cognateset_ID"] = merged["Cognateset_ID"].astype(int)

    # 4. Exclude loans (by cognateset)
    n_before_loan = len(merged)
    merged = merged[~merged["Cognateset_ID"].isin(loan_cogset_ids)].copy()
    n_loans_excluded = n_before_loan - len(merged)

    # 5. Attach language metadata
    merged = merged.merge(
        langs[["ID", "Glottocode", "Latitude", "Longitude"]],
        left_on="Language_ID",
        right_on="ID",
        suffixes=("", "_lang"),
        how="left",
    )
    merged = merged.dropna(subset=["Glottocode", "Latitude", "Longitude"]).copy()

    # 6. Deduplicate per Glottocode
    def majority_cognate(series: pd.Series) -> int:
        vc = series.value_counts()
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

    # 7. Global mode cognate
    mode_cognate = int(by_gc["cognate"].value_counts().index[0])
    retention = (by_gc["cognate"].values == mode_cognate).astype(int)
    n_retention_1 = int(retention.sum())
    retention_rate = float(n_retention_1 / n) if n else float("nan")

    # 8. Homeland distance
    dist = haversine_km(
        homeland_lat, homeland_lon,
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

    return {
        "role": role,
        "parameter_id": param_id,
        "n": int(n),
        "n_retention_1": n_retention_1,
        "retention_rate": round(retention_rate, 4) if np.isfinite(retention_rate) else None,
        "r": round(rho, 4) if np.isfinite(rho) else None,
        "p": round(pval, 6) if np.isfinite(pval) else None,
        "ci_lower": round(ci_lower, 4) if np.isfinite(ci_lower) else None,
        "ci_upper": round(ci_upper, 4) if np.isfinite(ci_upper) else None,
        "proto_form_cognate_id": mode_cognate,
        "n_loans_excluded": n_loans_excluded,
    }


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def ci_overlap(lo_a: float, hi_a: float, lo_b: float, hi_b: float) -> bool:
    """Do intervals [lo_a, hi_a] and [lo_b, hi_b] overlap?"""
    return not (hi_a < lo_b or hi_b < lo_a)


def summarise_concept(label: str, steppe: dict, anatolian: dict) -> dict:
    r_s = steppe["r"]
    r_a = anatolian["r"]
    delta = None
    sign_preserved = None
    overlap = None
    if r_s is not None and r_a is not None:
        delta = round(r_a - r_s, 4)
        sign_preserved = bool(np.sign(r_s) == np.sign(r_a)) and (abs(r_s) > 0.01 and abs(r_a) > 0.01)
        if (
            steppe["ci_lower"] is not None and steppe["ci_upper"] is not None
            and anatolian["ci_lower"] is not None and anatolian["ci_upper"] is not None
        ):
            overlap = ci_overlap(
                steppe["ci_lower"], steppe["ci_upper"],
                anatolian["ci_lower"], anatolian["ci_upper"],
            )

    # One-line interpretation
    if sign_preserved and overlap:
        interp = (
            f"Robust: sign preserved and 95% CIs overlap "
            f"(Δρ={delta:+.3f}); {label} gradient is homeland-insensitive."
        )
    elif sign_preserved and not overlap:
        interp = (
            f"Partially robust: sign preserved but CIs do not overlap "
            f"(Δρ={delta:+.3f}); magnitude shifts with homeland."
        )
    elif sign_preserved is False:
        interp = (
            f"Fragile: sign flips under Anatolian homeland "
            f"(Steppe ρ={r_s:+.3f}, Anatolian ρ={r_a:+.3f}, Δρ={delta:+.3f})."
        )
    else:
        interp = "Could not compare (missing values)."

    return {
        "steppe":    {k: steppe[k]    for k in ("r", "p", "ci_lower", "ci_upper", "n")},
        "anatolian": {k: anatolian[k] for k in ("r", "p", "ci_lower", "ci_upper", "n")},
        "delta_r": delta,
        "sign_preserved": sign_preserved,
        "ci_overlap": overlap,
        "interpretation": interp,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 78)
    print("IE Migration Gradient — Homeland Sensitivity (Steppe vs Anatolian)")
    print("=" * 78)

    # Load existing Steppe result (for sanity-checking our replication)
    with STEPPE_JSON.open("r", encoding="utf-8") as fh:
        steppe_existing = json.load(fh)
    existing_concepts = steppe_existing["concepts"]

    langs, params, forms, cogs, loan_cogset_ids = load_iecor()
    print(f"IECoR loaded: {len(langs)} languages, {len(params)} parameters, "
          f"{len(forms)} forms, {len(cogs)} cognates, {len(loan_cogset_ids)} loan cognatesets")

    slate = load_anchor_slate()
    slate = slate.sort_values(
        by="role", key=lambda s: s.map({"anchor": 0, "control": 1}).fillna(2)
    ).reset_index(drop=True)
    print(f"Anchor slate: {len(slate)} concepts\n")

    # Recompute both homelands with fresh RNGs per homeland for determinism
    results_by_homeland: dict[str, dict[str, dict]] = {"steppe": {}, "anatolian": {}}

    for key in ("steppe", "anatolian"):
        hl = HOMELANDS[key]
        rng = np.random.default_rng(SEED)
        for _, row in slate.iterrows():
            label = row["concept"]
            meaning = row["iecor_meaning"]
            role = row["role"]
            res = analyze_concept_homeland(
                concept_label=label,
                iecor_meaning=meaning,
                role=role,
                homeland_lat=hl["lat"],
                homeland_lon=hl["lon"],
                langs=langs,
                params=params,
                forms=forms,
                cogs=cogs,
                loan_cogset_ids=loan_cogset_ids,
                rng=rng,
            )
            results_by_homeland[key][label] = res

    # ---- Sanity: our Steppe replication must match the on-disk numbers ----
    print("Sanity check: replicated Steppe ρ vs. on-disk Steppe ρ")
    print(f"{'concept':<10} {'replicated':>12} {'on-disk':>10} {'diff':>8}")
    print("-" * 44)
    max_diff = 0.0
    for label in results_by_homeland["steppe"]:
        r_new = results_by_homeland["steppe"][label]["r"]
        r_old = existing_concepts[label]["spearman_r"]
        diff = abs(r_new - r_old) if (r_new is not None and r_old is not None) else float("nan")
        max_diff = max(max_diff, diff if np.isfinite(diff) else 0.0)
        print(f"{label:<10} {r_new:>+12.4f} {r_old:>+10.4f} {diff:>8.4f}")
    if max_diff > 1e-3:
        raise RuntimeError(
            f"Steppe replication diverges from on-disk result (max |Δρ|={max_diff:.4f}); "
            f"methodology mismatch — STOP."
        )
    print(f"\nSteppe replication OK (max |Δρ|={max_diff:.6f} <= 1e-3)\n")

    # ---- Side-by-side comparison table ----
    concepts_compared: dict[str, dict] = {}
    print(f"{'concept':<10} {'role':<8} {'n':>4} | "
          f"{'ρ_steppe':>10} {'ρ_anatol':>10} {'Δρ':>7} | "
          f"{'sign':>5} {'CIovl':>6}")
    print("-" * 72)
    n_sign_preserved = 0
    n_ci_overlap = 0
    for label in results_by_homeland["steppe"]:
        cmp = summarise_concept(
            label,
            results_by_homeland["steppe"][label],
            results_by_homeland["anatolian"][label],
        )
        concepts_compared[label] = cmp
        role = results_by_homeland["steppe"][label]["role"]
        n = results_by_homeland["steppe"][label]["n"]
        rs = cmp["steppe"]["r"]
        ra = cmp["anatolian"]["r"]
        dr = cmp["delta_r"]
        sp = cmp["sign_preserved"]
        ov = cmp["ci_overlap"]
        if sp:
            n_sign_preserved += 1
        if ov:
            n_ci_overlap += 1
        print(
            f"{label:<10} {role:<8} {n:>4} | "
            f"{rs:>+10.4f} {ra:>+10.4f} {dr:>+7.3f} | "
            f"{'yes' if sp else 'NO':>5} {'yes' if ov else 'NO':>6}"
        )
    print()

    # ---- MOUNTAIN verdict (the headline question) ----
    mt = concepts_compared["MOUNTAIN"]
    print("MOUNTAIN verdict:")
    print(f"  Steppe    ρ = {mt['steppe']['r']:+.4f} "
          f"[{mt['steppe']['ci_lower']:+.3f}, {mt['steppe']['ci_upper']:+.3f}]")
    print(f"  Anatolian ρ = {mt['anatolian']['r']:+.4f} "
          f"[{mt['anatolian']['ci_lower']:+.3f}, {mt['anatolian']['ci_upper']:+.3f}]")
    print(f"  Δρ = {mt['delta_r']:+.4f}   sign_preserved={mt['sign_preserved']}   "
          f"ci_overlap={mt['ci_overlap']}")
    print(f"  -> {mt['interpretation']}")
    print()

    # ---- Overall conclusion ----
    n_concepts = len(concepts_compared)
    if mt["sign_preserved"] and mt["ci_overlap"]:
        mt_verdict = "robust"
    elif mt["sign_preserved"]:
        mt_verdict = "partially robust"
    else:
        mt_verdict = "fragile"
    conclusion = (
        f"Of {n_concepts} IE concepts, {n_sign_preserved} preserve the sign of ρ "
        f"and {n_ci_overlap} have overlapping 95% CIs across the two homeland "
        f"hypotheses. MOUNTAIN is {mt_verdict} to homeland choice "
        f"(Steppe ρ={mt['steppe']['r']:+.3f}, Anatolian ρ={mt['anatolian']['r']:+.3f}, "
        f"Δρ={mt['delta_r']:+.3f}), indicating the anchor reflects a structural "
        f"geographic signal rather than a homeland-contingent artefact."
        if mt_verdict == "robust" else
        f"Of {n_concepts} IE concepts, {n_sign_preserved} preserve the sign of ρ "
        f"and {n_ci_overlap} have overlapping 95% CIs across the two homeland "
        f"hypotheses. MOUNTAIN is {mt_verdict} to homeland choice "
        f"(Steppe ρ={mt['steppe']['r']:+.3f}, Anatolian ρ={mt['anatolian']['r']:+.3f}, "
        f"Δρ={mt['delta_r']:+.3f})."
    )

    payload = {
        "family": "Indo-European",
        "comparison": "Steppe vs Anatolian homeland",
        "homelands": {
            "steppe":    {"name": HOMELANDS["steppe"]["name"],
                          "lat": HOMELANDS["steppe"]["lat"],
                          "lon": HOMELANDS["steppe"]["lon"],
                          "source": HOMELANDS["steppe"]["source"]},
            "anatolian": {"name": HOMELANDS["anatolian"]["name"],
                          "lat": HOMELANDS["anatolian"]["lat"],
                          "lon": HOMELANDS["anatolian"]["lon"],
                          "source": HOMELANDS["anatolian"]["source"]},
        },
        "dataset": steppe_existing["dataset"],
        "n_languages_total": steppe_existing["n_languages_total"],
        "concepts": concepts_compared,
        "overall_robustness": {
            "n_concepts_total": n_concepts,
            "n_concepts_sign_preserved": n_sign_preserved,
            "n_concepts_ci_overlap": n_ci_overlap,
            "mountain_verdict": mt_verdict,
            "conclusion": conclusion,
        },
        "method": {
            "dedup": "by Glottocode, majority-vote cognate class",
            "retention_definition": "cognate class equals global mode for concept",
            "distance": "haversine km from homeland",
            "bootstrap": {"n": N_BOOTSTRAP, "seed": SEED, "method": "percentile"},
            "loans_excluded": True,
            "sanity_check": f"Steppe replication matches on-disk result to max |Δρ|={max_diff:.2e}",
        },
    }

    with OUT_JSON.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"Wrote {OUT_JSON}")
    print()
    print("Overall conclusion:")
    print(f"  {conclusion}")


if __name__ == "__main__":
    main()
