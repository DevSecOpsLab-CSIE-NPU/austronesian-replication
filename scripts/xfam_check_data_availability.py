#!/usr/bin/env python3
"""Cross-family data availability checker.
==========================================
For each (family, concept) pair in data/external/anchor_concepts.csv, report:
  - Whether the concept exists in ASJP (100-concept Swadesh list)
  - Language coverage in the target family (from ASJP)
  - Cognate-code availability (ASJP has none; flagged as gap)
  - What external dataset (IECoR / ST-cognate) is required to fill the gap

Output: results/xfam_data_inventory.json + console summary.

This is a pre-flight check before launching the cross-family migration-gradient
pipeline. Once IECoR and a ST cognate dataset land in data/raw/, rerun to
confirm anchor concept coverage.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

ASJP_DIR = ROOT / "data" / "raw" / "asjp"
EXT_DIR = ROOT / "data" / "external"
RESULTS_DIR = ROOT / "results"

FAMILIES = {
    "Austronesian": "Austronesian",
    "Indo-European": "Indo-European",
    "Sino-Tibetan": "Sino-Tibetan",
}


def load_asjp():
    langs = pd.read_csv(
        ASJP_DIR / "languages.csv",
        usecols=["ID", "Glottocode", "Family", "Latitude", "Longitude"],
    )
    params = pd.read_csv(ASJP_DIR / "parameters.csv")
    forms = pd.read_csv(
        ASJP_DIR / "forms.csv",
        usecols=["Language_ID", "Parameter_ID", "Cognacy", "Loan"],
        low_memory=False,
    )
    return langs, params, forms


def check_asjp_coverage(row, langs, params, forms):
    """Return coverage stats for one (family, concept) pair in ASJP."""
    asjp_id = row.get("asjp_id")
    family = row["family"]

    fam_langs = langs[langs["Family"].astype(str).str.contains(family, case=False, na=False)]
    n_fam_langs = len(fam_langs)

    if pd.isna(asjp_id) or asjp_id == "":
        return {
            "in_asjp": False,
            "asjp_id": None,
            "n_family_languages": int(n_fam_langs),
            "n_coverage": 0,
            "coverage_pct": 0.0,
            "has_cognate_codes": False,
        }

    pid = int(asjp_id)
    fam_lang_ids = set(fam_langs["ID"])
    concept_forms = forms[(forms["Parameter_ID"] == pid) & (forms["Language_ID"].isin(fam_lang_ids))]
    n_covered = concept_forms["Language_ID"].nunique()

    cognacy_nonempty = (
        concept_forms["Cognacy"].astype(str).str.strip().replace("nan", "").astype(bool).sum()
    )

    return {
        "in_asjp": True,
        "asjp_id": pid,
        "asjp_gloss": params.loc[params["ID"] == pid, "Name"].iloc[0] if pid in params["ID"].values else None,
        "n_family_languages": int(n_fam_langs),
        "n_coverage": int(n_covered),
        "coverage_pct": round(100 * n_covered / max(n_fam_langs, 1), 1),
        "has_cognate_codes": int(cognacy_nonempty) > 0,
        "cognacy_annotated_forms": int(cognacy_nonempty),
    }


def recommend_source(row, asjp_stats):
    """Return the recommended external data source for this concept."""
    family = row["family"]
    in_asjp_with_cognate = asjp_stats["in_asjp"] and asjp_stats["has_cognate_codes"]

    if family == "Austronesian":
        return "ABVD (already loaded; data/raw/cldf/)"
    if family == "Indo-European":
        if in_asjp_with_cognate:
            return "ASJP (usable)"
        return "IECoR (Heggarty et al. 2023) — required"
    if family == "Sino-Tibetan":
        if in_asjp_with_cognate:
            return "ASJP (usable)"
        return "Sagart 2019 PNAS / Zhang 2019 Nature ST dataset — required"
    return "unknown"


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    anchors = pd.read_csv(EXT_DIR / "anchor_concepts.csv")
    langs, params, forms = load_asjp()

    print(f"ASJP inventory: {len(langs):,} languages, {len(params)} concepts, {len(forms):,} forms")
    print(f"ASJP Cognacy-annotated forms: {int(forms['Cognacy'].astype(str).str.strip().replace('nan','').astype(bool).sum()):,}")
    print()

    rows = []
    for _, row in anchors.iterrows():
        stats = check_asjp_coverage(row, langs, params, forms)
        stats["family"] = row["family"]
        stats["role"] = row["role"]
        stats["concept"] = row["concept"]
        stats["recommended_source"] = recommend_source(row, stats)
        rows.append(stats)

    out = pd.DataFrame(rows)

    print("=" * 96)
    print(f"{'Family':<16} {'Role':<8} {'Concept':<12} {'ASJP':<6} {'Cov%':<7} {'Cog?':<6} Source")
    print("-" * 96)
    for _, r in out.iterrows():
        print(
            f"{r['family']:<16} {r['role']:<8} {r['concept']:<12} "
            f"{'yes' if r['in_asjp'] else 'no':<6} "
            f"{r['coverage_pct']:<7} "
            f"{'yes' if r['has_cognate_codes'] else 'no':<6} "
            f"{r['recommended_source']}"
        )

    summary = {
        "asjp_languages_total": int(len(langs)),
        "asjp_concepts_total": int(len(params)),
        "asjp_forms_total": int(len(forms)),
        "asjp_cognacy_coverage_pct": round(
            100 * forms["Cognacy"].astype(str).str.strip().replace("nan", "").astype(bool).sum() / len(forms),
            3,
        ),
        "per_concept": rows,
        "missing_external_datasets": sorted(
            {r["recommended_source"] for r in rows if "required" in r["recommended_source"]}
        ),
    }

    out_path = RESULTS_DIR / "xfam_data_inventory.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print()
    print(f"Wrote {out_path}")

    missing = summary["missing_external_datasets"]
    if missing:
        print()
        print("GAPS — external datasets required before running migration-gradient pipeline:")
        for m in missing:
            print(f"  - {m}")


if __name__ == "__main__":
    main()
