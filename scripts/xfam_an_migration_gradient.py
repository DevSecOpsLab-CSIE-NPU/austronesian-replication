#!/usr/bin/env python3
"""Cross-family meta-analysis: Austronesian migration-gradient baseline.

Computes per-concept proto-form retention gradients (Spearman ρ between
haversine distance from Taiwan and binary retention) for the Austronesian
anchor (SEA) and body-relative controls (LEFT/RIGHT/ABOVE/BELOW), using the
same methodology that will be applied to Indo-European (RIVER anchor) and
Sino-Tibetan (RICE anchor) in the forthcoming cross-family paper.

Method (applied identically to all three families):
  1. Deduplicate by Glottocode: for each (Glottocode, concept), take the most
     frequent cognate class across all ABVD dialect entries mapping to that
     Glottocode.
  2. Proto-form retention R_j ∈ {0, 1}: R_j = 1 iff language j's cognate class
     equals the globally most frequent cognate class for that concept.
  3. Homeland distance: haversine km from Taiwan (23.5°N, 121.0°E) to each
     language's (lat, lon) from ABVD/Glottolog metadata.
  4. Migration gradient: Spearman ρ between dist_homeland_km and retention
     across languages with data for the concept. Report n, ρ, two-tailed p.
  5. Bootstrap 95% CI: 500 resamples, np.random.default_rng(42),
     percentile method.

Input:
  - data/processed/clean_wordlist.csv  (ABVD cleaned wordlist)
  - data/processed/languages.csv       (ABVD + Glottolog metadata)
  - data/processed/meanings.csv        (meaning_id ↔ concepticon gloss)
  - data/external/anchor_concepts.csv  (cross-family anchor/control slate)

Output:
  results/xfam_an_migration_gradient.json
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
DATA_EXTERNAL = ROOT / "data" / "external"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# Homeland coordinates per cross-family spec.
HOMELAND_NAME = "Taiwan"
HOMELAND_LAT = 23.5
HOMELAND_LON = 121.0

# Bootstrap configuration.
N_BOOTSTRAP = 500
SEED = 42

# Mapping from anchor_concepts.csv "concept" labels to ABVD meanings.csv ids.
# Concepticon gloss match (meanings.csv ↔ anchor_concepts.csv concepticon_gloss):
#   SEA     → 124_sea      (concepticon 1474 in ABVD; anchor CSV says 1622 but
#                           ABVD meanings.csv records SEA as id 1474 — we
#                           match on gloss SEA which is unambiguous)
#   LEFT    → 2_left
#   RIGHT   → 3_right
#   ABOVE   → 175_above
#   BELOW   → 176_below     (ABVD gloss "BELOW OR UNDER")
CONCEPT_TO_MEANING_ID = {
    "SEA": "124_sea",
    "LEFT": "2_left",
    "RIGHT": "3_right",
    "ABOVE": "175_above",
    "BELOW": "176_below",
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    r_earth = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    )
    return r_earth * 2 * atan2(sqrt(a), sqrt(1 - a))


def load_anchor_slate(family: str) -> pd.DataFrame:
    """Load the anchor_concepts.csv slate for a given family."""
    path = DATA_EXTERNAL / "anchor_concepts.csv"
    if not path.exists():
        raise FileNotFoundError(f"Anchor concept slate not found at {path}")
    slate = pd.read_csv(path)
    slate = slate[slate["family"] == family].copy()
    if slate.empty:
        raise ValueError(f"No rows for family={family!r} in {path}")
    # Drop "excluded" rows (none for AN but keep for generality).
    slate = slate[slate["role"].isin(["anchor", "control"])].copy()
    return slate.reset_index(drop=True)


def build_per_language_table(
    wl: pd.DataFrame,
    lang: pd.DataFrame,
    concept_to_meaning: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Build the per-(glottocode × concept) retention table.

    Steps:
      1. Join wordlist to language metadata and keep only entries with a
         Glottocode and geographic coordinates.
      2. Explode comma-separated cognate-class strings so each row carries a
         single class (standard JLE pre-processing).
      3. For each (glottocode, concept) pair, choose the most frequent
         cognate class across all ABVD dialect entries mapping to that
         Glottocode (majority-vote dedup).
      4. Identify the global mode (proto-form) per concept across all
         languages; mark retention = 1 iff the language's dedup class
         equals that mode.
    """
    merged = wl.merge(
        lang[["id", "glottocode", "latitude", "longitude"]].dropna(
            subset=["glottocode", "latitude", "longitude"]
        ),
        left_on="language_id",
        right_on="id",
        how="inner",
    )
    merged["cognate_class"] = merged["cognate_class"].astype(str)
    merged = merged.assign(
        cognate_class=merged["cognate_class"].str.split(r",\s*")
    ).explode("cognate_class")
    merged["cognate_class"] = merged["cognate_class"].str.strip()
    merged = merged[
        merged["cognate_class"].notna()
        & (merged["cognate_class"] != "")
        & (merged["cognate_class"].str.lower() != "nan")
    ]

    rows: list[dict] = []
    proto_form_by_concept: dict[str, str] = {}

    for concept, meaning_id in concept_to_meaning.items():
        sub = merged[merged["meaning_id"] == meaning_id].copy()
        if sub.empty:
            proto_form_by_concept[concept] = ""
            continue

        # Step 3: dedup by Glottocode — pick the per-language majority-vote
        # cognate class across all ABVD dialect entries for that Glottocode.
        # We use the earliest lat/lon observed (within a Glottocode, ABVD
        # records carry nearly identical coordinates by design).
        per_glotto: dict[str, dict] = {}
        for gc, grp in sub.groupby("glottocode"):
            cls_counts = Counter(grp["cognate_class"])
            # idxmax-equivalent with deterministic tie-break: sort first by
            # descending count, then by cognate_class string.
            top_class = sorted(
                cls_counts.items(), key=lambda kv: (-kv[1], kv[0])
            )[0][0]
            per_glotto[gc] = {
                "glottocode": gc,
                "concept": concept,
                "cognate_class": top_class,
                "latitude": float(grp["latitude"].iloc[0]),
                "longitude": float(grp["longitude"].iloc[0]),
            }

        # Step 4: global mode across deduplicated per-Glottocode classes.
        global_counts = Counter(r["cognate_class"] for r in per_glotto.values())
        proto_form = sorted(
            global_counts.items(), key=lambda kv: (-kv[1], kv[0])
        )[0][0]
        proto_form_by_concept[concept] = proto_form

        for r in per_glotto.values():
            r["retention"] = int(r["cognate_class"] == proto_form)
            rows.append(r)

    table = pd.DataFrame(rows)
    return table, proto_form_by_concept


def add_homeland_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Add haversine km from the homeland to each row."""
    df = df.copy()
    df["dist_homeland_km"] = df.apply(
        lambda row: haversine_km(
            HOMELAND_LAT, HOMELAND_LON, row["latitude"], row["longitude"]
        ),
        axis=1,
    )
    return df


def bootstrap_spearman_ci(
    dist_km: np.ndarray,
    retention: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    """Percentile bootstrap CI for Spearman ρ."""
    rng = np.random.default_rng(seed)
    n = len(dist_km)
    if n < 3:
        return float("nan"), float("nan")
    rs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        # Guard against degenerate resample (all retentions identical).
        if len(np.unique(retention[idx])) < 2 or len(np.unique(dist_km[idx])) < 2:
            rs[b] = np.nan
            continue
        r_b, _ = spearmanr(dist_km[idx], retention[idx])
        rs[b] = r_b
    rs = rs[~np.isnan(rs)]
    if rs.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def analyze_family(
    family: str,
    wl_path: Path,
    lang_path: Path,
    homeland_name: str,
    homeland_lat: float,
    homeland_lon: float,
) -> dict:
    """Run the full per-concept gradient pipeline for one family."""
    print(f"\nLoading ABVD tables...")
    wl = pd.read_csv(wl_path, dtype={"language_id": str}, low_memory=False)
    lang = pd.read_csv(lang_path, dtype={"id": str}, low_memory=False)
    print(f"  wordlist rows:   {len(wl):,}")
    print(f"  language rows:   {len(lang):,}")

    slate = load_anchor_slate(family)
    print(f"  concepts in slate for {family}: {list(slate['concept'])}")

    concept_to_meaning = {
        c: CONCEPT_TO_MEANING_ID[c]
        for c in slate["concept"]
        if c in CONCEPT_TO_MEANING_ID
    }
    missing = set(slate["concept"]) - set(CONCEPT_TO_MEANING_ID)
    if missing:
        print(f"  WARNING: concepts without ABVD mapping: {missing}")

    table, proto_forms = build_per_language_table(wl, lang, concept_to_meaning)
    table = add_homeland_distance(table)
    n_languages_total = int(table["glottocode"].nunique())
    print(f"  deduplicated languages (all concepts pooled): {n_languages_total}")

    roles = dict(zip(slate["concept"], slate["role"]))
    concepts_out: dict[str, dict] = {}

    for concept, meaning_id in concept_to_meaning.items():
        sub = table[table["concept"] == concept].copy()
        if sub.empty:
            concepts_out[concept] = {
                "role": roles.get(concept, "unknown"),
                "n": 0,
                "notes": "no ABVD rows matched for this concept",
            }
            continue

        n = int(len(sub))
        retention = sub["retention"].to_numpy(dtype=int)
        dist = sub["dist_homeland_km"].to_numpy(dtype=float)

        r, p = spearmanr(dist, retention)
        ci_lo, ci_hi = bootstrap_spearman_ci(dist, retention, N_BOOTSTRAP, SEED)

        concepts_out[concept] = {
            "role": roles.get(concept, "unknown"),
            "n": n,
            "n_retention_1": int(retention.sum()),
            "retention_rate": round(float(retention.mean()), 4),
            "spearman_r": round(float(r), 4),
            "spearman_p": float(p),
            "ci_lower": round(ci_lo, 4),
            "ci_upper": round(ci_hi, 4),
            "proto_form_cognate_id": proto_forms.get(concept, ""),
        }

    out = {
        "family": family,
        "homeland": {
            "name": homeland_name,
            "lat": homeland_lat,
            "lon": homeland_lon,
        },
        "dataset": "ABVD (Lexibank CLDF)",
        "n_languages_total": n_languages_total,
        "concepts": concepts_out,
        "method": {
            "dedup": "by Glottocode, majority-vote cognate class",
            "retention_definition": "cognate class equals global mode for concept",
            "distance": "haversine km from homeland",
            "bootstrap": {
                "n": N_BOOTSTRAP,
                "seed": SEED,
                "method": "percentile",
            },
        },
    }
    return out


def print_summary_table(result: dict) -> None:
    print("\n" + "=" * 78)
    print(f"SUMMARY  family={result['family']}  homeland={result['homeland']['name']} "
          f"({result['homeland']['lat']}°N, {result['homeland']['lon']}°E)")
    print("=" * 78)
    header = f"  {'concept':<7s} {'role':<7s} {'n':>5s} {'retention':>10s} {'ρ':>8s} {'p':>10s} {'95% CI':>22s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for concept, v in result["concepts"].items():
        if v.get("n", 0) == 0:
            print(f"  {concept:<7s} {v.get('role', '?'):<7s}  (no data)")
            continue
        ci = f"[{v['ci_lower']:+.3f}, {v['ci_upper']:+.3f}]"
        print(
            f"  {concept:<7s} {v['role']:<7s} {v['n']:>5d} "
            f"{v['retention_rate']:>10.4f} {v['spearman_r']:>+8.4f} "
            f"{v['spearman_p']:>10.2e} {ci:>22s}"
        )
    print("=" * 78)


def main() -> None:
    wl_path = DATA_PROCESSED / "clean_wordlist.csv"
    lang_path = DATA_PROCESSED / "languages.csv"
    if not wl_path.exists() or not lang_path.exists():
        raise FileNotFoundError(
            f"Expected processed ABVD data at {wl_path} / {lang_path}"
        )

    result = analyze_family(
        family="Austronesian",
        wl_path=wl_path,
        lang_path=lang_path,
        homeland_name=HOMELAND_NAME,
        homeland_lat=HOMELAND_LAT,
        homeland_lon=HOMELAND_LON,
    )

    print_summary_table(result)

    # Sanity check echo.
    sea = result["concepts"].get("SEA", {})
    if sea.get("n", 0) > 0:
        sign = "POSITIVE" if sea["spearman_r"] > 0 else "NEGATIVE"
        print(
            f"\n  SEA gradient: ρ={sea['spearman_r']:+.4f} "
            f"(p={sea['spearman_p']:.2e}, n={sea['n']}) "
            f"[{sign}, expected ~+0.40]"
        )

    out_path = RESULTS / "xfam_an_migration_gradient.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
