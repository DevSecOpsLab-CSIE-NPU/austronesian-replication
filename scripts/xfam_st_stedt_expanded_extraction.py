#!/usr/bin/env python3
"""Sino-Tibetan expanded STEDT extraction: MILLET / TEA / SILKWORM / HORSE / WHEAT.

Extension of ``scripts/xfam_st_stedt_full_extraction.py``. That script covers
RICE / SALT / BARLEY / YAK and established an empirical pipeline at ~84%
STEDT-name -> Glottocode match rate. This script reuses that pipeline
unchanged (same alias table, same Method A + Method B, same homeland,
same 500-resample bootstrap with seed 42) and adds five culturally-loaded
anchor candidates drawn from Sagart (2005) / Sagart (2019)'s discussion of
the proto-Sinitic agricultural package and its post-contact additions.

Pre-registered directional predictions (stated here BEFORE the gradients
are computed, so the ledger is not retroactively rationalised):

  * MILLET    : positive (+). North-China millet agriculture was the
                proto-Sinitic staple *before* the southern rice dispersal
                (Sagart 2005). If any ST cultural concept survives as an
                expansion-anchor gradient, MILLET is the strongest
                theoretical candidate.
  * TEA       : uncertain (?). Tea cultivation is a later-stage cultural
                layer, likely post-expansion, so the gradient may be
                weak or null.
  * SILKWORM  : uncertain (?). Silk culture originated in the Sinitic
                core; retention may mirror proto-Sinitic conservatism
                but only among northern ST languages.
  * HORSE     : negative (-). In many ST languages 'horse' is a
                loanword from Indo-European / Altaic contact zones;
                retention of a native ST proto-form may be noisy or
                inverted.
  * WHEAT     : similar to HORSE. West-Asian crop origin, variable loan
                status across the family; direction uncertain but a
                positive gradient would be surprising.

Pipeline (inherits 100% from full_extraction.py, imported by module)
--------
1. Fetch gloss-level AJAX blobs (polite 0.5s rate limit, cached to
   data/raw/stedt/<concept>_ajax_raw.json).
2. Filter rows whose exact gloss matches the inclusion set per concept.
3. Name -> Glottocode via the shared alias table (zhang2019 /
   peirosst / sagartst roster).
4. Method A = STEDT expert etyma tag as cognate class.
   Method B = single-linkage cluster on cleaned reflexes by normalised
              Levenshtein distance, threshold 0.4.
5. Spearman rho of (haversine distance, retention) + 500-resample
   percentile 95% CI, seed 42.

Outputs
-------
data/raw/stedt/millet_forms.csv   + millet_ajax_raw.json
data/raw/stedt/tea_forms.csv      + tea_ajax_raw.json
data/raw/stedt/silkworm_forms.csv + silkworm_ajax_raw.json
data/raw/stedt/horse_forms.csv    + horse_ajax_raw.json
data/raw/stedt/wheat_forms.csv    + wheat_ajax_raw.json
results/xfam_st_stedt_expanded.json

If any concept returns zero STEDT hits or extraction fails (after 3
retries with exponential backoff), the failure is recorded in the JSON
under that concept's ``status`` field and the script continues to the
next concept rather than aborting.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ---------------------------------------------------------------------- reuse
# Import the full_extraction module by file path (it is not a package
# member). Every helper we need -- fetch_concept, match_stedt_language,
# _clean_reflex, nld_single_link_cluster, haversine, majority_vote,
# gradient, build_glottocode_lookup -- lives there, already tested.
_full_path = ROOT / "scripts" / "xfam_st_stedt_full_extraction.py"
_spec = importlib.util.spec_from_file_location(
    "xfam_st_stedt_full_extraction", _full_path,
)
assert _spec is not None and _spec.loader is not None
_full = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_full)

fetch_concept = _full.fetch_concept
match_stedt_language = _full.match_stedt_language
_clean_reflex = _full._clean_reflex
nld_single_link_cluster = _full.nld_single_link_cluster
haversine = _full.haversine
majority_vote = _full.majority_vote
gradient = _full.gradient
build_glottocode_lookup = _full.build_glottocode_lookup

STEDT_BASE = _full.STEDT_BASE
STEDT_AJAX = _full.STEDT_AJAX
HOMELAND_LAT = _full.HOMELAND_LAT
HOMELAND_LON = _full.HOMELAND_LON
HOMELAND_NAME = _full.HOMELAND_NAME
SEED = _full.SEED
N_BOOTSTRAP = _full.N_BOOTSTRAP
NLD_COGNATE_THRESHOLD = _full.NLD_COGNATE_THRESHOLD

# Keep the user-agent aligned with this script's version bump. The host
# still sees the same contact address, so rate-limit policy is unchanged.
_full.USER_AGENT = (
    "Austronesian-research-tool/0.3 "
    "(contact: augchao@gms.npu.edu.tw)"
)

STEDT_DIR = ROOT / "data" / "raw" / "stedt"
STEDT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUT_JSON = RESULTS_DIR / "xfam_st_stedt_expanded.json"

# --------------------------------------------------------------- concept cfg
# Each concept is fetched via (potentially) multiple query glosses; all
# raw responses are concatenated and de-duplicated by lexicon.rn before
# the canonical-gloss inclusion filter runs. This matches how the
# existing pipeline widens the gloss inclusion set.
CONCEPTS = [
    {
        "concept": "MILLET",
        "role": "anchor (pre-registered positive prediction)",
        "query_glosses": ["millet", "foxtail millet", "broomcorn millet"],
        "include_glosses": {
            "millet",
            "millet (foxtail)",
            "millet (broomcorn)",
            "millet (panicum)",
            "millet (setaria)",
            "millet plant",
            "millet grain",
            "millet / cereal",
            "foxtail millet",
            "broomcorn millet",
            "panicum millet",
            "setaria millet",
            "glutinous millet",
            "non-glutinous millet",
        },
        "csv": STEDT_DIR / "millet_forms.csv",
        "raw": STEDT_DIR / "millet_ajax_raw.json",
        "predicted_sign": "+",
    },
    {
        "concept": "TEA",
        "role": "later-stage cultural concept",
        "query_glosses": ["tea", "tea (plant)"],
        "include_glosses": {
            "tea",
            "tea (plant)",
            "tea plant",
            "tea leaf",
            "tea leaves",
            "tea (leaf)",
            "tea (leaves)",
            "tea / leaves",
            "tea, green",
            "green tea",
        },
        "csv": STEDT_DIR / "tea_forms.csv",
        "raw": STEDT_DIR / "tea_ajax_raw.json",
        "predicted_sign": "?",
    },
    {
        "concept": "SILKWORM",
        "role": "Sinitic-core cultural concept",
        "query_glosses": ["silkworm", "silk"],
        "include_glosses": {
            "silkworm",
            "silk worm",
            "silk-worm",
            "silkworm (larva)",
            "silk",
            "silk thread",
            "silk (thread)",
            "raw silk",
        },
        "csv": STEDT_DIR / "silkworm_forms.csv",
        "raw": STEDT_DIR / "silkworm_ajax_raw.json",
        "predicted_sign": "?",
    },
    {
        "concept": "HORSE",
        "role": "loan-suspect animal concept",
        "query_glosses": ["horse", "pony"],
        "include_glosses": {
            "horse",
            "horse (animal)",
            "horse / pony",
            "pony",
            "wild horse",
            "horse (wild)",
            "male horse",
            "female horse",
            "stallion",
            "mare",
        },
        "csv": STEDT_DIR / "horse_forms.csv",
        "raw": STEDT_DIR / "horse_ajax_raw.json",
        "predicted_sign": "-",
    },
    {
        "concept": "WHEAT",
        "role": "loan-suspect crop concept",
        "query_glosses": ["wheat", "grain"],
        "include_glosses": {
            "wheat",
            "wheat (plant)",
            "wheat plant",
            "wheat grain",
            "wheat (grain)",
            "wheat flour",
            "wheat / grain",
            "grain (wheat)",
        },
        "csv": STEDT_DIR / "wheat_forms.csv",
        "raw": STEDT_DIR / "wheat_ajax_raw.json",
        "predicted_sign": "-",
    },
]


# ---------------------------------------------------------- multi-gloss fetch
def fetch_concept_multi(concept: str, query_glosses: list[str],
                        cache_path: Path) -> dict:
    """Fetch and merge multiple gloss queries for one concept.

    Stores the merged payload (union of rows, dedup by lexicon.rn) at
    ``cache_path``. Individual per-gloss responses are cached alongside
    so re-runs are free.
    """
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as fh:
            return json.load(fh)

    merged_rows: list[list] = []
    fields: list[str] | None = None
    seen_rn: set = set()
    per_gloss_meta: list[dict] = []

    for g in query_glosses:
        slug = g.replace(" ", "_").replace("(", "").replace(")", "")
        side_cache = cache_path.with_name(
            f"{concept.lower()}__{slug}_ajax_raw.json"
        )
        try:
            data = fetch_concept(g, side_cache)
        except Exception as e:
            per_gloss_meta.append({
                "query_gloss": g,
                "status": "fetch_failed",
                "error": f"{type(e).__name__}: {e}",
            })
            continue
        if fields is None:
            fields = data["fields"]
        IDX_rn = fields.index("lexicon.rn")
        added = 0
        for r in data["data"]:
            rn = r[IDX_rn]
            if rn in seen_rn:
                continue
            seen_rn.add(rn)
            merged_rows.append(r)
            added += 1
        per_gloss_meta.append({
            "query_gloss": g,
            "status": "ok",
            "n_returned": len(data["data"]),
            "n_new_after_dedup": added,
        })

    if fields is None:
        # All queries failed.
        raise RuntimeError(
            f"All query_glosses failed for concept {concept}: "
            f"{per_gloss_meta}"
        )

    merged = {
        "fields": fields,
        "data": merged_rows,
        "_per_gloss_meta": per_gloss_meta,
    }
    with cache_path.open("w", encoding="utf-8") as fh:
        json.dump(merged, fh, ensure_ascii=False)
    return merged


# -------------------------------------------------------- per-concept driver
def process_concept(cfg: dict, lookup: dict[str, tuple]) -> dict:
    concept = cfg["concept"]
    print(f"\n=== {concept} =====================================")
    print(f"  role           : {cfg['role']}")
    print(f"  predicted sign : {cfg['predicted_sign']}")
    print(f"  query glosses  : {cfg['query_glosses']}")

    try:
        data = fetch_concept_multi(concept, cfg["query_glosses"], cfg["raw"])
    except Exception as e:
        print(f"  FATAL fetch: {type(e).__name__}: {e}")
        return {
            "role": cfg["role"],
            "query_glosses": cfg["query_glosses"],
            "predicted_sign": cfg["predicted_sign"],
            "status": "extraction_failed",
            "error": f"{type(e).__name__}: {e}",
        }

    rows = data["data"]
    fields = data["fields"]
    IDX = {name: i for i, name in enumerate(fields)}
    print(f"  merged raw STEDT rows          : {len(rows)}")

    if not rows:
        print("  zero STEDT rows returned; skipping.")
        return {
            "role": cfg["role"],
            "query_glosses": cfg["query_glosses"],
            "predicted_sign": cfg["predicted_sign"],
            "status": "zero_stedt_hits",
            "per_gloss_fetch_meta": data.get("_per_gloss_meta", []),
        }

    include = {g.lower() for g in cfg["include_glosses"]}
    exact = [r for r in rows
             if str(r[IDX["lexicon.gloss"]]).strip().lower() in include]
    print(f"  filtered to canonical gloss set "
          f"({len(cfg['include_glosses'])} tags): {len(exact)}")

    out_records: list[dict] = []
    unmatched: set[str] = set()
    match_stats: Counter = Counter()
    for r in exact:
        lg = r[IDX["languagenames.language"]]
        m = match_stedt_language(lg, lookup)
        if m is None:
            unmatched.add(lg)
            match_stats["unmatched"] += 1
            continue
        gc, la, lo, src, mode = m
        match_stats[mode.split(":")[0]] += 1
        raw_reflex = r[IDX["lexicon.reflex"]]
        etyma = r[IDX["analysis"]]
        etyma_str = "" if etyma is None else str(etyma).strip()
        out_records.append({
            "stedt_rn": r[IDX["lexicon.rn"]],
            "stedt_etyma_tag": etyma_str,
            "reflex": raw_reflex,
            "reflex_clean": _clean_reflex(raw_reflex),
            "gloss": r[IDX["lexicon.gloss"]],
            "stedt_lgid": r[IDX["languagenames.lgid"]],
            "stedt_language": lg,
            "stedt_group": r[IDX["languagegroups.grp"]],
            "citation": r[IDX["citation"]],
            "srcabbr": r[IDX["languagenames.srcabbr"]],
            "glottocode": gc,
            "latitude": la,
            "longitude": lo,
            "gc_source": src,
            "match_mode": mode,
            "stedt_url": (STEDT_BASE
                          + "/gnis?lexicon.lgid="
                          + str(r[IDX["languagenames.lgid"]])),
        })
    print(f"  rows mapped to Glottocode       : {len(out_records)}")
    print(f"  distinct STEDT names unmatched  : {len(unmatched)}")
    print(f"  match breakdown                 : {dict(match_stats)}")

    # Sanity check: name-match rate should be comparable to the full
    # pipeline's ~84% for RICE. We flag (not abort) if it drops below 50%.
    total = len(exact)
    matched = len(out_records)
    match_rate = (matched / total) if total else 0.0
    if total and match_rate < 0.5:
        print(f"  WARNING: match rate {match_rate:.1%} < 50% floor; "
              f"reporting anyway but flagging in JSON.")

    if not out_records:
        return {
            "role": cfg["role"],
            "query_glosses": cfg["query_glosses"],
            "predicted_sign": cfg["predicted_sign"],
            "status": "no_forms_mapped",
            "n_raw_forms": len(exact),
            "n_forms_matched_to_glottocode": 0,
            "match_stats": dict(match_stats),
            "match_rate": round(match_rate, 4),
            "unmatched_names": sorted(unmatched),
            "per_gloss_fetch_meta": data.get("_per_gloss_meta", []),
        }

    df = pd.DataFrame(out_records)
    df.to_csv(cfg["csv"], index=False)
    print(f"  wrote {cfg['csv'].relative_to(ROOT)} ({len(df)} rows)")

    # Method A: STEDT etyma tag
    tag = df["stedt_etyma_tag"].fillna("").astype(str).str.strip()
    df_a = df[(tag != "") & (tag.str.lower() != "nan")].copy()
    if len(df_a):
        per_a = (
            df_a.groupby("glottocode")
            .agg(
                cognate_class=("stedt_etyma_tag",
                               lambda s: majority_vote(list(s))),
                lat=("latitude", "mean"),
                lon=("longitude", "mean"),
            ).reset_index()
        )
        per_a["dist_km"] = per_a.apply(
            lambda r: haversine(HOMELAND_LAT, HOMELAND_LON,
                                r["lat"], r["lon"]), axis=1)
        rng = np.random.default_rng(SEED)
        res_a = gradient(per_a, "cognate_class", rng)
    else:
        res_a = {"n": 0, "note": "no rows with etyma tag"}

    # Method B: NLD single-link on cleaned reflexes
    reflexes_clean = df["reflex_clean"].astype(str).tolist()
    clusters = nld_single_link_cluster(reflexes_clean, NLD_COGNATE_THRESHOLD)
    df_b = df.copy()
    df_b["nld_cluster"] = clusters
    per_b = (
        df_b.groupby("glottocode")
        .agg(
            cognate_class=("nld_cluster",
                           lambda s: majority_vote(list(s))),
            lat=("latitude", "mean"),
            lon=("longitude", "mean"),
        ).reset_index()
    )
    per_b["dist_km"] = per_b.apply(
        lambda r: haversine(HOMELAND_LAT, HOMELAND_LON,
                            r["lat"], r["lon"]), axis=1)
    rng = np.random.default_rng(SEED)
    res_b = gradient(per_b, "cognate_class", rng)

    out = {
        "role": cfg["role"],
        "query_glosses": cfg["query_glosses"],
        "include_glosses": sorted(cfg["include_glosses"]),
        "predicted_sign": cfg["predicted_sign"],
        "n_raw_forms": int(len(exact)),
        "n_forms_matched_to_glottocode": int(len(df)),
        "n_unique_glottocodes": int(df["glottocode"].nunique()),
        "match_stats": dict(match_stats),
        "match_rate": round(match_rate, 4),
        "method_A_expert_etyma": {
            "description": ("STEDT proto-etyma tag as cognate class; "
                            "rows missing a tag excluded"),
            "n_rows_with_tag": int(len(df_a)) if len(df_a) else 0,
            **res_a,
        },
        "method_B_nld_cluster": {
            "description": ("Single-link clusters of cleaned reflexes by "
                            "normalised Levenshtein distance"),
            "threshold": NLD_COGNATE_THRESHOLD,
            **res_b,
        },
        "per_gloss_fetch_meta": data.get("_per_gloss_meta", []),
        "unmatched_names": sorted(unmatched),
    }

    def _fmt(res):
        return (f"n={res.get('n')} r={res.get('spearman_r')} "
                f"p={res.get('spearman_p')} "
                f"CI=[{res.get('ci_lower')}, {res.get('ci_upper')}]")
    print(f"  Method A (etyma tag): {_fmt(res_a)}")
    print(f"  Method B (NLD<=0.4) : {_fmt(res_b)}")
    return out


# -------------------------------------------------------------- empirical sign
def _empirical_sign(method_result: dict) -> str:
    r = method_result.get("spearman_r")
    p = method_result.get("spearman_p")
    if r is None or p is None:
        return "na"
    if p >= 0.05:
        return "0"  # non-significant
    return "+" if r > 0 else "-"


def _prediction_match(predicted: str, empirical: str) -> str:
    if empirical == "na":
        return "no-data"
    if predicted == "?":
        return "no-commitment"
    if predicted == empirical:
        return "match"
    if empirical == "0":
        return "null-result"
    return "mismatch"


# ------------------------------------------------------------------ main
def main() -> None:
    print("=" * 78)
    print("Sino-Tibetan expanded STEDT extraction")
    print("  concepts: MILLET + TEA + SILKWORM + HORSE + WHEAT")
    print(f"  homeland: {HOMELAND_NAME} "
          f"({HOMELAND_LAT} N, {HOMELAND_LON} E)")
    print("=" * 78)

    lookup, roster = build_glottocode_lookup()
    print(f"  glottocode lookup entries           : {len(lookup)}")
    print(f"  target ST roster size (Glottocodes) : {len(roster)}")

    per_concept: dict[str, dict] = {}
    unmatched_by_concept: dict[str, list[str]] = {}
    for cfg in CONCEPTS:
        res = process_concept(cfg, lookup)
        per_concept[cfg["concept"]] = res
        unmatched_by_concept[cfg["concept"]] = res.pop("unmatched_names",
                                                       []) or []

    # Pre-registration ledger: pred-sign vs empirical-sign per concept.
    predictions_ledger = {}
    for cfg in CONCEPTS:
        c = cfg["concept"]
        r = per_concept.get(c, {})
        mB = r.get("method_B_nld_cluster", {})
        mA = r.get("method_A_expert_etyma", {})
        emp_b = _empirical_sign(mB)
        emp_a = _empirical_sign(mA)
        predictions_ledger[c] = {
            "predicted_sign": cfg["predicted_sign"],
            "empirical_sign_method_A": emp_a,
            "empirical_sign_method_B": emp_b,
            "prediction_vs_method_B": _prediction_match(
                cfg["predicted_sign"], emp_b,
            ),
            "n_method_A": mA.get("n"),
            "n_method_B": mB.get("n"),
            "r_method_A": mA.get("spearman_r"),
            "r_method_B": mB.get("spearman_r"),
            "p_method_A": mA.get("spearman_p"),
            "p_method_B": mB.get("spearman_p"),
        }

    # New positive-significant anchors survive the pre-registration test
    # if predicted "+" AND empirical method-B sign is "+".
    positive_anchors = [
        c for c, v in predictions_ledger.items()
        if v["predicted_sign"] == "+"
        and v["empirical_sign_method_B"] == "+"
        and v["prediction_vs_method_B"] == "match"
    ]

    # Any concept, regardless of prediction, that shows significant
    # positive gradient on method B. Paper 2's ST story pivots on this
    # list being non-empty.
    any_pos_significant = [
        c for c, v in predictions_ledger.items()
        if v["empirical_sign_method_B"] == "+"
    ]

    output = {
        "family": "Sino-Tibetan",
        "data_source": "STEDT bulk via stedtdb.johnblowe.com AJAX",
        "data_source_url_template":
            STEDT_AJAX + "?tbl=lexicon&s=<concept>",
        "extraction_date": date.today().isoformat(),
        "extension_of": "results/xfam_st_stedt_full.json",
        "homeland": {
            "name": HOMELAND_NAME,
            "lat": HOMELAND_LAT,
            "lon": HOMELAND_LON,
        },
        "target_roster_size": len(roster),
        "method": {
            "name_matching": (
                "normalise -> exact -> drop-parens -> drop-brackets "
                "-> alias table -> token>=4 chars -> "
                "difflib cutoff=0.85"
            ),
            "homeland_distance": "haversine km",
            "retention_rule": (
                "cognate class == global-mode class "
                "(ties broken lexicographically)"
            ),
            "bootstrap": {
                "n": N_BOOTSTRAP,
                "seed": SEED,
                "method": "percentile",
            },
            "nld_threshold": NLD_COGNATE_THRESHOLD,
            "glottocode_lookup_sources": [
                "data/raw/zhang2019/zhang2019_data.xlsx Language Info",
                "data/raw/peirosst/cldf/languages.csv",
                "data/raw/sagartst/cldf/languages.csv",
            ],
        },
        "predictions": predictions_ledger,
        "concepts": per_concept,
        "unmatched_stedt_names": unmatched_by_concept,
        "comparison_to_previous": {
            "stedt_full_RICE_methodB": {
                "n": 99, "r": -0.18,
                "note": "refuted; RICE has no positive gradient at scale",
            },
        },
        "verdict": {
            "new_positive_predicted_anchors": positive_anchors,
            "any_method_B_significant_positive": any_pos_significant,
            "paper_2_ST_story_updated": (
                "new_positive_anchor_candidate"
                if any_pos_significant
                else "still_nothing_positive"
            ),
        },
    }
    with OUT_JSON.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    # ---------------------------------------------------------- console
    print("\n" + "=" * 78)
    print("FINAL SUMMARY")
    print("=" * 78)
    header = (f"{'concept':<10} {'pred':>4} {'n_forms':>8} {'n_langs':>8}  "
              f"{'methA r(p,n)':<32}  {'methB r(p,n)':<32}  "
              f"{'verdict(B)':<14}")
    print(header)
    print("-" * len(header))
    for cfg in CONCEPTS:
        c = cfg["concept"]
        r = per_concept.get(c, {})
        nf = r.get("n_forms_matched_to_glottocode", 0)
        ng = r.get("n_unique_glottocodes", 0)
        a = r.get("method_A_expert_etyma", {})
        b = r.get("method_B_nld_cluster", {})
        a_str = (f"r={a.get('spearman_r')} "
                 f"p={a.get('spearman_p')} n={a.get('n')}")
        b_str = (f"r={b.get('spearman_r')} "
                 f"p={b.get('spearman_p')} n={b.get('n')}")
        v = predictions_ledger[c]["prediction_vs_method_B"]
        print(f"{c:<10} {cfg['predicted_sign']:>4} {nf:>8} {ng:>8}  "
              f"{a_str:<32}  {b_str:<32}  {v:<14}")

    print()
    if positive_anchors:
        print(f"Pre-registered positive prediction confirmed for: "
              f"{', '.join(positive_anchors)}")
    else:
        print("No pre-registered positive prediction confirmed.")
    if any_pos_significant:
        print(f"Any method-B significant positive (post-hoc): "
              f"{', '.join(any_pos_significant)}")
    else:
        print("No concept reaches method-B significance with rho>0.")

    print(f"\nST-story verdict: "
          f"{output['verdict']['paper_2_ST_story_updated']}")
    print(f"\nWrote {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
