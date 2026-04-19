#!/usr/bin/env python3
"""Sino-Tibetan RICE migration-gradient POC using STEDT data.

Current ST RICE result from sagartst (50 langs) is n=23, rho=+0.271, p=0.21 --
direction correct but underpowered. STEDT (Sino-Tibetan Etymological Dictionary
and Thesaurus) is the only comprehensive ST lexical source that includes RICE.
This script is a proof-of-concept extraction to see whether a RICE-anchor test
is feasible at larger n.

Approach
--------
1. Pull RICE forms from STEDT's AJAX endpoint (lexicon table; gloss="rice").
   Endpoint: https://stedtdb.johnblowe.com/search/ajax?tbl=lexicon&s=rice
   STEDT redirects from stedt.berkeley.edu/~stedt-cgi/ to the johnblowe mirror.

2. Filter to rows with exact gloss "rice" (2626 broader matches, 137 exact).

3. Map STEDT language names -> Glottocode/Lat/Lon using a roster assembled from:
     - data/raw/zhang2019/zhang2019_data.xlsx  (109 ST, Glottocode + lat/lon)
     - data/raw/peirosst/cldf/languages.csv   ( 91 ST with latlon)
     - data/raw/sagartst/cldf/languages.csv   ( 50 ST, Glottocode + lat/lon)
   Names are normalised (lowercase, strip parens/brackets/underscores) and
   matched exact -> base-without-parens -> alias -> token -> difflib fuzzy.

4. Cognate classification -- two independent methods:
     A. STEDT etyma-tag (field "analysis" in AJAX reply). This is STEDT's own
        proto-reconstruction pointer; rows sharing a tag are cognate per STEDT.
     B. Normalised Levenshtein clustering (threshold 0.4) on STEDT reflexes --
        gives an LexStat-style automatic baseline when etyma tags are missing.
   Per-Glottocode: majority-vote cognate class.

5. Global-mode cognate class = proto. Retention R_j = 1 iff language j's class
   equals the global mode. Haversine from Upper Yellow River (35 N, 104 E).
   Spearman rho(distance, retention), two-tailed p, 500-resample percentile
   bootstrap 95% CI, seed 42.

Output
------
- data/raw/stedt/rice_forms.csv   -- per-row extraction with form, gloss,
  stedt_lgid, stedt_lg_name, glottocode, lat, lon, etyma_tag, source URL.
- results/xfam_st_stedt_rice_poc.json -- gradient result + comparison to
  sagartst baseline (n=23, r=0.271, p=0.21).
"""

from __future__ import annotations

import difflib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from austronesian.analysis.distance import (  # noqa: E402
    normalized_levenshtein_distance,
)

# ---- constants --------------------------------------------------------------
STEDT_BASE = "https://stedtdb.johnblowe.com"
STEDT_AJAX = STEDT_BASE + "/search/ajax"
USER_AGENT = "Austronesian-research-tool/0.1 (contact: augchao@gms.npu.edu.tw)"
REQUEST_DELAY_S = 0.5  # polite to STEDT academic server

HOMELAND_LAT = 35.0
HOMELAND_LON = 104.0
HOMELAND_NAME = "Upper Yellow River basin"

SEED = 42
N_BOOTSTRAP = 500
NLD_COGNATE_THRESHOLD = 0.4  # single-link clustering cutoff

STEDT_DIR = ROOT / "data" / "raw" / "stedt"
STEDT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

RICE_RAW_JSON = STEDT_DIR / "rice_ajax_raw.json"
RICE_CSV = STEDT_DIR / "rice_forms.csv"
OUT_JSON = RESULTS / "xfam_st_stedt_rice_poc.json"

PREV_SAGART = {"previous_r": 0.271, "previous_n": 23, "previous_p": 0.21}

# STEDT-spelling aliases. Only used if direct + fuzzy match fail.
# Keys are normalised (lowercase, parens/brackets removed).
STEDT_NAME_ALIASES = {
    "rgyalrong": "japhug",
    "cuona menba": "tawang",
    "nishing": "nisi",
    "kaman miju": "miju",
    "darang taraon": "idu",
    "tsangla central": "central monpa",
    "yi nanhua": "yi",
    "yi xide": "yi",
    "yi liangshan": "yi",
    "pumi lanping": "pumi",
    "pumi jiulong": "pumi",
    "hani pijo": "hani",
    "hani khatu": "hani",
    "hani wordlist": "hani",
    "jinuo youle": "jinuo",
    "aka hruso": "hruso",
}


# ---- STEDT fetch ------------------------------------------------------------
def fetch_stedt_rice(force_refresh: bool = False) -> dict:
    """Pull lexicon rows with gloss matching "rice"; cache to disk."""
    if RICE_RAW_JSON.exists() and not force_refresh:
        with RICE_RAW_JSON.open(encoding="utf-8") as fh:
            return json.load(fh)

    params = {"tbl": "lexicon", "s": "rice", "f": ""}
    url = STEDT_AJAX + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    time.sleep(REQUEST_DELAY_S)
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8")
    data = json.loads(raw)
    with RICE_RAW_JSON.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)
    return data


# ---- language-name -> glottocode lookup -------------------------------------
def _norm(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"[\(\)\[\]]", "", s)
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def build_glottocode_lookup() -> dict[str, tuple]:
    """Aggregate ST Glottocode+latlon from zhang2019, peirosst, sagartst."""
    lookup: dict[str, tuple] = {}

    def _add(name, gc, la, lo, src):
        if not isinstance(name, str) or not name.strip():
            return
        if gc is None or (isinstance(gc, float) and np.isnan(gc)):
            return
        if la is None or lo is None:
            return
        nn = _norm(name)
        if nn and nn not in lookup:
            lookup[nn] = (str(gc), float(la), float(lo), src)

    # 1) Zhang 2019 (109 ST langs, Glottocode + Lat/Long + STEDT metadata)
    z19_xlsx = ROOT / "data" / "raw" / "zhang2019" / "zhang2019_data.xlsx"
    if z19_xlsx.exists():
        z19 = pd.read_excel(z19_xlsx, sheet_name="Language Info")
        for _, r in z19.iterrows():
            if pd.notna(r["Glottocode"]) and pd.notna(r["Latitude"]):
                _add(r["Name"], r["Glottocode"], r["Latitude"], r["Longitude"],
                     "zhang2019")
                if "Alternative Name" in r and pd.notna(r["Alternative Name"]):
                    for alt in str(r["Alternative Name"]).split(","):
                        _add(alt.strip(), r["Glottocode"],
                             r["Latitude"], r["Longitude"], "zhang2019_alt")

    # 2) peirosst
    pst_csv = ROOT / "data" / "raw" / "peirosst" / "cldf" / "languages.csv"
    if pst_csv.exists():
        p = pd.read_csv(pst_csv)
        for _, r in p.iterrows():
            if pd.notna(r["Glottocode"]) and pd.notna(r["Latitude"]):
                _add(r["Name"], r["Glottocode"], r["Latitude"], r["Longitude"],
                     "peirosst")
                if "Glottolog_Name" in r and pd.notna(r["Glottolog_Name"]):
                    _add(r["Glottolog_Name"], r["Glottocode"],
                         r["Latitude"], r["Longitude"], "peirosst_gl")

    # 3) sagartst
    sst_csv = ROOT / "data" / "raw" / "sagartst" / "cldf" / "languages.csv"
    if sst_csv.exists():
        s = pd.read_csv(sst_csv)
        for _, r in s.iterrows():
            if pd.notna(r["Glottocode"]) and pd.notna(r["Latitude"]):
                _add(r["Name"], r["Glottocode"], r["Latitude"], r["Longitude"],
                     "sagartst")
                if "Glottolog_Name" in r and pd.notna(r["Glottolog_Name"]):
                    _add(r["Glottolog_Name"], r["Glottocode"],
                         r["Latitude"], r["Longitude"], "sagartst_gl")
                if "Name_in_Source" in r and pd.notna(r["Name_in_Source"]):
                    _add(r["Name_in_Source"], r["Glottocode"],
                         r["Latitude"], r["Longitude"], "sagartst_src")

    return lookup


def match_stedt_language(stedt_name: str,
                         lookup: dict[str, tuple]) -> tuple | None:
    """Return (glottocode, lat, lon, src, match_mode) or None."""
    if stedt_name.startswith("*"):  # STEDT proto-forms; skip
        return None
    nn = _norm(stedt_name)
    if nn in lookup:
        return (*lookup[nn], "exact")

    base = re.sub(r"\s*\([^)]*\)", "", stedt_name).strip()
    nb = _norm(base)
    if nb in lookup:
        return (*lookup[nb], "base")

    if nb in STEDT_NAME_ALIASES and STEDT_NAME_ALIASES[nb] in lookup:
        return (*lookup[STEDT_NAME_ALIASES[nb]], "alias")
    if nn in STEDT_NAME_ALIASES and STEDT_NAME_ALIASES[nn] in lookup:
        return (*lookup[STEDT_NAME_ALIASES[nn]], "alias")

    # Token match: split on non-word, try each token >= 4 chars
    plain = re.sub(r"[\[\]\(\)]", "", stedt_name)
    for tok in re.split(r"[,/\s]+", plain):
        tn = _norm(tok)
        if tn and len(tn) >= 4 and tn in lookup:
            return (*lookup[tn], "token:" + tok)

    # difflib fuzzy, cutoff 0.85 to avoid false positives
    close = difflib.get_close_matches(nn, lookup.keys(), n=1, cutoff=0.85)
    if close:
        return (*lookup[close[0]], "fuzzy:" + close[0])
    return None


# ---- cognate clustering -----------------------------------------------------
def nld_single_link_cluster(reflexes: list[str],
                            threshold: float) -> list[int]:
    """Single-linkage cluster of reflexes by normalised Levenshtein distance.

    Returns a list of cluster ids (0-indexed), aligned with reflexes.
    """
    n = len(reflexes)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if normalized_levenshtein_distance(reflexes[i], reflexes[j]) \
                    <= threshold:
                union(i, j)
    roots = [find(i) for i in range(n)]
    rank = {r: k for k, r in enumerate(sorted(set(roots)))}
    return [rank[r] for r in roots]


# ---- gradient analysis ------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))
         * np.sin(dlon / 2) ** 2)
    return float(R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


def majority_vote(classes: list) -> str:
    counts = Counter(classes)
    top = max(counts.values())
    tied = sorted(str(c) for c, v in counts.items() if v == top)
    return tied[0]


def gradient(per_lang: pd.DataFrame, class_col: str,
             rng: np.random.Generator) -> dict:
    """Compute Spearman rho(distance, retention) + bootstrap CI."""
    n = len(per_lang)
    if n < 5:
        return {"n": int(n), "note": "n<5, skipped"}
    class_counts = Counter(per_lang[class_col])
    top = max(class_counts.values())
    tied = sorted(str(c) for c, v in class_counts.items() if v == top)
    proto = tied[0]
    retained = (per_lang[class_col].astype(str) == proto).astype(int).to_numpy()
    dist = per_lang["dist_km"].to_numpy(dtype=float)
    if retained.std() == 0 or dist.std() == 0:
        r_obs, p_obs = float("nan"), float("nan")
    else:
        r_obs, p_obs = spearmanr(dist, retained)
        r_obs, p_obs = float(r_obs), float(p_obs)

    boot = np.empty(N_BOOTSTRAP)
    idx = np.arange(n)
    for b in range(N_BOOTSTRAP):
        samp = rng.choice(idx, size=n, replace=True)
        d_b, r_b = dist[samp], retained[samp]
        if r_b.std() == 0 or d_b.std() == 0:
            boot[b] = np.nan
            continue
        rr, _ = spearmanr(d_b, r_b)
        boot[b] = rr
    boot = boot[~np.isnan(boot)]
    ci_lo = float(np.percentile(boot, 2.5)) if boot.size >= 10 else float("nan")
    ci_hi = float(np.percentile(boot, 97.5)) if boot.size >= 10 else float("nan")

    return {
        "n": int(n),
        "n_cognate_classes": int(len(class_counts)),
        "retention_rate": float(retained.mean()),
        "spearman_r": None if np.isnan(r_obs) else round(r_obs, 4),
        "spearman_p": None if np.isnan(p_obs) else round(p_obs, 6),
        "ci_lower": None if np.isnan(ci_lo) else round(ci_lo, 4),
        "ci_upper": None if np.isnan(ci_hi) else round(ci_hi, 4),
        "proto_cognate_class": proto,
        "n_tied_top_classes": int(len(tied)),
    }


# ---- main -------------------------------------------------------------------
def main() -> None:
    print("=" * 78)
    print("ST RICE migration-gradient POC via STEDT")
    print("Homeland:", HOMELAND_NAME, f"({HOMELAND_LAT} N, {HOMELAND_LON} E)")
    print("=" * 78)

    # 1. Fetch STEDT rice
    try:
        data = fetch_stedt_rice()
    except Exception as e:
        # Feasibility failure report
        fail = {
            "family": "Sino-Tibetan",
            "concept": "RICE",
            "status": "extraction_failed",
            "access_method_attempted": (
                "HTTP GET "
                "https://stedtdb.johnblowe.com/search/ajax?tbl=lexicon&s=rice"
            ),
            "obstacles": f"{type(e).__name__}: {e}",
            "alternative_paths": [
                "Email STEDT maintainers (stedt-project-admin) for SQL dump",
                "Dryad archive doi:10.6078/D1159Q (svn code snapshot, no data)",
                "Use digling/sinotibetan GitHub TSVs (40 doculects, 240 concepts)",
            ],
        }
        with OUT_JSON.open("w", encoding="utf-8") as fh:
            json.dump(fail, fh, indent=2, ensure_ascii=False)
        print("FAILED:", fail["obstacles"])
        print("wrote", OUT_JSON)
        return

    rows = data["data"]
    fields = data["fields"]
    print(f"  STEDT rows (s=rice broad match): {len(rows)}")

    def _clean_reflex(s: str) -> str:
        """Strip STEDT morpheme markers, tone digits, and IPA superscripts.

        STEDT reflexes carry notation like 'em-|bĩ' (morpheme), 'tshen⁵⁵'
        (tone digits), 'ˉᵐbraˀ s' (modifiers). Auto-cognate clustering by NLD
        is severely hurt by this noise, so we normalise before clustering.
        """
        s = str(s)
        s = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹ˊˋˆˇ◦ˉ˜ʔʰʱː]", "", s)
        s = re.sub(r"\d+", "", s)
        s = s.replace("|", "").replace("-", "")
        s = re.sub(r"\s+", "", s)
        return s.lower()

    # Field indices (per STEDT AJAX response)
    # ['lexicon.rn','analysis','lexicon.reflex','lexicon.gloss','lexicon.gfn',
    #  'languagenames.lgid','languagenames.language','languagegroups.grpid',
    #  'languagegroups.grpno','languagegroups.grp','citation',
    #  'languagenames.srcabbr','lexicon.srcid','lexicon.semkey',
    #  'chapters.chaptertitle','num_notes']
    IDX = {name: i for i, name in enumerate(fields)}

    # 2. Filter to exact rice gloss
    exact = [r for r in rows if r[IDX["lexicon.gloss"]].strip().lower() == "rice"]
    print(f"  exact gloss=='rice': {len(exact)}")

    # 3. Build glottocode lookup and match each row
    lookup = build_glottocode_lookup()
    print(f"  Glottocode lookup entries (from zhang2019+peirosst+sagartst): "
          f"{len(lookup)}")

    out_records: list[dict] = []
    unmatched_langs: set[str] = set()
    match_stats: Counter = Counter()
    for r in exact:
        lg = r[IDX["languagenames.language"]]
        m = match_stedt_language(lg, lookup)
        if m is None:
            unmatched_langs.add(lg)
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
    print(f"  rows mapped to Glottocode: {len(out_records)}")
    print(f"  distinct STEDT langs unmatched: {len(unmatched_langs)}")
    print(f"  match-mode breakdown: {dict(match_stats)}")

    if not out_records:
        fail = {
            "family": "Sino-Tibetan",
            "concept": "RICE",
            "status": "extraction_failed",
            "access_method_attempted":
                "STEDT AJAX tbl=lexicon&s=rice (successful, "
                f"{len(rows)} rows) but none mapped to Glottocode",
            "obstacles":
                "STEDT language names do not intersect zhang2019/peirosst/"
                "sagartst Glottocode roster after normalisation+fuzzy matching",
            "alternative_paths": [
                "Manually curate STEDT lgid -> Glottocode table "
                "(~300 languages) using Glottolog 5.0 JSON",
                "Use LingPy auto-cognate plus Glottolog geocoder",
            ],
        }
        with OUT_JSON.open("w", encoding="utf-8") as fh:
            json.dump(fail, fh, indent=2, ensure_ascii=False)
        print("wrote", OUT_JSON)
        return

    # 4. Write per-row CSV (all matched rows)
    df = pd.DataFrame(out_records)
    df.to_csv(RICE_CSV, index=False)
    print(f"  wrote {RICE_CSV} ({len(df)} rows)")

    # 5. Per-Glottocode cognate class (two methods)
    #    Method A: STEDT etyma tag (drop rows with no tag; "nan" is NOT a tag)
    tag_str = df["stedt_etyma_tag"].fillna("").astype(str).str.strip()
    df_a = df[(tag_str != "") & (tag_str.str.lower() != "nan")].copy()
    per_lang_a = (
        df_a.groupby("glottocode")
        .agg(
            cognate_class=("stedt_etyma_tag",
                           lambda s: majority_vote(list(s))),
            lat=("latitude", "mean"),
            lon=("longitude", "mean"),
            stedt_langs=("stedt_language",
                         lambda s: "|".join(sorted(set(s)))),
        )
        .reset_index()
    )
    per_lang_a["dist_km"] = per_lang_a.apply(
        lambda r: haversine(HOMELAND_LAT, HOMELAND_LON, r["lat"], r["lon"]),
        axis=1,
    )

    #    Method B: normalised Levenshtein single-link clusters on CLEANED
    #    reflexes (tone digits, morpheme bars, modifiers stripped).
    reflexes_clean = df["reflex_clean"].astype(str).tolist()
    clusters = nld_single_link_cluster(reflexes_clean, NLD_COGNATE_THRESHOLD)
    df_b = df.copy()
    df_b["nld_cluster"] = clusters
    per_lang_b = (
        df_b.groupby("glottocode")
        .agg(
            cognate_class=("nld_cluster",
                           lambda s: majority_vote(list(s))),
            lat=("latitude", "mean"),
            lon=("longitude", "mean"),
        )
        .reset_index()
    )
    per_lang_b["dist_km"] = per_lang_b.apply(
        lambda r: haversine(HOMELAND_LAT, HOMELAND_LON, r["lat"], r["lon"]),
        axis=1,
    )

    # 6. Gradients
    rng = np.random.default_rng(SEED)
    res_a = gradient(per_lang_a, "cognate_class", rng)
    rng = np.random.default_rng(SEED)  # reset for B
    res_b = gradient(per_lang_b, "cognate_class", rng)

    # Primary = STEDT etyma tag if n_A >= 10, else NLD. Etyma tags are
    # philology-grade; NLD clusters on noisy STEDT reflexes are brittle.
    if res_a.get("n", 0) >= 10:
        primary_method = "STEDT-etyma-tag"
        primary_res = res_a
    else:
        primary_method = "LexStat-NLD"
        primary_res = res_b

    # 7. Assemble final JSON
    output = {
        "family": "Sino-Tibetan",
        "concept": "RICE",
        "data_source": "STEDT (stedtdb.johnblowe.com), gloss='rice', "
                       "AJAX tbl=lexicon",
        "data_source_url": STEDT_AJAX + "?tbl=lexicon&s=rice",
        "n_languages_queried": len(set(r[IDX["languagenames.language"]]
                                       for r in exact)),
        "n_forms_extracted": int(len(df)),
        "n_stedt_langs_unmatched": len(unmatched_langs),
        "n_glottocodes_with_form": int(df["glottocode"].nunique()),
        "match_stats": dict(match_stats),
        "primary_cognate_method": primary_method,
        "results_by_method": {
            "STEDT-etyma-tag": {
                "description":
                    "STEDT proto-etyma tag as cognate class; rows missing a "
                    "tag were excluded",
                "n_rows_with_tag": int(len(df_a)),
                **res_a,
            },
            "LexStat-NLD": {
                "description":
                    f"Single-link clusters of reflexes by normalised "
                    f"Levenshtein distance <= {NLD_COGNATE_THRESHOLD}",
                "nld_threshold": NLD_COGNATE_THRESHOLD,
                **res_b,
            },
        },
        # Top-level shorthand matching the schema contract
        "n_cognate_classes": primary_res.get("n_cognate_classes"),
        "retention_rate": primary_res.get("retention_rate"),
        "spearman_r": primary_res.get("spearman_r"),
        "spearman_p": primary_res.get("spearman_p"),
        "ci_lower": primary_res.get("ci_lower"),
        "ci_upper": primary_res.get("ci_upper"),
        "homeland": {
            "name": HOMELAND_NAME,
            "lat": HOMELAND_LAT,
            "lon": HOMELAND_LON,
        },
        "comparison_to_sagartst": PREV_SAGART,
        "method": {
            "homeland_distance": "haversine km",
            "retention_rule":
                "cognate class == global-mode class (ties broken "
                "lexicographically)",
            "bootstrap": {"n": N_BOOTSTRAP, "seed": SEED,
                          "method": "percentile"},
            "glottocode_lookup_sources": [
                "data/raw/zhang2019/zhang2019_data.xlsx Language Info",
                "data/raw/peirosst/cldf/languages.csv",
                "data/raw/sagartst/cldf/languages.csv",
            ],
            "name_matching": (
                "normalise -> exact -> drop-parens -> alias table -> "
                "token>=4 chars -> difflib cutoff=0.85"
            ),
        },
        "unmatched_stedt_languages": sorted(unmatched_langs),
    }

    with OUT_JSON.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    # Console summary
    print()
    print("-" * 78)
    print(f"  Method A (STEDT etyma tag): n={res_a.get('n')}  "
          f"r={res_a.get('spearman_r')}  p={res_a.get('spearman_p')}  "
          f"CI=[{res_a.get('ci_lower')}, {res_a.get('ci_upper')}]")
    print(f"  Method B (NLD <= {NLD_COGNATE_THRESHOLD}):   "
          f"n={res_b.get('n')}  r={res_b.get('spearman_r')}  "
          f"p={res_b.get('spearman_p')}  CI=[{res_b.get('ci_lower')}, "
          f"{res_b.get('ci_upper')}]")
    print("-" * 78)
    print(f"  sagartst baseline: n=23  r=+0.271  p=0.21  "
          f"(primary -> {primary_method})")
    print(f"\nSaved: {OUT_JSON}")
    print(f"Saved: {RICE_CSV}")


if __name__ == "__main__":
    main()
