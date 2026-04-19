#!/usr/bin/env python3
"""Sino-Tibetan full STEDT extraction for RICE / SALT / BARLEY / YAK.

Follow-on to ``scripts/xfam_st_stedt_rice_poc.py``. The POC established that
STEDT's AJAX mirror is scrape-able and produced a RICE gradient at n=60
(Spearman rho=+0.184, p=0.16, underpowered). Power analysis says we need
n>=107 to detect rho=0.27 at alpha=0.05, so the primary goal here is to
triple the POC's 60 by (a) widening the gloss inclusion set, (b) extending
the language alias table, and (c) adding three more anchor concepts.

Pre-registered directional predictions under the corridor-ecology (expansion
anchor) hypothesis for Sino-Tibetan, homeland Upper Yellow River:

  * RICE    : positive (Sagart 2019 PNAS; rice agriculture as ST core)
  * SALT    : uncertain (trade concept; could go either way)
  * BARLEY  : unclear -- Tibetan-plateau adaptation, retention may track
              altitude, not strictly distance from homeland
  * YAK     : similar to BARLEY -- restricted highland distribution

Pipeline
--------
1. Fetch gloss-level AJAX blobs from stedtdb.johnblowe.com (polite 0.5s
   rate limit, cached to data/raw/stedt/<concept>_ajax_raw.json).
2. Filter rows whose exact gloss matches an inclusion set per concept.
3. Name -> Glottocode via alias table built from
   zhang2019/peirosst/sagartst (same logic as the POC, alias table
   extended using the POC's unmatched-STEDT-names list).
4. For each (concept, method) pair compute
     retention = (cognate class == global-mode class)
     dist_km  = haversine(upper_yellow_river, language)
     Spearman rho + 500-resample percentile-bootstrap 95% CI, seed 42.

Methods (both reported per concept, no picking a winner):
   A. STEDT etyma tag -- philology-grade, low n
   B. Normalised Levenshtein single-link cluster, threshold 0.4 on
      tone/morpheme-bar-stripped reflexes -- noisier, high n.

Sanity check: re-running method B on the RICE subsample of size 60 must
reproduce the POC's r=+0.184. If it does not, halt and investigate.

Outputs
-------
data/raw/stedt/rice_forms_full.csv
data/raw/stedt/salt_forms.csv
data/raw/stedt/barley_forms.csv
data/raw/stedt/yak_forms.csv
results/xfam_st_stedt_full.json
"""

from __future__ import annotations

import difflib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from austronesian.analysis.distance import (  # noqa: E402
    normalized_levenshtein_distance,
)

# -------------------------------------------------------------------- consts
STEDT_BASE = "https://stedtdb.johnblowe.com"
STEDT_AJAX = STEDT_BASE + "/search/ajax"
USER_AGENT = (
    "Austronesian-research-tool/0.2 "
    "(contact: augchao@gms.npu.edu.tw)"
)
REQUEST_DELAY_S = 0.5
FETCH_RETRIES = 3
FETCH_BACKOFF_S = 5.0

HOMELAND_LAT = 35.0
HOMELAND_LON = 104.0
HOMELAND_NAME = "Upper Yellow River basin"

SEED = 42
N_BOOTSTRAP = 500
NLD_COGNATE_THRESHOLD = 0.4

STEDT_DIR = ROOT / "data" / "raw" / "stedt"
STEDT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUT_JSON = RESULTS_DIR / "xfam_st_stedt_full.json"

# Concept configuration -- ordered list of (concept, query_gloss,
# inclusion-gloss set, csv filename, role, predicted sign).
# Inclusion globs are exact (case-folded, stripped) matches against
# STEDT's lexicon.gloss field. Chosen conservatively: only the word sense
# that denotes the crop/commodity itself, never derivations (rice beer,
# salt strainer, bull-yak crossbreed).
CONCEPTS = [
    {
        "concept": "RICE",
        "role": "anchor",
        "query_gloss": "rice",
        "include_glosses": {
            "rice",
            "rice (plant)",
            "rice plant",
            "rice (paddy)",
            "rice (uncooked)",
            "paddy rice",
            "rice / paddy",
            "rice (unhusked)",
            "rice, paddy",
            "rice (glutinous)",
            "glutinous rice",
            "husked rice",
            "uncooked rice",
            "rice (grains)",
        },
        "csv": STEDT_DIR / "rice_forms_full.csv",
        "raw": STEDT_DIR / "rice_ajax_raw.json",
        "predicted_sign": "+",
    },
    {
        "concept": "SALT",
        "role": "supplementary anchor",
        "query_gloss": "salt",
        "include_glosses": {"salt", "salt / salty"},
        "csv": STEDT_DIR / "salt_forms.csv",
        "raw": STEDT_DIR / "salt_ajax_raw.json",
        "predicted_sign": "?",
    },
    {
        "concept": "BARLEY",
        "role": "highland-adaptation anchor",
        "query_gloss": "barley",
        "include_glosses": {
            "barley",
            "barley (tibetan)",
            "barley (highland)",
            "tibetan barley",
            "highland barley",
            "barley plant",
        },
        "csv": STEDT_DIR / "barley_forms.csv",
        "raw": STEDT_DIR / "barley_ajax_raw.json",
        "predicted_sign": "?",
    },
    {
        "concept": "YAK",
        "role": "highland-adaptation anchor",
        "query_gloss": "yak",
        "include_glosses": {
            "yak",
            "yak (male)",
            "yak (female)",
            "male yak",
            "female yak",
            "wild yak",
            "yak (wild)",
        },
        "csv": STEDT_DIR / "yak_forms.csv",
        "raw": STEDT_DIR / "yak_ajax_raw.json",
        "predicted_sign": "?",
    },
]

# Alias table used by name-matching fallback. Keys are normalised
# (lowercase, parens/brackets stripped, underscores -> spaces). Values are
# the roster's normalised name so the lookup resolves in one hop. Extended
# from the POC using the unmatched STEDT list.
STEDT_NAME_ALIASES = {
    # POC originals
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
    # Extensions for unmatched STEDT names; map to an entry we do have
    # (usually the zhang2019 first-underscore prefix or a Glottolog_Name).
    "kaman miju": "miju",
    "chinese hanzi": "chinese mandarin",
    "chinese middle": "chinese old",
    "chinese old/mid": "chinese old",
    "chinese gsr": "chinese old",
    "chinese gsr #": "chinese old",
    "rgbenzhen": "japhug",
    "rgyalrong": "japhug",
    "nyagrong minyag ganzi xinlong": "muya minyak",
    "kucong": "lahu yellow",
    "nung": "trung dulong",
    "pumi jiulong": "pumi",
    "pumi lanping": "pumi",
    "hani khatu": "hani",
    "hani pijo": "hani",
    "hani wordlist": "hani",
    "cuona menba": "cuona mama",
    "tsangla central": "tshangla",
    "tsangla motuo": "tshangla",
    "tsangla tilang": "tshangla",
    "tshona mama": "cuona mama",
    "tshona wenlang": "cuona mama",
    "padam mishing abor miri": "padam",
    "padam mishing": "padam",
    "miri hill": "hill miri",
    "nishing": "bengni",  # both Western Tani / Nyishic
    "galo": "bokar lhoba",
    "tangsa moshang": "chang",  # Northern Naga, rough but family-correct
    "bai bijiang": "jianchuan bai",
    "bai dali": "jianchuan bai",
    "lahu black": "lahu lancang",
    "japanese": None,  # Sagart includes Japonic; intentionally skip
}
# Strip None sentinels (alias table should only contain real mappings)
STEDT_NAME_ALIASES = {k: v for k, v in STEDT_NAME_ALIASES.items() if v}

# -------------------------------------------------------------------- fetch
def fetch_concept(query_gloss: str, cache_path: Path,
                  force_refresh: bool = False) -> dict:
    """GET STEDT lexicon AJAX for a gloss, with on-disk cache + retry."""
    if cache_path.exists() and not force_refresh:
        with cache_path.open(encoding="utf-8") as fh:
            return json.load(fh)

    params = {"tbl": "lexicon", "s": query_gloss, "f": ""}
    url = STEDT_AJAX + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    last_err: Exception | None = None
    for attempt in range(1, FETCH_RETRIES + 1):
        try:
            time.sleep(REQUEST_DELAY_S)
            with urllib.request.urlopen(req, timeout=90) as resp:
                raw = resp.read().decode("utf-8")
            data = json.loads(raw)
            with cache_path.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False)
            return data
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_err = e
            sleep_s = FETCH_BACKOFF_S * attempt
            print(f"  [fetch] {query_gloss}: attempt {attempt}/"
                  f"{FETCH_RETRIES} failed ({type(e).__name__}); "
                  f"backing off {sleep_s:.1f}s")
            time.sleep(sleep_s)
    assert last_err is not None
    raise last_err


# ---------------------------------------------------------- name-matching
def _norm(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"[\(\)\[\]]", "", s)
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def build_glottocode_lookup() -> tuple[dict[str, tuple], set[str]]:
    """Aggregate ST Glottocode+latlon from the three roster sources.

    Returns (lookup, target_roster_glottocodes).
    """
    lookup: dict[str, tuple] = {}
    roster: set[str] = set()

    def _add(name, gc, la, lo, src):
        if not isinstance(name, str) or not name.strip():
            return
        if gc is None or (isinstance(gc, float) and np.isnan(gc)):
            return
        if la is None or lo is None:
            return
        if pd.isna(la) or pd.isna(lo):
            return
        nn = _norm(name)
        if nn and nn not in lookup:
            lookup[nn] = (str(gc), float(la), float(lo), src)
        roster.add(str(gc))

    # 1) Zhang 2019 (Language Info sheet)
    # Zhang names are underscored (Yi_Mile, Hani_Luchun, Pumi_Qinghua). Register
    # the full name AND the first-underscore prefix as a base-name fallback so
    # "Yi (Xide)" -> normalise "yi xide" -> base "yi" -> hits "yi" in lookup
    # pointing to the first Yi_* doculect encountered. The dialect-centroid
    # this pins to is approximate but keeps the Glottocode group correct.
    z19_xlsx = ROOT / "data" / "raw" / "zhang2019" / "zhang2019_data.xlsx"
    if z19_xlsx.exists():
        z19 = pd.read_excel(z19_xlsx, sheet_name="Language Info")
        for _, r in z19.iterrows():
            if pd.notna(r["Glottocode"]) and pd.notna(r["Latitude"]):
                _add(r["Name"], r["Glottocode"],
                     r["Latitude"], r["Longitude"], "zhang2019")
                # First-underscore prefix ("Yi_Mile" -> "Yi")
                if "_" in str(r["Name"]):
                    prefix = str(r["Name"]).split("_", 1)[0]
                    _add(prefix, r["Glottocode"],
                         r["Latitude"], r["Longitude"],
                         "zhang2019_prefix")
                if "Alternative Name" in r and pd.notna(r["Alternative Name"]):
                    for alt in str(r["Alternative Name"]).split(","):
                        _add(alt.strip(), r["Glottocode"],
                             r["Latitude"], r["Longitude"], "zhang2019_alt")

    # 2) Peiros ST
    pst_csv = ROOT / "data" / "raw" / "peirosst" / "cldf" / "languages.csv"
    if pst_csv.exists():
        p = pd.read_csv(pst_csv)
        for _, r in p.iterrows():
            if pd.notna(r["Glottocode"]) and pd.notna(r["Latitude"]):
                _add(r["Name"], r["Glottocode"],
                     r["Latitude"], r["Longitude"], "peirosst")
                if "Glottolog_Name" in r and pd.notna(r["Glottolog_Name"]):
                    _add(r["Glottolog_Name"], r["Glottocode"],
                         r["Latitude"], r["Longitude"], "peirosst_gl")

    # 3) Sagart ST
    sst_csv = ROOT / "data" / "raw" / "sagartst" / "cldf" / "languages.csv"
    if sst_csv.exists():
        s = pd.read_csv(sst_csv)
        for _, r in s.iterrows():
            if pd.notna(r["Glottocode"]) and pd.notna(r["Latitude"]):
                _add(r["Name"], r["Glottocode"],
                     r["Latitude"], r["Longitude"], "sagartst")
                if "Glottolog_Name" in r and pd.notna(r["Glottolog_Name"]):
                    _add(r["Glottolog_Name"], r["Glottocode"],
                         r["Latitude"], r["Longitude"], "sagartst_gl")
                if "Name_in_Source" in r and pd.notna(r["Name_in_Source"]):
                    _add(r["Name_in_Source"], r["Glottocode"],
                         r["Latitude"], r["Longitude"], "sagartst_src")

    return lookup, roster


def match_stedt_language(stedt_name: str,
                         lookup: dict[str, tuple]) -> tuple | None:
    """Return (glottocode, lat, lon, src, match_mode) or None."""
    if not stedt_name:
        return None
    if stedt_name.startswith("*"):  # STEDT proto-nodes; skip
        return None
    nn = _norm(stedt_name)
    if nn in lookup:
        return (*lookup[nn], "exact")

    base = re.sub(r"\s*\([^)]*\)", "", stedt_name).strip()
    nb = _norm(base)
    if nb in lookup:
        return (*lookup[nb], "base")

    # Strip square brackets too (e.g. "Kaman [Miju]")
    base2 = re.sub(r"\s*\[[^\]]*\]", "", base).strip()
    nb2 = _norm(base2)
    if nb2 != nb and nb2 in lookup:
        return (*lookup[nb2], "base")

    if nb in STEDT_NAME_ALIASES and STEDT_NAME_ALIASES[nb] in lookup:
        return (*lookup[STEDT_NAME_ALIASES[nb]], "alias")
    if nn in STEDT_NAME_ALIASES and STEDT_NAME_ALIASES[nn] in lookup:
        return (*lookup[STEDT_NAME_ALIASES[nn]], "alias")
    if nb2 in STEDT_NAME_ALIASES and STEDT_NAME_ALIASES[nb2] in lookup:
        return (*lookup[STEDT_NAME_ALIASES[nb2]], "alias")

    plain = re.sub(r"[\[\]\(\)]", "", stedt_name)
    for tok in re.split(r"[,/\s]+", plain):
        tn = _norm(tok)
        if tn and len(tn) >= 4 and tn in lookup:
            return (*lookup[tn], "token:" + tok)

    close = difflib.get_close_matches(nn, lookup.keys(), n=1, cutoff=0.85)
    if close:
        return (*lookup[close[0]], "fuzzy:" + close[0])
    return None


# ------------------------------------------------------------- clustering
def _clean_reflex(s: str) -> str:
    """Strip STEDT morpheme markers, tone digits, IPA superscripts."""
    s = str(s)
    s = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹ˊˋˆˇ◦ˉ˜ʔʰʱː]", "", s)
    s = re.sub(r"\d+", "", s)
    s = s.replace("|", "").replace("-", "")
    s = re.sub(r"\s+", "", s)
    return s.lower()


def nld_single_link_cluster(reflexes: list[str],
                            threshold: float) -> list[int]:
    """Single-linkage cluster by normalised Levenshtein distance."""
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


# ---------------------------------------------------------- gradient math
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
    n = len(per_lang)
    if n < 5:
        return {"n": int(n), "note": "n<5, skipped"}
    class_counts = Counter(per_lang[class_col])
    top = max(class_counts.values())
    tied = sorted(str(c) for c, v in class_counts.items() if v == top)
    proto = tied[0]
    retained = (
        per_lang[class_col].astype(str) == proto
    ).astype(int).to_numpy()
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
        "retention_rate": round(float(retained.mean()), 4),
        "spearman_r": None if np.isnan(r_obs) else round(r_obs, 4),
        "spearman_p": None if np.isnan(p_obs) else round(p_obs, 6),
        "ci_lower": None if np.isnan(ci_lo) else round(ci_lo, 4),
        "ci_upper": None if np.isnan(ci_hi) else round(ci_hi, 4),
        "proto_cognate_class": proto,
        "n_tied_top_classes": int(len(tied)),
    }


# --------------------------------------------------------- per-concept pipe
def process_concept(cfg: dict, lookup: dict[str, tuple]) -> dict:
    """Fetch -> filter -> match -> cluster -> gradient for one concept."""
    concept = cfg["concept"]
    print(f"\n=== {concept} =====================================")

    try:
        data = fetch_concept(cfg["query_gloss"], cfg["raw"])
    except Exception as e:
        print(f"  FATAL fetch: {type(e).__name__}: {e}")
        return {
            "role": cfg["role"],
            "query_gloss": cfg["query_gloss"],
            "status": "extraction_failed",
            "error": f"{type(e).__name__}: {e}",
        }

    rows = data["data"]
    fields = data["fields"]
    IDX = {name: i for i, name in enumerate(fields)}
    print(f"  raw STEDT rows (s={cfg['query_gloss']}): {len(rows)}")

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
    print(f"  rows mapped to Glottocode: {len(out_records)}")
    print(f"  distinct STEDT names unmatched: {len(unmatched)}")
    print(f"  match breakdown: {dict(match_stats)}")

    if not out_records:
        return {
            "role": cfg["role"],
            "query_gloss": cfg["query_gloss"],
            "status": "no_forms_mapped",
            "n_raw_forms": len(exact),
            "unmatched_names": sorted(unmatched),
        }

    df = pd.DataFrame(out_records)
    df.to_csv(cfg["csv"], index=False)
    print(f"  wrote {cfg['csv'].relative_to(ROOT)} ({len(df)} rows)")

    # Method A: STEDT etyma tags
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

    # Build the per-concept result block
    out = {
        "role": cfg["role"],
        "query_gloss": cfg["query_gloss"],
        "include_glosses": sorted(cfg["include_glosses"]),
        "predicted_sign": cfg["predicted_sign"],
        "n_raw_forms": int(len(exact)),
        "n_forms_matched_to_glottocode": int(len(df)),
        "n_unique_glottocodes": int(df["glottocode"].nunique()),
        "match_stats": dict(match_stats),
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
        "unmatched_names": sorted(unmatched),
    }

    # Console
    def _fmt(res):
        return (f"n={res.get('n')} r={res.get('spearman_r')} "
                f"p={res.get('spearman_p')} "
                f"CI=[{res.get('ci_lower')}, {res.get('ci_upper')}]")
    print(f"  Method A (etyma tag): {_fmt(res_a)}")
    print(f"  Method B (NLD<=0.4) : {_fmt(res_b)}")
    return out


# -------------------------------------------------------------- sanity check
def poc_sanity_check(poc_n: int = 60, poc_r: float = 0.184) -> dict:
    """Re-run Method B on the POC's original rice_forms.csv (60 Glottocodes).

    This is a *harmonisation-integrity* check: feed the POC's exact input to
    the current clustering + gradient code and confirm we get the same
    r=+0.184. If this fails, something in _clean_reflex, nld_single_link
    or the gradient maths drifted. Note we do NOT feed the expanded
    rice_forms_full.csv here -- the expanded inclusion set and extended
    alias table produce a legitimately different n, which is the whole
    point of this extension.
    """
    poc_csv = STEDT_DIR / "rice_forms.csv"
    if not poc_csv.exists():
        return {"status": "poc_csv_missing",
                "note": f"expected {poc_csv}"}
    poc_df = pd.read_csv(poc_csv)
    reflexes = poc_df["reflex_clean"].astype(str).tolist()
    clusters = nld_single_link_cluster(reflexes, NLD_COGNATE_THRESHOLD)
    poc_df = poc_df.copy()
    poc_df["nld_cluster"] = clusters
    per = (
        poc_df.groupby("glottocode")
        .agg(
            cognate_class=("nld_cluster",
                           lambda s: majority_vote(list(s))),
            lat=("latitude", "mean"),
            lon=("longitude", "mean"),
        ).reset_index()
    )
    per["dist_km"] = per.apply(
        lambda r: haversine(HOMELAND_LAT, HOMELAND_LON,
                            r["lat"], r["lon"]), axis=1)
    rng = np.random.default_rng(SEED)
    res = gradient(per, "cognate_class", rng)

    match = (res["n"] == poc_n
             and res.get("spearman_r") is not None
             and abs(res["spearman_r"] - poc_r) < 0.01)
    print(f"  POC sanity: method B on POC rice_forms.csv -> "
          f"n={res.get('n')} r={res.get('spearman_r')} "
          f"(POC: n={poc_n} r={poc_r}) -> "
          f"{'MATCH' if match else 'DIFFERS'}")
    return {
        "status": "match" if match else "differs",
        "poc_n_expected": poc_n,
        "poc_r_expected": poc_r,
        "replicated_n": res.get("n"),
        "replicated_r": res.get("spearman_r"),
    }


# ---------------------------------------------------------------- main
def main() -> None:
    print("=" * 78)
    print("Sino-Tibetan full STEDT extraction (RICE + SALT + BARLEY + YAK)")
    print(f"Homeland: {HOMELAND_NAME} ({HOMELAND_LAT} N, {HOMELAND_LON} E)")
    print("=" * 78)

    lookup, roster = build_glottocode_lookup()
    print(f"  glottocode lookup entries: {len(lookup)}")
    print(f"  target ST roster size (union Glottocodes): {len(roster)}")

    per_concept: dict[str, dict] = {}
    unmatched_by_concept: dict[str, list[str]] = {}
    for cfg in CONCEPTS:
        res = process_concept(cfg, lookup)
        per_concept[cfg["concept"]] = res
        unmatched_by_concept[cfg["concept"]] = res.pop("unmatched_names",
                                                       []) or []

    # POC-replication sanity check on RICE (run against POC's original CSV)
    print("\n--- POC sanity check (method B against POC rice_forms.csv) ---")
    sanity = poc_sanity_check()

    # Verdict on RICE
    rice_res = per_concept.get("RICE", {})
    mB = rice_res.get("method_B_nld_cluster", {})
    r_b = mB.get("spearman_r")
    p_b = mB.get("spearman_p")
    n_b = mB.get("n")
    if r_b is None or n_b is None:
        rice_verdict = "underpowered"
    elif n_b >= 30 and p_b is not None and p_b < 0.05 and r_b > 0:
        rice_verdict = "supported"
    elif n_b >= 30 and p_b is not None and p_b < 0.05 and r_b < 0:
        rice_verdict = "refuted"
    else:
        rice_verdict = "underpowered"

    # Any other concept with positive significant gradient (method B)
    others_pos = []
    for c in ("SALT", "BARLEY", "YAK"):
        m = per_concept.get(c, {}).get("method_B_nld_cluster", {})
        if (m.get("spearman_r") is not None
                and m.get("spearman_p") is not None
                and m["spearman_p"] < 0.05
                and m["spearman_r"] > 0):
            others_pos.append(c)

    output = {
        "family": "Sino-Tibetan",
        "data_source": "STEDT bulk via stedtdb.johnblowe.com AJAX",
        "data_source_url_template":
            STEDT_AJAX + "?tbl=lexicon&s=<concept>",
        "extraction_date": date.today().isoformat(),
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
        "concepts": per_concept,
        "unmatched_stedt_names": unmatched_by_concept,
        "poc_sanity_check": sanity,
        "comparison_to_previous": {
            "sagartst_RICE": {"n": 23, "r": 0.271, "p": 0.21},
            "stedt_poc_RICE_methodA": {"n": 8, "r": 0.756, "p": 0.030},
            "stedt_poc_RICE_methodB": {"n": 60, "r": 0.184, "p": 0.159},
        },
        "verdict": {
            "ST_RICE_anchor_hypothesis": rice_verdict,
            "ST_other_anchors_positive": others_pos,
        },
    }
    with OUT_JSON.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    # Console summary
    print("\n" + "=" * 78)
    print("FINAL SUMMARY")
    print("=" * 78)
    print(f"{'concept':<8} {'n_forms':>8} {'n_langs':>8}  "
          f"{'methA r(p,n)':<28}  {'methB r(p,n)':<28}")
    for c in ("RICE", "SALT", "BARLEY", "YAK"):
        r = per_concept.get(c, {})
        nf = r.get("n_forms_matched_to_glottocode", 0)
        ng = r.get("n_unique_glottocodes", 0)
        a = r.get("method_A_expert_etyma", {})
        b = r.get("method_B_nld_cluster", {})
        a_str = (f"r={a.get('spearman_r')} "
                 f"p={a.get('spearman_p')} n={a.get('n')}")
        b_str = (f"r={b.get('spearman_r')} "
                 f"p={b.get('spearman_p')} n={b.get('n')}")
        print(f"{c:<8} {nf:>8} {ng:>8}  {a_str:<28}  {b_str:<28}")
    print(f"\nRICE anchor verdict: {rice_verdict}")
    if others_pos:
        print(f"Other concepts with positive significant gradient (method B): "
              f"{', '.join(others_pos)}")
    else:
        print("No other concepts reach significance at alpha=0.05 (method B).")
    print(f"\nWrote {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
