"""Pool three Sino-Tibetan CLDF datasets (sagartst / zhangst / peirosst) to test
the migration-gradient anchor for MOUNTAIN (+ TREE, FIRE, WATER, STONE as controls).

Procedure per dataset:
  1. Join forms.csv with cognates.csv on Form_ID.
  2. Filter to target parameter (e.g. MOUNTAIN).
  3. Exclude loans (Loan in {True, 'True', 1}).
  4. Attach language geo (Lat, Lon, Glottocode). For zhangst, geo comes from
     data/raw/zhang2019/zhang2019_data.xlsx (sheet 'Language Info') joined on
     language name (case-insensitive, underscore-normalised, with fuzzy fallback).
     For peirosst, drop rows with missing geo.
  5. Dedup by Glottocode (majority-vote Cognateset_ID).

Pooling:
  6. Union per-Glottocode tables. On Glottocode overlap, prefer
     sagartst > zhangst > peirosst.
  7. Cross-dataset cognate harmonisation: cluster pooled Forms by Normalised
     Levenshtein Distance <= 0.35 into a single harmonised_cognate_id.
  8. Proto-form retention R_j = 1 iff harmonised_cognate_id == globally most
     frequent class.
  9. Haversine distance from Upper Yellow River homeland (35 N, 104 E).
 10. Spearman rho(dist, retention), 500-bootstrap percentile 95% CI, seed 42.

Also reports per-dataset sub-totals (sagartst-only, zhangst-only, peirosst-only).

Outputs:
  results/xfam_st_pooled_mountain.json
  results/xfam_st_pooled_forms.csv
"""
from __future__ import annotations

import json
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Prefer package NLD; fallback to difflib ratio (task-sanctioned)
try:
    from austronesian.analysis.distance import normalized_levenshtein_distance as _nld

    NLD_SOURCE = "austronesian.analysis.distance.normalized_levenshtein_distance"
except Exception:  # pragma: no cover
    from difflib import SequenceMatcher

    def _nld(a: str, b: str) -> float:
        return 1.0 - SequenceMatcher(None, a, b).ratio()

    NLD_SOURCE = "difflib.SequenceMatcher fallback"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HOMELAND = {"name": "Upper Yellow River basin", "lat": 35.0, "lon": 104.0}
NLD_THRESHOLD = 0.35
N_BOOT = 500
SEED = 42

DATASETS = ["sagartst", "zhangst", "peirosst"]  # pooling priority order

CONCEPT_PARAMS: dict[str, dict[str, str]] = {
    "MOUNTAIN": {"sagartst": "132_themountain", "zhangst": "56_mountain", "peirosst": "55_mountain"},
    "TREE":     {"sagartst": "224_thetree",    "zhangst": "90_tree",     "peirosst": "90_tree"},
    "FIRE":     {"sagartst": "54_thefire",     "zhangst": "27_fire",     "peirosst": "28_fire"},
    "WATER":    {"sagartst": "229_thewater",   "zhangst": "93_water",    "peirosst": "94_water"},
    "STONE":    {"sagartst": "203_thestoneapieceof", "zhangst": "82_stone", "peirosst": "81_stone"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _strip_diacritics(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _norm_name(s) -> str:
    s = str(s).strip().lower().replace("_", " ")
    s = _strip_diacritics(s)
    s = re.sub(r"\s+", " ", s)
    return s


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    lat1r, lon1r, lat2r, lon2r = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def is_loan(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    return s in {"true", "1", "yes", "y"}


def majority_cognate(items: list) -> object:
    cnt = Counter(items)
    # In ties Counter.most_common is stable; that's acceptable
    return cnt.most_common(1)[0][0]


def cluster_forms_nld(forms: list[str], threshold: float = NLD_THRESHOLD) -> list[int]:
    """Transitive-closure clustering by NLD <= threshold (single-link)."""
    n = len(forms)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if _nld(forms[i], forms[j]) <= threshold:
                union(i, j)
    roots = [find(i) for i in range(n)]
    remap: dict[int, int] = {}
    labels = []
    for r in roots:
        if r not in remap:
            remap[r] = len(remap)
        labels.append(remap[r])
    return labels


def spearman_with_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict:
    n = len(x)
    if n < 3 or np.unique(y).size < 2:
        return {"n": int(n), "spearman_r": float("nan"), "spearman_p": float("nan"),
                "ci_lower": float("nan"), "ci_upper": float("nan")}
    r, p = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    rs = []
    idx = np.arange(n)
    for _ in range(n_boot):
        s = rng.choice(idx, size=n, replace=True)
        if np.unique(y[s]).size < 2:
            continue
        rb, _ = spearmanr(x[s], y[s])
        if not np.isnan(rb):
            rs.append(rb)
    if len(rs) < 10:
        lo = hi = float("nan")
    else:
        lo, hi = np.percentile(rs, [2.5, 97.5])
    return {"n": int(n), "spearman_r": float(r), "spearman_p": float(p),
            "ci_lower": float(lo), "ci_upper": float(hi)}


# ---------------------------------------------------------------------------
# Per-dataset extraction
# ---------------------------------------------------------------------------
def _zhang_geo_map(zhangst_langs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Attach Glottocode/Lat/Long to zhangst languages from zhang2019 xlsx.

    zhangst languages.csv has these columns empty. The xlsx 'Language Info'
    sheet has a near-identical list. Match by normalised name; for remaining
    unmatched zhangst names fall back to token-based fuzzy match (prefix + first
    token agreement), taking the closest xlsx row.
    """
    xlsx_path = ROOT / "data/raw/zhang2019/zhang2019_data.xlsx"
    zx = pd.read_excel(xlsx_path, sheet_name="Language Info")
    zx = zx[["Name", "Glottocode", "Latitude", "Longitude"]].copy()
    zx["k"] = zx["Name"].apply(_norm_name)

    zl = zhangst_langs.copy()
    zl["k"] = zl["Name"].apply(_norm_name)

    merged = zl.merge(zx[["k", "Glottocode", "Latitude", "Longitude"]], on="k",
                      how="left", suffixes=("", "_x"))
    merged["Glottocode"] = merged["Glottocode_x"]
    merged["Latitude"] = merged["Latitude_x"]
    merged["Longitude"] = merged["Longitude_x"]
    merged = merged.drop(columns=[c for c in ["Glottocode_x", "Latitude_x", "Longitude_x"] if c in merged])

    # Fuzzy fallback for unmatched
    unmatched_mask = merged["Glottocode"].isna()
    xlsx_keys = zx["k"].tolist()
    for idx in merged[unmatched_mask].index:
        name = merged.at[idx, "Name"]
        k = _norm_name(name)
        head = k.split(" ")[0]
        # Look for xlsx keys sharing the head token and with best NLD
        candidates = [(xk, _nld(k, xk)) for xk in xlsx_keys if xk.split(" ")[0] == head]
        if not candidates:
            # Also consider xlsx entries whose first token is a substring of k
            candidates = [(xk, _nld(k, xk)) for xk in xlsx_keys
                          if xk.split(" ")[0] in k]
        if not candidates:
            continue
        best_k, best_d = min(candidates, key=lambda t: t[1])
        if best_d <= 0.6:  # permissive for one-row language-name join
            row = zx[zx["k"] == best_k].iloc[0]
            merged.at[idx, "Glottocode"] = row["Glottocode"]
            merged.at[idx, "Latitude"] = row["Latitude"]
            merged.at[idx, "Longitude"] = row["Longitude"]

    matched = int(merged["Glottocode"].notna().sum())
    unmatched = int(merged["Glottocode"].isna().sum())
    unmatched_names = merged.loc[merged["Glottocode"].isna(), "Name"].tolist()
    diag = {"matched": matched, "unmatched": unmatched, "unmatched_names": unmatched_names}
    return merged[["ID", "Name", "Glottocode", "Latitude", "Longitude"]], diag


def load_language_geo(dataset: str, diagnostics: dict) -> tuple[pd.DataFrame, int]:
    """Return languages dataframe with [ID, Name, Glottocode, Latitude, Longitude].

    Returns (df, dropped_count) where df has valid geo for every row.
    """
    path = ROOT / f"data/raw/{dataset}/cldf/languages.csv"
    langs = pd.read_csv(path)
    if dataset == "zhangst":
        langs, zdiag = _zhang_geo_map(langs)
        diagnostics["zhangst_geo_join"] = zdiag
    keep_cols = ["ID", "Name", "Glottocode", "Latitude", "Longitude"]
    langs = langs[keep_cols]
    before = len(langs)
    valid = langs.dropna(subset=["Glottocode", "Latitude", "Longitude"]).copy()
    dropped = before - len(valid)
    return valid, dropped


def extract_dataset_concept(
    dataset: str,
    param_id: str,
    lang_geo: pd.DataFrame,
) -> pd.DataFrame:
    """Return per-language-per-concept rows for one dataset.

    Columns: glottocode, language_name, source_dataset, concept_param_id,
             raw_cognate_id, form, lat, lon.
    """
    forms = pd.read_csv(ROOT / f"data/raw/{dataset}/cldf/forms.csv", low_memory=False)
    cogs = pd.read_csv(ROOT / f"data/raw/{dataset}/cldf/cognates.csv", low_memory=False)
    f = forms[forms["Parameter_ID"] == param_id].copy()
    f = f[~f["Loan"].apply(is_loan)]
    merged = f.merge(cogs[["Form_ID", "Cognateset_ID"]], left_on="ID", right_on="Form_ID", how="left")
    # drop any row without a cognate assignment
    merged = merged.dropna(subset=["Cognateset_ID"])
    # Attach geo
    merged = merged.merge(lang_geo, left_on="Language_ID", right_on="ID",
                          how="inner", suffixes=("", "_lang"))
    # Dedup by Glottocode (majority Cognateset_ID; first Form/Name)
    rows = []
    for gc, grp in merged.groupby("Glottocode"):
        cog = majority_cognate(grp["Cognateset_ID"].tolist())
        # Form representative: first form under the chosen cognate
        sub = grp[grp["Cognateset_ID"] == cog]
        form = sub["Form"].iloc[0] if "Form" in sub and pd.notna(sub["Form"].iloc[0]) else sub["Value"].iloc[0]
        rows.append({
            "glottocode": gc,
            "language_name": grp["Name"].iloc[0],
            "source_dataset": dataset,
            "concept_param_id": param_id,
            "raw_cognate_id": str(cog),
            "form": str(form),
            "lat": float(grp["Latitude"].iloc[0]),
            "lon": float(grp["Longitude"].iloc[0]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pooling + harmonisation + stats
# ---------------------------------------------------------------------------
def pool_concept(
    per_ds: dict[str, pd.DataFrame],
    priority: list[str] = DATASETS,
) -> pd.DataFrame:
    """Union per-Glottocode tables; on conflict keep higher-priority dataset."""
    seen: dict[str, dict] = {}
    for ds in priority:
        df = per_ds.get(ds)
        if df is None or df.empty:
            continue
        for _, r in df.iterrows():
            gc = r["glottocode"]
            if gc not in seen:
                seen[gc] = r.to_dict()
    return pd.DataFrame(list(seen.values()))


def harmonise_cognates(pooled: pd.DataFrame, threshold: float = NLD_THRESHOLD) -> pd.DataFrame:
    pooled = pooled.copy()
    forms = [str(x).strip() for x in pooled["form"].tolist()]
    labels = cluster_forms_nld(forms, threshold=threshold)
    pooled["harmonised_cognate_id"] = [f"H{i:04d}" for i in labels]
    return pooled


def compute_retention_and_distance(pooled: pd.DataFrame) -> pd.DataFrame:
    pooled = pooled.copy()
    if pooled.empty:
        pooled["dist_homeland_km"] = []
        pooled["retention"] = []
        return pooled
    counts = Counter(pooled["harmonised_cognate_id"].tolist())
    top = counts.most_common(1)[0][0]
    pooled["retention"] = (pooled["harmonised_cognate_id"] == top).astype(int)
    pooled["dist_homeland_km"] = [
        haversine_km(HOMELAND["lat"], HOMELAND["lon"], lat, lon)
        for lat, lon in zip(pooled["lat"], pooled["lon"])
    ]
    return pooled


def per_dataset_gradient(df: pd.DataFrame) -> dict:
    """Gradient on one dataset's (unharmonised) subset using raw_cognate_id."""
    if df.empty:
        return {"n": 0, "spearman_r": float("nan"), "spearman_p": float("nan"),
                "retention_rate": float("nan")}
    counts = Counter(df["raw_cognate_id"].tolist())
    top = counts.most_common(1)[0][0]
    retention = (df["raw_cognate_id"] == top).astype(int).to_numpy()
    dist = np.array([
        haversine_km(HOMELAND["lat"], HOMELAND["lon"], la, lo)
        for la, lo in zip(df["lat"], df["lon"])
    ])
    stats = spearman_with_bootstrap(dist, retention)
    stats["retention_rate"] = float(retention.mean())
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    diagnostics: dict = {"nld_source": NLD_SOURCE}

    # Language geos
    lang_geos: dict[str, pd.DataFrame] = {}
    for ds in DATASETS:
        geo, dropped = load_language_geo(ds, diagnostics)
        lang_geos[ds] = geo
        if ds == "peirosst":
            diagnostics["peirosst_geo_dropped"] = dropped

    all_concept_results: dict[str, dict] = {}
    all_pooled_rows: list[pd.DataFrame] = []

    for concept, params in CONCEPT_PARAMS.items():
        per_ds: dict[str, pd.DataFrame] = {}
        per_ds_stats: dict[str, dict] = {}
        for ds in DATASETS:
            pid = params[ds]
            df = extract_dataset_concept(ds, pid, lang_geos[ds])
            per_ds[ds] = df
            per_ds_stats[ds] = per_dataset_gradient(df)

        pooled = pool_concept(per_ds)
        pooled = harmonise_cognates(pooled)
        pooled = compute_retention_and_distance(pooled)

        # Pooled stats
        if not pooled.empty:
            dist = pooled["dist_homeland_km"].to_numpy()
            ret = pooled["retention"].to_numpy()
            pooled_stats = spearman_with_bootstrap(dist, ret)
            pooled_stats["retention_rate"] = float(ret.mean())
        else:
            pooled_stats = {"n": 0, "spearman_r": float("nan"), "spearman_p": float("nan"),
                            "ci_lower": float("nan"), "ci_upper": float("nan"),
                            "retention_rate": float("nan")}
        pooled_stats["source_counts"] = {
            ds: int((pooled["source_dataset"] == ds).sum()) if not pooled.empty else 0
            for ds in DATASETS
        }

        all_concept_results[concept] = {
            "pooled": pooled_stats,
            **{ds: per_ds_stats[ds] for ds in DATASETS},
        }

        # Save pooled rows for the CSV
        if not pooled.empty:
            out = pooled.copy()
            out["concept"] = concept
            out = out[[
                "glottocode", "language_name", "source_dataset", "concept",
                "raw_cognate_id", "harmonised_cognate_id", "form",
                "lat", "lon", "dist_homeland_km", "retention",
            ]]
            all_pooled_rows.append(out)

    diagnostics["pooled_glottocodes_total"] = int(
        all_concept_results["MOUNTAIN"]["pooled"]["n"]
    )

    # -----------------------------------------------------------------
    # Write outputs
    # -----------------------------------------------------------------
    out_json = {
        "family": "Sino-Tibetan",
        "homeland": HOMELAND,
        "datasets_pooled": DATASETS,
        "harmonisation": {"method": "NLD cluster", "threshold": NLD_THRESHOLD},
        "concepts": all_concept_results,
        "diagnostics": diagnostics,
    }
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "xfam_st_pooled_mountain.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out_json, fh, indent=2, ensure_ascii=False, default=float)
    print(f"Wrote {out_path}")

    if all_pooled_rows:
        csv_path = results_dir / "xfam_st_pooled_forms.csv"
        pd.concat(all_pooled_rows, ignore_index=True).to_csv(csv_path, index=False)
        print(f"Wrote {csv_path}")

    # -----------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------
    print()
    print(f"NLD source: {NLD_SOURCE}")
    zj = diagnostics.get("zhangst_geo_join", {})
    print(f"zhangst name-join: matched={zj.get('matched')}, unmatched={zj.get('unmatched')}")
    print(f"peirosst geo-dropped rows: {diagnostics.get('peirosst_geo_dropped')}")
    print()
    header = f"{'concept':<10}{'sagartst':>18}{'zhangst':>18}{'peirosst':>18}{'POOLED':>22}"
    print(header)
    print("-" * len(header))
    for concept in CONCEPT_PARAMS:
        res = all_concept_results[concept]
        def fmt(d):
            if d["n"] == 0 or np.isnan(d.get("spearman_r", np.nan)):
                return f"n={d['n']}, rho=NA"
            return f"n={d['n']}, rho={d['spearman_r']:+.3f}"
        p = res["pooled"]
        p_str = (f"n={p['n']}, rho={p['spearman_r']:+.3f}, p={p['spearman_p']:.2e}"
                 if p["n"] and not np.isnan(p.get("spearman_r", np.nan))
                 else f"n={p['n']}, rho=NA")
        print(f"{concept:<10}{fmt(res['sagartst']):>18}{fmt(res['zhangst']):>18}{fmt(res['peirosst']):>18}{p_str:>22}")

    # Sanity check
    sag_m = all_concept_results["MOUNTAIN"]["sagartst"]
    print()
    print(f"Sanity: sagartst-only MOUNTAIN n={sag_m['n']}, rho={sag_m['spearman_r']:+.3f} "
          f"(prior expectation ~n=36, rho~-0.05)")


if __name__ == "__main__":
    main()
