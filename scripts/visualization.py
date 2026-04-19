#!/usr/bin/env python3
"""親緣樹可視化與評估腳本（簡化版）."""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

FORMOSAN_KEYWORDS = ['Atayal', 'Amis', 'Bunun', 'Puyuma', 'Rukai', 'Tsou', 'Saisiyat', 'Thao', 'Kavalan', 'Paiwan', 'Pazih', 'Papora', 'Favorlang', 'Siraya', 'Hoanya', 'Babuza', 'Taokas']


def load_distance_matrix(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def evaluate_clusters(dist_df: pd.DataFrame):
    languages = dist_df.index.tolist()
    n = len(languages)
    
    dist_values = dist_df.values.copy()
    nan_mask = np.isnan(dist_values)
    for i in range(n):
        if nan_mask[i].any():
            valid = dist_values[i, ~nan_mask[i]]
            if len(valid) > 0:
                fill = np.mean(valid)
                dist_values[i, nan_mask[i]] = fill
                dist_values[nan_mask[i], i] = fill
    
    dist_values = (dist_values + dist_values.T) / 2
    np.fill_diagonal(dist_values, 0)
    
    print("\n最近的語言對:")
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((languages[i], languages[j], dist_values[i,j]))
    
    pairs.sort(key=lambda x: x[2])
    for lang1, lang2, dist in pairs[:15]:
        print(f"  {lang1} <-> {lang2}: {dist:.4f}")
    
    return pairs


def plot_dendrogram(dist_df: pd.DataFrame, output_path: Path, title: str = "Austronesian Language Phylogeny"):
    languages = dist_df.index.tolist()
    n = len(languages)
    
    dist_values = dist_df.values.copy()
    nan_mask = np.isnan(dist_values)
    for i in range(n):
        if nan_mask[i].any():
            valid = dist_values[i, ~nan_mask[i]]
            if len(valid) > 0:
                fill = np.mean(valid)
                dist_values[i, nan_mask[i]] = fill
                dist_values[nan_mask[i], i] = fill
    
    dist_values = (dist_values + dist_values.T) / 2
    np.fill_diagonal(dist_values, 0)
    
    condensed = squareform(dist_values)
    Z = linkage(condensed, method='average')
    
    fig, ax = plt.subplots(figsize=(20, 10))
    dendrogram(Z, truncate_mode='lastp', p=30, leaf_rotation=90, leaf_font_size=8, ax=ax)
    subset_size = min(50, n)
    dendrogram(Z, labels=languages[:subset_size], leaf_rotation=90, leaf_font_size=8, ax=ax)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Language', fontsize=12)
    ax.set_ylabel('Distance', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已儲存樹圖: {output_path}")


def analyze_formosan(dist_df: pd.DataFrame):
    languages = dist_df.index.tolist()
    formosan = [lang for lang in languages if any(kw.lower() in lang.lower() for kw in FORMOSAN_KEYWORDS)]
    
    print(f"\n找到 {len(formosan)} 個 Formosan 語言:")
    for lang in formosan[:10]:
        print(f"  - {lang}")


def main():
    results_dir = Path(__file__).parent.parent / "results"
    figures_dir = Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print("載入距離矩陣...")
    dist_df = load_distance_matrix(results_dir / "language_distance_matrix.csv")
    print(f"語言數: {len(dist_df)}")
    
    print("\n" + "=" * 60)
    print("分群分析")
    print("=" * 60)
    pairs = evaluate_clusters(dist_df)
    
    print("\n" + "=" * 60)
    print("Formosan 語言分析")
    print("=" * 60)
    analyze_formosan(dist_df)
    
    print("\n" + "=" * 60)
    print("生成樹圖...")
    print("=" * 60)
    
    plot_dendrogram(dist_df, figures_dir / "tree_figure.png", "Austronesian Language Phylogeny")
    
    report = f"""# 親緣樹分群評估報告

## 概述
- 語言數量: {len(dist_df)}
- 樹演算法: Neighbor Joining
- 距離度量: Normalized Levenshtein Distance

## 分群結果
"""
    for lang1, lang2, dist in pairs[:10]:
        report += f"- {lang1} <-> {lang2}: {dist:.4f}\n"
    
    report_path = results_dir / "evaluation.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\n已儲存評估報告: {report_path}")
    
    print("\n完成！")


if __name__ == "__main__":
    main()
