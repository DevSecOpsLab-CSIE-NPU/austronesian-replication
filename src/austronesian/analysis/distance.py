#!/usr/bin/env python3
"""詞彙距離計算模組 - Levenshtein/Normalized Distance."""

from typing import List, Tuple
import numpy as np


def levenshtein_distance(s1: str, s2: str) -> int:
    """計算兩個字串之間的 Levenshtein 編輯距離.
    
    參數:
        s1: 第一個字串
        s2: 第二個字串
        
    返回:
        編輯距離（整數）
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def normalized_levenshtein_distance(s1: str, s2: str) -> float:
    """計算標準化 Levenshtein 距離 (0-1).
    
    距離 = 編輯距離 / max(len(s1), len(s2))
    
    參數:
        s1: 第一個字串
        s2: 第二個字串
        
    返回:
        標準化距離（0-1，0 表示完全相同）
    """
    if not s1 and not s2:
        return 0.0
    if not s1 or not s2:
        return 1.0
    
    distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return distance / max_len


def levenshtein_similarity(s1: str, s2: str) -> float:
    """計算 Levenshtein相似度 (0-1).
    
    相似度 = 1 - 標準化距離
    
    參數:
        s1: 第一個字串
        s2: 第二個字串
        
    返回:
        相似度（0-1，1 表示完全相同）
    """
    return 1.0 - normalized_levenshtein_distance(s1, s2)


def damerau_levenshtein_distance(s1: str, s2: str) -> int:
    """計算 Damerau-Levenshtein 距離（含相鄰交換）.
    
    參數:
        s1: 第一個字串
        s2: 第二個字串
        
    返回:
        編輯距離（包含相鄰交換）
    """
    if len(s1) < len(s2):
        return damerau_levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    # 使用優化過的動態規劃
    INF = len(s1) + len(s2)
    dp = [[INF] * (len(s2) + 2) for _ in range(len(s1) + 2)]
    
    for i in range(len(s1) + 2):
        dp[i][0] = i
    for j in range(len(s2) + 2):
        dp[0][j] = j
    
    da = {}
    for c in s1:
        da[c] = 0
    for c in s2:
        da[c] = 0
    
    for i in range(1, len(s1) + 1):
        db = 0
        for j in range(1, len(s2) + 1):
            i1 = da[s2[j-1]]
            j1 = db
            
            cost = 1
            if s1[i-1] == s2[j-1]:
                cost = 0
                db = j
            
            dp[i][j] = min(
                dp[i-1][j] + 1,      # 刪除
                dp[i][j-1] + 1,      # 插入
                dp[i-1][j-1] + cost, # 替換
                dp[i1][j1] + (i - i1 - 1) + 1 + (j - j1 - 1)  # 交換
            )
        
        da[s1[i-1]] = i
    
    return dp[len(s1)][len(s2)]


def compute_distance_matrix(forms: List[str], method: str = "levenshtein") -> np.ndarray:
    """計算詞彙距離矩陣.
    
    參數:
        forms: 詞彙列表
        method: 距離方法 ("levenshtein", "normalized", "damerau")
        
    返回:
        距離矩陣 (numpy array)
    """
    n = len(forms)
    matrix = np.zeros((n, n))
    
    if method == "levenshtein":
        dist_func = levenshtein_distance
    elif method == "normalized":
        dist_func = normalized_levenshtein_distance
    elif method == "damerau":
        dist_func = damerau_levenshtein_distance
    else:
        raise ValueError(f"Unknown method: {method}")
    
    for i in range(n):
        for j in range(i + 1, n):
            d = dist_func(forms[i], forms[j])
            matrix[i, j] = d
            matrix[j, i] = d
    
    return matrix


if __name__ == "__main__":
    # 測試
    test_pairs = [
        ("mata", "mata"),
        ("mata", "masa"),
        ("mata", "kata"),
        ("mata", "matai"),
        ("hand", "hand"),
        ("hand", "land"),
    ]
    
    print("Levenshtein 距離測試：")
    print("-" * 50)
    for s1, s2 in test_pairs:
        lev = levenshtein_distance(s1, s2)
        norm = normalized_levenshtein_distance(s1, s2)
        sim = levenshtein_similarity(s1, s2)
        print(f"{s1:10} vs {s2:10} | Lev={lev:2} | Norm={norm:.3f} | Sim={sim:.3f}")
