#!/usr/bin/env python3
"""語音轉寫正規化模組 - IPA/ASJP 轉換與正規化."""

import re
from typing import Dict, Optional


# IPA 到 ASJP 的標準轉換表
# 參考：Brown et al. (2013); Wichmann et al. (2016)
IPA_TO_ASJP: Dict[str, str] = {
    #母音 (Vowels)
    #高前元音
    'i': 'i', 'ɪ': 'i', 'y': 'i', 'Y': 'i', 'iː': 'i', 'iː': 'i',
    #中高前元音
    'e': 'e', 'eː': 'e', 'ø': 'e', 'øː': 'e',
    #低前元音
    'æ': 'E', 'ɛ': 'E', 'œ': 'E', 'Œ': 'E', 'æː': 'E', 'ɛː': 'E',
    #中央元音
    'ɨ': '3', 'ɘ': '3', 'ə': '3', 'ɜ': '3', 'ʉ': '3', 'ɵ': '3', 'ɞ': '3', 'ɚ': '3',
    'ɯ': '3', 'ɤ': '3', 'ɞ': '3',
    #低元音
    'a': 'a', 'aː': 'a', 'ɐ': 'a', 'ɑ': 'a', 'ɑː': 'a', 'ɶ': 'a',
    #高後元音
    'u': 'u', 'uː': 'u', 'ʊ': 'u', 'ʊː': 'u',
    #中高後元音
    'o': 'o', 'oː': 'o', 'ɤ': '3',  # o 可能有爭議，但通常作 o
    #低後元音
    'ɔ': 'O', 'ɔː': 'O',
    
    #子音 (Consonants)
    #塞音
    'p': 'p', 'b': 'b', 't': 't', 'd': 'd', 'ʈ': 't', 'ɖ': 'd',
    'c': 'k', 'ɟ': 'g', 'k': 'k', 'g': 'g', 'q': 'k', 'ɢ': 'g',
    #擦音
    'f': 'f', 'v': 'v', 'ɸ': 'f', 'β': 'v',
    'θ': 's', 'ð': 'z',  # 英語 th -> s/z
    's': 's', 'z': 'z', 'ʃ': 'S', 'ʒ': 'Z', 'ʂ': 'S', 'ʐ': 'Z',
    'x': 'k', 'ɣ': 'g', 'χ': 'k', 'ʁ': 'r',
    'h': 'h', 'ɦ': 'h',
    #鼻音
    'm': 'm', 'n': 'n', 'ɱ': 'm', 'ɳ': 'N', 'ɲ': 'N', 'ŋ': 'N', 'ɴ': 'N',
    #邊音
    'l': 'l', 'ɬ': 'l', 'ɮ': 'l', 'ɭ': 'L', 'ʎ': 'L', 'λ': 'L',
    #顫音/閃音
    'r': 'r', 'ʀ': 'r', 'ɾ': 'r', 'ɽ': 'r',
    #近音/滑音
    'j': 'j', 'ʝ': 'j', 'w': 'w', 'ʍ': 'w', 'ɰ': 'g',
    #喉塞音
    'ʔ': '7',
    #氣聲/嘎裂聲標記（通常省略）
    'ʰ': '', 'ʷ': '', 'ʲ': '', 'ˤ': '',
    
    #其他符號
    #連音符
    '-': '-', 'ː': '',  # 長音符號省略
    'ˈ': '', 'ˌ': '',  # 重音符號省略
    '|': '',  # 次重音
    '.': '',  # 音節邊界
}

# 常见组合 (Common clusters)
IPA_CLUSTERS: Dict[str, str] = {
    'tʃ': 'tS', 'dʒ': 'dZ', 'ts': 'tS', 'dz': 'dZ',
    'tɕ': 'tS', 'dʑ': 'dZ',
    'tʂ': 'tS', 'dʐ': 'dZ',
    'pf': 'p', 'bv': 'b',  # 唇齒塞擦音
    # 氣聲組合
    'bʱ': 'b', 'dʱ': 'd', 'gʱ': 'g',
    'dʒ': 'dZ', 'tʃ': 'tS',
}

# 需要移除的 diacritics
DIACRITICS_TO_REMOVE: str = 'ˈˌʰʷʲˤː̥ː̤ː̩ː̯ː̘ː̹ː̜ː̟ː̠ː̈ː̽'


def normalize_diacritics(text: str) -> str:
    """移除 diacritics（附加符號）."""
    # 移除變音符號
    text = text.replace('́', '')  # acute
    text = text.replace('̀', '')  # grave
    text = text.replace('̂', '')  # circumflex
    text = text.replace('̄', '')  # macron below
    text = text.replace('̃', '')  # tilde
    text = text.replace('̈', '')  # diaeresis
    text = text.replace('̇', '')  # dot above
    text = text.replace('̊', '')  # ring above
    text = text.replace('̌', '')  # caron
    text = text.replace('̑', '')  # inverted breve
    text = text.replace('̡', '')  # tongue root
    text = text.replace('̢', '')  # rhoticity
    text = text.replace('̤', '')  # breathy
    text = text.replace('̥', '')  # voiceless
    text = text.replace('̦', '')  # voiced
    text = text.replace('̩', '')  # syllabic
    text = text.replace('̯', '')  # non-syllabic
    text = text.replace('̹', '')  # more rounded
    text = text.replace('̜', '')  # less rounded
    text = text.replace('̟', '')  # advanced
    text = text.replace('̠', '')  # retracted
    text = text.replace('̽', '')  # mid
    return text


def normalize_asjp(text: str) -> str:
    """將任意語音轉寫轉換為 ASJP 格式.
    
    這是一個最大匹配替換，會盡可能將 IPA 轉換為 ASJP.
    """
    if not text:
        return ""
    
    # 移除空白
    text = text.strip()
    
    # 移除附加符號
    text = normalize_diacritics(text)
    
    # 處理常見組合 (先處理較長的)
    sorted_clusters = sorted(IPA_CLUSTERS.keys(), key=len, reverse=True)
    for cluster, replacement in sorted_clusters:
        text = text.replace(cluster, replacement)
    
    # 處理單一符號
    result = []
    i = 0
    while i < len(text):
        matched = False
        # 嘗試匹配較長的符號
        for length in range(min(4, len(text) - i), 0, -1):
            substr = text[i:i+length]
            if substr in IPA_TO_ASJP:
                result.append(IPA_TO_ASJP[substr])
                i += length
                matched = True
                break
        if not matched:
            # 保留未知字符（可能是特殊符號）
            if text[i] not in ' \t\n,;:.()[]{}':
                result.append(text[i])
            i += 1
    
    return ''.join(result)


def normalize_phonetic(text: str) -> str:
    """通用語音正規化：移除附加符號並標準化."""
    if not text:
        return ""
    
    text = text.strip()
    
    # 移除附加符號
    text = normalize_diacritics(text)
    
    # 移除重音符號
    text = text.replace('ˈ', '').replace('ˌ', '')
    
    # 移除音節邊界標記
    text = text.replace('.', '')
    
    # 移除長音符號
    text = text.replace('ː', '')
    
    # 標準化空白
    text = re.sub(r'\s+', '', text)
    
    return text


def clean_form(form: str) -> str:
    """清理詞彙形式：移除空格和標點."""
    if not form:
        return ""
    
    # 移除前後空白
    form = form.strip()
    
    # 移除常見的詞彙邊界標記
    # 如：-kuna (後綴), 'pukɨl (前綴)
    # 保留主要的詞彙形式
    
    return form


def extract_asjp_word(text: str) -> str:
    """從混合文本中提取 ASJP 格式的詞彙.
    
    例如：'pukɨl -> puk3l
    """
    if not text:
        return ""
    
    # 首先正規化為 ASJP
    asjp = normalize_asjp(text)
    
    # 移除非 ASJP 字符（保留基本拉丁字母和數字）
    asjp = re.sub(r'[^a-zA-Z0-9]', '', asjp)
    
    return asjp.lower()


if __name__ == "__main__":
    # 測試
    test_cases = [
        "mata",
        "taʔara", 
        "pukɨl",
        "matɑ",
        "kōrero",
        "ngā",
        "ʔā",
        "fijian",
        "taʔara",
    ]
    
    print("ASJP 轉換測試：")
    print("-" * 40)
    for form in test_cases:
        asjp = normalize_asjp(form)
        print(f"{form:15} -> {asjp}")
