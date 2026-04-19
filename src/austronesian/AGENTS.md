# src/austronesian/ — Python 套件

計算語言學研究工具包，可編輯安裝（`pip install -e ".[dev]"`）。
CLI 入口：`austronesian` → `cli/main.py:main()`

---

## STRUCTURE

```
austronesian/
├── analysis/
│   ├── distance.py    # Levenshtein / SCA / Weighted 三種距離計算
│   ├── phonetics.py   # IPA → ASJP 轉換與正規化
│   ├── cognates.py    # 同源詞啟發式偵測
│   ├── roots.py       # Proto-Austronesian 詞根重構輔助
│   └── sound_change.py # 音變對應分析
├── databases/
│   ├── abvd.py        # ABVD REST API 客戶端
│   └── acd.py         # ACD HTML scraper
├── models/
│   ├── language.py    # Language dataclass
│   ├── lexeme.py      # Lexeme dataclass
│   └── cognate.py     # CognateSet dataclass
└── cli/
    └── main.py        # CLI 入口點
```

---

## WHERE TO LOOK

| 需求 | 位置 |
|------|------|
| 修改距離公式 | `analysis/distance.py` — `normalized_levenshtein_distance()`, `sound_class_distance()`, `weighted_levenshtein_distance()` |
| IPA → ASJP 對照表 | `analysis/phonetics.py` — `IPA_TO_ASJP` dict |
| ABVD API 呼叫 | `databases/abvd.py` — `ABVDClient` class |
| 資料模型定義 | `models/*.py` |

---

## KEY CONVENTIONS

- **中文 docstring**：分析模組用中文，模型/資料庫模組用英文
- **回傳型別**：所有距離函數回傳 `float ∈ [0, 1]`
- **ASJP 字串**：距離計算的輸入皆為 ASJP 格式，不是原始 IPA
- **距離函數純函數**：無 side effect，可直接被 `scripts/` 呼叫

---

## ANTI-PATTERNS

- **勿在套件內直接讀 `data/` 或 `results/`**（資料讀取由 `scripts/` 負責）
- **勿修改 `ABVDClient` 的速率限制**（ABVD 伺服器有 rate limit，目前設 0.5s/request）
- **`models/` 是純 dataclass**，不含計算邏輯（邏輯在 `analysis/`）
