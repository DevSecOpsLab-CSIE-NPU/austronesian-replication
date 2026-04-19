# Replication Package: *A Cross-Family Migration-Gradient Workflow*

**Paper**: Chao, A. F.-Y. (2026). A Cross-Family Migration-Gradient Workflow with Pre-Registered Replication and Null Reporting. *Digital Scholarship in the Humanities* (DSH). Oxford University Press.

**Repository**: https://github.com/DevSecOpsLab-CSIE-NPU/austronesian-replication

---

## Overview

This repository provides the complete replication package for the above paper. It contains all analysis scripts, pre-computed results, pre-registration documents, and figures for the six-family cross-linguistic migration-gradient workflow.

**Families covered**: Austronesian, Indo-European, Sino-Tibetan, Bantu, Uralic (pre-registered null), Turkic (pre-registered null)

**Core method**: CI-disjoint signal classification — each (family, concept) pair is mechanically labelled as *selective anchor*, *founder-effect baseline*, *borrowing signature*, *ceiling effect*, or *null/untestable* based on 500-bootstrap 95% confidence intervals.

---

## Repository Structure

```
├── scripts/
│   ├── xfam_an_migration_gradient.py        # Austronesian
│   ├── xfam_an_expanded_concepts.py
│   ├── xfam_ie_migration_gradient.py        # Indo-European
│   ├── xfam_ie_cultural_concepts.py
│   ├── xfam_ie_homeland_sensitivity.py
│   ├── xfam_st_migration_gradient.py        # Sino-Tibetan
│   ├── xfam_st_pooled_mountain.py
│   ├── xfam_st_stedt_full_extraction.py
│   ├── xfam_st_stedt_expanded_extraction.py
│   ├── xfam_st_stedt_rice_poc.py
│   ├── xfam_bantu_migration_gradient.py     # Bantu
│   ├── xfam_bantu_cultural_concepts.py
│   ├── xfam_5th_uralic_migration_gradient.py   # Pre-registered null
│   ├── xfam_6th_turkic_migration_gradient.py   # Pre-registered null
│   ├── xfam_meta_analysis_v4.py            # Cross-family summary (primary)
│   ├── xfam_robustness_alt_tests.py        # Inferential robustness
│   ├── xfam_check_data_availability.py
│   ├── draw_stedt_pipeline.py              # Figure: STEDT pipeline
│   ├── generate_decision_figure.py         # Figure: forest plot
│   ├── generate_new_figures.py
│   └── visualization.py
├── src/austronesian/                        # Python package (distance functions, ABVD client)
├── results/                                 # Pre-computed outputs (JSON + CSV)
│   ├── xfam_meta_v4.json                   # Primary cross-family results
│   ├── xfam_meta_v4_long.csv
│   ├── xfam_robustness.json                # Robustness check results
│   ├── xfam_robustness_table.csv
│   └── xfam_*.json / xfam_*.csv           # Per-family outputs
├── figures/
│   ├── fig_xfam_v4_forest.png             # Figure 1: forest plot
│   ├── fig_xfam_v4_scorecard.png          # Figure 2: scorecard
│   └── fig_stedt_pipeline.png             # Figure 3: STEDT pipeline
├── preregistration/
│   ├── paper-2-5th-family-preregistration.md   # Uralic (null verdict)
│   └── paper-2-6th-family-preregistration.md   # Turkic (null verdict)
└── pyproject.toml
```

---

## Requirements

- Python 3.9+

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Key dependencies: `pandas`, `numpy`, `scipy`, `matplotlib`, `requests`, `tqdm`

---

## Data Sources

The workflow ingests data from Lexibank CLDF datasets. These must be downloaded separately:

| Family | Dataset | Source |
|--------|---------|--------|
| Austronesian | ABVD | https://abvd.eva.mpg.de/ |
| Indo-European | IECoR | https://github.com/lexibank/iecor |
| Sino-Tibetan | sagartst / zhangst / STEDT | https://github.com/lexibank/sagartst ; https://stedt.berkeley.edu/ |
| Bantu | Grollemund 2015 | https://github.com/lexibank/grollemund |
| Uralic | DIACL | https://github.com/diacl |
| Turkic | (see script for source) | — |

Pre-computed results for all families are included in `results/` — re-running ingestion scripts is not required for replication.

---

## Pipeline Execution

### Per-family gradient scripts

```bash
# Austronesian
python scripts/xfam_an_migration_gradient.py

# Indo-European (primary Kurgan homeland)
python scripts/xfam_ie_migration_gradient.py
# Sensitivity: Anatolian alternative homeland
python scripts/xfam_ie_homeland_sensitivity.py

# Sino-Tibetan (STEDT + Lexibank)
python scripts/xfam_st_stedt_full_extraction.py
python scripts/xfam_st_migration_gradient.py

# Bantu
python scripts/xfam_bantu_migration_gradient.py

# Pre-registered families (Uralic, Turkic)
python scripts/xfam_5th_uralic_migration_gradient.py
python scripts/xfam_6th_turkic_migration_gradient.py
```

### Cross-family meta-analysis

```bash
python scripts/xfam_meta_analysis_v4.py   # → results/xfam_meta_v4.json
```

### Robustness checks

```bash
python scripts/xfam_robustness_alt_tests.py  # → results/xfam_robustness.json
```

### Figure regeneration

```bash
python scripts/generate_decision_figure.py   # forest plot
python scripts/draw_stedt_pipeline.py        # STEDT pipeline diagram
```

---

## Key Constants (fixed across all scripts)

| Constant | Value | Description |
|----------|-------|-------------|
| `RANDOM_SEED` | 42 | `np.random.default_rng(42)` |
| `N_BOOTSTRAP` | 500 | Bootstrap resamples for CI |
| `NLD_THRESHOLD` | 0.4 | STEDT Method B cognate clustering |

---

## Pre-Registration

Two families (Uralic, Turkic) were pre-registered before gradient computation. The pre-registration files in `preregistration/` are version-controlled with no edits after the "Pre-registration signed off" line. Both produced null verdicts — the workflow is not a yes-machine.

---

## Software Versions

Python 3.12; pandas 2.3; numpy 2.4; scipy 1.17; matplotlib 3.10.

---

## License

Code: MIT License  
Data: Subject to respective source database terms (ABVD, STEDT, Lexibank — academic, non-commercial)
