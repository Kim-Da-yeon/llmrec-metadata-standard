# Standard-Aware LLM Graph Augmentation for Recommendation

**Diagnosing and normalizing the standard compliance of LLM-generated recommendation metadata (ISO 3166-1 / BCP-47 / ISO 8601)**

> Introduction to Standards — Term Paper · Target system: [LLMRec (HKUDS, WSDM '24)](https://github.com/HKUDS/LLMRec) · Data: Netflix Prize subset (17,366 items)

---

## Overview

LLM-based recommenders augment sparse item metadata (country, language, etc.) with LLM-generated text. But LLMs emit **free text**, not controlled vocabulary, so the same country appears as `USA` / `United States` / `America` — inconsistent surface forms that break cross-system interoperability and data integration.

This work **quantitatively diagnoses** the standard compliance of LLMRec's generated Netflix metadata and improves it with an ISO 3166-1 / BCP-47 normalization layer.

## Key Results

### ✅ Standard normalization — interoperability 0 → 80.3% (core contribution)

| Field | Format compliance (Raw → Norm) | Unique surface → canonical | Compression |
|---|---|---|---|
| country | 0.000 → **0.858** | 1,166 → 57 | **20.5×** |
| language | 0.000 → **0.866** | 326 → 52 | **6.3×** |
| year | 0.999 → 0.999 | — | — |
| director | 0.936 → 0.936 (no standard) | — | — |
| **Overall SCR** | 0.484 → **0.914** | — | — |
| **Interoperability (IS)** | 0.000 → **0.803** | — | — |

- A single country scattered across **1,166 surface forms** (20.5× redundancy) → unified to canonical standard codes
- After normalization, **80.3%** of items convert to valid schema.org/Movie JSON-LD across all four fields
- Standardization also acts as an **LLM error detector**: e.g. **724 column-shift errors** (a country injected into the language field) automatically caught

### ⚠️ Recommendation performance — no significant improvement (honest reporting)

The +2.67% (R@20) gain observed with a single seed (2022) **did not replicate across multiple seeds (0–9)**.

| Metric | Vanilla (mean±std) | Std-Aware (mean±std) | Δ% | p (paired t-test) |
|---|---|---|---|---|
| R@20 | 0.07588±0.00489 | 0.07696±0.00475 | +1.43 | 0.293 |
| N@20 | 0.03029±0.00186 | 0.03045±0.00247 | +0.51 | 0.783 |

**All 7 metrics p > 0.05** (every 95% CI includes 0). A controlled experiment (EXP-A) shows the observed shift comes from **attribute-embedding smoothing (centroid denoising)**, not from the semantics of standard codes — standard and random centroids are statistically indistinguishable because the data is dominated by US/English (head-dominance).

→ **Conclusion:** the verified value of standardization lies in **data quality and interoperability**; a direct link to recommendation accuracy is not confirmed on this dataset.

## Repository Structure

```
.
├── term-paper.md / .pdf               # Term paper (full write-up)
├── images/                            # Figures
└── work/
    ├── normalization.py               # Standard normalization module (ISO 3166-1 / BCP-47)
    ├── compliance_metrics.py          # Standard-compliance diagnostics
    ├── build_*_data.py                # Data builders (standardized / centroid variants)
    ├── run_experiment.py              # LLMRec training & evaluation
    ├── aggregate_seeds.py             # Multi-seed aggregation + paired t-test
    ├── cold_start_eval.py             # Cold-start diagnostics
    ├── make_figures.py                # Figure generation
    ├── prompt_templates.md            # Standard-aware prompt design document
    ├── run_*.sh                       # Experiment runners
    └── results/                       # Experiment outputs (JSON / logs / summary)
        ├── RESULTS.md                 # Multi-seed validation (honest null result)
        ├── compliance_summary.json    # Standard-compliance diagnostics output
        ├── expB/                      # Multi-seed (Vanilla vs Std-Aware)
        └── expD/                      # Augment-clean comparison
```

## Reproduce

```bash
# Standard-compliance diagnostics
python3 work/compliance_metrics.py

# Aggregate multi-seed results and run the statistical test
python3 work/aggregate_seeds.py
```

> Environment: Python 3.12 · PyTorch 2.11+cu130 · pycountry, scipy
> LLMRec training requires the public Netflix augmented data from [HKUDS/LLMRec](https://github.com/HKUDS/LLMRec).

## Standards

- **ISO 3166-1 alpha-2** — country codes
- **BCP 47** — language tags
- **ISO 8601** — year (datePublished)
- **schema.org/Movie** (JSON-LD) — interoperability validation schema
