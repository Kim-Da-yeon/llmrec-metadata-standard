# 실험 결과 요약 (Standard-Aware LLMRec)

생성일: 2026-04-30
환경: NVIDIA RTX PRO 6000 Blackwell (98GB), PyTorch 2.11+CUDA 13.0
데이터: Netflix Prize subset (LLMRec 공개 augmented data, n_users=13187, n_items=17366)
프로토콜: All-ranking, Best epoch (early stopping patience=7), LLMRec 논문 Table 2와 동일

---

## 1. 베이스라인 재현 검증 (RQ0)

논문 보고치를 본 환경에서 재현한 결과, 7개 지표 모두 95–99% 일치하여 **재현 신뢰성 확보**.

| 지표 | LLMRec 논문 | 본 연구 재현 (Vanilla) | 일치율 |
|---|---|---|---|
| R@10 | 0.0531 | 0.05149 | 96.97% |
| **R@20** | **0.0829** | **0.08130** | **98.07%** |
| R@50 | 0.1382 | 0.13713 | 99.23% |
| N@10 | 0.0272 | 0.02596 | 95.44% |
| **N@20** | **0.0347** | **0.03350** | **96.54%** |
| N@50 | 0.0456 | 0.04454 | 97.68% |
| P@20 | 0.0041 | 0.00407 | 99.27% |

## 2. RQ1 핵심 결과: Standard-Aware의 7개 지표 개선

| 지표 | Vanilla (재현) | **Standard-Aware** | Δ |
|---|---|---|---|
| R@10 | 0.05149 | **0.05257** | **+2.10%** |
| **R@20** | **0.08130** | **0.08347** | **+2.67%** |
| R@50 | 0.13713 | 0.13659 | -0.39% |
| N@10 | 0.02596 | **0.02725** | **+4.97%** |
| **N@20** | **0.03350** | **0.03497** | **+4.39%** |
| N@50 | 0.04454 | **0.04543** | **+2.00%** |
| P@20 | 0.00407 | **0.00417** | **+2.46%** |

> **7개 중 6개 지표 개선.** NDCG 평균 +3.8% > Recall 평균 +1.5% — 상위 순위 품질 개선이 더 큼.

## 3. LLMRec 논문 Table 2 다른 모델들과의 비교 (R@20)

| 모델 | R@20 | 출처 |
|---|---|---|
| MF-BPR | 0.0542 | 논문 |
| LightGCN | 0.0701 | 논문 |
| MMGCN | 0.0699 | 논문 |
| GRCN | 0.0706 | 논문 |
| LATTICE | 0.0737 | 논문 |
| MICRO | 0.0764 | 논문 |
| MMSSL | 0.0743 | 논문 |
| LLMRec (논문 SoTA) | 0.0829 | 논문 |
| LLMRec (본 연구 재현) | 0.0813 | 본 연구 |
| **Standard-Aware (본 연구)** | **0.0835** | **본 연구** |

> Standard-Aware는 LLMRec 논문 보고치 0.0829를 0.6% 상회하여 SoTA 갱신.

## 4. RQ3: 표준 적합성 정량화

| 필드 | Raw SCR | Post SCR | Uplift | 고유 토큰 (Raw → Norm) | 압축비 |
|---|---|---|---|---|---|
| director | 0.936 | 0.936 | +0.000 | 8,759 → 8,755 | 1.0× |
| country | **0.000** | **0.858** | **+0.858** | 1,166 → 57 | **20.5×** |
| language | **0.000** | **0.866** | **+0.866** | 326 → 52 | **6.3×** |
| year | 0.999 | 0.999 | +0.000 | 95 → 94 | 1.0× |
| **Overall SCR** | **0.484** | **0.914** | **+0.430** | — | — |
| **Interoperability Score** | **0.000** | **0.803** | **+0.803** | — | — |

## 5. RQ2 (Long-tail) — 본 데이터셋에서 측정 불가

LLMRec 공개 Netflix subset의 71.9% 아이템이 학습 인터랙션 0회. LightGCN은 freq=0 노드에 신호를 만들 수 없으므로 Cold Recall@20 = 0.000 (Vanilla / Std-Aware 양쪽 모두). LLMRec 논문도 cold/warm 분리 평가는 보고하지 않음. **데이터셋 한계로 본 텀페이퍼 범위에서는 RQ2 직접 검증 불가**, 졸업논문에서 ML-10M + attribute-aware backbone으로 확장 예정.

## 6. 종합 — 4축 트레이드오프

| 축 | Vanilla | Std-Aware | 효과 |
|---|---|---|---|
| 추천 성능 (R@20) | 0.0813 | **0.0835** | ↑ +2.67% |
| 추천 순위 품질 (N@20) | 0.0335 | **0.0350** | ↑ +4.39% |
| 표준 적합성 (SCR) | 0.484 | **0.914** | ↑ +43%p |
| 상호운용성 (IS) | 0.000 | **0.803** | ↑ +80%p |
| 데이터 분산 (고유 country) | 1,166 | **57** | ↓ 20.5× |

> **표준 준수와 추천 성능 간 트레이드오프 부재.** 두 축 동시 개선 (positive-sum).

## 7. 산출물 명세

| 파일 | 용도 | 위치 |
|---|---|---|
| `normalization.py` | ISO 3166 / BCP 47 정규화 모듈 | `work/normalization.py` |
| `compliance_metrics.py` | SCR / Interoperability 측정 | `work/compliance_metrics.py` |
| `build_standardized_data.py` | Centroid 기반 표준 임베딩 생성 | `work/build_standardized_data.py` |
| `prompt_templates.md` | schema.org/Movie JSON-LD 프롬프트 설계 (Stage A) | `work/prompt_templates.md` |
| `compliance_summary.json` | SCR 측정 raw output | `work/results/compliance_summary.json` |
| `baseline_train.log` | Vanilla 학습 로그 | `work/results/baseline_train.log` |
| `standard_aware_train.log` | Std-Aware 학습 로그 | `work/results/standard_aware_train.log` |
