# LLM 생성 추천 메타데이터의 표준 적합성 진단과 정규화

**ISO 3166-1 / BCP-47 / ISO 8601 기반 LLM 증강 추천 데이터의 품질·상호운용성 개선**

> 표준학개론 텀페이퍼 · 대상 시스템: [LLMRec (HKUDS)](https://github.com/HKUDS/LLMRec) · 데이터: Netflix Prize subset (17,366 items)

---

## 개요

LLM 기반 추천 시스템은 부족한 항목 메타데이터(국가·언어 등)를 LLM으로 생성·증강한다. 그러나 LLM은 통제 어휘가 아니라 **자유 텍스트**를 출력하므로, 같은 국가가 `USA` / `United States` / `America`처럼 비일관 형식으로 산출되어 시스템 간 상호운용과 데이터 통합을 저해한다.

본 연구는 LLM이 생성한 Netflix 메타데이터의 **표준 적합성을 정량 진단**하고, ISO 3166-1 / BCP-47 정규화 레이어로 이를 개선한다.

## 핵심 결과

### ✅ 표준 정규화 — 상호운용성 0 → 80.3% (핵심 기여)

| 필드 | 형식 적합률 (Raw → Norm) | 고유 표면형 → 정준 | 압축비 |
|---|---|---|---|
| country | 0.000 → **0.858** | 1,166 → 57 | **20.5×** |
| language | 0.000 → **0.866** | 326 → 52 | **6.3×** |
| year | 0.999 → 0.999 | — | — |
| director | 0.936 → 0.936 (표준 없음) | — | — |
| **전체 SCR** | 0.484 → **0.914** | — | — |
| **상호운용성 (IS)** | 0.000 → **0.803** | — | — |

- 같은 국가가 **1,166개 표면형**으로 난립(20.5× 중복) → 정준 표준 코드로 통합
- 정규화 후 **80.3%** 항목이 4개 필드 모두 유효한 schema.org/Movie JSON-LD로 변환 가능
- 표준화가 **LLM 생성 오류 검출기**로도 기능: language 칸에 국가가 삽입된 **컬럼시프트 오류 724건** 등 자동 탐지

### ⚠️ 추천 성능 — 유의한 개선 없음 (정직 보고)

단일 seed(2022)에서 관측된 Standard-Aware의 +2.67%(R@20) 개선은 **다중 seed(0–9)에서 통계적으로 재현되지 않았다**.

| 지표 | Vanilla (mean±std) | Std-Aware (mean±std) | Δ% | p (paired t-test) |
|---|---|---|---|---|
| R@20 | 0.07588±0.00489 | 0.07696±0.00475 | +1.43 | 0.293 |
| N@20 | 0.03029±0.00186 | 0.03045±0.00247 | +0.51 | 0.783 |

**7개 지표 전부 p > 0.05** (95% CI 모두 0 포함). 통제 실험(EXP-A) 결과, 관측된 변화는 표준 코드의 의미성이 아니라 **속성 임베딩 평활화(centroid denoising) 효과**에 기인하며, 표준 centroid와 랜덤 centroid가 통계적으로 구분되지 않는다(데이터가 US/English에 지배되는 head-dominance).

→ **결론:** 표준화의 검증된 가치는 **데이터 품질·상호운용성**에 있으며, 추천 정확도와의 직접 연계는 본 데이터셋에서 확인되지 않았다.

## 저장소 구조

```
.
├── 메타데이터_표준화_정리.md          # 표준 적합성 진단·정규화 연구 정리 (핵심 결과)
├── 표준학개론_텀페이퍼.md / .pdf       # 텀페이퍼
├── 표준학개론_프로포절_최종.md / .pdf  # 연구 제안서
├── images/                            # 논문 그림
└── work/
    ├── normalization.py               # 표준 정규화 모듈 (ISO 3166-1 / BCP-47)
    ├── compliance_metrics.py          # 표준 적합성 진단 스크립트
    ├── build_*_data.py                # 데이터 빌드 (standardized / centroid 변형)
    ├── run_experiment.py              # LLMRec 학습·평가 실행
    ├── aggregate_seeds.py             # 다중 seed 집계 · paired t-test
    ├── cold_start_eval.py             # cold-start 진단
    ├── make_figures.py                # 그림 생성
    ├── run_*.sh                       # 실험 실행 스크립트
    └── results/                       # 실험 결과 (JSON / 로그 / 요약)
        ├── RESULTS_SUMMARY.md         # 단일 seed 결과 요약
        ├── RESULTS_ADDENDUM.md        # 다중 seed 검증 (null result 정직 보고)
        ├── compliance_summary.json    # 표준 적합성 진단 결과
        ├── expB/                      # 다중 seed (Vanilla vs Std-Aware)
        └── expD/                      # augment clean 비교
```

## 재현

```bash
# 표준 적합성 진단
python3 work/compliance_metrics.py

# 다중 seed 결과 집계 및 통계 검정
python3 work/aggregate_seeds.py
```

> 실행 환경: Python 3.12 · PyTorch 2.11+cu130 · pycountry, scipy
> LLMRec 학습에는 [HKUDS/LLMRec](https://github.com/HKUDS/LLMRec)의 공개 Netflix augmented data가 필요하다.

## 표준

- **ISO 3166-1 alpha-2** — 국가 코드
- **BCP 47** — 언어 태그
- **ISO 8601** — 연도(datePublished)
- **schema.org/Movie** (JSON-LD) — 상호운용성 검증 스키마
