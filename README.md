# LLM 생성 추천 메타데이터의 국제 표준 정규화

ISO 3166-1 · BCP 47 · ISO 8601 적합성 진단 및 정규화 레이어.
대상 시스템 [LLMRec (HKUDS, WSDM '24)](https://github.com/HKUDS/LLMRec) · 데이터 Netflix Prize 부분집합 17,366건 · 표준학 기말 과제 · 2026

---

## 결과

### 표준 정규화 — 상호운용성 0 → 80.3%

| 필드 | 형식 적합률 (원본 → 정규화) | 표면형 → 표준 코드 | 압축비 |
|---|---|---|---|
| country | 0.000 → 0.858 | 1,166 → 57 | 20.5× |
| language | 0.000 → 0.866 | 326 → 52 | 6.3× |
| year | 0.999 → 0.999 | — | — |
| director | 0.936 → 0.936 (해당 표준 없음) | — | — |
| 전체 SCR | 0.484 → 0.914 | — | — |
| 상호운용성 IS | 0.000 → 0.803 | — | — |

- 한 국가가 표면형 1,166개로 분산 (중복 20.5배) → 표준 코드로 통합
- 정규화 후 4개 필드 전부에서 schema.org/Movie JSON-LD 변환 가능 항목 80.3%
- 부수 효과로 LLM 오류 검출: language 필드에 국가명이 들어간 column-shift 724건 자동 적발

### 추천 성능 — 유의한 개선 없음

단일 seed(2022)의 +2.67% (R@20)는 seed 0–9에서 재현되지 않음.

| 지표 | Vanilla (mean±std) | Std-Aware (mean±std) | Δ% | p (paired t-test) |
|---|---|---|---|---|
| R@20 | 0.07588±0.00489 | 0.07696±0.00475 | +1.43 | 0.293 |
| N@20 | 0.03029±0.00186 | 0.03045±0.00247 | +0.51 | 0.783 |

- 7개 지표 전부 p > 0.05, 95% CI 전부 0 포함
- 통제 실험(EXP-A): 관측된 변화는 표준 코드의 의미가 아니라 속성 임베딩 평활(centroid denoising)에서 기인. 표준 centroid와 무작위 centroid가 통계적으로 구분되지 않음 — 데이터가 US/English에 편중된 head-dominance 때문
- 검증된 기여는 데이터 품질과 상호운용성. 추천 정확도와의 직접적 연결은 이 데이터셋에서 미확인

## 구성

```
term-paper.md / .pdf              기말 보고서
images/                           그림
work/
  normalization.py                정규화 모듈 (ISO 3166-1 · BCP 47)
  compliance_metrics.py           표준 적합성 진단
  build_*_data.py                 데이터 생성 (표준화 / centroid 변형)
  run_experiment.py               LLMRec 학습·평가
  aggregate_seeds.py              다중 seed 집계 + paired t-test
  cold_start_eval.py              콜드스타트 진단
  make_figures.py                 그림 생성
  prompt_templates.md             표준 인지 프롬프트 설계
  results/
    RESULTS.md                    다중 seed 검증 결과
    compliance_summary.json       적합성 진단 출력
    expB/                         Vanilla vs Std-Aware 다중 seed
    expD/                         augment-clean 비교
```

## 실행

```bash
python3 work/compliance_metrics.py   # 표준 적합성 진단
python3 work/aggregate_seeds.py      # 다중 seed 집계 및 검정
```

Python 3.12 · PyTorch 2.11+cu130 · pycountry · scipy.
LLMRec 학습에는 [HKUDS/LLMRec](https://github.com/HKUDS/LLMRec)의 Netflix augmented 데이터 필요.

## 표준

ISO 3166-1 alpha-2 (국가) · BCP 47 (언어) · ISO 8601 (연도) · schema.org/Movie JSON-LD (상호운용성 검증 스키마)

