#!/usr/bin/env bash
# 추가 실험 일괄 실행 스크립트 (학회논문 보강용)
# 우선순위: EXP-A > EXP-B > EXP-C-1 > EXP-C-2
#
# 주의: 학습 1회당 best-epoch까지 수십 분 소요될 수 있음(GPU). 전체는 매우 오래 걸린다.
#       필요한 블록만 골라 실행하거나, 백그라운드/세션 분리 실행을 권장.
set -e

WORK="~/LLREC_표준학/work"
LLMREC="~/LLMRec"
# 이 프로젝트의 의존성(torch+cu130, pycountry, scipy)이 설치된 인터프리터
PY="/usr/bin/python3.12"

cd "$WORK"

# =========================================================
# EXP-A: 표준 centroid vs 랜덤 centroid ablation (최우선)
#   - netflix_std 는 seed 별로 재학습(데이터 동일, 학습 seed만 변경)
#   - netflix_rand 는 seed 별로 데이터 자체를 새로 생성 후 학습
#   - aggregate: --vanilla netflix_std --std netflix_rand
# =========================================================
exp_A() {
  for s in 0 1 2 3 4; do
    # 1) 랜덤 centroid 데이터 생성 (country+language, 표준과 동일 필드/크기분포)
    cd "$WORK"
    $PY build_random_centroid_data.py --seed "$s" --out "data/netflix_rand_seed$s"
    # 2) 학습 + 결과 저장 (tag 고정 → seed 별 비교 가능)
    $PY run_experiment.py --dataset "netflix_rand_seed$s" --seed "$s" --exp expA --tag netflix_rand
    # 3) 표준 baseline (동일 seed)
    $PY run_experiment.py --dataset netflix_std --seed "$s" --exp expA --tag netflix_std
  done
  $PY aggregate_seeds.py --dir results/expA --vanilla netflix_std --std netflix_rand --seeds 0 1 2 3 4
}

# =========================================================
# EXP-B: 다중 seed 안정성 + paired t-test
# =========================================================
exp_B() {
  for s in 0 1 2 3 4 5 6 7 8 9; do
    $PY run_experiment.py --dataset netflix     --seed "$s" --exp expB
    $PY run_experiment.py --dataset netflix_std --seed "$s" --exp expB
  done
  $PY aggregate_seeds.py --dir results/expB --vanilla netflix --std netflix_std \
      --seeds 0 1 2 3 4 5 6 7 8 9
}

# =========================================================
# EXP-C-1: per-K (5,10,20,50). 측정만 추가 (모델 선택은 R@20 고정 유지)
# =========================================================
exp_C1() {
  for s in 0 1 2; do
    $PY run_experiment.py --dataset netflix     --seed "$s" --exp expC1 --ks '[5,10,20,50]'
    $PY run_experiment.py --dataset netflix_std --seed "$s" --exp expC1 --ks '[5,10,20,50]'
  done
}

# =========================================================
# EXP-C-2: 콜드/웜 진단표
# =========================================================
exp_C2() {
  $PY coldwarm_diagnostic.py --seed 0 --datasets netflix netflix_std
}

# 실행할 블록 선택 (인자 없으면 안내만)
if [ $# -eq 0 ]; then
  echo "usage: bash run_all.sh [A|B|C1|C2|all]"
  exit 0
fi
for arg in "$@"; do
  case "$arg" in
    A)  exp_A ;;
    B)  exp_B ;;
    C1) exp_C1 ;;
    C2) exp_C2 ;;
    all) exp_A; exp_B; exp_C1; exp_C2 ;;
    *) echo "unknown: $arg" ;;
  esac
done
