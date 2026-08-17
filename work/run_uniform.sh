#!/usr/bin/env bash
# EXP-A 보강: 균등 크기 랜덤 centroid 대조군 (gs=10,30,100 × seed 0-4)
set -e
WORK="~/LLREC_표준학/work"
PY="/usr/bin/python3.12"
cd "$WORK"

for gs in 30 10 100; do
  for s in 0 1 2 3 4; do
    $PY build_uniform_centroid_data.py --seed "$s" --group-size "$gs" \
        --out "data/netflix_unif_gs${gs}_seed${s}"
    $PY run_experiment.py --dataset "netflix_unif_gs${gs}_seed${s}" --seed "$s" \
        --exp expA --tag "netflix_unif_gs${gs}"
  done
  # 표준 baseline 과 paired 비교 (std 는 EXP-A 에서 이미 생성됨)
  $PY aggregate_seeds.py --dir results/expA \
      --vanilla netflix_std --std "netflix_unif_gs${gs}" --seeds 0 1 2 3 4
done
echo "ALL UNIFORM DONE"
