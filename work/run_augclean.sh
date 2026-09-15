#!/usr/bin/env bash
# EXP-D: 증강쌍 통계 정제 2x2 (raw/clean x 기본/업가중), seed 0-4
set -e
WORK="$HOME/LLREC_표준학/work"; PY="/usr/bin/python3.12"; cd "$WORK"
UP="0.05"   # 업가중 aug_mf_rate (기본 0.012)
for s in 0 1 2 3 4; do
  $PY run_experiment.py --dataset netflix          --seed "$s" --exp expD --tag raw_def
  $PY run_experiment.py --dataset netflix          --seed "$s" --exp expD --tag raw_up   --aug_mf_rate "$UP"
  $PY run_experiment.py --dataset netflix_augclean --seed "$s" --exp expD --tag clean_def
  $PY run_experiment.py --dataset netflix_augclean --seed "$s" --exp expD --tag clean_up --aug_mf_rate "$UP"
done
echo "=== 정제 효과 (기본 가중): raw_def vs clean_def ==="
$PY aggregate_seeds.py --dir results/expD --vanilla raw_def --std clean_def --seeds 0 1 2 3 4
echo "=== 정제 효과 (업가중): raw_up vs clean_up ==="
$PY aggregate_seeds.py --dir results/expD --vanilla raw_up --std clean_up --seeds 0 1 2 3 4
echo "=== 업가중 효과 (정제 데이터): clean_def vs clean_up ==="
$PY aggregate_seeds.py --dir results/expD --vanilla clean_def --std clean_up --seeds 0 1 2 3 4
echo "ALL EXP-D DONE"
