#!/usr/bin/env bash
# 성능 탐색: 신호 강화 HP 스윕 (seed 0, 탐색용)
set -e
WORK="~/LLREC_표준학/work"; PY="/usr/bin/python3.12"; cd "$WORK"
S=0
run(){ $PY run_experiment.py --dataset netflix --seed $S --exp sweep --tag "$1" ${2:+--set "$2"}; }

run base
run mcr0.1   "model_cat_rate=0.1"
run mcr0.2   "model_cat_rate=0.2"
run mcr0.4   "model_cat_rate=0.4"
run L2       "layers=2"
run prune0.5 "prune_loss_drop_rate=0.5"
run ucr4     "user_cat_rate=4.0"
run mcr0.2_L2 "model_cat_rate=0.2,layers=2"
echo "SWEEP DONE"
