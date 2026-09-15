"""
coldwarm_diagnostic.py
EXP-C-2: 콜드/웜 진단표 (한계 보고용).

현재 데이터는 freq=0 아이템이 대부분이라 cold recall 측정이 의미가 없다.
새 실험을 만들지 않고, freq 구간별(0 / 1-5 / 6+) 아이템 수·테스트 노출·Recall@20 을
표로 만들어 "왜 측정이 불가/무의미한지"를 정량적으로 보여주는 진단표를 생성한다.

출력 컬럼: freq_bucket | n_items | %items | n_test_pos | R@20(vanilla) | R@20(std)
  - n_test_pos : 해당 구간 아이템이 test 정답(positive)으로 등장한 횟수
  - R@20       : 해당 구간 정답에 대한 pooled Recall@20 (cold_start_eval.py 와 동일 방식)

freq 카운트는 데이터셋 무관(원본 train interaction) 이므로 vanilla 기준 1회 계산하고,
R@20 만 두 모델(netflix, netflix_std)에서 각각 측정한다.

사용법:
    python coldwarm_diagnostic.py --seed 0
    python coldwarm_diagnostic.py --seed 0 --datasets netflix netflix_std
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

LLMREC = os.path.expanduser("~/LLMRec")
WORK = os.path.dirname(os.path.abspath(__file__))

BUCKETS = [
    ("freq=0", lambda f: f == 0),
    ("freq=1-5", lambda f: 1 <= f <= 5),
    ("freq=6+", lambda f: f >= 6),
]

FIXED = [
    "--lr", "0.0001", "--batch_size", "1024", "--embed_size", "64",
    "--prune_loss_drop_rate", "0.71", "--aug_sample_rate", "0.1",
    "--early_stopping_patience", "7",
]


def train_and_score(dataset, seed, epoch, K=20):
    """주어진 데이터셋을 학습 후 사용자별 top-K(학습 아이템 마스킹) 반환.
    반환: (topk[n_users, K] ndarray, data_generator)"""
    os.chdir(LLMREC)
    sys.argv = ["main.py", "--dataset", dataset, "--seed", str(seed),
                "--Ks", "[10,20,50]", "--epoch", str(epoch), "--verbose", "5"] + FIXED

    import importlib
    import utility.parser as p_mod; importlib.reload(p_mod)
    import utility.batch_test as bt_mod; importlib.reload(bt_mod)
    import utility.load_data as ld_mod; importlib.reload(ld_mod)
    import Models as m_mod; importlib.reload(m_mod)
    import main as main_mod; importlib.reload(main_mod)

    main_mod.set_seed(seed)
    dg = bt_mod.data_generator
    cfg = {"n_users": dg.n_users, "n_items": dg.n_items}
    trainer = main_mod.Trainer(data_config=cfg)
    trainer.train()

    trainer.model_mm.eval()
    with torch.no_grad():
        ua, ia, *_ = trainer.model_mm(trainer.ui_graph, trainer.iu_graph,
                                      trainer.image_ui_graph, trainer.image_iu_graph,
                                      trainer.text_ui_graph, trainer.text_iu_graph)
    scores = torch.mm(ua, ia.T)

    # 학습 아이템 마스킹 (LLMRec testing convention)
    n_items = dg.n_items
    rows, cols = [], []
    for u, items in dg.train_items.items():
        for it in items:
            if int(it) < n_items:
                rows.append(int(u)); cols.append(int(it))
    if rows:
        scores[rows, cols] = -1e9
    topk = torch.topk(scores, K, dim=1).indices.cpu().numpy()
    return topk, dg


def item_freq(dg):
    n_items = dg.n_items
    freq = defaultdict(int)
    for u, items in dg.train_items.items():
        for it in items:
            freq[int(it)] += 1
    return np.array([freq.get(i, 0) for i in range(n_items)])


def bucket_recall(topk, dg, freqs):
    """구간별 pooled Recall@20 과 test positive 수 집계."""
    bucket_of = {}
    for i, f in enumerate(freqs):
        for name, pred in BUCKETS:
            if pred(f):
                bucket_of[i] = name
                break
    hits = defaultdict(int)
    total = defaultdict(int)
    for u, gt in dg.test_set.items():
        u = int(u)
        gt_set = set(int(x) for x in gt)
        if not gt_set:
            continue
        pred = set(int(x) for x in topk[u])
        for it in gt_set:
            b = bucket_of.get(it)
            if b is None:
                continue
            total[b] += 1
            if it in pred:
                hits[b] += 1
    return hits, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epoch", type=int, default=200)
    ap.add_argument("--datasets", nargs="+", default=["netflix", "netflix_std"],
                    help="[vanilla, std] 순서로 2개")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    sys.path.insert(0, LLMREC)

    vanilla_ds, std_ds = args.datasets[0], args.datasets[1]

    # vanilla 학습/스코어 → freq 기준(데이터셋 무관)도 여기서 산출
    topk_v, dg_v = train_and_score(vanilla_ds, args.seed, args.epoch)
    freqs = item_freq(dg_v)
    hits_v, total_v = bucket_recall(topk_v, dg_v, freqs)

    # std 학습/스코어 (freq/test_set 은 동일하므로 R@20 만 사용)
    topk_s, dg_s = train_and_score(std_ds, args.seed, args.epoch)
    hits_s, total_s = bucket_recall(topk_s, dg_s, freqs)

    n_items_total = len(freqs)
    print(f"\n진단표 (seed={args.seed}, n_items={n_items_total})")
    print(f"{'freq_bucket':12}{'n_items':>9}{'%items':>9}{'n_test_pos':>12}"
          f"{'R@20(van)':>12}{'R@20(std)':>12}")
    print("-" * 66)
    rows = []
    for name, pred in BUCKETS:
        n_items = int(np.sum([1 for f in freqs if pred(f)]))
        pct = 100.0 * n_items / n_items_total
        ntp = total_v.get(name, 0)
        rv = hits_v.get(name, 0) / ntp if ntp else float("nan")
        rs = hits_s.get(name, 0) / ntp if ntp else float("nan")
        print(f"{name:12}{n_items:>9}{pct:>8.1f}%{ntp:>12}{rv:>12.5f}{rs:>12.5f}")
        rows.append({
            "freq_bucket": name,
            "n_items": n_items,
            "pct_items": round(pct, 2),
            "n_test_pos": ntp,
            "R@20_vanilla": None if ntp == 0 else round(rv, 5),
            "R@20_std": None if ntp == 0 else round(rs, 5),
        })

    out_path = args.out or os.path.join(WORK, "results", "expC2", f"coldwarm_seed{args.seed}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"seed": args.seed, "n_items": n_items_total,
                   "vanilla": vanilla_ds, "std": std_ds, "buckets": rows},
                  f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
