"""
Long-tail / Cold-Item Recall@20 evaluation.

We define "cold" items as those with the fewest training interactions.
Bucket the test set's positive items by their training-interaction count,
then compute Recall@20 restricted to cold buckets.

This is a post-hoc evaluation on top of an already-trained checkpoint:
we re-run inference inside main.py logic but split the recall numerator
by item warmth.
"""

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import torch

# add LLMRec to import path
sys.path.insert(0, os.path.expanduser("~/LLMRec"))
os.chdir(os.path.expanduser("~/LLMRec"))


def evaluate(dataset_name, ckpt_args=None):
    # patch sys.argv so LLMRec's parser picks up our dataset
    sys.argv = ["main.py", "--dataset", dataset_name, "--epoch", "200",
                "--verbose", "5", "--batch_size", "1024"]
    # re-import main fresh
    import importlib
    import utility.parser as p_mod; importlib.reload(p_mod)
    import utility.batch_test as bt_mod; importlib.reload(bt_mod)
    import utility.load_data as ld_mod; importlib.reload(ld_mod)
    import Models as m_mod; importlib.reload(m_mod)
    import main as main_mod; importlib.reload(main_mod)

    args = p_mod.parse_args()
    data_generator = bt_mod.data_generator
    train_items = data_generator.train_items
    test_set = data_generator.test_set

    # interaction count per item from training
    item_freq = defaultdict(int)
    for u, items in train_items.items():
        for it in items:
            item_freq[int(it)] += 1
    n_items = data_generator.n_items
    for it in range(n_items):
        item_freq.setdefault(it, 0)

    # Bucket items by warmth
    freqs = np.array([item_freq[i] for i in range(n_items)])
    # use absolute thresholds because the distribution is very long-tailed
    # cold = freq <= 1 (long-tail); warm = freq >= 5 (head)
    cold_items = set(np.where(freqs <= 1)[0].tolist())
    warm_items = set(np.where(freqs >= 5)[0].tolist())
    print(f"Items: total={n_items}, cold(freq<=1)={len(cold_items)}, warm(freq>=5)={len(warm_items)}")
    p20 = float(np.percentile(freqs, 20))
    p50 = float(np.percentile(freqs, 50))

    # Train (1 epoch is enough since we only need any ckpt; better: rely on cached behavior)
    # Actually run a short training to load the model in trained state — we'll use the same code path.
    # For simplicity: do a fresh training to convergence isn't needed; we just want recall buckets
    # on the *converged* checkpoint. So we'll run main_mod.Trainer.train() with default settings.
    config = {"n_users": data_generator.n_users, "n_items": data_generator.n_items}
    trainer = main_mod.Trainer(data_config=config)
    trainer.train()

    # After training, get embeddings
    trainer.model_mm.eval()
    with torch.no_grad():
        ua, ia, *_ = trainer.model_mm(trainer.ui_graph, trainer.iu_graph,
                                       trainer.image_ui_graph, trainer.image_iu_graph,
                                       trainer.text_ui_graph, trainer.text_iu_graph)
    scores = torch.mm(ua, ia.T)  # n_users x n_items

    # Mask training items (LLMRec testing convention)
    rows, cols = [], []
    for u, items in train_items.items():
        for it in items:
            if int(it) < n_items:
                rows.append(int(u))
                cols.append(int(it))
    if rows:
        scores[rows, cols] = -1e9

    K = 20
    topk = torch.topk(scores, K, dim=1).indices.cpu().numpy()

    overall_hits = overall_total = 0
    cold_hits = cold_total = 0
    warm_hits = warm_total = 0
    for u, gt in test_set.items():
        u = int(u)
        gt_set = set(int(x) for x in gt)
        if not gt_set:
            continue
        pred = set(int(x) for x in topk[u])
        gt_cold = gt_set & cold_items
        gt_warm = gt_set & warm_items

        if gt_set:
            overall_hits += len(pred & gt_set)
            overall_total += len(gt_set)
        if gt_cold:
            cold_hits += len(pred & gt_cold)
            cold_total += len(gt_cold)
        if gt_warm:
            warm_hits += len(pred & gt_warm)
            warm_total += len(gt_warm)

    out = {
        "dataset": dataset_name,
        "recall@20_overall": overall_hits / max(overall_total, 1),
        "recall@20_cold_items": cold_hits / max(cold_total, 1),
        "recall@20_warm_items": warm_hits / max(warm_total, 1),
        "n_test_users": len(test_set),
        "n_cold_items": len(cold_items),
        "n_warm_items": len(warm_items),
        "p20_freq_threshold": float(p20),
        "p50_freq_threshold": float(p50),
    }
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=os.path.expanduser("~/LLREC_표준학/work/results/cold_start_{ds}.json"))
    a = ap.parse_args()
    res = evaluate(a.dataset)
    out_path = a.out.format(ds=a.dataset)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved: {out_path}")
