"""
aggregate_seeds.py
EXP-B: Vanilla vs Standard-Aware 다중 seed 결과 집계 + paired t-test.
(EXP-A 에도 재사용 가능: --vanilla netflix_std --std netflix_rand)

전제: run_experiment.py 가 results/{exp}/{tag}_seed{seed}.json 으로 저장.
      json 내용 예) {"R@10":..., "R@20":..., ..., "P@20":...}

사용법:
    # EXP-B: Vanilla vs Std-Aware
    python aggregate_seeds.py --dir results/expB \
        --vanilla netflix --std netflix_std --seeds 0 1 2 3 4 5 6 7 8 9

    # EXP-A: Std-Aware(표준) vs Random-centroid
    python aggregate_seeds.py --dir results/expA \
        --vanilla netflix_std --std netflix_rand --seeds 0 1 2 3 4
"""
import argparse
import json
import os

import numpy as np

try:
    from scipy import stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

METRICS = ["R@10", "R@20", "R@50", "N@10", "N@20", "N@50", "P@20"]


def load(dirp, tag, seed):
    path = os.path.join(dirp, f"{tag}_seed{seed}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"결과 파일 없음: {path}\n  → run_experiment.py 로 해당 (dataset,seed) 먼저 실행하세요."
        )
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--vanilla", default="netflix", help="기준(베이스라인) tag")
    ap.add_argument("--std", default="netflix_std", help="비교 대상 tag")
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    args = ap.parse_args()

    van = {m: [] for m in METRICS}
    std = {m: [] for m in METRICS}
    for s in args.seeds:
        v = load(args.dir, args.vanilla, s)
        d = load(args.dir, args.std, s)
        for m in METRICS:
            if v.get(m) is None or d.get(m) is None:
                continue
            van[m].append(v[m])
            std[m].append(d[m])

    if not HAVE_SCIPY:
        print("[warn] scipy 미설치 → p-value 계산 생략 (pip install scipy)")

    print(f"\n{args.vanilla}(=Vanilla) vs {args.std}(=Std-Aware), n_seeds={len(args.seeds)}")
    print(f"{'Metric':8} {'Vanilla(mean±std)':22} {'Std-Aware(mean±std)':22} {'Δ%':>8} {'p':>8}")
    print("-" * 74)
    rows = []
    for m in METRICS:
        a = np.array(van[m], dtype=float)
        b = np.array(std[m], dtype=float)
        if len(a) == 0:
            continue
        diff = b - a
        delta_pct = 100.0 * diff.mean() / a.mean() if a.mean() else float("nan")
        if HAVE_SCIPY and len(a) > 1:
            _, p = stats.ttest_rel(b, a)
        else:
            p = float("nan")
        # paired difference 의 95% CI (정규근사)
        ci = 1.96 * diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else 0.0
        sig = "" if np.isnan(p) else (" *" if p < 0.05 else "")
        std_a = a.std(ddof=1) if len(a) > 1 else 0.0
        std_b = b.std(ddof=1) if len(b) > 1 else 0.0
        print(f"{m:8} {a.mean():.5f}±{std_a:.5f}   "
              f"{b.mean():.5f}±{std_b:.5f}   "
              f"{delta_pct:+7.2f} {p:8.4f}{sig}")
        rows.append({
            "metric": m,
            "vanilla_mean": round(float(a.mean()), 5),
            "vanilla_std": round(float(std_a), 5),
            "std_mean": round(float(b.mean()), 5),
            "std_std": round(float(std_b), 5),
            "delta_pct": round(delta_pct, 2),
            "diff_mean": round(float(diff.mean()), 6),
            "diff_95ci": round(float(ci), 6),
            "ci_excludes_zero": bool(abs(diff.mean()) > ci) if len(diff) > 1 else None,
            "p_value": None if np.isnan(p) else round(float(p), 4),
            "n_seeds": int(len(a)),
        })

    out = os.path.join(args.dir, "aggregate.json")
    with open(out, "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {out}")
    print("(* = paired t-test p<0.05 / ci_excludes_zero=True 면 95% CI 가 0 미포함)")


if __name__ == "__main__":
    main()
