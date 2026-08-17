"""
run_experiment.py
하나의 (dataset, seed) 학습을 돌리고 best-epoch 기준 7개 지표를 JSON 으로 저장한다.
EXP-A / EXP-B / EXP-C-1 공용 러너. (main.py 가 JSON 을 저장하지 않으므로 이 래퍼가 담당)

  # EXP-B (다중 seed)
  python run_experiment.py --dataset netflix     --seed 0 --exp expB
  python run_experiment.py --dataset netflix_std --seed 0 --exp expB

  # EXP-C-1 (per-K: 5,10,20,50)
  python run_experiment.py --dataset netflix     --seed 0 --exp expC1 --ks '[5,10,20,50]'

  # EXP-A (랜덤 centroid 대조군). tag 를 고정하면 aggregate_seeds 로 std vs rand 비교 가능
  python run_experiment.py --dataset netflix_rand_seed0 --seed 0 --exp expA --tag netflix_rand
  python run_experiment.py --dataset netflix_std        --seed 0 --exp expA --tag netflix_std

저장 경로: work/results/{exp}/{tag}_seed{seed}.json   (tag 기본값 = dataset)
저장 내용: aggregate_seeds.py 가 읽는 R@10/20/50, N@10/20/50, P@20 + 메타데이터.

best-epoch 선택 기준은 main.py 가 R@20 으로 고정(Ks 순서/길이 무관) → C-1 도 동일 기준.
"""
import argparse
import json
import os
import sys

LLMREC = "~/LLMRec"
WORK = os.path.dirname(os.path.abspath(__file__))  # chdir 전에 계산

# 명세 고정 하이퍼파라미터
FIXED = [
    "--lr", "0.0001",
    "--batch_size", "1024",
    "--embed_size", "64",
    "--prune_loss_drop_rate", "0.71",
    "--aug_sample_rate", "0.1",
    "--early_stopping_patience", "7",
]


def extract_metrics(ret, Ks):
    """ret['recall'/'ndcg'/'precision'] (각 len=len(Ks)) → 명세 7개 지표 dict."""
    def at(metric, k):
        return float(ret[metric][Ks.index(k)]) if k in Ks else None
    return {
        "R@10": at("recall", 10), "R@20": at("recall", 20), "R@50": at("recall", 50),
        "N@10": at("ndcg", 10),  "N@20": at("ndcg", 20),  "N@50": at("ndcg", 50),
        "P@20": at("precision", 20),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--exp", required=True, help="결과 하위폴더 (expA/expB/expC1)")
    ap.add_argument("--tag", default=None, help="출력 파일명 tag (기본=dataset)")
    ap.add_argument("--ks", default="[10,20,50]")
    ap.add_argument("--epoch", type=int, default=200)
    ap.add_argument("--aug_mf_rate", default=None, help="증강쌍 결합 가중치 override (기본 0.012)")
    ap.add_argument("--aug_sample_rate", default=None, help="증강 샘플 비율 override (기본 0.1)")
    ap.add_argument("--set", default=None, dest="overrides",
                    help="임의 HP override, 'k=v,k=v' 형식 (예: model_cat_rate=0.2,layers=2)")
    args = ap.parse_args()
    tag = args.tag or args.dataset

    sys.path.insert(0, LLMREC)
    os.chdir(LLMREC)
    sys.argv = ["main.py", "--dataset", args.dataset, "--seed", str(args.seed),
                "--Ks", args.ks, "--epoch", str(args.epoch), "--verbose", "5"] + FIXED
    if args.aug_mf_rate is not None:
        sys.argv += ["--aug_mf_rate", str(args.aug_mf_rate)]
    if args.aug_sample_rate is not None:
        sys.argv += ["--aug_sample_rate", str(args.aug_sample_rate)]
    if args.overrides:
        for kv in args.overrides.split(","):
            k, v = kv.split("=")
            sys.argv += ["--" + k.strip(), v.strip()]

    # cold_start_eval.py 와 동일한 재로딩 순서 (batch_test 가 import 시점에 args 로
    # data_generator 를 만들기 때문에 sys.argv 설정 후 import 해야 함).
    import importlib
    import utility.parser as p_mod; importlib.reload(p_mod)
    import utility.batch_test as bt_mod; importlib.reload(bt_mod)
    import utility.load_data as ld_mod; importlib.reload(ld_mod)
    import Models as m_mod; importlib.reload(m_mod)
    import main as main_mod; importlib.reload(main_mod)

    Ks = eval(args.ks)
    main_mod.set_seed(args.seed)
    dg = bt_mod.data_generator
    cfg = {"n_users": dg.n_users, "n_items": dg.n_items}

    trainer = main_mod.Trainer(data_config=cfg)
    trainer.train()

    ret = trainer.best_test_ret
    if ret is None:
        raise RuntimeError("best_test_ret is None — 학습이 평가 결과를 만들지 못했습니다.")
    metrics = extract_metrics(ret, Ks)

    record = {
        "dataset": args.dataset,
        "tag": tag,
        "seed": args.seed,
        "exp": args.exp,
        "Ks": Ks,
        "best_epoch": int(trainer.best_epoch),
        **metrics,
        "full": {
            "recall": [float(x) for x in ret["recall"]],
            "ndcg": [float(x) for x in ret["ndcg"]],
            "precision": [float(x) for x in ret["precision"]],
            "hit_ratio": [float(x) for x in ret["hit_ratio"]],
        },
        "hparams": {
            "lr": 1e-4, "batch_size": 1024, "embed_dim": 64,
            "prune_loss_drop_rate": 0.71, "aug_sample_rate": 0.1,
            "early_stopping_patience": 7, "epoch": args.epoch,
        },
    }

    out_dir = os.path.join(WORK, "results", args.exp)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{tag}_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    print(f"\n[saved] {out_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
