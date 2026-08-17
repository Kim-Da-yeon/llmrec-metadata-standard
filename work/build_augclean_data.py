"""
build_augclean_data.py
EXP-D: LLM이 생성한 증강 상호작용쌍(augmented_sample_dict)을 통계적으로 정제.

LLMRec은 LLM이 만든 (user -> {0:pos, 1:neg}) 가짜 학습쌍을 품질검증 없이 그대로
쓴다. 진단 결과 (1) pos의 36.9%가 이미 본 아이템(중복), (2) 소수 인기 아이템 쏠림.
→ API 없이, 기존 임베딩만으로 통계적 신뢰도를 추정해 정제한다.

정제 규칙:
  1) 중복 제거: pos가 유저의 train 이력에 이미 있으면 폐기 (증강 가치 없음)
  2) 신뢰도 필터: 신뢰도 = cos(content[pos], 유저 취향벡터),
     유저 취향벡터 = 유저가 본 아이템 content 임베딩 평균.
     남은 쌍 중 신뢰도 하위 q분위를 폐기.
  content = L2정규화(image_feat) ⊕ L2정규화(text_feat)  (멀티모달 피처, API 불필요)

폐기는 키 삭제가 아니라 pos/neg를 sentinel(n_items)로 설정 → main.py의 기존 필터
(`augmented_sample_dict[user][0] < n_items`)가 자동으로 건너뛴다. (main.py 수정 불필요)

사용법:
  python build_augclean_data.py --out data/netflix_augclean --drop_quantile 0.25
"""
import argparse
import json
import os
import pickle
import shutil

import numpy as np

LLMREC = "~/LLMRec"


def l2norm(x, axis=1, eps=1e-8):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=os.path.join(LLMREC, "data/netflix"))
    p.add_argument("--out", "--dst", dest="dst", required=True)
    p.add_argument("--drop_quantile", type=float, default=0.25,
                   help="중복 제거 후, 신뢰도 하위 q분위를 추가 폐기 (0이면 신뢰도필터 끔)")
    args = p.parse_args()
    dst = args.dst if os.path.isabs(args.dst) else os.path.join(LLMREC, args.dst)
    os.makedirs(dst, exist_ok=True)

    # 1) 나머지 파일 심볼릭 링크 (augmented_sample_dict 만 교체)
    for f in os.listdir(args.src):
        sp_, dp_ = os.path.join(args.src, f), os.path.join(dst, f)
        if f == "augmented_sample_dict":
            continue
        if os.path.islink(dp_) or os.path.isfile(dp_):
            os.remove(dp_)
        elif os.path.isdir(dp_):
            shutil.rmtree(dp_)
        os.symlink(sp_, dp_)

    # 2) 로드
    asd = pickle.load(open(os.path.join(args.src, "augmented_sample_dict"), "rb"))
    train = json.load(open(os.path.join(args.src, "train.json")))
    img = np.load(os.path.join(args.src, "image_feat.npy"))
    txt = np.load(os.path.join(args.src, "text_feat.npy"))
    n_items = img.shape[0]
    content = np.concatenate([l2norm(img), l2norm(txt)], axis=1)
    content = l2norm(content)

    # 3) 유저 취향벡터 = 본 아이템 content 평균
    taste = {}
    for u, items in train.items():
        its = [int(i) for i in items if int(i) < n_items]
        if its:
            taste[int(u)] = l2norm(content[its].mean(0, keepdims=True))[0]

    SENT = n_items  # sentinel → main.py 가 건너뜀
    cleaned = {}
    n_dup = n_lowconf = n_kept = n_total = 0
    confs = []  # 1차(중복제거 후) 신뢰도 수집 → 분위 임계 계산
    pair_conf = {}
    for u, d in asd.items():
        pos, neg = int(d[0]), int(d[1])
        n_total += 1
        hist = set(int(i) for i in train.get(str(u), []))
        if pos in hist or pos >= n_items:
            cleaned[u] = {0: SENT, 1: SENT}     # 중복 → 폐기
            n_dup += 1
            continue
        c = float(content[pos] @ taste[int(u)]) if int(u) in taste else 0.0
        pair_conf[u] = (pos, neg, c)
        confs.append(c)

    thr = np.quantile(confs, args.drop_quantile) if (confs and args.drop_quantile > 0) else -np.inf
    for u, (pos, neg, c) in pair_conf.items():
        if c < thr:
            cleaned[u] = {0: SENT, 1: SENT}     # 저신뢰 → 폐기
            n_lowconf += 1
        else:
            cleaned[u] = {0: pos, 1: neg}       # 유지
            n_kept += 1

    with open(os.path.join(dst, "augmented_sample_dict"), "wb") as f:
        pickle.dump(cleaned, f)

    print(f"증강쌍 정제 결과 (n={n_total}):")
    print(f"  중복(이미 본 pos) 폐기 : {n_dup} ({n_dup/n_total*100:.1f}%)")
    print(f"  저신뢰(하위{args.drop_quantile:.0%}) 폐기: {n_lowconf} ({n_lowconf/n_total*100:.1f}%)  [임계 cos={thr:.4f}]")
    print(f"  최종 유지            : {n_kept} ({n_kept/n_total*100:.1f}%)")
    print(f"Wrote {os.path.join(dst,'augmented_sample_dict')}")
    print(f"\nRun: python main.py --dataset {os.path.basename(dst)}")


if __name__ == "__main__":
    main()
