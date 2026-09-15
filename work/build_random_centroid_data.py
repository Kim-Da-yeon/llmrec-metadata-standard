"""
build_random_centroid_data.py
EXP-A: 표준 centroid의 대조군(랜덤 centroid) 데이터셋 생성.

build_standardized_data.py 와 '그룹 멤버십'만 다르고 나머지는 모두 동일하게 유지한다.
- 표준 정규화로 얻은 그룹의 '크기 분포'는 그대로 보존하고, 멤버십만 무작위로 섞는다.
- 정규화 실패(C_i=None) 아이템은 표준 버전과 동일하게 원본 임베딩 유지.
- director / title / year 임베딩은 표준 버전과 동일하게 불변.
→ '표준 코드의 의미성' 외 다른 변수가 모두 통제됨.

[중요 — 명세 스켈레톤과의 차이]
표준 버전(build_standardized_data.py)은 country 와 language **두 필드**를 centroid
평활화한다. std vs rand 가 공정한 ablation 이 되려면 대조군도 동일하게 두 필드를
(각 필드 자기 자신의 그룹-크기 분포에 맞춰) 랜덤 평활화해야 한다. 그래서 기본값은
--fields country,language 이다. country 만 대조하려면 --fields country 로 실행.

데이터 포맷:
  augmented_atttribute_embedding_dict = {field: {item_idx(int): 1536-d vec}}
  (스켈레톤은 (N,D) ndarray 를 가정했지만 실제 저장 포맷은 idx→vec dict 이다.)

사용법:
    python build_random_centroid_data.py --seed 0 --out data/netflix_rand_seed0
    # (--out 이 상대경로면 ~/LLMRec 기준으로 해석)
"""
import argparse
import os
import pickle
import shutil
from collections import defaultdict

import numpy as np

from normalization import normalize_country, normalize_language

LLMREC = os.path.expanduser("~/LLMRec")


def build_random_centroid_embeddings(field_embeddings, normalized_codes, rng):
    """표준 그룹의 크기 분포는 보존하고 멤버십만 무작위로 섞은 centroid 임베딩 생성.

    field_embeddings : {item_idx: 1536-d ndarray}
    normalized_codes : {item_idx: code or None}
    rng              : np.random.Generator
    반환             : ({item_idx: ndarray}, n_groups)
    """
    # 1) 표준 그룹의 크기 분포 추출 (정규화 성공 아이템만 대상)
    real_groups = defaultdict(list)
    for idx, code in normalized_codes.items():
        if code is not None and idx in field_embeddings:
            real_groups[code].append(idx)
    group_sizes = sorted((len(v) for v in real_groups.values()), reverse=True)

    # 2) 평활화 대상(=정규화 성공 아이템; 표준 버전과 동일 집합)을 섞어 동일 크기로 분할
    eligible = [idx for idx, code in normalized_codes.items()
                if code is not None and idx in field_embeddings]
    shuffled = list(eligible)
    rng.shuffle(shuffled)

    fake_groups, pos = [], 0
    for size in group_sizes:          # sum(group_sizes) == len(eligible) 보장 → 잔여 없음
        fake_groups.append(shuffled[pos:pos + size])
        pos += size

    # 3) 기본은 원본 임베딩(정규화 실패 아이템은 그대로 유지됨)
    new_embed = {idx: np.asarray(vec, dtype=np.float32)
                 for idx, vec in field_embeddings.items()}

    # 4) 각 가짜 그룹의 centroid 계산 후 그룹 전원에 할당
    for grp in fake_groups:
        if not grp:
            continue
        centroid = np.mean(
            np.stack([np.asarray(field_embeddings[i], dtype=np.float32) for i in grp]),
            axis=0,
        )
        for i in grp:
            new_embed[i] = centroid

    return new_embed, len(fake_groups)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=os.path.join(LLMREC, "data/netflix"))
    p.add_argument("--out", "--dst", dest="dst", required=True,
                   help="출력 데이터셋 폴더 (상대경로면 ~/LLMRec 기준)")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--fields", default="country,language",
                   help="랜덤 평활화할 필드(쉼표구분). 표준버전과 일치시키려면 country,language")
    args = p.parse_args()

    dst = args.dst if os.path.isabs(args.dst) else os.path.join(LLMREC, args.dst)
    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    rng = np.random.default_rng(args.seed)

    os.makedirs(dst, exist_ok=True)

    # 1. embedding dict 를 제외한 나머지 파일은 원본으로 심볼릭 링크
    for f in os.listdir(args.src):
        src_path = os.path.join(args.src, f)
        dst_path = os.path.join(dst, f)
        if f == "augmented_atttribute_embedding_dict":
            continue  # 새로 작성
        if os.path.islink(dst_path) or os.path.isfile(dst_path):
            os.remove(dst_path)
        elif os.path.isdir(dst_path):
            shutil.rmtree(dst_path)
        os.symlink(src_path, dst_path)

    # 2. 원본 속성/임베딩 로드
    attr = pickle.load(open(os.path.join(args.src, "augmented_attribute_dict"), "rb"))
    emb = pickle.load(open(os.path.join(args.src, "augmented_atttribute_embedding_dict"), "rb"))

    # 3. 정규화 코드맵 (build_standardized_data.py 와 동일 로직
    #    → 동일한 그룹-크기 분포 / 동일한 None 집합 보장)
    code_maps = {
        "country": {idx: normalize_country(rec.get(1))[0] for idx, rec in attr.items()},
        "language": {idx: normalize_language(rec.get(2))[0] for idx, rec in attr.items()},
    }

    # 4. 새 임베딩 dict: 기본은 원본 복사, 지정 필드만 랜덤 centroid 로 교체
    new_emb = dict(emb)  # year / title / director 등 불변
    for field in fields:
        if field not in code_maps:
            raise ValueError(f"Unsupported field for randomization: {field}")
        codes = code_maps[field]
        n_norm = sum(1 for v in codes.values() if v)
        rand_field, n_groups = build_random_centroid_embeddings(emb[field], codes, rng)
        new_emb[field] = rand_field
        print(f"[{field}] normalized={n_norm} ({n_norm / len(attr) * 100:.1f}%), "
              f"random groups={n_groups}  (크기분포=표준과 동일, 멤버십=랜덤 seed={args.seed})")

    out_path = os.path.join(dst, "augmented_atttribute_embedding_dict")
    with open(out_path, "wb") as f:
        pickle.dump(new_emb, f)
    print(f"Wrote {out_path}")

    # 5. total_embed_dict 캐시 제거 → main.py 가 새 임베딩으로 재생성
    cache = os.path.join(dst, "augmented_total_embed_dict")
    if os.path.islink(cache) or os.path.exists(cache):
        os.remove(cache)
        print(f"Removed {cache} (main.py 가 재생성)")

    print(f"\nDone. Run:\n  python main.py --dataset {os.path.basename(dst)} --seed {args.seed}")


if __name__ == "__main__":
    main()
