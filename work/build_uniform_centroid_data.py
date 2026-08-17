"""
build_uniform_centroid_data.py
EXP-A 보강: 균등 크기 랜덤 centroid 대조군 데이터셋 생성.

목적
----
기존 EXP-A(표준 vs 크기분포-보존 랜덤)에서 둘이 거의 동일했던 이유가
head-dominance(US/English 거대 그룹) 때문이라는 '해석'을 '입증'한다.
크기분포를 보존하지 않고 모든 그룹을 '균등 크기'로 무작위 분할한 대조군도
표준/기존랜덤과 차이가 없으면 → 그룹 구성 방식과 무관하게 centroid 효과가
동일하다는 것이 증명됨.

build_random_centroid_data.py 와 모든 처리(데이터 포맷, country+language 두 필드,
정규화 실패 아이템 원본 유지, director/title/year 불변, 심볼릭 링크 스캐폴딩)가
동일하고, 오직 '그룹화 방식'만 다르다:
  - random : 표준 그룹 크기분포 보존, 멤버십만 무작위
  - uniform: 크기분포 무시, 전부 group_size 단위로 균등 분할

데이터 포맷:
  augmented_atttribute_embedding_dict = {field: {item_idx(int): 1536-d vec}}

사용법:
    python build_uniform_centroid_data.py --seed 0 --group-size 30 --out data/netflix_unif_gs30_seed0
"""
import argparse
import os
import pickle
import shutil

import numpy as np

from normalization import normalize_country, normalize_language

LLMREC = "~/LLMRec"


def build_uniform_centroid_embeddings(field_embeddings, normalized_codes, rng, group_size):
    """정규화 성공 아이템을 섞어 group_size 단위로 균등 분할 후 centroid 할당.

    field_embeddings : {item_idx: 1536-d ndarray}
    normalized_codes : {item_idx: code or None}
    rng              : np.random.Generator
    group_size       : 균등 그룹 1개의 크기
    반환             : ({item_idx: ndarray}, n_groups)
    """
    # 평활화 대상 = 정규화 성공 아이템 (표준/기존랜덤과 동일 집합)
    eligible = [idx for idx, code in normalized_codes.items()
                if code is not None and idx in field_embeddings]
    shuffled = list(eligible)
    rng.shuffle(shuffled)

    # 균등 분할 (마지막 그룹만 잔여로 group_size 미만일 수 있음)
    groups = [shuffled[i:i + group_size] for i in range(0, len(shuffled), group_size)]

    # 기본은 원본(정규화 실패 아이템은 그대로 유지됨)
    new_embed = {idx: np.asarray(vec, dtype=np.float32)
                 for idx, vec in field_embeddings.items()}

    for grp in groups:
        if not grp:
            continue
        centroid = np.mean(
            np.stack([np.asarray(field_embeddings[i], dtype=np.float32) for i in grp]),
            axis=0,
        )
        for i in grp:
            new_embed[i] = centroid

    return new_embed, len(groups)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=os.path.join(LLMREC, "data/netflix"))
    p.add_argument("--out", "--dst", dest="dst", required=True,
                   help="출력 데이터셋 폴더 (상대경로면 ~/LLMRec 기준)")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--group-size", type=int, default=30,
                   help="균등 그룹 1개의 크기 (표준 그룹 평균 크기와 비슷한 30 권장)")
    p.add_argument("--fields", default="country,language",
                   help="평활화할 필드(쉼표구분). 표준/기존랜덤과 일치시키려면 country,language")
    args = p.parse_args()

    dst = args.dst if os.path.isabs(args.dst) else os.path.join(LLMREC, args.dst)
    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    rng = np.random.default_rng(args.seed)

    os.makedirs(dst, exist_ok=True)

    # 1. embedding dict 제외 나머지 파일 심볼릭 링크
    for f in os.listdir(args.src):
        src_path = os.path.join(args.src, f)
        dst_path = os.path.join(dst, f)
        if f == "augmented_atttribute_embedding_dict":
            continue
        if os.path.islink(dst_path) or os.path.isfile(dst_path):
            os.remove(dst_path)
        elif os.path.isdir(dst_path):
            shutil.rmtree(dst_path)
        os.symlink(src_path, dst_path)

    # 2. 로드
    attr = pickle.load(open(os.path.join(args.src, "augmented_attribute_dict"), "rb"))
    emb = pickle.load(open(os.path.join(args.src, "augmented_atttribute_embedding_dict"), "rb"))

    # 3. 정규화 코드맵 (성공/실패 집합을 표준·기존랜덤과 동일하게 유지)
    code_maps = {
        "country": {idx: normalize_country(rec.get(1))[0] for idx, rec in attr.items()},
        "language": {idx: normalize_language(rec.get(2))[0] for idx, rec in attr.items()},
    }

    # 4. 새 임베딩: 기본 원본 복사, 지정 필드만 균등 랜덤 centroid 로 교체
    new_emb = dict(emb)  # director/title/year 불변
    for field in fields:
        if field not in code_maps:
            raise ValueError(f"Unsupported field: {field}")
        codes = code_maps[field]
        n_norm = sum(1 for v in codes.values() if v)
        uni_field, n_groups = build_uniform_centroid_embeddings(
            emb[field], codes, rng, args.group_size)
        new_emb[field] = uni_field
        print(f"[{field}] eligible={n_norm} ({n_norm / len(attr) * 100:.1f}%), "
              f"uniform groups={n_groups} (size={args.group_size}, seed={args.seed})")

    out_path = os.path.join(dst, "augmented_atttribute_embedding_dict")
    with open(out_path, "wb") as f:
        pickle.dump(new_emb, f)
    print(f"Wrote {out_path}")

    # 5. total_embed_dict 캐시 제거 → main.py 가 재생성
    cache = os.path.join(dst, "augmented_total_embed_dict")
    if os.path.islink(cache) or os.path.exists(cache):
        os.remove(cache)
        print(f"Removed {cache} (main.py 가 재생성)")

    print(f"\nDone. Run:\n  python main.py --dataset {os.path.basename(dst)} --seed {args.seed}")


if __name__ == "__main__":
    main()
