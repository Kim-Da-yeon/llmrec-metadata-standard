"""
build_composite_centroid_data.py
"표준화를 더" : 단일필드(country) 대신 다필드 '조합 표준코드'로 더 잘게 표준화.

동기: 단일 country 표준화는 US 거대그룹(69%)에 지배되어 효과가 소멸했다(head-dominance).
조합 코드(country×decade, country×language×decade)는 최대그룹을 ~23%로 쪼개 쏠림을
완화 → 표준의 의미적 그룹화가 살아날 여지를 만든다.

build_standardized_data.py 와 동일하되 그룹 정의만 '조합 코드'로 바꾼다.
country/language 임베딩을 조합그룹 centroid 로 치환, 조합코드 불완전(any None) 아이템은
원본 유지. director/title/year 불변.

사용법:
  python build_composite_centroid_data.py --mode cd  --out data/netflix_comp_cd
  python build_composite_centroid_data.py --mode cld --out data/netflix_comp_cld
"""
import argparse, os, pickle, shutil
from collections import defaultdict
import numpy as np
import pandas as pd
from normalization import normalize_country, normalize_language, normalize_year

LLMREC = "~/LLMRec"


def composite_codes(attr, year, mode):
    codes = {}
    for i, r in attr.items():
        c = normalize_country(r.get(1))[0]
        l = normalize_language(r.get(2))[0]
        y = normalize_year(year.get(i))[0]
        dec = (int(y) // 10 * 10) if y else None
        if mode == "cd":
            key = (c, dec) if (c and dec) else None
        elif mode == "cld":
            key = (c, l, dec) if (c and l and dec) else None
        elif mode == "cl":
            key = (c, l) if (c and l) else None
        else:
            raise ValueError(mode)
        codes[i] = key
    return codes


def centroid_by_code(field_emb, codes):
    grouped = defaultdict(list)
    for idx, code in codes.items():
        if code is not None and idx in field_emb:
            grouped[code].append(idx)
    new = {idx: np.asarray(v, np.float32) for idx, v in field_emb.items()}
    n_grp = 0
    for code, mem in grouped.items():
        cen = np.mean(np.stack([np.asarray(field_emb[i], np.float32) for i in mem]), axis=0)
        for i in mem:
            new[i] = cen
        n_grp += 1
    return new, n_grp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=os.path.join(LLMREC, "data/netflix"))
    p.add_argument("--out", "--dst", dest="dst", required=True)
    p.add_argument("--mode", choices=["cd", "cld", "cl"], default="cd")
    p.add_argument("--fields", default="country,language")
    args = p.parse_args()
    dst = args.dst if os.path.isabs(args.dst) else os.path.join(LLMREC, args.dst)
    fields = [f.strip() for f in args.fields.split(",")]
    os.makedirs(dst, exist_ok=True)

    for f in os.listdir(args.src):
        sp_, dp_ = os.path.join(args.src, f), os.path.join(dst, f)
        if f == "augmented_atttribute_embedding_dict":
            continue
        if os.path.islink(dp_) or os.path.isfile(dp_):
            os.remove(dp_)
        elif os.path.isdir(dp_):
            shutil.rmtree(dp_)
        os.symlink(sp_, dp_)

    attr = pickle.load(open(os.path.join(args.src, "augmented_attribute_dict"), "rb"))
    emb = pickle.load(open(os.path.join(args.src, "augmented_atttribute_embedding_dict"), "rb"))
    df = pd.read_csv(os.path.join(args.src, "item_attribute_filter.csv"), names=["id", "year", "title"])
    year = {i: (df.iloc[i]["year"] if i < len(df) else None) for i in range(len(attr))}

    codes = composite_codes(attr, year, args.mode)
    n_complete = sum(1 for v in codes.values() if v is not None)
    print(f"[mode={args.mode}] 조합코드 완전 아이템: {n_complete}/{len(attr)} ({n_complete/len(attr)*100:.1f}%)")

    new_emb = dict(emb)
    for field in fields:
        ne, ng = centroid_by_code(emb[field], codes)
        new_emb[field] = ne
        print(f"  [{field}] 조합그룹 {ng}개로 centroid 치환")

    with open(os.path.join(dst, "augmented_atttribute_embedding_dict"), "wb") as f:
        pickle.dump(new_emb, f)
    cache = os.path.join(dst, "augmented_total_embed_dict")
    if os.path.islink(cache) or os.path.exists(cache):
        os.remove(cache)
    print(f"Wrote {dst}")


if __name__ == "__main__":
    main()
