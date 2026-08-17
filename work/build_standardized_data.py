"""
Build a Standard-Aware copy of LLMRec's augmented_atttribute_embedding_dict.

For each field that has a target standard (country, language):
  - Group items by their normalized canonical code.
  - Replace each item's 1536-dim attribute embedding with the centroid
    of the embeddings of all items sharing that normalized code.
  - Items that fail to normalize (None) keep their original embedding,
    so we don't manufacture signal for invalid entries.

For non-standardized fields (director, title, year), embeddings are
copied unchanged. This isolates the *standardization* effect on
recommendation performance.

Output is written to a sibling dataset folder so main.py can be invoked
with --dataset {NEW_NAME}.
"""

import argparse
import os
import pickle
import shutil
from collections import defaultdict

import numpy as np

from normalization import normalize_country, normalize_language


def build_centroid_embeddings(field_embeddings, normalized_codes):
    """field_embeddings: {item_idx: 1536-dim ndarray}
       normalized_codes: {item_idx: code or None}
       Returns {item_idx: centroid embedding}, with un-normalizable items
       left at their original embedding."""
    grouped = defaultdict(list)
    for idx, code in normalized_codes.items():
        if code is not None and idx in field_embeddings:
            grouped[code].append(np.asarray(field_embeddings[idx], dtype=np.float32))

    centroids = {code: np.mean(np.stack(vecs), axis=0) for code, vecs in grouped.items()}

    new_embed = {}
    for idx, vec in field_embeddings.items():
        code = normalized_codes.get(idx)
        if code is not None and code in centroids:
            new_embed[idx] = centroids[code]
        else:
            new_embed[idx] = np.asarray(vec, dtype=np.float32)
    return new_embed, centroids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="~/LLMRec/data/netflix")
    p.add_argument("--dst", default="~/LLMRec/data/netflix_std")
    args = p.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    # 1. Copy/symlink everything else (CF data, image/text feats, train/val/test splits)
    for f in os.listdir(args.src):
        src_path = os.path.join(args.src, f)
        dst_path = os.path.join(args.dst, f)
        if f == "augmented_atttribute_embedding_dict":
            continue  # we will write a new one
        if os.path.exists(dst_path) or os.path.islink(dst_path):
            os.remove(dst_path) if os.path.isfile(dst_path) else shutil.rmtree(dst_path)
        os.symlink(src_path, dst_path)

    # 2. Load source attribute dict and embedding dict
    attr = pickle.load(open(os.path.join(args.src, "augmented_attribute_dict"), "rb"))
    emb = pickle.load(open(os.path.join(args.src, "augmented_atttribute_embedding_dict"), "rb"))

    # 3. Build normalized code maps
    country_codes = {idx: normalize_country(rec.get(1))[0] for idx, rec in attr.items()}
    language_codes = {idx: normalize_language(rec.get(2))[0] for idx, rec in attr.items()}

    n_country_norm = sum(1 for v in country_codes.values() if v)
    n_lang_norm = sum(1 for v in language_codes.values() if v)
    print(f"Items: {len(attr)}")
    print(f"Country normalized: {n_country_norm} ({n_country_norm/len(attr)*100:.1f}%)")
    print(f"Language normalized: {n_lang_norm} ({n_lang_norm/len(attr)*100:.1f}%)")

    # 4. Build standardized embeddings
    new_country_emb, country_centroids = build_centroid_embeddings(emb["country"], country_codes)
    new_lang_emb, lang_centroids = build_centroid_embeddings(emb["language"], language_codes)
    print(f"Country centroids: {len(country_centroids)} unique codes")
    print(f"Language centroids: {len(lang_centroids)} unique codes")

    # 5. Assemble new embedding dict (other fields unchanged)
    new_emb = {
        "year": emb["year"],
        "title": emb["title"],
        "director": emb["director"],
        "country": new_country_emb,
        "language": new_lang_emb,
    }

    out_path = os.path.join(args.dst, "augmented_atttribute_embedding_dict")
    with open(out_path, "wb") as f:
        pickle.dump(new_emb, f)
    print(f"Wrote {out_path}")

    # 6. Drop pre-built s_*adj_mat.npz / total_embed_dict to force LLMRec to rebuild
    for f in ["augmented_total_embed_dict"]:
        target = os.path.join(args.dst, f)
        if os.path.islink(target):
            os.remove(target)
            print(f"Removed symlink {target} (will be rebuilt by main.py)")

    print("\nDone. Run:")
    print(f"  python3 main.py --dataset {os.path.basename(args.dst)}")


if __name__ == "__main__":
    main()
