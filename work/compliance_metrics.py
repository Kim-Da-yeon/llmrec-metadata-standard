"""
Standard Compliance Rate (SCR) and Interoperability Score (IS).

SCR_field = (# items whose raw value is already in canonical standard form) / (# items)
SCR_overall = mean over fields

Interoperability Score (per item) is binary:
  1 if the standardized record produces a valid schema.org/Movie JSON-LD
    (all targeted fields present and resolvable to canonical codes), else 0.
  IS = mean over items.

Run as a script over the LLMRec netflix augmented_attribute_dict.
"""

import argparse
import json
import pickle
from collections import Counter

from normalization import (
    normalize_country,
    normalize_language,
    normalize_year,
    normalize_director,
)


SCHEMA_ORG_REQUIRED = ["director", "countryOfOrigin", "inLanguage", "datePublished"]


def load_attribute_dict(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_item_csv(path):
    import pandas as pd
    return pd.read_csv(path, names=["id", "year", "title"])


def compute_compliance(attr_dict, year_lookup=None):
    """Returns per-field compliance stats.

    attr_dict: {item_idx: {0: director, 1: country, 2: language}} (Netflix LLM format)
    year_lookup: optional {item_idx: year} for datePublished compliance.
    """
    n = len(attr_dict)
    compliant = Counter()
    normalized_total = Counter()
    raw_unique = {f: set() for f in ["director", "country", "language", "year"]}
    norm_unique = {f: set() for f in ["director", "country", "language", "year"]}

    interoperability_hits = 0

    for idx, rec in attr_dict.items():
        director_raw = rec.get(0)
        country_raw = rec.get(1)
        lang_raw = rec.get(2)
        year_raw = year_lookup.get(idx) if year_lookup is not None else None

        d_norm, d_comp = normalize_director(director_raw)
        c_norm, c_comp = normalize_country(country_raw)
        l_norm, l_comp = normalize_language(lang_raw)
        y_norm, y_comp = normalize_year(year_raw)

        if d_comp: compliant["director"] += 1
        if c_comp: compliant["country"] += 1
        if l_comp: compliant["language"] += 1
        if y_comp: compliant["year"] += 1

        if d_norm is not None: normalized_total["director"] += 1
        if c_norm is not None: normalized_total["country"] += 1
        if l_norm is not None: normalized_total["language"] += 1
        if y_norm is not None: normalized_total["year"] += 1

        if director_raw is not None: raw_unique["director"].add(str(director_raw))
        if country_raw is not None: raw_unique["country"].add(str(country_raw))
        if lang_raw is not None: raw_unique["language"].add(str(lang_raw))
        if year_raw is not None: raw_unique["year"].add(str(year_raw))

        if d_norm: norm_unique["director"].add(d_norm)
        if c_norm: norm_unique["country"].add(c_norm)
        if l_norm: norm_unique["language"].add(l_norm)
        if y_norm: norm_unique["year"].add(y_norm)

        # Interoperability: all 4 schema.org fields resolve
        if d_norm and c_norm and l_norm and y_norm:
            interoperability_hits += 1

    fields = ["director", "country", "language", "year"]
    rows = []
    for f in fields:
        rows.append({
            "field": f,
            "raw_compliance_rate": compliant[f] / n,
            "post_normalization_rate": normalized_total[f] / n,
            "uplift": (normalized_total[f] - compliant[f]) / n,
            "raw_unique": len(raw_unique[f]),
            "normalized_unique": len(norm_unique[f]),
            "compression_ratio": (len(raw_unique[f]) / len(norm_unique[f])) if norm_unique[f] else None,
        })

    summary = {
        "n_items": n,
        "per_field": rows,
        "scr_raw_overall": sum(r["raw_compliance_rate"] for r in rows) / len(rows),
        "scr_post_overall": sum(r["post_normalization_rate"] for r in rows) / len(rows),
        "interoperability_score_raw": 0.0,  # compute below
        "interoperability_score_post": interoperability_hits / n,
    }
    # raw IS: count items where ALL 4 fields are already compliant
    raw_io = 0
    for idx, rec in attr_dict.items():
        d_raw = normalize_director(rec.get(0))[1]
        c_raw = normalize_country(rec.get(1))[1]
        l_raw = normalize_language(rec.get(2))[1]
        y_raw = (year_lookup is not None) and normalize_year(year_lookup.get(idx))[1]
        if d_raw and c_raw and l_raw and y_raw:
            raw_io += 1
    summary["interoperability_score_raw"] = raw_io / n
    return summary


def pretty_print(summary):
    print(f"=== Standard Compliance Rate (n={summary['n_items']}) ===")
    print(f"{'field':<10}{'raw SCR':>10}{'post SCR':>12}{'uplift':>10}"
          f"{'|raw|':>8}{'|norm|':>8}{'compress':>10}")
    for r in summary["per_field"]:
        cr = f"{r['compression_ratio']:.1f}x" if r['compression_ratio'] else "n/a"
        print(f"{r['field']:<10}{r['raw_compliance_rate']:>10.3f}"
              f"{r['post_normalization_rate']:>12.3f}{r['uplift']:>10.3f}"
              f"{r['raw_unique']:>8}{r['normalized_unique']:>8}{cr:>10}")
    print()
    print(f"Overall SCR  : raw {summary['scr_raw_overall']:.3f} -> post {summary['scr_post_overall']:.3f}")
    print(f"Interop Score: raw {summary['interoperability_score_raw']:.3f} "
          f"-> post {summary['interoperability_score_post']:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--attr", default="~/LLMRec/data/netflix/augmented_attribute_dict")
    p.add_argument("--csv", default="~/LLMRec/data/netflix/item_attribute_filter.csv")
    p.add_argument("--out", default="~/LLREC_표준학/work/results/compliance_summary.json")
    args = p.parse_args()

    attr = load_attribute_dict(args.attr)
    df = load_item_csv(args.csv)
    year_lookup = {int(row["id"]): row["year"] for _, row in df.iterrows() if not (row["year"] != row["year"])}
    # The CSV id is the dataset's item index already (per LLMRec docs).
    # Build year_lookup keyed by row position, not by id, to match attr_dict keys.
    year_lookup = {i: row["year"] for i, row in df.iterrows()}

    summary = compute_compliance(attr, year_lookup)
    pretty_print(summary)

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {args.out}")
