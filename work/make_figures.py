"""Generate custom figures for the term paper."""
import os
import pickle
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.patches as mpatches
import numpy as np

# Korean font
for fp in [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
]:
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.expanduser("~/LLREC_표준학/images")
os.makedirs(OUT, exist_ok=True)

# ============ Load data for Figure A ============
import sys
sys.path.insert(0, os.path.expanduser("~/LLREC_표준학/work"))
from normalization import normalize_country

attr = pickle.load(open(os.path.expanduser("~/LLMRec/data/netflix/augmented_attribute_dict"), "rb"))
raw_countries = [v[1] for v in attr.values()]
norm_countries = [normalize_country(c)[0] or "(invalid)" for c in raw_countries]

raw_top = Counter(raw_countries).most_common(8)
norm_top = Counter(norm_countries).most_common(8)


# ============ Figure A: country token before/after ============
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

raw_labels = [r[0] for r in raw_top]
raw_counts = [r[1] for r in raw_top]
colors_raw = ["#d62728" if l == "country" else "#1f77b4" for l in raw_labels]
ax1.barh(range(len(raw_labels)), raw_counts, color=colors_raw)
ax1.set_yticks(range(len(raw_labels)))
ax1.set_yticklabels([f"'{l}'" for l in raw_labels])
ax1.invert_yaxis()
ax1.set_xlabel("아이템 수 (count)")
ax1.set_title(f"정규화 전 (Raw)\n총 {len(set(raw_countries))}개 고유 토큰", fontsize=12)
for i, c in enumerate(raw_counts):
    ax1.text(c + 50, i, f" {c:,}", va="center", fontsize=10)

norm_labels = [n[0] for n in norm_top]
norm_counts = [n[1] for n in norm_top]
colors_norm = ["#2ca02c" if l != "(invalid)" else "#7f7f7f" for l in norm_labels]
ax2.barh(range(len(norm_labels)), norm_counts, color=colors_norm)
ax2.set_yticks(range(len(norm_labels)))
ax2.set_yticklabels([f"'{l}'" for l in norm_labels])
ax2.invert_yaxis()
ax2.set_xlabel("아이템 수 (count)")
ax2.set_title(f"정규화 후 (ISO 3166-1 alpha-2)\n총 57개 ISO 코드 (20.5× 압축)", fontsize=12)
for i, c in enumerate(norm_counts):
    ax2.text(c + 50, i, f" {c:,}", va="center", fontsize=10)

# add legend for the red bar in raw
red_patch = mpatches.Patch(color="#d62728", label="LLM 환각 (placeholder)")
ax1.legend(handles=[red_patch], loc="lower right", fontsize=9)

fig.suptitle("Figure A. Country 필드 정규화 전/후 상위 8개 토큰 분포 (n=17,366)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/figure_A_country_tokens.png", dpi=140, bbox_inches="tight")
plt.close()
print(f"Saved {OUT}/figure_A_country_tokens.png")


# ============ Figure B: pipeline comparison ============
fig, ax = plt.subplots(figsize=(13, 5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis("off")

def box(x, y, w, h, text, color="#cce5ff", fontsize=10, edge="#333"):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                                   linewidth=1.5, edgecolor=edge, facecolor=color)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize)

def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.6, color="#333"))

# Vanilla pipeline (top row, y=5.5)
ax.text(0.2, 7.2, "Vanilla LLMRec (논문)", fontsize=12, fontweight="bold", color="#666")
box(0.5, 5.5, 2.0, 1.2, "GPT-3.5\n자유 텍스트\n프롬프트", color="#fff2cc")
box(3.0, 5.5, 2.5, 1.2, "free text 출력\n'USA', 'United States'\n'country' (환각)...", color="#ffe6e6")
box(6.0, 5.5, 2.5, 1.2, "ada-002 임베딩\n(1,166개 분산점)", color="#ffe6e6")
box(9.0, 5.5, 2.5, 1.2, "LLMRec\n(LightGCN)", color="#cce5ff")
box(12.0, 5.5, 1.7, 1.2, "추천\n출력", color="#d4edda")
arrow(2.5, 6.1, 3.0, 6.1)
arrow(5.5, 6.1, 6.0, 6.1)
arrow(8.5, 6.1, 9.0, 6.1)
arrow(11.5, 6.1, 12.0, 6.1)

# Standard-Aware pipeline (bottom row, y=1.5)
ax.text(0.2, 3.2, "Standard-Aware LLMRec (본 연구)", fontsize=12, fontweight="bold", color="#1a5490")
box(0.5, 1.5, 2.0, 1.2, "GPT-3.5\n자유 텍스트\n프롬프트", color="#fff2cc")
box(3.0, 1.5, 2.5, 1.2, "free text 출력\n(동일)", color="#ffe6e6")
box(6.0, 1.5, 2.5, 1.2, "정규화 모듈\nISO 3166 / BCP 47\npycountry+aliases\n환각 거부", color="#d4edda", edge="#1a5490")
box(9.0, 1.5, 2.5, 1.2, "Centroid 임베딩\n(57개 점)", color="#d4edda", edge="#1a5490")
box(12.0, 1.5, 1.7, 1.2, "LLMRec\n(LightGCN)\n→추천", color="#cce5ff")
arrow(2.5, 2.1, 3.0, 2.1)
arrow(5.5, 2.1, 6.0, 2.1)
arrow(8.5, 2.1, 9.0, 2.1)
arrow(11.5, 2.1, 12.0, 2.1)

# Highlight delta
ax.annotate("본 연구 추가", xy=(7.25, 1.45), xytext=(7.25, 0.4),
            ha="center", fontsize=11, color="#1a5490", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#1a5490", lw=1.4))

ax.text(7, 7.7, "Figure B. Vanilla vs Standard-Aware 파이프라인 비교",
        ha="center", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/figure_B_pipeline.png", dpi=140, bbox_inches="tight")
plt.close()
print(f"Saved {OUT}/figure_B_pipeline.png")


# ============ Figure D: 7 metrics comparison ============
metrics = ["R@10", "R@20", "R@50", "N@10", "N@20", "N@50", "P@20"]
vanilla = [0.05149, 0.08130, 0.13713, 0.02596, 0.03350, 0.04454, 0.00407]
std     = [0.05257, 0.08347, 0.13659, 0.02725, 0.03497, 0.04543, 0.00417]
deltas = [(s-v)/v*100 for s, v in zip(std, vanilla)]

x = np.arange(len(metrics))
w = 0.36
fig, ax = plt.subplots(figsize=(11, 5.5))

# normalize each metric to its vanilla value=1 for visual comparison
v_norm = [1.0]*len(metrics)
s_norm = [s/v for s, v in zip(std, vanilla)]
b1 = ax.bar(x - w/2, v_norm, w, label="Vanilla LLMRec (재현)", color="#7f7f7f", edgecolor="#333")
b2 = ax.bar(x + w/2, s_norm, w, label="Standard-Aware (본 연구)", color="#2ca02c", edgecolor="#1a5490")

ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylabel("정규화 값 (Vanilla = 1.0)")
ax.axhline(1.0, color="#333", lw=0.6, ls="--", alpha=0.5)
ax.set_ylim(0.96, 1.075)
ax.set_title("Figure D. Standard-Aware의 7개 추천 지표 개선 (Vanilla 대비 상대값)",
             fontsize=13, fontweight="bold")

# annotate Δ%
for xi, di in zip(x, deltas):
    color = "#1a5490" if di > 0 else "#d62728"
    ax.text(xi + w/2, s_norm[list(x).index(xi)] + 0.002, f"{di:+.2f}%",
            ha="center", fontsize=9.5, color=color, fontweight="bold")

# annotate raw values inside bars
for xi, v_, s_ in zip(x, vanilla, std):
    ax.text(xi - w/2, 0.965, f"{v_:.4f}", ha="center", fontsize=8, color="white")
    ax.text(xi + w/2, 0.965, f"{s_:.4f}", ha="center", fontsize=8, color="white")

ax.legend(loc="upper left", fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/figure_D_metrics.png", dpi=140, bbox_inches="tight")
plt.close()
print(f"Saved {OUT}/figure_D_metrics.png")


# ============ Figure E: SCR & IS ============
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# SCR per field
fields = ["director", "country", "language", "year (datePublished)"]
raw_scr = [0.936, 0.000, 0.000, 0.999]
post_scr = [0.936, 0.858, 0.866, 0.999]
x = np.arange(len(fields))
w = 0.36
ax1.bar(x - w/2, raw_scr, w, label="Raw (LLMRec 원본)", color="#7f7f7f", edgecolor="#333")
ax1.bar(x + w/2, post_scr, w, label="Standard-Aware (정규화 후)", color="#2ca02c", edgecolor="#1a5490")
for xi, v_ in zip(x, raw_scr):
    ax1.text(xi - w/2, v_ + 0.02, f"{v_:.3f}", ha="center", fontsize=9)
for xi, v_ in zip(x, post_scr):
    ax1.text(xi + w/2, v_ + 0.02, f"{v_:.3f}", ha="center", fontsize=9, color="#1a5490", fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(fields, rotation=15, ha="right", fontsize=10)
ax1.set_ylabel("Standard Compliance Rate")
ax1.set_ylim(0, 1.1)
ax1.set_title("(a) 필드별 SCR — country/language\n원본은 0% 표준 준수", fontsize=12)
ax1.legend(loc="upper center", fontsize=10)
ax1.grid(axis="y", alpha=0.3)

# Overall SCR & IS
labels = ["Overall SCR", "Interoperability\nScore (IS)"]
raw_o = [0.484, 0.000]
post_o = [0.914, 0.803]
x2 = np.arange(len(labels))
ax2.bar(x2 - w/2, raw_o, w, label="Raw", color="#7f7f7f", edgecolor="#333")
ax2.bar(x2 + w/2, post_o, w, label="Standard-Aware", color="#2ca02c", edgecolor="#1a5490")
for xi, v_ in zip(x2, raw_o):
    ax2.text(xi - w/2, v_ + 0.02, f"{v_:.3f}", ha="center", fontsize=10)
for xi, v_ in zip(x2, post_o):
    ax2.text(xi + w/2, v_ + 0.02, f"{v_:.3f}", ha="center", fontsize=10, color="#1a5490", fontweight="bold")
# arrows showing uplift
ax2.annotate("", xy=(0+w/2, post_o[0]-0.05), xytext=(0-w/2, raw_o[0]+0.05),
             arrowprops=dict(arrowstyle="->", color="#1a5490", lw=2))
ax2.text(0, 0.7, "+43%p", ha="center", color="#1a5490", fontweight="bold", fontsize=11)
ax2.annotate("", xy=(1+w/2, post_o[1]-0.05), xytext=(1-w/2, raw_o[1]+0.05),
             arrowprops=dict(arrowstyle="->", color="#1a5490", lw=2))
ax2.text(1, 0.4, "+80%p", ha="center", color="#1a5490", fontweight="bold", fontsize=11)
ax2.set_xticks(x2)
ax2.set_xticklabels(labels, fontsize=11)
ax2.set_ylabel("점수")
ax2.set_ylim(0, 1.1)
ax2.set_title("(b) 종합 SCR + Interoperability Score", fontsize=12)
ax2.legend(loc="upper left", fontsize=10)
ax2.grid(axis="y", alpha=0.3)

fig.suptitle("Figure E. 표준 적합성·상호운용성 정량화 (n=17,366)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/figure_E_compliance.png", dpi=140, bbox_inches="tight")
plt.close()
print(f"Saved {OUT}/figure_E_compliance.png")

print("\nAll figures generated.")
