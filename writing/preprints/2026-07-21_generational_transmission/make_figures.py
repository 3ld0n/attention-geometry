#!/usr/bin/env python3
"""Figures for the exp-085 generational-transmission preprint.

Reads measurement JSONs from the experiment directories and produces:
  fig_ladder.{pdf,png} — conformal head count per training corpus (the ladder)
  fig_heads.{pdf,png}  — per-head conformal populations, C-NAT s0 vs C-generated
  fig_mi.{pdf,png}     — mutual-information profiles, C-NAT vs C-generated
"""

import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
EXP085 = HERE / "../../../research/physics/experiments/exp-085_generational_transmission"
EXP062 = HERE / "../../../research/physics/experiments/exp-062_corpus_statistics"

plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

# --- Figure 1: the formation ladder ---
# (corpus, n_conformal, color, label group)
ladder = [
    ("C-SR\n(Markov)", 0, "tab:gray"),
    ("C-PL15\n($\\hat\\beta$=0.34)", 2, "tab:gray"),
    ("C-PL25\n($\\hat\\beta$=0.49)", 0, "tab:gray"),
    ("C-PL40\n($\\hat\\beta$=0.79)", 5, "tab:gray"),
    ("C-PCFG\n($\\hat\\beta$=2.97)", 0, "tab:orange"),
    ("C-generated s0/s1/s2\n($\\hat\\beta$=0.92)", None, "tab:red"),
    ("C-NAT s0/s1/s2\n($\\hat\\beta$=1.38)", None, "tab:blue"),
]
fig, ax = plt.subplots(figsize=(7.4, 4.2))
xs = np.arange(len(ladder))
for i, (name, n, color) in enumerate(ladder):
    if n is not None:
        ax.bar(i, n, color=color, width=0.62)
# C-generated three seeds as grouped narrow bars
for j, n in enumerate([7, 7, 3]):
    ax.bar(len(ladder) - 2 + (j - 1) * 0.21, n, color="tab:red", width=0.19)
# C-NAT three seeds as grouped narrow bars
for j, n in enumerate([15, 11, 13]):
    ax.bar(len(ladder) - 1 + (j - 1) * 0.21, n, color="tab:blue", width=0.19)
ax.axhline(10, color="k", lw=1.0, ls="--", label="formation criterion (10/48)")
ax.set_xticks(xs)
ax.set_xticklabels([name for name, _, _ in ladder], fontsize=8.5)
ax.set_ylabel("Conformal heads (of 48)")
handles = [
    plt.Rectangle((0, 0), 1, 1, color="tab:gray"),
    plt.Rectangle((0, 0), 1, 1, color="tab:orange"),
    plt.Rectangle((0, 0), 1, 1, color="tab:red"),
    plt.Rectangle((0, 0), 1, 1, color="tab:blue"),
    plt.Line2D([0], [0], color="k", ls="--"),
]
ax.legend(
    handles,
    ["engineered statistics (exp-062)", "hierarchical grammar (exp-084)",
     "model-generated (this work)", "natural text (exp-062)",
     "formation criterion (10/48)"],
    loc="upper left", fontsize=8.5,
)
ax.set_ylim(0, 17)
fig.tight_layout()
fig.savefig(HERE / "fig_ladder.pdf")
fig.savefig(HERE / "fig_ladder.png")

# --- Figure 2: per-head populations ---
cgen = json.loads((EXP085 / "measurements/run_Cgen_s0_final.json").read_text())
cnat = json.loads((EXP062 / "measurements/run_CNAT_s0_final.json").read_text())

fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.9), sharey=True, sharex=True)
for ax, data, title, color in (
    (axes[0], cnat, "C-NAT s0 (parent corpus): 15/48", "tab:blue"),
    (axes[1], cgen, "C-generated (child corpus): 7/48", "tab:red"),
):
    conf = [(h["layer"], h["delta"]) for h in data["heads"] if h["conformal"]]
    rest = [(h["layer"], h["delta"]) for h in data["heads"]
            if not h["conformal"] and h["delta"] is not None and 0 <= h["delta"] <= 1.2]
    ax.scatter([l for l, _ in rest], [d for _, d in rest],
               marker="x", s=28, color="0.75", label="non-conformal ($R^2<0.90$)")
    ax.scatter([l for l, _ in conf], [d for _, d in conf],
               marker="o", s=48, color=color, label="conformal head")
    ax.axhspan(0.20, 0.30, color="tab:gray", alpha=0.15)
    ax.axhline(0.25, color="k", lw=0.8, ls="--")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Layer")
    ax.set_xticks(range(6))
    ax.legend(loc="upper left", fontsize=8)
axes[0].set_ylabel(r"Fitted exponent $\Delta$")
fig.tight_layout()
fig.savefig(HERE / "fig_heads.pdf")
fig.savefig(HERE / "fig_heads.png")

# --- Figure 3: MI profiles ---
gen_mi = json.loads((EXP085 / "C-generated_s0.mi.json").read_text())
nat_mi = json.loads((EXP062 / "corpora/C-NAT_full.mi.json").read_text())

fig, ax = plt.subplots(figsize=(6.2, 4.2))
for mi, label, color in (
    (nat_mi, r"C-NAT (TinyStories), $\hat\beta = 1.38$", "tab:blue"),
    (gen_mi, r"C-generated, $\hat\beta = 0.92$", "tab:red"),
):
    prof = mi["profile"]
    ds = sorted(int(d) for d in prof)
    ys = [prof[str(d)]["mi_corr"] for d in ds]
    keep = [(d, y) for d, y in zip(ds, ys) if y > 0]
    ax.loglog([d for d, _ in keep], [y for _, y in keep], "o-", color=color,
              label=label, ms=4.5)
ax.set_xlabel("Token distance $d$")
ax.set_ylabel("Shuffle-corrected mutual information (nats)")
ax.legend(loc="lower left", fontsize=9)
fig.tight_layout()
fig.savefig(HERE / "fig_mi.pdf")
fig.savefig(HERE / "fig_mi.png")

print("figures written")
