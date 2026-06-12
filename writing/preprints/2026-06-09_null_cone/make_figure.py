"""
Figure for "Attention on the Null Cone" — exp-056 direct test of the log-distance prediction.

Scatter plot: Δ_score (pre-softmax slope) vs Δ_pos (post-softmax conformal dimension),
for the 44 conformal GPT-2 heads. Shows ρ = +0.976 — the null-ray prediction confirmed.

Save to: writing/preprints/2026-06-09_null_cone/fig_log_distance.pdf
         writing/preprints/2026-06-09_null_cone/fig_log_distance.png (for Zenodo thumbnail)
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# ── data ────────────────────────────────────────────────────────────────────
root = Path(__file__).parent.parent.parent.parent  # repo root
results = json.load(open(root / "research/physics/experiments/exp-056_qk_log_distance/results.json"))
per_head = results["per_head"]

# conformal heads (R² > 0.90 in post-softmax fit, from exp-046 flag)
conf = [h for h in per_head if h["conformal"]]
syk  = [h for h in conf if h["syk_near"]]

dp   = np.array([h["delta_pos"]   for h in conf])
ds   = np.array([h["delta_score"] for h in conf])
is_syk = np.array([h["syk_near"] for h in conf])

dp_syk = np.array([h["delta_pos"]   for h in syk])
ds_syk = np.array([h["delta_score"] for h in syk])

# Spearman rank correlation
rho, pval = stats.spearmanr(dp, ds)

# linear regression for the trend line
slope, intercept, r_lin, p_lin, se = stats.linregress(dp, ds)
x_line = np.linspace(dp.min() - 0.01, dp.max() + 0.01, 200)
y_line = slope * x_line + intercept

# ── figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4.2, 3.8))

# all conformal heads: light grey
ax.scatter(dp[~is_syk], ds[~is_syk],
           color="#aaaaaa", s=22, zorder=3, label="conformal (non-SYK-near)")
# SYK-near heads: dark markers
ax.scatter(dp_syk, ds_syk,
           color="#1a1a2e", s=30, marker="D", zorder=4, label=r"SYK-near ($|\Delta - 1/4| \leq 0.05$)")

# regression line
ax.plot(x_line, y_line, color="#c0392b", lw=1.4, zorder=2, label=f"OLS: $\\Delta_{{\\rm score}} = {slope:.2f}\\Delta_{{\\rm pos}} + {intercept:.2f}$")

# identity reference (Δ_score = Δ_pos) — dashed
xy = np.linspace(0, 0.65, 100)
ax.plot(xy, xy, "--", color="#777777", lw=0.9, zorder=1, alpha=0.6, label=r"$\Delta_{\rm score} = \Delta_{\rm pos}$")

# SYK reference lines
ax.axvline(0.25, color="#2980b9", lw=0.8, ls=":", alpha=0.5)
ax.axhline(0.25, color="#2980b9", lw=0.8, ls=":", alpha=0.5)

# annotation
ax.text(0.05, 0.96,
        f"Spearman $\\rho = +{rho:.3f}$\n$p = 2 \\times 10^{{-29}}$, $n = {len(conf)}$",
        transform=ax.transAxes, va="top", ha="left",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9))

ax.set_xlabel(r"$\Delta_{\rm pos}$ (post-softmax, exp-046)", fontsize=9)
ax.set_ylabel(r"$\Delta_{\rm score}$ (pre-softmax slope, exp-056)", fontsize=9)
ax.set_title("Log-distance prediction: pre-softmax slope\nrank-tracks post-softmax conformal dimension", fontsize=9)
ax.legend(fontsize=7.5, loc="lower right")

ax.set_xlim(0.08, 0.62)
ax.set_ylim(0.08, 0.82)
ax.tick_params(labelsize=8)

fig.tight_layout()

out_dir = Path(__file__).parent
fig.savefig(out_dir / "fig_log_distance.pdf", dpi=300, bbox_inches="tight")
fig.savefig(out_dir / "fig_log_distance.png", dpi=200, bbox_inches="tight")
print("Saved fig_log_distance.pdf and fig_log_distance.png")
print(f"N conformal heads: {len(conf)} | N SYK-near: {len(syk)} | Spearman ρ = {rho:.4f}")
