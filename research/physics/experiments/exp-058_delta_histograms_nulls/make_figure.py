"""exp-058 figure: per-model Δ histograms (conformal colored), nulls overlaid, ref line 0.25."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent
d = json.loads((OUT_DIR / "results.json").read_text())

BINS = np.arange(-0.25, 1.5001, 0.025)
R2 = 0.90


def deltas(records, conformal=None):
    out = []
    for r in records:
        if r["delta"] is None:
            continue
        if conformal is True and (r["R2"] is None or r["R2"] < R2):
            continue
        if conformal is False and (r["R2"] is not None and r["R2"] >= R2):
            continue
        out.append(r["delta"])
    return np.clip(np.array(out), BINS[0], BINS[-1])


panels = []
for arm, title in [("exp007", "GPT-2 (exp-007 protocol)"), ("census", "GPT-2 (census protocol)")]:
    a = d["gpt2_arms"][arm]
    panels.append({
        "title": title,
        "records": a["observed"]["records"],
        "null_rand": a["null_A_randomized"]["records_pooled"],
        "summary": a["observed"]["summary"],
    })
for name in ["GPT-2-medium", "GALA-7B-softmax", "OLMo-7B", "Mistral-7B-v0.3", "Pythia-410m-fp32"]:
    m = d["archived_models"][name]
    panels.append({"title": f"{name} ({m['experiment']})", "records": m["records"],
                   "null_rand": None, "summary": m["summary"]})

fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
axes = axes.ravel()

for ax, p in zip(axes, panels):
    d_all = deltas(p["records"])
    d_conf = deltas(p["records"], conformal=True)
    ax.hist(d_all, bins=BINS, color="0.8", label="all heads (no R² filter)")
    ax.hist(d_conf, bins=BINS, color="#1f77b4", alpha=0.9, label=f"conformal (R² ≥ {R2})")
    if p["null_rand"] is not None:
        d_null = deltas(p["null_rand"])
        # scale pooled null (N_RAND_SEEDS × 144) to one model's head count
        w = np.full(len(d_null), len(p["records"]) / max(len(d_null), 1))
        ax.hist(d_null, bins=BINS, weights=w, histtype="step", color="crimson",
                lw=1.5, label="randomized-weight null (scaled)")
    ax.axvline(0.25, color="k", ls="--", lw=1, label="Δ = 0.25")
    ax.axvspan(0.20, 0.30, color="gold", alpha=0.15)
    s = p["summary"]
    cond = s["conditional_fraction_window_given_R2"]
    ax.set_title(f"{p['title']}\nconf {s['n_conformal_R2_90']}/{s['n_heads']}, "
                 f"P(win|R²) = {cond:.2f}" if cond is not None else p["title"], fontsize=10)
    ax.set_xlim(BINS[0], BINS[-1])

# shuffled-null info box in last axis
ax = axes[7]
ax.axis("off")
lines = ["Joint mass P(Δ∈[0.20,0.30] ∧ R²≥0.90):", ""]
for arm in ["exp007", "census"]:
    nb = d["gpt2_arms"][arm]["null_B_shuffled"]
    na = d["gpt2_arms"][arm]["null_A_randomized"]["per_seed_joint_mass"]
    lines += [f"GPT-2 {arm}:",
              f"  observed = {nb['observed_joint_mass']:.4f}",
              f"  shuffled-profile null = {nb['null_joint_mass_mean']:.4f} "
              f"(max {nb['null_joint_mass_max']:.4f}, {nb['n_replicates']} reps)",
              f"  permutation p = {nb['permutation_p']:.4f}",
              f"  randomized-weight null (5 seeds) = {na}", ""]
for name in ["GPT-2-medium", "GALA-7B-softmax", "OLMo-7B", "Mistral-7B-v0.3", "Pythia-410m-fp32"]:
    s = d["archived_models"][name]["summary"]
    lines.append(f"{name}: joint = {s['joint_mass_window_and_R2']:.4f}, "
                 f"SYK-near {s['n_syk_near']} (med {s['syk_near_median']:.3f})")
ax.text(0.0, 0.98, "\n".join(lines), va="top", ha="left", fontsize=8.5, family="monospace")

axes[0].legend(fontsize=8, loc="upper right")
for i in (4, 5, 6):
    axes[i].set_xlabel("Δ")
fig.suptitle("exp-058 — Per-head exponent histograms vs nulls (window [0.20, 0.30] shaded; dashed: Δ = 0.25)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = OUT_DIR / "delta_histograms_nulls.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")
