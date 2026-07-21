#!/usr/bin/env python3
"""Figures for 'Latent Iteration as Renormalization' preprint.

Reads exp-089 results.json / results_control.json and produces:
  fig_rg_flow.{pdf,png}   — Δ_med vs recurrence step (NAT / RAND / control)
  fig_syk_count.{pdf,png} — SYK-near head count vs recurrence step
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
EXP = HERE / "../../../research/physics/experiments/exp-089_huginn_latent_rg_flow"

main = json.loads((EXP / "results.json").read_text())
ctrl = json.loads((EXP / "results_control.json").read_text())

nat = main["nat_summary"]
rand = main["rand_summary"]
# control run stores its natural-text summary under the same key
cnat = ctrl.get("nat_summary") or ctrl.get("control_summary")

steps = nat["step_nums"]

plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

# --- Figure 1: Delta_med flow ---
fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.axhspan(0.20, 0.30, color="tab:gray", alpha=0.15, label="SYK-near window")
ax.axhline(0.25, color="k", lw=0.8, ls="--", label=r"SYK $q{=}4$: $\Delta = 1/4$")
ax.axhline(0.50, color="k", lw=0.8, ls=":", label=r"$q{=}2$: $\Delta = 1/2$")
ax.plot(steps, nat["delta_meds"], "o-", color="tab:blue", label="Natural text (NAT)")
ax.plot(steps, rand["delta_meds"], "s-", color="tab:orange", label="Random tokens (RAND)")
ax.plot(cnat["step_nums"], cnat["delta_meds"], "^-", color="tab:red",
        label="Randomized weights (control)")
ax.set_xscale("log", base=2)
ax.set_xticks(steps)
ax.set_xticklabels([str(s) for s in steps])
ax.set_xlabel("Recurrence step $r$")
ax.set_ylabel(r"$\Delta_{\mathrm{med}}$ (median per-head exponent)")
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(HERE / "fig_rg_flow.pdf")
fig.savefig(HERE / "fig_rg_flow.png")

# --- Figure 2: SYK-near head count ---
fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.plot(steps, nat["syk_counts_per_step"], "o-", color="tab:blue",
        label="Natural text (NAT)")
ax.plot(steps, rand["syk_counts_per_step"], "s-", color="tab:orange",
        label="Random tokens (RAND)")
ax.plot(cnat["step_nums"], cnat["syk_counts_per_step"], "^-", color="tab:red",
        label="Randomized weights (control)")
ax.set_xscale("log", base=2)
ax.set_xticks(steps)
ax.set_xticklabels([str(s) for s in steps])
ax.set_xlabel("Recurrence step $r$")
ax.set_ylabel("SYK-near heads (of 4400 measurements)")
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(HERE / "fig_syk_count.pdf")
fig.savefig(HERE / "fig_syk_count.png")

print("NAT delta:", nat["delta_meds"])
print("RAND delta:", rand["delta_meds"])
print("CTRL delta:", cnat["delta_meds"])
print("NAT syk:", nat["syk_counts_per_step"])
print("RAND syk:", rand["syk_counts_per_step"])
print("CTRL syk:", cnat["syk_counts_per_step"])
print("rho NAT conv/emerg:", nat["rho_convergence"], nat.get("rho_emergence"))
print("rho RAND conv/emerg:", rand["rho_convergence"], rand.get("rho_emergence"))
print("figures written")
