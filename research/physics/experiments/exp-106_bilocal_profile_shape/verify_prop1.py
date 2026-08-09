"""Direct verification of Proposition 1 on real GPT-2 data, plus a figure.

Proposition 1 [EXACT]: for row-stochastic A, G = A K A^T satisfies
G = mean(K) * 11^T + A (K - mean(K) 11^T) A^T, and mean(K_V) = ||vbar||^2 exactly.
Both halves are checked entry-wise on one real forward pass, to floating precision.

Run: python verify_prop1.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

HERE = Path(__file__).resolve().parent
KIT = HERE.parent.parent / "replication" / "measure_conformal_heads.py"
spec = importlib.util.spec_from_file_location("census_kit", KIT)
kit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kit)

SEQ_LEN, SEED = kit.SEQ_LEN, kit.SEED
FIT_LO, FIT_HI = kit.FIT_LO, kit.FIT_HI


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = AutoConfig.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2", dtype=torch.float32, attn_implementation="eager").to(device).eval()
    n_head, d = cfg.num_attention_heads, cfg.hidden_size

    cap = {}

    def hook(ell):
        def post(mod, inputs, output):
            v = output.detach()[..., 2 * d:3 * d]
            b, L, dd = v.shape
            cap[ell] = v.view(b, L, n_head, dd // n_head).permute(0, 2, 1, 3)
        return post

    handles = [model.transformer.h[e].attn.c_attn.register_forward_hook(hook(e))
               for e in range(cfg.num_hidden_layers)]

    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    ids = torch.from_numpy(
        rng.integers(0, cfg.vocab_size, size=(1, SEQ_LEN)).astype(np.int64)).to(device)
    with torch.no_grad():
        out = model(ids, output_attentions=True)
    for h in handles:
        h.remove()

    report = {}
    print("Proposition 1, entry-wise, on the first census input (seed 42):")
    print("  L/H    max|mean(K_V) - ||vbar||^2|   max|G - (mu + A Ktil A^T)|   "
          "rel. to |G|")
    worst_mu, worst_id = 0.0, 0.0
    for ell in (0, 2, 5, 7, 10):
        A = out.attentions[ell][0].float().cpu().double()
        v = cap[ell][0].float().cpu().double()
        for hh in range(n_head):
            Ah, vh = A[hh], v[hh]
            K = vh @ vh.T
            mu_direct = float(K.mean())
            vbar = vh.mean(dim=0)
            mu_theory = float(vbar @ vbar)
            e_mu = abs(mu_direct - mu_theory)

            G = (Ah @ vh) @ (Ah @ vh).T
            Ktil = K - mu_theory
            G_recon = mu_theory + Ah @ Ktil @ Ah.T
            e_id = float(torch.max(torch.abs(G - G_recon)))
            rel = e_id / float(torch.max(torch.abs(G)))
            worst_mu = max(worst_mu, e_mu / max(abs(mu_theory), 1e-30))
            worst_id = max(worst_id, rel)
            if hh in (1, 6):
                print(f"  {ell:2d}/{hh:<3d}  {e_mu:.3e}                    "
                      f"{e_id:.3e}                {rel:.3e}")
    print(f"\n  worst relative error, mu = ||vbar||^2      : {worst_mu:.3e}")
    print(f"  worst relative error, the decomposition    : {worst_id:.3e}")
    report["worst_rel_err_mu"] = worst_mu
    report["worst_rel_err_decomposition"] = worst_id

    # ---- the shape, for the record and for a figure -------------------------
    z = np.load(HERE / "profiles_forward_gpt2.npz")
    Gm = z["G_out"] / 50.0
    vbar_sq = z["vbar_sq"]
    picks = [(2, 1), (5, 0), (7, 11), (10, 8)]   # SYK-near heads
    lines = {}
    print("\nSYK-near heads: G's profile against its exact floor")
    print("  L/H    lag 8      lag 32     lag 128    lag 256    ||vbar||^2")
    for ell, hh in picks:
        p, mu = Gm[ell, hh], float(vbar_sq[ell, hh])
        print(f"  {ell:2d}/{hh:<3d} {p[8]:.4f}   {p[32]:.4f}   {p[128]:.4f}   "
              f"{p[256]:.4f}   {mu:.4f}")
        lines[f"L{ell}H{hh}"] = {"profile": p[:FIT_HI + 1].tolist(), "floor": mu}
    report["syk_near_shapes"] = lines
    json.dump(report, open(HERE / "verify_prop1.json", "w"), indent=1)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharex=True)
        lags = np.arange(1, FIT_HI + 1)
        for ax, (ell, hh) in zip(axes, picks):
            p, mu = Gm[ell, hh], float(vbar_sq[ell, hh])
            ax.plot(lags, p[1:FIT_HI + 1], lw=1.6, label=r"$P_G(s)$ measured")
            ax.axhline(mu, ls="--", lw=1.4, color="crimson",
                       label=r"exact floor $\|\bar v\|^2$")
            ax.set_xscale("log")
            ax.set_title(f"GPT-2 L{ell}H{hh}  ($\\Delta_A$ SYK-near)", fontsize=10)
            ax.set_xlabel("lag $s$")
            ax.axvspan(FIT_LO, FIT_HI, color="0.9", zorder=0)
        axes[0].set_ylabel(r"$P_G(s)$")
        axes[0].legend(fontsize=8, loc="lower left")
        fig.suptitle("The bilocal's lag profile sits BELOW its own exact floor: "
                     "the connected part is negative (GPT-2, census protocol)",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(HERE / "fig_floor_above_profile.png", dpi=140)
        print(f"\nwrote {HERE / 'fig_floor_above_profile.png'}")
    except Exception as exc:            # figure is a convenience, not a result
        print(f"\n(figure skipped: {exc})")


if __name__ == "__main__":
    main()
