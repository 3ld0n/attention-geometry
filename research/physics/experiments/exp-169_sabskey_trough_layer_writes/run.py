"""
exp-169 — S_abskey trough mechanism: layer-0 write counterfactual test.

Pre-registration: prereg.md (committed b48b2b6 before this script was written).

Amendment A-1 (pre-run): The trough contribution metric is defined as
    contribution[h] = (trough_full[h] - trough_condition[h]) / trough_full[h]
where trough_X[h] = X[h, 0] - min(X[h, 1:512]).
This is equivalent to S_full - S_condition and is cleaner than the prereg's
"max causal[0:512] - min causal[0:512]" formula; no hypotheses changed.

Four conditions:
  normal       — standard forward pass (validate against exp-164 kbar)
  no_attn0     — zero attn_block0 residual write (h[0].attn output → 0)
  no_mlp0      — zero MLP_block0 residual write (h[0].mlp output → 0)
  uniform_wpe  — replace wpe with mean(wpe, dim=0) repeated across positions

Usage:
    .venv/bin/python3 research/physics/experiments/exp-169_sabskey_trough_layer_writes/run.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ── Config ───────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
EXPS_DIR   = SCRIPT_DIR.parent
EXP164_DIR = EXPS_DIR / "exp-164_l3h4_bilinear_decomposition"
KBAR164    = EXP164_DIR / "kbar_qbar.npz"
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEQ_LEN    = 1024
N_SEQS     = 200
BATCH_SIZE = 10
SEED       = 42

HEAD_MAP = {
    "L2H1":  (2,  1),
    "L3H4":  (3,  4),
    "L5H0":  (5,  0),
    "L7H11": (7, 11),
    "L10H8": (10, 8),
}
STRUCTURAL_HEADS = list(HEAD_MAP.keys())
D_HEAD  = 64
D_MODEL = 768
MQ_START, MQ_END = 256, 1024   # mean-query window (matches exp-163/164)

# exp-168 wpe trough fractions (for H3 combination check)
EXP168_WPE_FRACTION = {
    "L2H1":  0.629,
    "L3H4":  0.479,
    "L5H0":  0.604,
    "L7H11": 0.655,
    "L10H8": 0.359,
}

# exp-168 S_abskey sampled values for K1 cross-check
# (from results.json: h2_results shows these were computed fresh; cross-check against exp-163)
EXP164_KBAR_SAMPLES = {
    "L2H1":  {"positions": [0, 511, 1023], "expected_s_abskey": [2.413, -0.117, 2.455]},
    "L3H4":  {"positions": [0, 511, 1023], "expected_s_abskey": [5.170, -0.322, 3.843]},
    "L5H0":  {"positions": [0, 511, 1023], "expected_s_abskey": [5.438, -0.368, 3.146]},
    "L7H11": {"positions": [0, 511, 1023], "expected_s_abskey": [3.640, -0.082, 2.528]},
    "L10H8": {"positions": [0, 511, 1023], "expected_s_abskey": [4.695, -0.542, 4.442]},
}


# ── Data ─────────────────────────────────────────────────────────────────────

def make_random_seqs(tokenizer, n: int, seq_len: int, seed: int = 42) -> torch.Tensor:
    """Exact replica of exp-164's make_random_seqs."""
    rng     = np.random.default_rng(seed)
    exclude = {tokenizer.eos_token_id, tokenizer.bos_token_id,
               tokenizer.pad_token_id} - {None}
    valid   = [i for i in range(tokenizer.vocab_size) if i not in exclude]
    ids     = rng.choice(valid, size=(n, seq_len), replace=True)
    return torch.tensor(ids, dtype=torch.long)


# ── kbar extraction ───────────────────────────────────────────────────────────

def collect_kbar_qbar(
    model: GPT2LMHeadModel,
    all_seqs: torch.Tensor,
    condition: str,
    zero_attn0: bool = False,
    zero_mlp0: bool = False,
    uniform_wpe: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Run forward passes under the given condition, accumulate kbar and qbar.

    Returns:
        kbar: {head_name: array[SEQ_LEN, D_HEAD]}
        qbar: {head_name: array[SEQ_LEN, D_HEAD]}
    """
    kbar_sum = {h: np.zeros((SEQ_LEN, D_HEAD), dtype=np.float64) for h in STRUCTURAL_HEADS}
    qbar_sum = {h: np.zeros((SEQ_LEN, D_HEAD), dtype=np.float64) for h in STRUCTURAL_HEADS}
    seq_count = 0

    # Patch: uniform wpe — save original and replace
    orig_wpe_weight = None
    if uniform_wpe:
        orig_wpe_weight = model.transformer.wpe.weight.data.clone()
        mean_wpe = orig_wpe_weight.mean(dim=0, keepdim=True).expand(SEQ_LEN, -1)
        model.transformer.wpe.weight.data.copy_(mean_wpe)

    # Counterfactual hooks for zeroing layer-0 writes
    hooks = []
    if zero_attn0:
        def _zero_attn0(module, input, output):
            # GPT2Attention returns (attn_output, present_kv, [attn_weights])
            return (torch.zeros_like(output[0]),) + output[1:]
        h = model.transformer.h[0].attn.register_forward_hook(_zero_attn0)
        hooks.append(h)
    if zero_mlp0:
        def _zero_mlp0(module, input, output):
            return torch.zeros_like(output)
        h = model.transformer.h[0].mlp.register_forward_hook(_zero_mlp0)
        hooks.append(h)

    # QK extraction hooks (match exp-164 exactly)
    captured: dict[str, dict] = {}

    def make_qk_hook(head_name: str, layer: int, head: int):
        def hook(module, inputs, output):
            hidden = inputs[0]
            B, T, C = hidden.shape
            # Re-project to get Q, K (same as exp-164)
            with torch.no_grad():
                qkv = module.c_attn(hidden)
                q_all, k_all, _ = qkv.split(C, dim=-1)
                def to_head(x):
                    x = x.view(B, T, model.config.n_head, D_HEAD)
                    return x.permute(0, 2, 1, 3)  # [B, n_heads, T, d_head]
                q_h = to_head(q_all)[:, head, :, :].cpu().float().numpy()  # [B, T, d_head]
                k_h = to_head(k_all)[:, head, :, :].cpu().float().numpy()
            captured[head_name] = {"q": q_h, "k": k_h}
        return hook

    qk_hooks = []
    for head_name, (layer, head) in HEAD_MAP.items():
        h = model.transformer.h[layer].attn.register_forward_hook(
            make_qk_hook(head_name, layer, head)
        )
        qk_hooks.append(h)

    model.eval()
    with torch.no_grad():
        for b0 in range(0, N_SEQS, BATCH_SIZE):
            b1    = min(b0 + BATCH_SIZE, N_SEQS)
            batch = all_seqs[b0:b1]  # [Bsz, T]

            captured.clear()
            _ = model(batch)

            Bsz = b1 - b0
            for head_name in STRUCTURAL_HEADS:
                k_h = captured[head_name]["k"]  # [Bsz, T, d_head]
                q_h = captured[head_name]["q"]  # [Bsz, T, d_head]
                kbar_sum[head_name] += k_h.sum(axis=0)   # sum over batch → [T, d_head]
                qbar_sum[head_name] += q_h.sum(axis=0)
            seq_count += Bsz

    for h_hook in hooks + qk_hooks:
        h_hook.remove()

    # Restore wpe if patched
    if orig_wpe_weight is not None:
        model.transformer.wpe.weight.data.copy_(orig_wpe_weight)

    kbar = {h: kbar_sum[h] / seq_count for h in STRUCTURAL_HEADS}
    qbar = {h: qbar_sum[h] / seq_count for h in STRUCTURAL_HEADS}
    return kbar, qbar


# ── Score computation ─────────────────────────────────────────────────────────

def compute_s_abskey(kbar_h: np.ndarray, m_q: np.ndarray) -> np.ndarray:
    """S_abskey[a] = m_q · (kbar[a] - m_k) / sqrt(d_head).

    m_k = kbar[MQ_START:MQ_END].mean(0)  — matches exp-163/164/168 protocol.
    """
    m_k = kbar_h[MQ_START:MQ_END].mean(axis=0)
    dk  = kbar_h - m_k[np.newaxis, :]
    return (dk @ m_q) / np.sqrt(D_HEAD)


def trough_depth(s: np.ndarray) -> float:
    """Trough depth = s[0] - min(s[1:512]).

    Matches exp-168's trough_abskey metric. Positive when s dips below s[0].
    """
    return float(s[0] - s[1:512].min())


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    print("exp-169 | S_abskey trough mechanism — layer-0 write counterfactual")
    print("Pre-registration: prereg.md (b48b2b6)")
    print("Amendment A-1: trough contribution = (trough_full - trough_cond) / trough_full")
    print()

    # ── Load model and sequences ──────────────────────────────────────────────
    print("Loading GPT-2 small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model     = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    print("  ✓")

    print(f"Generating {N_SEQS} random sequences (seed={SEED}, SEQ_LEN={SEQ_LEN})...")
    all_seqs = make_random_seqs(tokenizer, N_SEQS, SEQ_LEN, seed=SEED)
    print("  ✓")

    # ── Run all four conditions ───────────────────────────────────────────────
    conditions = {
        "normal":       dict(zero_attn0=False, zero_mlp0=False, uniform_wpe=False),
        "no_attn0":     dict(zero_attn0=True,  zero_mlp0=False, uniform_wpe=False),
        "no_mlp0":      dict(zero_attn0=False, zero_mlp0=True,  uniform_wpe=False),
        "uniform_wpe":  dict(zero_attn0=False, zero_mlp0=False, uniform_wpe=True),
    }

    kbar_all: dict[str, dict[str, np.ndarray]] = {}
    qbar_all: dict[str, dict[str, np.ndarray]] = {}
    for cond_name, kwargs in conditions.items():
        print(f"\nRunning condition: {cond_name} ...")
        t_cond = time.time()
        kb, qb = collect_kbar_qbar(model, all_seqs, condition=cond_name, **kwargs)
        kbar_all[cond_name] = kb
        qbar_all[cond_name] = qb
        print(f"  ✓ ({time.time() - t_cond:.1f}s)")

    # ── K1: Cross-check normal kbar against exp-164 ───────────────────────────
    print("\nK1: Cross-checking normal kbar against exp-164...")
    exp164 = np.load(KBAR164)

    m_q = {h: qbar_all["normal"][h][MQ_START:MQ_END].mean(axis=0) for h in STRUCTURAL_HEADS}
    s_normal = {h: compute_s_abskey(kbar_all["normal"][h], m_q[h]) for h in STRUCTURAL_HEADS}

    k1_max_err = 0.0
    k1_pass = True
    for head_name in STRUCTURAL_HEADS:
        kbar_164 = exp164[f"kbar_{head_name}"]
        diff = np.abs(kbar_all["normal"][head_name] - kbar_164).max()
        k1_max_err = max(k1_max_err, diff)
        if diff > 0.5:
            print(f"  K1 FIRES: {head_name} kbar max diff = {diff:.4f} > 0.5 THRESHOLD")
            k1_pass = False
        else:
            print(f"  {head_name}: max kbar diff from exp-164 = {diff:.6f} ✓")

    if not k1_pass:
        print("K1 fired — data mismatch. Aborting further analysis.")
        raise SystemExit(1)
    print(f"  K1 passed (max error = {k1_max_err:.6f})")

    # ── Compute S_abskey for all conditions using m_q from normal run ─────────
    print("\nComputing S_abskey for all conditions (using m_q from normal run)...")
    s_all: dict[str, dict[str, np.ndarray]] = {}
    for cond_name in conditions:
        s_all[cond_name] = {
            h: compute_s_abskey(kbar_all[cond_name][h], m_q[h])
            for h in STRUCTURAL_HEADS
        }

    # ── Measure trough depths and contributions ───────────────────────────────
    print("\nMeasuring trough depths...")

    trough_full   = {h: trough_depth(s_all["normal"][h])    for h in STRUCTURAL_HEADS}
    trough_no_a0  = {h: trough_depth(s_all["no_attn0"][h])  for h in STRUCTURAL_HEADS}
    trough_no_m0  = {h: trough_depth(s_all["no_mlp0"][h])   for h in STRUCTURAL_HEADS}
    trough_unif   = {h: trough_depth(s_all["uniform_wpe"][h]) for h in STRUCTURAL_HEADS}

    # Contribution = (trough_full - trough_condition) / trough_full
    # Positive → component ADDS to the trough; Negative → component SUPPRESSES it
    attn0_contrib = {
        h: (trough_full[h] - trough_no_a0[h]) / trough_full[h]
        for h in STRUCTURAL_HEADS
    }
    mlp0_contrib = {
        h: (trough_full[h] - trough_no_m0[h]) / trough_full[h]
        for h in STRUCTURAL_HEADS
    }
    wpe_true_contrib = {
        h: (trough_full[h] - trough_unif[h]) / trough_full[h]
        for h in STRUCTURAL_HEADS
    }

    print("\n--- Trough depths and contributions ---")
    print(f"{'Head':<8} {'full':>8} {'no_attn0':>10} {'no_mlp0':>9} {'unif_wpe':>10}")
    for h in STRUCTURAL_HEADS:
        print(f"{h:<8} {trough_full[h]:>8.3f} {trough_no_a0[h]:>10.3f} "
              f"{trough_no_m0[h]:>9.3f} {trough_unif[h]:>10.3f}")

    print(f"\n{'Head':<8} {'attn0':>8} {'mlp0':>8} {'wpe_true':>10} {'wpe_lin':>9}")
    print(f"{'':8} {'contrib':>8} {'contrib':>8} {'contrib':>10} {'(exp168)':>9}")
    for h in STRUCTURAL_HEADS:
        print(f"{h:<8} {attn0_contrib[h]:>+8.3f} {mlp0_contrib[h]:>+8.3f} "
              f"{wpe_true_contrib[h]:>+10.3f} {EXP168_WPE_FRACTION[h]:>+9.3f}")

    # ── H1: attn0 causal ≥ 30% on ≥ 3/5 ──────────────────────────────────────
    h1_pass_heads = [h for h in STRUCTURAL_HEADS if attn0_contrib[h] >= 0.30]
    h1_confirmed  = len(h1_pass_heads) >= 3
    print(f"\nH1 (attn0 ≥30%): {'CONFIRMED' if h1_confirmed else 'NOT CONFIRMED'} "
          f"— {len(h1_pass_heads)}/5 heads pass "
          f"({', '.join(h1_pass_heads) if h1_pass_heads else 'none'})")

    # ── H2: MLP0 causal ≥ 30% on ≥ 3/5 ──────────────────────────────────────
    h2_pass_heads = [h for h in STRUCTURAL_HEADS if mlp0_contrib[h] >= 0.30]
    h2_confirmed  = len(h2_pass_heads) >= 3
    print(f"H2 (MLP0 ≥30%): {'CONFIRMED' if h2_confirmed else 'NOT CONFIRMED'} "
          f"— {len(h2_pass_heads)}/5 heads pass "
          f"({', '.join(h2_pass_heads) if h2_pass_heads else 'none'})")

    # ── H3: combined wpe_lin + attn0 + mlp0 ≥ 70% on ≥ 3/5 ─────────────────
    combined = {
        h: EXP168_WPE_FRACTION[h] + attn0_contrib[h] + mlp0_contrib[h]
        for h in STRUCTURAL_HEADS
    }
    h3_pass_heads = [h for h in STRUCTURAL_HEADS if combined[h] >= 0.70]
    h3_confirmed  = len(h3_pass_heads) >= 3
    print(f"H3 (wpe_lin+attn0+MLP0 ≥70%): {'CONFIRMED' if h3_confirmed else 'NOT CONFIRMED'} "
          f"— {len(h3_pass_heads)}/5 heads pass "
          f"({', '.join(h3_pass_heads) if h3_pass_heads else 'none'})")
    for h in STRUCTURAL_HEADS:
        print(f"  {h}: combined = {combined[h]:+.3f} "
              f"(wpe_lin {EXP168_WPE_FRACTION[h]:+.3f} + "
              f"attn0 {attn0_contrib[h]:+.3f} + mlp0 {mlp0_contrib[h]:+.3f})")

    # ── H4: wpe true causal ≥ 30% MORE than wpe linear (exp-168) ────────────
    wpe_uplift = {
        h: (wpe_true_contrib[h] - EXP168_WPE_FRACTION[h]) / EXP168_WPE_FRACTION[h]
        for h in STRUCTURAL_HEADS
    }
    h4_pass_heads = [h for h in STRUCTURAL_HEADS if wpe_uplift[h] >= 0.30]
    h4_confirmed  = len(h4_pass_heads) >= 3
    print(f"H4 (wpe_true ≥30% uplift over wpe_lin): {'CONFIRMED' if h4_confirmed else 'NOT CONFIRMED'} "
          f"— {len(h4_pass_heads)}/5 heads pass")
    for h in STRUCTURAL_HEADS:
        print(f"  {h}: wpe_true={wpe_true_contrib[h]:+.3f}, wpe_lin={EXP168_WPE_FRACTION[h]:+.3f}, "
              f"uplift={wpe_uplift[h]:+.3f} {'✓' if wpe_uplift[h] >= 0.30 else '✗'}")

    # ── K2: Combined coverage ─────────────────────────────────────────────────
    wpe_effective = {h: max(wpe_true_contrib[h], EXP168_WPE_FRACTION[h]) for h in STRUCTURAL_HEADS}
    combined_true = {
        h: wpe_effective[h] + attn0_contrib[h] + mlp0_contrib[h]
        for h in STRUCTURAL_HEADS
    }
    k2_pass_heads = [h for h in STRUCTURAL_HEADS if combined_true[h] >= 0.70]
    if len(k2_pass_heads) < 3 and not h4_confirmed:
        print("\nK2: FIRES — combined (wpe_true + attn0 + mlp0) < 70% on ≥ 3/5 heads AND H4 fails.")
        print("  Mechanism is in layers 1–(L_head−1) or non-additive. Queue a follow-on.")
    else:
        print(f"\nK2: Does not fire — combined_true coverage ≥ 70% on {len(k2_pass_heads)}/5 heads.")

    # ── Monotonicity check ────────────────────────────────────────────────────
    def mono_frac(s: np.ndarray) -> float:
        return float((np.diff(s[1:]) >= 0).mean())

    print("\n--- Monotonicity fraction (positions 1–1023) ---")
    print(f"{'Head':<8} {'normal':>8} {'no_attn0':>10} {'no_mlp0':>9} {'unif_wpe':>10}")
    for h in STRUCTURAL_HEADS:
        print(f"{h:<8} "
              f"{mono_frac(s_all['normal'][h]):>8.3f} "
              f"{mono_frac(s_all['no_attn0'][h]):>10.3f} "
              f"{mono_frac(s_all['no_mlp0'][h]):>9.3f} "
              f"{mono_frac(s_all['uniform_wpe'][h]):>10.3f}")

    # ── Save results ──────────────────────────────────────────────────────────
    elapsed = time.time() - t0

    per_head = {}
    for h in STRUCTURAL_HEADS:
        per_head[h] = {
            "trough_full":          float(trough_full[h]),
            "trough_no_attn0":      float(trough_no_a0[h]),
            "trough_no_mlp0":       float(trough_no_m0[h]),
            "trough_uniform_wpe":   float(trough_unif[h]),
            "attn0_contrib":        float(attn0_contrib[h]),
            "mlp0_contrib":         float(mlp0_contrib[h]),
            "wpe_true_contrib":     float(wpe_true_contrib[h]),
            "wpe_lin_contrib_exp168": float(EXP168_WPE_FRACTION[h]),
            "wpe_uplift":           float(wpe_uplift[h]),
            "combined_lin_contrib": float(combined[h]),
            "combined_true_contrib": float(combined_true[h]),
            "mono_frac_normal":     mono_frac(s_all["normal"][h]),
            "mono_frac_no_attn0":   mono_frac(s_all["no_attn0"][h]),
            "mono_frac_no_mlp0":    mono_frac(s_all["no_mlp0"][h]),
            "mono_frac_uniform_wpe": mono_frac(s_all["uniform_wpe"][h]),
            "h1_pass": attn0_contrib[h] >= 0.30,
            "h2_pass": mlp0_contrib[h] >= 0.30,
            "h3_pass": combined[h] >= 0.70,
            "h4_pass": wpe_uplift[h] >= 0.30,
        }

    # Determine overall verdict
    if h1_confirmed and h2_confirmed:
        verdict = "partial"  # both layer-write components contribute
    elif h2_confirmed and not h1_confirmed:
        verdict = "partial"  # MLP0 drives trough, attn0 secondary
    elif h1_confirmed and not h2_confirmed:
        verdict = "partial"  # attn0 drives trough, MLP0 secondary
    elif h4_confirmed:
        verdict = "partial"  # wpe true causal explains more than linear approx (LN amplification)
    elif h3_confirmed:
        verdict = "confirmed"  # combined coverage ≥ 70% — mechanism understood to first order
    else:
        verdict = "inconclusive"  # K2 fires, mechanism elsewhere

    results = {
        "experiment":     "exp-169",
        "date":           "2026-10-09",
        "prereg_commit":  "b48b2b6",
        "prereg_evidence": "git-attested — prereg.md committed before run.py was written",
        "amendment_a1":   "Trough contribution metric clarified (see run.py header); no hypotheses changed",
        "analysis_only":  False,
        "model":          "gpt2",
        "n_seqs":         N_SEQS,
        "seq_len":        SEQ_LEN,
        "seed":           SEED,
        "k1_max_kbar_diff": float(k1_max_err),
        "k1_pass":        k1_pass,
        "per_head":       per_head,
        "h1_n_confirmed": len(h1_pass_heads),
        "h1_confirmed":   h1_confirmed,
        "h2_n_confirmed": len(h2_pass_heads),
        "h2_confirmed":   h2_confirmed,
        "h3_n_confirmed": len(h3_pass_heads),
        "h3_confirmed":   h3_confirmed,
        "h4_n_confirmed": len(h4_pass_heads),
        "h4_confirmed":   h4_confirmed,
        "verdict":        verdict,
        "elapsed_s":      round(elapsed, 1),
    }

    out_path = RESULTS_DIR / "results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nResults saved to {out_path}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"\nVerdict: {verdict.upper()}")


if __name__ == "__main__":
    main()
