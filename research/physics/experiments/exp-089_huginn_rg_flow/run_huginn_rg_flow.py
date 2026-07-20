"""
exp-089: Latent-Iteration RG Flow in Huginn

Tests whether inference-time recurrence in a trained depth-recurrent transformer
produces RG flow toward the conformal fixed point (Δ → 0.25).

Pre-registration: prereg.md (committed before model download)
Protocol: see prereg.md for full specification.
"""

import json
import sys
import time
import math
import torch
import numpy as np
from pathlib import Path
from scipy import stats
from transformers import AutoModelForCausalLM

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = Path(__file__).resolve().parent
RESULTS_FILE = RESULTS_DIR / "results.json"
MODEL_ID = "tomg-group-umd/huginn-0125"

# ── Protocol constants (from prereg.md) ────────────────────────────────────────
N_SEQ = 20          # number of random token sequences
SEQ_LEN = 128       # sequence length
NUM_STEPS = 64      # total recurrence steps
CUTOFF_LOW = 3      # minimum distance for power-law fit
MAX_DIST = 32       # maximum distance for power-law fit
R2_THRESHOLD = 0.90 # conformal head threshold
DELTA_LOW = 0.20    # SYK-near lower bound
DELTA_HIGH = 0.30   # SYK-near upper bound
REPORT_STEPS = [1, 2, 4, 8, 16, 32, 64]  # steps to highlight in summary


# ── BCFT fitting ────────────────────────────────────────────────────────────────

def fit_power_law(profile: np.ndarray, cutoff_low: int, max_dist: int) -> dict:
    """
    Fit A(d) ~ C * d^{-2Delta} to the mean distance-decay profile.
    profile[d] = mean attention weight at distance d (0-indexed, d=0 unused).
    Returns dict with keys: delta, r2, valid.
    """
    ds = np.arange(cutoff_low, max_dist + 1)
    vals = profile[cutoff_low:max_dist + 1]
    valid_mask = (vals > 0) & np.isfinite(vals)
    if valid_mask.sum() < 5:
        return {"delta": float("nan"), "r2": 0.0, "valid": False}
    log_d = np.log(ds[valid_mask])
    log_A = np.log(vals[valid_mask])
    slope, intercept, r, p, _ = stats.linregress(log_d, log_A)
    delta = -slope / 2.0
    r2 = r ** 2
    return {"delta": float(delta), "r2": float(r2), "valid": True}


def fit_all_heads(profiles_by_step_layer: dict, n_heads: int, cutoff_low: int, max_dist: int) -> dict:
    """
    For each (step, layer), fit BCFT for all heads.
    profiles_by_step_layer: {(step, layer): np.ndarray of shape (n_heads, max_dist+1)}
    Returns: {(step, layer): list of {delta, r2, syk_near} per head}
    """
    results = {}
    for (step, layer), profiles in profiles_by_step_layer.items():
        head_results = []
        for h in range(n_heads):
            fit = fit_power_law(profiles[h], cutoff_low, max_dist)
            syk_near = (
                fit["valid"]
                and fit["r2"] >= R2_THRESHOLD
                and DELTA_LOW <= fit["delta"] <= DELTA_HIGH
            )
            head_results.append({
                "delta": fit["delta"],
                "r2": fit["r2"],
                "syk_near": syk_near,
                "valid": fit["valid"],
            })
        results[(step, layer)] = head_results
    return results


# ── Attention capture ───────────────────────────────────────────────────────────

def capture_attention_profiles(model, input_ids: torch.Tensor, num_steps: int, device: str) -> dict:
    """
    Run Huginn with attention capture, returning distance-averaged profiles.

    Returns: profiles dict {(step, layer): np.ndarray of shape (n_heads, max_dist+1)}
    """
    n_layers_recurrent = model.config.n_layers_in_recurrent_block   # 4
    n_prelude = model.config.n_layers_in_prelude                     # 2
    n_heads = model.config.num_attention_heads                       # 55

    # Import from the model's custom module (loaded via trust_remote_code)
    # After AutoModelForCausalLM.from_pretrained, the custom classes are available
    # via the model's class hierarchy
    attn_class = type(model.transformer.core_block[0].attn)
    apply_rotary = None
    # Find apply_rotary_emb_complex_like in the module
    import importlib
    for mod_name, mod in sys.modules.items():
        if "raven_modeling" in mod_name:
            apply_rotary = getattr(mod, "apply_rotary_emb_complex_like", None)
            if apply_rotary is not None:
                break
    if apply_rotary is None:
        raise RuntimeError("Could not find apply_rotary_emb_complex_like in loaded modules")

    # Storage: profiles[step][layer] = sum of mean_A(d) across sequences, shape (n_heads, max_dist+1)
    profiles = {}
    n_seq_processed = [0]

    # Precompute causal mask
    causal_mask = torch.triu(
        torch.full((SEQ_LEN, SEQ_LEN), float("-inf")), diagonal=1
    ).to(device)

    original_forward = attn_class.forward

    def patched_forward(self, x, freqs_cis, block_idx, mask=None, past_key_values=None):
        B, S_len, E = x.shape
        q, k, v = self.Wqkv(x).split(self.chunks, dim=2)
        q = q.view(B, S_len, self.n_head, self.head_dim)
        k = k.view(B, S_len, self.n_kv_heads, self.head_dim)
        v = v.view(B, S_len, self.n_kv_heads, self.head_dim)
        if self.config.qk_bias:
            q_bias, k_bias = self.qk_bias.split(1, dim=0)
            q, k = (q + q_bias).to(q.dtype), (k + k_bias).to(q.dtype)
        q, k = apply_rotary(q, k, freqs_cis=freqs_cis)

        # Decode step and layer from block_idx
        bid = int(block_idx.item())
        if n_prelude <= bid:  # core_block layers (coda uses negative idx, so bid < 0 for coda)
            adjusted = bid - n_prelude
            if adjusted >= 0:
                step_i = adjusted // n_layers_recurrent
                layer_i = adjusted % n_layers_recurrent
                if 0 <= step_i < num_steps:
                    # Compute causal attention weights
                    qT = q.transpose(1, 2).float()  # (B, H, S, d)
                    kT = k.transpose(1, 2).float()
                    scale = self.head_dim ** -0.5
                    logits = qT @ kT.transpose(-2, -1) * scale  # (B, H, S, S)

                    # Apply causal mask (reuse precomputed; handle length mismatch)
                    cm = causal_mask[:S_len, :S_len]
                    attn_w = torch.softmax(logits + cm, dim=-1)  # (B, H, S, S)
                    attn_cpu = attn_w.detach().float().cpu().numpy()  # (B, H, S, S)

                    key = (step_i, layer_i)
                    if key not in profiles:
                        profiles[key] = np.zeros((n_heads, MAX_DIST + 1), dtype=np.float64)

                    # Accumulate distance-averaged profile
                    # For d = cutoff_low .. max_dist, average over batch and valid query positions
                    for d in range(CUTOFF_LOW, MAX_DIST + 1):
                        if d >= S_len:
                            break
                        # pairs: q_idx = d..S_len-1, k_idx = q_idx - d
                        q_idxs = np.arange(d, S_len)
                        k_idxs = q_idxs - d
                        # attn_cpu[:, :, q_idxs, k_idxs] → (B, H, n_valid)
                        vals = attn_cpu[:, :, q_idxs, k_idxs]  # (B, H, n_valid)
                        profiles[key][:, d] += vals.mean(axis=(0, 2))  # (H,)

        # Original computation continues
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, block_idx)
        if mask is not None:
            from torch.nn.attention.flex_attention import flex_attention
            y = flex_attention(q, k, v, block_mask=mask)
        else:
            if q.shape[2] < k.shape[2]:
                if q.shape[2] > 1:
                    from torch.nn.attention import bias as attn_bias
                    bias_obj = attn_bias.causal_lower_right(q.shape[2], k.shape[2])
                    y = torch.nn.functional.scaled_dot_product_attention(
                        q, k, v, bias_obj, dropout_p=0.0
                    )
                else:
                    y = torch.nn.functional.scaled_dot_product_attention(
                        q, k, v, dropout_p=0.0, is_causal=False
                    )
            else:
                y = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, dropout_p=0.0, is_causal=True
                )
        return self.proj(y.transpose(1, 2).reshape(B, S_len, E).contiguous())

    attn_class.forward = patched_forward
    try:
        with torch.no_grad():
            model(input_ids, num_steps=num_steps)
    finally:
        attn_class.forward = original_forward

    return profiles


# ── Randomization ───────────────────────────────────────────────────────────────

def randomize_core_block(model):
    """
    Reinitialize only the core_block weights, leaving prelude/coda unchanged.
    Uses the model's own init scheme.
    """
    for block in model.transformer.core_block:
        model._init_weights(block.attn.Wqkv)
        if hasattr(block.attn, 'proj'):
            model._init_weights(block.attn.proj)
        model._init_weights(block.mlp.fc)
        model._init_weights(block.mlp.proj)
        # Reinit layer norms to ones
        torch.nn.init.ones_(block.norm_1.weight)
        torch.nn.init.ones_(block.norm_2.weight)
        torch.nn.init.ones_(block.norm_3.weight)
        torch.nn.init.ones_(block.norm_4.weight)
        # QK bias
        if hasattr(block.attn, 'qk_bias'):
            torch.nn.init.zeros_(block.attn.qk_bias)
    print("Core block weights randomized.")


# ── Summary statistics ──────────────────────────────────────────────────────────

def summarize_per_step(head_results: dict, n_layers_recurrent: int, n_heads: int) -> dict:
    """
    Aggregate SYK_near counts per recurrence step (summed over all layers).
    Returns: {step: {"n_syk": int, "n_conformal": int, "total": int, "fraction_syk": float}}
    """
    step_summary = {}
    for (step, layer), heads in head_results.items():
        if step not in step_summary:
            step_summary[step] = {"n_syk": 0, "n_conformal": 0, "total": 0}
        for h in heads:
            step_summary[step]["total"] += 1
            if h.get("r2", 0) >= R2_THRESHOLD and h.get("valid", False):
                step_summary[step]["n_conformal"] += 1
            if h.get("syk_near", False):
                step_summary[step]["n_syk"] += 1
    for step in step_summary:
        total = step_summary[step]["total"]
        step_summary[step]["fraction_syk"] = step_summary[step]["n_syk"] / total if total > 0 else 0.0
    return step_summary


def compute_spearman(step_summary: dict) -> dict:
    """Compute Spearman ρ(step, SYK_near_fraction)."""
    steps = sorted(step_summary.keys())
    fracs = [step_summary[s]["fraction_syk"] for s in steps]
    rho, p = stats.spearmanr(steps, fracs)
    return {"rho": float(rho), "p": float(p), "steps": steps, "fracs": fracs}


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load model ──────────────────────────────────────────────────────────────
    print(f"\nLoading {MODEL_ID} in bfloat16...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    model = model.to(device)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    n_layers_recurrent = model.config.n_layers_in_recurrent_block   # 4
    n_heads = model.config.num_attention_heads                       # 55
    print(f"  Architecture: {n_layers_recurrent} recurrent layers × {n_heads} heads")
    print(f"  Total heads per step: {n_layers_recurrent * n_heads}")

    # ── Prepare random inputs ───────────────────────────────────────────────────
    torch.manual_seed(89)
    np.random.seed(89)
    input_ids = torch.randint(
        0, model.config.padded_vocab_size, (N_SEQ, SEQ_LEN), device=device
    )
    print(f"\nInput: {N_SEQ} sequences × {SEQ_LEN} tokens (random), seed=89")

    # ── Trained condition ───────────────────────────────────────────────────────
    print(f"\n=== TRAINED condition: {NUM_STEPS} recurrence steps ===")
    t0 = time.time()
    trained_profiles = capture_attention_profiles(model, input_ids, NUM_STEPS, device)
    print(f"  Capture done in {time.time() - t0:.1f}s")
    print(f"  Profile keys captured: {len(trained_profiles)}")

    print("  Fitting BCFT for trained condition...")
    trained_head_results = fit_all_heads(trained_profiles, n_heads, CUTOFF_LOW, MAX_DIST)
    trained_step_summary = summarize_per_step(trained_head_results, n_layers_recurrent, n_heads)
    trained_spearman = compute_spearman(trained_step_summary)

    print(f"\n  Trained Spearman ρ={trained_spearman['rho']:.3f}, p={trained_spearman['p']:.4f}")
    print(f"  Step | SYK_near | Fraction")
    for s in REPORT_STEPS:
        if s - 1 in trained_step_summary:
            sm = trained_step_summary[s - 1]
            print(f"  {s:4d} | {sm['n_syk']:8d} | {sm['fraction_syk']:.4f}")
    print(f"  At step {NUM_STEPS}: {trained_step_summary.get(NUM_STEPS-1, {}).get('n_syk', 'N/A')} SYK-near / {n_layers_recurrent * n_heads} total")

    # ── Randomized condition ────────────────────────────────────────────────────
    print(f"\n=== RANDOM condition: randomizing core_block weights ===")
    randomize_core_block(model)

    t0 = time.time()
    rand_profiles = capture_attention_profiles(model, input_ids, NUM_STEPS, device)
    print(f"  Capture done in {time.time() - t0:.1f}s")

    print("  Fitting BCFT for random condition...")
    rand_head_results = fit_all_heads(rand_profiles, n_heads, CUTOFF_LOW, MAX_DIST)
    rand_step_summary = summarize_per_step(rand_head_results, n_layers_recurrent, n_heads)
    rand_spearman = compute_spearman(rand_step_summary)

    print(f"\n  Random Spearman ρ={rand_spearman['rho']:.3f}, p={rand_spearman['p']:.4f}")
    print(f"  Step | SYK_near | Fraction")
    for s in REPORT_STEPS:
        if s - 1 in rand_step_summary:
            sm = rand_step_summary[s - 1]
            print(f"  {s:4d} | {sm['n_syk']:8d} | {sm['fraction_syk']:.4f}")

    # ── Registered verdicts ─────────────────────────────────────────────────────
    print("\n=== REGISTERED VERDICTS ===")

    # H_rg_flow
    h_rg_flow = trained_spearman["rho"] > 0 and trained_spearman["p"] < 0.05
    print(f"H_rg_flow: {'KEEP' if h_rg_flow else 'KILL'} "
          f"(trained ρ={trained_spearman['rho']:.3f}, p={trained_spearman['p']:.4f})")

    # H_rand_flat
    h_rand_flat = rand_spearman["rho"] <= 0 or rand_spearman["p"] >= 0.05
    print(f"H_rand_flat: {'KEEP' if h_rand_flat else 'FAIL'} "
          f"(random ρ={rand_spearman['rho']:.3f}, p={rand_spearman['p']:.4f})")

    # H_comparison (step 64 = index 63)
    trained_final = trained_step_summary.get(63, {}).get("fraction_syk", 0.0)
    rand_final = rand_step_summary.get(63, {}).get("fraction_syk", 0.0)
    h_comparison = trained_final > rand_final
    print(f"H_comparison: {'KEEP' if h_comparison else 'FAIL'} "
          f"(trained@64={trained_final:.4f} vs rand@64={rand_final:.4f})")

    # H_initial (step 1 = index 0)
    trained_init = trained_step_summary.get(0, {}).get("fraction_syk", 0.0)
    rand_init = rand_step_summary.get(0, {}).get("fraction_syk", 0.0)
    h_initial = trained_init > rand_init
    print(f"H_initial: {'KEEP' if h_initial else 'FAIL'} "
          f"(trained@1={trained_init:.4f} vs rand@1={rand_init:.4f})")

    # Overall verdict
    if h_rg_flow and h_rand_flat and h_comparison:
        overall = "CONFIRMED"
    elif h_rg_flow and h_comparison:
        overall = "PARTIAL_CONFIRMED"
    elif h_rg_flow and not h_comparison:
        overall = "PARTIAL_NO_ADVANTAGE"
    elif not h_rg_flow and h_rand_flat:
        overall = "FLATLINE"
    elif not h_rg_flow and not h_rand_flat:
        overall = "ARCHITECTURAL_FLOW"
    else:
        overall = "INCONCLUSIVE"
    print(f"\nOverall verdict: {overall}")

    # ── Save results ────────────────────────────────────────────────────────────
    # Convert step_summary keys from int to str for JSON
    def step_summary_to_json(ss):
        return {str(k): v for k, v in ss.items()}

    results = {
        "experiment": "exp-089",
        "model": MODEL_ID,
        "protocol": {
            "N_seq": N_SEQ,
            "seq_len": SEQ_LEN,
            "num_steps": NUM_STEPS,
            "cutoff_low": CUTOFF_LOW,
            "max_dist": MAX_DIST,
            "r2_threshold": R2_THRESHOLD,
            "delta_range": [DELTA_LOW, DELTA_HIGH],
            "seed": 89,
        },
        "architecture": {
            "n_layers_recurrent": n_layers_recurrent,
            "n_heads": n_heads,
            "total_heads_per_step": n_layers_recurrent * n_heads,
            "head_dim": model.config.head_dim,
            "n_embd": model.config.n_embd,
        },
        "trained": {
            "spearman_rho": trained_spearman["rho"],
            "spearman_p": trained_spearman["p"],
            "step_summary": step_summary_to_json(trained_step_summary),
        },
        "random": {
            "spearman_rho": rand_spearman["rho"],
            "spearman_p": rand_spearman["p"],
            "step_summary": step_summary_to_json(rand_step_summary),
        },
        "verdicts": {
            "H_rg_flow": "KEEP" if h_rg_flow else "KILL",
            "H_rand_flat": "KEEP" if h_rand_flat else "FAIL",
            "H_comparison": "KEEP" if h_comparison else "FAIL",
            "H_initial": "KEEP" if h_initial else "FAIL",
            "overall": overall,
        },
        "report_steps": REPORT_STEPS,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
