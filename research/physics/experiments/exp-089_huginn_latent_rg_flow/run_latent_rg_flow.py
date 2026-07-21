"""
exp-089: Huginn Latent-Iteration RG Flow
Pre-registered 2026-07-20.

Extracts per-step attention patterns from Huginn's recurrent core block
and fits BCFT power-law to measure Δ as a function of recurrence depth.
"""

import torch
import math
import json
import numpy as np
from scipy import stats
from pathlib import Path
from datetime import datetime, timezone
import sys

# ── Configuration ────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
RESULTS_FILE = SCRIPT_DIR / "results.json"
MODEL_ID = "tomg-group-umd/huginn-0125"
DEVICE_STR = "mps" if torch.backends.mps.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)
TORCH_DTYPE = torch.bfloat16
SEED = 89
N_SEQS = 20
SEQ_LEN = 128
PROBE_STEPS = [1, 2, 4, 8, 16, 32]
N_LAYERS_CORE = 4       # n_layers_in_recurrent_block from config.json
N_HEADS = 55            # n_heads from config.json
HEAD_DIM = 96           # head_dim = n_embd // n_heads = 5280 // 55
SYK_WINDOW = (0.20, 0.30)
R2_THRESHOLD = 0.90
CUTOFF_LOW = 3          # minimum lag for power-law fit (match prior experiments)
MAX_LAG = 60
VOCAB_SIZE = 65536

# ── RoPE (match Huginn's rotary embedding implementation) ─────────────────────

def precompute_freqs_cis(dim: int, end: int, theta: float = 50000.0):
    """RoPE frequencies — matches Huginn rope_base=50000."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis  # (end, dim/2)


def apply_rotary_emb(xq, xk, freqs_cis):
    """Apply RoPE to Q and K. freqs_cis: (S, head_dim/2)."""
    # xq, xk: (B, S, n_heads, head_dim)
    B, S, H, D = xq.shape
    # Reshape to complex
    xq_ = xq.float().reshape(B, S, H, D // 2, 2)
    xk_ = xk.float().reshape(B, S, H, D // 2, 2)
    xq_complex = torch.view_as_complex(xq_)  # (B, S, H, D/2)
    xk_complex = torch.view_as_complex(xk_)  # (B, S, H, D/2)
    # freqs_cis: (S, D/2) → (1, S, 1, D/2)
    freqs = freqs_cis[:S].unsqueeze(0).unsqueeze(2)  # (1, S, 1, D/2)
    xq_rot = torch.view_as_real(xq_complex * freqs).reshape(B, S, H, D)
    xk_rot = torch.view_as_real(xk_complex * freqs).reshape(B, S, H, D)
    return xq_rot.to(xq.dtype), xk_rot.to(xk.dtype)


# ── BCFT fitting ──────────────────────────────────────────────────────────────

def compute_lag_profile(attn_matrix):
    """
    Compute mean attention weight as a function of positional lag.
    attn_matrix: (S, S) causal attention weights (rows=query, cols=key)
    Returns: (lags_array, means_array) for lags cutoff_low..max_lag
    """
    S = attn_matrix.shape[0]
    max_lag_actual = min(MAX_LAG, S - 1)
    lags, means = [], []
    for lag in range(CUTOFF_LOW, max_lag_actual + 1):
        # diagonal elements at distance lag below main diagonal
        diag = attn_matrix[lag:, :S - lag]  # (S-lag, S-lag) → we want the off-diag
        vals = np.diag(attn_matrix, -lag)  # (S-lag,) elements at lag below main diag
        if len(vals) > 0:
            lags.append(lag)
            means.append(float(np.mean(vals)))
    return np.array(lags), np.array(means)


def fit_delta(lags, means):
    """Fit power law A * lag^(-2Δ) in log-log space. Returns (Δ, R²)."""
    mask = means > 1e-12
    if mask.sum() < 5:
        return None, None
    log_lags = np.log(lags[mask])
    log_means = np.log(means[mask])
    try:
        slope, intercept, r, p, se = stats.linregress(log_lags, log_means)
    except Exception:
        return None, None
    delta = -slope / 2.0
    r2 = r ** 2
    return float(delta), float(r2)


# ── Manual attention computation from QK weights ─────────────────────────────

def compute_head_attentions_manual(module, x_normed, freqs_cis):
    """
    Compute per-head attention weight matrices from CausalSelfAttention.
    x_normed: (B, S, E) — normalized input to attention
    freqs_cis: (S, head_dim/2) — RoPE frequencies for this sequence
    Returns: np.ndarray of shape (n_heads, S, S)
    """
    B, S, E = x_normed.shape
    with torch.no_grad():
        # QKV
        qkv = module.Wqkv(x_normed)  # (B, S, (n_head + 2*n_kv_heads)*head_dim)
        q, k, v = qkv.split(module.chunks, dim=2)

        q = q.view(B, S, module.n_head, module.head_dim)
        k = k.view(B, S, module.n_kv_heads, module.head_dim)

        # QK bias
        if module.config.qk_bias:
            q_bias, k_bias = module.qk_bias.split(1, dim=0)
            q = q + q_bias.to(q.dtype)
            k = k + k_bias.to(k.dtype)

        # RoPE
        q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis)

        q_rot = q_rot.transpose(1, 2).float()  # (B, n_head, S, head_dim)
        k_rot = k_rot.transpose(1, 2).float()  # (B, n_kv_heads, S, head_dim)

        scale = 1.0 / math.sqrt(module.head_dim)
        scores = torch.matmul(q_rot, k_rot.transpose(-2, -1)) * scale  # (B, n_head, S, S)

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(S, S, dtype=torch.bool, device=scores.device), diagonal=1
        )
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)  # (B, n_head, S, S)

    # Return first element of batch
    return attn_weights[0].cpu().numpy()  # (n_head, S, S)


# ── Main extraction ───────────────────────────────────────────────────────────

def extract_per_step_deltas(model, input_ids, condition_label, freqs_cis_full):
    """
    Run model with num_steps=max(PROBE_STEPS) and extract per-step attention.
    Returns list of dicts: {step, layer, head, delta, r2, syk_near, condition}
    """
    max_steps = max(PROBE_STEPS)
    S = input_ids.shape[1]
    freqs_cis = freqs_cis_full[:, :S]  # (1, S, head_dim) → trim to seq len

    # Step tracking
    step_counter = [0]  # mutable for closure
    call_counter = [0]  # counts total hook calls; groups of N_LAYERS_CORE = one step
    collected = []  # list of (step_idx, layer_idx, attn_matrices) tuples
    freqs_cis_hook = [None]  # will be set from model

    def make_core_attn_hook(layer_idx):
        def hook_fn(module, inputs, output):
            # inputs: (x_normed, freqs_cis, block_idx, mask, past_key_values)
            x_normed = inputs[0]
            fc = inputs[1]  # freqs_cis used in this forward pass
            step_idx = call_counter[0] // N_LAYERS_CORE
            call_counter[0] += 1

            if step_idx + 1 not in PROBE_STEPS:
                return  # only collect at probe steps

            # Compute attention patterns manually
            attn = compute_head_attentions_manual(module, x_normed.detach(), fc.detach())
            collected.append((step_idx + 1, layer_idx, attn))  # step_idx is 0-indexed, step is 1-indexed

        return hook_fn

    # Register hooks on core block attention modules
    hooks = []
    for li, block in enumerate(model.transformer.core_block):
        h = block.attn.register_forward_hook(make_core_attn_hook(li))
        hooks.append(h)

    # Forward pass
    with torch.no_grad():
        # num_steps as plain int: Huginn's iterate_forward calls len(num_steps)
        # on anything with __len__; a 0-d tensor has the attr but raises TypeError.
        _ = model(
            input_ids.to(DEVICE),
            num_steps=max_steps,
            output_details={"return_logits": False, "return_latents": False, "return_head": False, "return_stats": False},
        )

    # Remove hooks
    for h in hooks:
        h.remove()

    # Fit BCFT for each collected (step, layer, head)
    results = []
    for (step, layer_idx, attn_mat) in collected:
        # attn_mat: (n_heads, S, S)
        for head_idx in range(N_HEADS):
            head_attn = attn_mat[head_idx]  # (S, S)
            lags, means = compute_lag_profile(head_attn)
            if len(lags) < 5:
                continue
            delta, r2 = fit_delta(lags, means)
            if delta is None:
                continue
            syk_near = (
                SYK_WINDOW[0] <= delta <= SYK_WINDOW[1]
                and r2 >= R2_THRESHOLD
            )
            results.append({
                "step": step,
                "layer": layer_idx,
                "head": head_idx,
                "delta": round(delta, 6),
                "r2": round(r2, 6),
                "syk_near": syk_near,
                "condition": condition_label,
            })

    return results


def main():
    print(f"exp-089: Huginn Latent-Iteration RG Flow")
    print(f"Device: {DEVICE_STR}")
    print(f"Probe steps: {PROBE_STEPS}")
    print(f"Seed: {SEED}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── Load model and tokenizer ──────────────────────────────────────────────
    print(f"\nLoading model: {MODEL_ID}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=TORCH_DTYPE,
        trust_remote_code=True,
    )
    model.eval()
    model.to(DEVICE)
    print(f"Model loaded. n_layers_in_recurrent_block={model.config.n_layers_in_recurrent_block}, "
          f"n_heads={model.config.n_heads}, head_dim={model.config.head_dim}")

    # Precompute RoPE frequencies
    freqs_cis_full = model.freqs_cis  # already on device, shape (1, block_size, head_dim/2) or similar
    # The model stores freqs_cis as a buffer — let's check shape
    print(f"freqs_cis shape: {freqs_cis_full.shape}")

    # ── Prepare inputs ────────────────────────────────────────────────────────
    print(f"\nPreparing inputs (N={N_SEQS}, seq_len={SEQ_LEN})")

    # Natural text: TinyStories
    try:
        from datasets import load_dataset
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        nat_texts = []
        for item in ds:
            nat_texts.append(item["text"])
            if len(nat_texts) >= N_SEQS * 3:  # oversample
                break
        nat_ids_list = []
        for text in nat_texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) >= SEQ_LEN:
                nat_ids_list.append(torch.tensor(ids[:SEQ_LEN]).unsqueeze(0))
            if len(nat_ids_list) >= N_SEQS:
                break
        nat_input_ids = torch.cat(nat_ids_list[:N_SEQS], dim=0)  # (N_SEQS, SEQ_LEN)
        print(f"Natural text inputs: {nat_input_ids.shape}")
    except Exception as e:
        print(f"TinyStories unavailable ({e}), using wikitext fallback")
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
        nat_texts = []
        for item in ds:
            if len(item["text"]) > 200:
                nat_texts.append(item["text"])
            if len(nat_texts) >= N_SEQS * 3:
                break
        nat_ids_list = []
        for text in nat_texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) >= SEQ_LEN:
                nat_ids_list.append(torch.tensor(ids[:SEQ_LEN]).unsqueeze(0))
            if len(nat_ids_list) >= N_SEQS:
                break
        nat_input_ids = torch.cat(nat_ids_list[:N_SEQS], dim=0)
        print(f"Natural text inputs (wikitext fallback): {nat_input_ids.shape}")

    # Random tokens
    torch.manual_seed(SEED)
    rand_input_ids = torch.randint(0, VOCAB_SIZE, (N_SEQS, SEQ_LEN))
    print(f"Random token inputs: {rand_input_ids.shape}")

    # ── Run extraction ────────────────────────────────────────────────────────
    all_results = []

    print(f"\n--- Natural text condition ---")
    for seq_i in range(N_SEQS):
        if seq_i % 5 == 0:
            print(f"  Sequence {seq_i+1}/{N_SEQS}...")
        ids = nat_input_ids[seq_i:seq_i+1]  # (1, SEQ_LEN)
        seq_results = extract_per_step_deltas(model, ids, "nat", freqs_cis_full)
        all_results.extend(seq_results)

    print(f"\n--- Random token condition ---")
    for seq_i in range(N_SEQS):
        if seq_i % 5 == 0:
            print(f"  Sequence {seq_i+1}/{N_SEQS}...")
        ids = rand_input_ids[seq_i:seq_i+1]  # (1, SEQ_LEN)
        seq_results = extract_per_step_deltas(model, ids, "rand", freqs_cis_full)
        all_results.extend(seq_results)

    print(f"\nTotal head measurements: {len(all_results)}")

    # ── Analysis ──────────────────────────────────────────────────────────────
    print("\n=== ANALYSIS ===")

    def analyze_condition(condition):
        cond_results = [r for r in all_results if r["condition"] == condition]
        per_step = {}
        for r in cond_results:
            s = r["step"]
            if s not in per_step:
                per_step[s] = {"deltas": [], "syk_near": []}
            per_step[s]["deltas"].append(r["delta"])
            per_step[s]["syk_near"].append(1 if r["syk_near"] else 0)

        step_nums = sorted(per_step.keys())
        delta_meds = [float(np.median(per_step[s]["deltas"])) for s in step_nums]
        syk_counts = [int(sum(per_step[s]["syk_near"])) for s in step_nums]
        n_per_step = [len(per_step[s]["deltas"]) for s in step_nums]

        print(f"\nCondition: {condition}")
        print(f"{'Step':>6} {'Δ_med':>8} {'SYK-near':>10} {'N':>6}")
        for i, s in enumerate(step_nums):
            print(f"{s:>6} {delta_meds[i]:>8.4f} {syk_counts[i]:>10} {n_per_step[i]:>6}")

        if len(step_nums) >= 3:
            rho_conv, _ = stats.spearmanr(step_nums, delta_meds)
            rho_emerg, _ = stats.spearmanr(step_nums, syk_counts)
            print(f"Spearman ρ(step, Δ_med) = {rho_conv:.4f}")
            print(f"Spearman ρ(step, syk_count) = {rho_emerg:.4f}")
        else:
            rho_conv, rho_emerg = None, None

        return {
            "condition": condition,
            "step_nums": step_nums,
            "delta_meds": delta_meds,
            "syk_counts_per_step": syk_counts,
            "n_per_step": n_per_step,
            "rho_convergence": rho_conv,
            "rho_emergence": rho_emerg,
        }

    nat_summary = analyze_condition("nat")
    rand_summary = analyze_condition("rand")

    # Verdict
    print("\n=== VERDICT ===")
    h_emergence_nat = nat_summary["rho_emergence"] is not None and nat_summary["rho_emergence"] > 0.5
    h_convergence_nat = nat_summary["rho_convergence"] is not None and nat_summary["rho_convergence"] < -0.5
    kill = (
        nat_summary["rho_emergence"] is not None and nat_summary["rho_emergence"] < 0.3
        and nat_summary["rho_convergence"] is not None and nat_summary["rho_convergence"] > -0.3
    )

    if kill:
        verdict = "FALSIFIED"
        print("H_emergence + H_convergence both fail → RG flow NOT confirmed at inference time")
    elif h_emergence_nat and h_convergence_nat:
        verdict = "CONFIRMED"
        print("H_emergence KEEP + H_convergence KEEP → RG flow confirmed at inference time")
    elif h_emergence_nat or h_convergence_nat:
        verdict = "PARTIAL"
        print(f"H_emergence {'KEEP' if h_emergence_nat else 'FAIL'}, H_convergence {'KEEP' if h_convergence_nat else 'FAIL'} → PARTIAL")
    else:
        verdict = "INCONCLUSIVE"
        print("Neither primary hypothesis confirmed, not within kill criterion")

    print(f"Verdict: {verdict}")

    # H_rand_contrast
    if nat_summary["rho_emergence"] is not None and rand_summary["rho_emergence"] is not None:
        contrast_emergence = nat_summary["rho_emergence"] > rand_summary["rho_emergence"]
        contrast_convergence = abs(nat_summary["rho_convergence"] or 0) > abs(rand_summary["rho_convergence"] or 0)
        print(f"H_rand_contrast: emergence contrast {'KEEP' if contrast_emergence else 'FAIL'}, "
              f"convergence contrast {'KEEP' if contrast_convergence else 'FAIL'}")

    # ── Save results ──────────────────────────────────────────────────────────
    output = {
        "experiment": "exp-089",
        "model": MODEL_ID,
        "date": datetime.now(timezone.utc).isoformat(),
        "config": {
            "probe_steps": PROBE_STEPS,
            "n_seqs": N_SEQS,
            "seq_len": SEQ_LEN,
            "seed": SEED,
            "syk_window": list(SYK_WINDOW),
            "r2_threshold": R2_THRESHOLD,
            "cutoff_low": CUTOFF_LOW,
            "max_lag": MAX_LAG,
        },
        "verdict": verdict,
        "nat_summary": nat_summary,
        "rand_summary": rand_summary,
        "raw_results": all_results,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {RESULTS_FILE}")
    print(f"Total measurements: {len(all_results)}")


if __name__ == "__main__":
    main()
