"""
exp-136 — J_eff threshold vs. formation phase transition (Pythia-70m)

Pre-registration: commit 425beb6 (pushed before this script was written).
Analysis-only: extracts weight matrices and embedding vectors from
Pythia-70m checkpoints; no inference, no new training.

Protocol exactly as specified in prereg.md.
"""

import json
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM
from scipy import stats

# ── Configuration ────────────────────────────────────────────────────────────

CHECKPOINTS = [0, 1, 4, 16, 64, 256, 1000, 4000, 16000, 64000, 143000]
N_TOKENS = 500   # number of sampled token indices
SEED = 136       # fixed across all checkpoints and heads (pre-registered)
MODEL_ID = "EleutherAI/pythia-70m"

# exp-086 n_syk_near results (random-token census, per-checkpoint)
# Source: exp-086_longitudinal_delta_spectrum/results.json
# Loaded below from the results file to avoid hard-coding stale numbers.
EXP086_RESULTS_PATH = Path(__file__).parent.parent / "exp-086_longitudinal_delta_spectrum" / "results.json"

# ── J_eff computation ─────────────────────────────────────────────────────────

def compute_jeff_squared(model, token_indices):
    """
    Compute J_eff² for every (layer, head) in the model.

    Returns:
        jeff_sq: np.array of shape [n_layers, n_heads]
        sigma_E2: scalar (embedding variance proxy / kinetic term)
    """
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    d_model = cfg.hidden_size
    d_k = d_model // n_heads  # 64

    jeff_sq = np.zeros((n_layers, n_heads))

    # Embedding vectors for sampled tokens
    wte = model.gpt_neox.embed_in.weight.detach().float()  # [V, d_model]
    x = wte[token_indices]  # [N, d_model]
    N = x.shape[0]

    # Embedding scale proxy (kinetic term)
    sigma_E2 = (wte ** 2).mean().item()

    for ell in range(n_layers):
        layer = model.gpt_neox.layers[ell]
        qkv_weight = layer.attention.query_key_value.weight.detach().float()
        # W_K occupies rows [d_model : 2*d_model]
        W_K_all = qkv_weight[d_model:2*d_model, :]  # [d_model, d_model]

        for h in range(n_heads):
            W_K_h = W_K_all[h*d_k:(h+1)*d_k, :]  # [d_k, d_model]

            # Key activations: k_a = (W_K_h @ x_a) / sqrt(d_k)
            k = (W_K_h @ x.T).T / (d_k ** 0.5)  # [N, d_k]

            # Raw key gram matrix
            K = k @ k.T  # [N, N]

            # Doubly-center K
            row_means = K.mean(dim=1, keepdim=True)
            col_means = K.mean(dim=0, keepdim=True)
            grand_mean = K.mean()
            dK = K - row_means - col_means + grand_mean  # [N, N]

            # Ω̂ = ||K @ δK||_F² / N²
            KdK = K @ dK
            Omega_hat = (KdK ** 2).sum().item() / (N ** 2)

            # σ_K² = mean squared entry of W_K_h
            sigma_K2 = (W_K_h ** 2).mean().item()

            jeff_sq[ell, h] = (sigma_K2 ** 2) * Omega_hat

    return jeff_sq, sigma_E2


def load_exp086_nsyk():
    """Load exp-086 n_syk_near series (random token, per checkpoint).
    
    Returns dict {step: n_syk_near} or None if not found.
    """
    if not EXP086_RESULTS_PATH.exists():
        print(f"Warning: exp-086 results not found at {EXP086_RESULTS_PATH}")
        return None
    with open(EXP086_RESULTS_PATH) as f:
        data = json.load(f)
    # Structure: data['results_rand'] = [{step, n_syk_near, delta_median, ...}, ...]
    results_rand = data.get("results_rand", [])
    step_to_nsyk = {}
    for item in results_rand:
        s = item.get("step")
        n = item.get("n_syk_near")
        if s is not None and n is not None:
            step_to_nsyk[s] = n
    return step_to_nsyk


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(SEED)
    # Sample token indices once, fixed for all checkpoints and heads
    # Pythia-70m vocab size = 50304 (padded from 50257)
    token_indices_np = rng.integers(0, 50257, size=N_TOKENS)
    token_indices = torch.tensor(token_indices_np, dtype=torch.long)

    results = {
        "experiment": "exp-136",
        "prereg_commit": "425beb6",
        "model": MODEL_ID,
        "n_tokens": N_TOKENS,
        "seed": SEED,
        "checkpoints": CHECKPOINTS,
        "per_checkpoint": []
    }

    print(f"Running exp-136: J_eff threshold test on {MODEL_ID}")
    print(f"  N={N_TOKENS} tokens, seed={SEED}, {len(CHECKPOINTS)} checkpoints")
    print()

    for step in CHECKPOINTS:
        revision = f"step{step}" if step > 0 else "step0"
        print(f"  Loading step={step} ({revision})...", end=" ", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=revision,
            dtype=torch.float32,
            trust_remote_code=True
        )
        model.eval()
        print("done.")

        with torch.no_grad():
            jeff_sq, sigma_E2 = compute_jeff_squared(model, token_indices)

        n_layers, n_heads = jeff_sq.shape
        jeff_sq_mean = float(jeff_sq.mean())
        jeff_sq_max = float(jeff_sq.max())
        jeff_sq_by_layer = jeff_sq.mean(axis=1).tolist()

        entry = {
            "step": step,
            "jeff_sq_mean": jeff_sq_mean,
            "jeff_sq_max": jeff_sq_max,
            "jeff_sq_by_layer": jeff_sq_by_layer,
            "jeff_sq_all": jeff_sq.tolist(),  # [n_layers, n_heads]
            "sigma_E2": sigma_E2,
        }
        results["per_checkpoint"].append(entry)

        print(f"    J_eff²_mean = {jeff_sq_mean:.4e},  σ_E² = {sigma_E2:.4e}")

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Hypothesis evaluation ─────────────────────────────────────────────────

    steps = [e["step"] for e in results["per_checkpoint"]]
    jeff_means = [e["jeff_sq_mean"] for e in results["per_checkpoint"]]
    jeff_means_arr = np.array(jeff_means)
    steps_arr = np.array(steps)
    sigma_E2s = [e["sigma_E2"] for e in results["per_checkpoint"]]

    print()
    print("=== Hypothesis Evaluation ===")

    # H1: Spearman ρ(step, J_eff²_mean)
    rho_h1, p_h1 = stats.spearmanr(steps_arr, jeff_means_arr)
    h1_passed = rho_h1 >= 0.80
    print(f"H1 (monotone growth): ρ = {rho_h1:.3f} (p={p_h1:.3e}) → {'CONFIRMED' if h1_passed else 'FALSIFIED' if rho_h1 < 0.50 else 'PARTIAL'}")

    # H2: First 2× crossing of J_eff²_mean
    j0 = jeff_means_arr[0]  # step=0
    print(f"  J_eff²(step=0) = {j0:.4e}")
    threshold = 2.0 * j0
    crossing_idx = None
    for i, (s, j) in enumerate(zip(steps_arr, jeff_means_arr)):
        if j > threshold:
            crossing_idx = i
            crossing_step = int(s)
            break
    if crossing_idx is not None:
        h2_passed = crossing_step in [64, 256, 1000, 4000]
        h2_status = "CONFIRMED" if h2_passed else ("FALSIFIED" if crossing_step not in range(32, 8001) else "PARTIAL")
        print(f"H2 (2× crossing): first s* = step {crossing_step} → {h2_status}")
    else:
        h2_status = "FALSIFIED"
        crossing_step = None
        print(f"H2 (2× crossing): J_eff² never doubles → FALSIFIED (K2)")

    # H3: Correlation with exp-086 n_syk_near
    step_to_nsyk = load_exp086_nsyk()
    rho_h3 = None
    p_h3 = None
    h3_status = "NOT_RUN"
    common_steps = []
    if step_to_nsyk is not None:
        print(f"  exp-086 n_syk_near loaded: {len(step_to_nsyk)} steps")
        common_steps = [s for s in steps if s in step_to_nsyk]
        if len(common_steps) >= 5:
            j_common = np.array([jeff_means_arr[steps.index(s)] for s in common_steps])
            n_common = np.array([step_to_nsyk[s] for s in common_steps])
            rho_h3, p_h3 = stats.spearmanr(np.log1p(j_common), n_common)
            h3_passed = rho_h3 >= 0.70
            h3_status = "CONFIRMED" if h3_passed else ("FALSIFIED" if rho_h3 < 0.50 else "PARTIAL")
            print(f"H3 (correlation with n_syk_near): ρ = {rho_h3:.3f} (p={p_h3:.3e}, {len(common_steps)} steps) → {h3_status}")
            print(f"  n_syk_near by step: {dict(zip(common_steps, n_common.tolist()))}")
        else:
            print(f"  Only {len(common_steps)} common steps — H3 not evaluable")
            h3_status = "NOT_EVALUABLE"

    # Normalized growth curve
    R = jeff_means_arr / jeff_means_arr[0]
    print()
    print("Normalized J_eff²(s) / J_eff²(0):")
    for s, r in zip(steps_arr, R):
        marker = " ← 2× crossing" if s == crossing_step else ""
        print(f"  step {s:>7d}: R = {r:.2f}{marker}")

    # ── Save results ──────────────────────────────────────────────────────────

    results["hypotheses"] = {
        "H1": {
            "spearman_rho": float(rho_h1),
            "p_value": float(p_h1),
            "criterion": "rho >= 0.80",
            "verdict": "CONFIRMED" if h1_passed else ("FALSIFIED" if rho_h1 < 0.50 else "PARTIAL"),
        },
        "H2": {
            "first_2x_crossing_step": crossing_step,
            "threshold": float(threshold),
            "criterion": "s* in {64, 256, 1000, 4000}",
            "verdict": h2_status,
        },
        "H3": {
            "spearman_rho": float(rho_h3) if rho_h3 is not None else None,
            "p_value": float(p_h3) if p_h3 is not None else None,
            "n_common_steps": len(common_steps),
            "criterion": "rho >= 0.70",
            "verdict": h3_status,
        },
    }
    results["normalized_growth"] = {str(int(s)): float(r) for s, r in zip(steps_arr, R)}
    results["sigma_E2_series"] = {str(int(e["step"])): float(e["sigma_E2"]) for e in results["per_checkpoint"]}

    out_path = Path(__file__).parent / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
