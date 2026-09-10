"""
exp-139 — Gauge-fixed recomputation: key Gram (exp-127 analogue) and
          σ_K² (exp-136 analogue) in the canonical QR gauge.

Pre-registration: attention-geometry 9ad9b7c (pushed before this script).
Analysis-only: GPT-2 small (already cached) + Pythia-70m checkpoints (already cached).

Canonical gauge: W_Q = U_Q R_Q (QR), then W_K^{can} = W_K R_Q^T,
  k^{can} = k R_Q^T  (GPT-2 convention, W_Q shape n_embd × d_k)
  k^{can} = k R_R    (Pythia convention, W_Q_h shape d_k × d_model,
                      QR of W_Q_h^T = Q_R R_R → W_K^{can} = R_R^T @ W_K_h)
"""

import json
import numpy as np
import torch
from pathlib import Path
from transformers import GPT2Model, GPT2Tokenizer, AutoModelForCausalLM
from datasets import load_dataset
from scipy.stats import mannwhitneyu, spearmanr

# ── Configuration ─────────────────────────────────────────────────────────────

SEED = 42
N_SEQ = 100
SEQ_LEN = 128
D_K_GPT2 = 64
N_LAYERS_GPT2 = 12
N_HEADS_GPT2 = 12

# Population definitions (from exp-127 / exp-126)
WIKI_HEADS = [
    (4, 10), (7, 1), (8, 2), (9, 4), (9, 6),
    (10, 1), (10, 2), (10, 10),
    (11, 0), (11, 1), (11, 2), (11, 4), (11, 5), (11, 6), (11, 7), (11, 9),
]
STRUCTURAL_HEADS = [(2, 1), (3, 4), (5, 0), (7, 11), (10, 8)]
# Same control selection as exp-126 (seed 42)
rng_ctrl = np.random.default_rng(42)
all_heads = [(l, h) for l in range(12) for h in range(12)]
window_set = set(WIKI_HEADS)
non_window = [x for x in all_heads if x not in window_set]
ctrl_idxs = sorted(rng_ctrl.choice(len(non_window), 16, replace=False).tolist())
CONTROL_HEADS = [non_window[i] for i in ctrl_idxs]

# Pythia exp-136 configuration
PYTHIA_CHECKPOINTS = [0, 1, 4, 16, 64, 256, 1000, 4000, 16000, 64000, 143000]
PYTHIA_N_TOKENS = 500
PYTHIA_SEED = 136
PYTHIA_MODEL_ID = "EleutherAI/pythia-70m"

EXP086_PATH = Path(__file__).parent.parent / "exp-086_longitudinal_delta_spectrum" / "results.json"


# ── Part A helpers ─────────────────────────────────────────────────────────────

def extract_gpt2_weights_per_head(model):
    """
    Return dict: (layer, head) -> (W_Q_h, W_K_h) each of shape (n_embd, d_k).
    GPT-2 Conv1D weight: shape (n_embd, 3*n_embd). Columns = output features.
    Q: cols [0 : n_embd], K: cols [n_embd : 2*n_embd].
    Head h: cols [h*d_k : (h+1)*d_k] within each third.
    """
    n_embd = 768
    d_k = D_K_GPT2
    weights = {}
    for ell in range(N_LAYERS_GPT2):
        w = model.h[ell].attn.c_attn.weight.detach().float()  # (n_embd, 3*n_embd)
        for h in range(N_HEADS_GPT2):
            W_Q_h = w[:, h * d_k: (h + 1) * d_k]          # (n_embd, d_k)
            W_K_h = w[:, n_embd + h * d_k: n_embd + (h + 1) * d_k]  # (n_embd, d_k)
            weights[(ell, h)] = (W_Q_h.numpy(), W_K_h.numpy())
    return weights


def canonical_R_from_W_Q(W_Q_h):
    """
    QR decomposition of W_Q_h (n_embd × d_k): W_Q_h = U_Q R_Q.
    Returns R_Q (d_k × d_k, upper triangular).
    Canonical key transform: k^{can} = k @ R_Q^T.
    """
    # Use thin QR (reduced)
    _, R_Q = np.linalg.qr(W_Q_h, mode='reduced')  # Q: (n_embd, d_k), R: (d_k, d_k)
    # Ensure positive diagonal (unique QR)
    signs = np.sign(np.diag(R_Q))
    signs[signs == 0] = 1.0
    R_Q = signs[:, None] * R_Q  # make diagonal positive
    return R_Q  # (d_k, d_k)


def compute_eig_metrics(eigs):
    """
    Same formula as exp-127: λ₁/Σλ and supra-MP fraction.
    eigs: 1D array of eigenvalues (from 128×128 Gram K_can K_can^T / d_k).
    K was 128 rows × 64 cols → p=64, n=128, MP upper edge = σ²(1+√(p/n))².
    """
    eigs = np.clip(eigs, 0, None)
    lam_sum = eigs.sum()
    lam_max = eigs.max()
    top_share = lam_max / lam_sum if lam_sum > 1e-12 else np.nan
    p, n_ = 64, 128
    sigma2 = lam_sum / p
    lam_plus = sigma2 * (1.0 + np.sqrt(p / n_)) ** 2
    supra_mp = float(np.sum(eigs > lam_plus)) / len(eigs)
    return {
        "top_share": float(top_share),
        "supra_mp_fraction": float(supra_mp),
        "lam_sum": float(lam_sum),
        "lam_max": float(lam_max),
        "lam_plus": float(lam_plus),
    }


# ── Part A — GPT-2 small canonical key Gram ───────────────────────────────────

def run_part_a():
    print("=" * 70)
    print("Part A — GPT-2 small: canonical key Gram eigenspectrum")
    print("=" * 70)

    print("Loading GPT-2 small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2Model.from_pretrained("gpt2")
    model.eval()
    print("  done.")

    # Extract per-head weights and compute R_Q for all heads
    print("Computing QR decompositions for all heads...")
    weights = extract_gpt2_weights_per_head(model)
    R_Q_per_head = {key: canonical_R_from_W_Q(wq) for key, (wq, _) in weights.items()}
    print("  done.")

    # Load WikiText-103 validation (same as exp-126)
    print("Loading WikiText-103 validation...")
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="validation")
    text = "\n".join(x for x in dataset["text"] if x.strip())
    tokens_full = tokenizer.encode(text)
    sequences = []
    for i in range(N_SEQ):
        start = i * SEQ_LEN
        end = start + SEQ_LEN
        if end > len(tokens_full):
            break
        sequences.append(tokens_full[start:end])
    print(f"  Using {len(sequences)} sequences × {SEQ_LEN} tokens.")

    # Determine all heads of interest
    all_heads_needed = set(WIKI_HEADS) | set(STRUCTURAL_HEADS) | set(CONTROL_HEADS)

    # Collect key activations per head via hooks
    # Store: (layer, head) -> list of K arrays (one per sequence, shape 128 × d_k)
    extracted_keys = {h: [] for h in all_heads_needed}
    handles = []

    def make_hook(layer_idx):
        def hook(module, input, output):
            # output: (batch=1, seq_len, 3*n_embd) from c_attn
            with torch.no_grad():
                qkv = output[0].detach().float()  # (seq_len, 3*n_embd)
                n_embd = 768
                k_all = qkv[:, n_embd: 2 * n_embd]  # (seq_len, n_embd)
                for h in range(N_HEADS_GPT2):
                    if (layer_idx, h) in all_heads_needed:
                        k_h = k_all[:, h * D_K_GPT2: (h + 1) * D_K_GPT2]  # (seq_len, d_k)
                        extracted_keys[(layer_idx, h)].append(k_h.numpy())
        return hook

    for ell in range(N_LAYERS_GPT2):
        if any((ell, h) in all_heads_needed for h in range(N_HEADS_GPT2)):
            h_ = model.h[ell].attn.c_attn
            handles.append(h_.register_forward_hook(make_hook(ell)))

    print("Running census forward passes...")
    with torch.no_grad():
        for seq_i, seq_tokens in enumerate(sequences):
            inp = torch.tensor([seq_tokens])
            model(inp)
    for h_ in handles:
        h_.remove()
    print("  Forward passes done.")

    # Compute canonical eigenvalues per head
    print("Computing canonical key Gram eigenvalues...")
    head_metrics = {}
    for key in all_heads_needed:
        R_Q = R_Q_per_head[key]  # (d_k, d_k)
        R_Qt = R_Q.T              # (d_k, d_k)
        seq_eigs = []
        for K_seq in extracted_keys[key]:  # K_seq: (128, 64)
            K_can = K_seq @ R_Qt  # (128, 64)
            G_can = K_can @ K_can.T / D_K_GPT2  # (128, 128)
            eigs = np.linalg.eigvalsh(G_can)  # ascending
            eigs = np.sort(eigs)[::-1]  # descending
            seq_eigs.append(eigs)
        mean_eigs = np.mean(seq_eigs, axis=0)
        head_metrics[key] = compute_eig_metrics(mean_eigs)

    print("  done.")

    # Aggregate by population
    def pop_stats(heads, metric):
        return [head_metrics[h][metric] for h in heads]

    wiki_ts = pop_stats(WIKI_HEADS, "top_share")
    struct_ts = pop_stats(STRUCTURAL_HEADS, "top_share")
    ctrl_ts = pop_stats(CONTROL_HEADS, "top_share")

    wiki_sm = pop_stats(WIKI_HEADS, "supra_mp_fraction")
    struct_sm = pop_stats(STRUCTURAL_HEADS, "supra_mp_fraction")
    ctrl_sm = pop_stats(CONTROL_HEADS, "supra_mp_fraction")

    # Statistical tests (same as exp-127)
    mw_ts = mannwhitneyu(wiki_ts, ctrl_ts, alternative="less")
    effect_ts = float(np.median(ctrl_ts) - np.median(wiki_ts))

    mw_sm = mannwhitneyu(wiki_sm, ctrl_sm, alternative="greater")
    effect_sm = float(np.median(wiki_sm) - np.median(ctrl_sm))

    mw_sw = mannwhitneyu(struct_ts, wiki_ts, alternative="greater")
    mw_sc = mannwhitneyu(struct_ts, ctrl_ts, alternative="greater")

    k1_fired = effect_ts < 0.05
    k2_fired = mw_ts.pvalue >= 0.05
    h1_confirmed = not k1_fired and not k2_fired

    # Report
    print()
    print("── λ₁/Σλ (top eigenvalue share, canonical gauge) ────────────────────")
    print(f"  Δ-window  (n={len(WIKI_HEADS)}): median={np.median(wiki_ts):.3f}  "
          f"range=[{min(wiki_ts):.3f}, {max(wiki_ts):.3f}]")
    print(f"  Structural(n={len(STRUCTURAL_HEADS)}): median={np.median(struct_ts):.3f}  "
          f"range=[{min(struct_ts):.3f}, {max(struct_ts):.3f}]")
    print(f"  Control   (n={len(CONTROL_HEADS)}): median={np.median(ctrl_ts):.3f}  "
          f"range=[{min(ctrl_ts):.3f}, {max(ctrl_ts):.3f}]")
    print(f"  P1 Mann-Whitney (window < control): U={mw_ts.statistic:.0f}, p={mw_ts.pvalue:.4f}")
    print(f"  Effect (median control − window): {effect_ts:.3f}")
    print()
    print("── Supra-MP fraction (canonical gauge) ──────────────────────────────")
    print(f"  Δ-window  : median={np.median(wiki_sm):.4f}")
    print(f"  Structural: median={np.median(struct_sm):.4f}")
    print(f"  Control   : median={np.median(ctrl_sm):.4f}")
    print(f"  P2 Mann-Whitney (window > control): U={mw_sm.statistic:.0f}, p={mw_sm.pvalue:.4f}")
    print(f"  Effect (median window − control): {effect_sm:.4f}")
    print()
    print("── Kill conditions ──────────────────────────────────────────────────")
    print(f"  K1 (effect < 0.05): FIRED={k1_fired}  (effect={effect_ts:.3f})")
    print(f"  K2 (P1 p ≥ 0.05) : FIRED={k2_fired}  (p={mw_ts.pvalue:.4f})")
    print(f"  H1: {'CONFIRMED' if h1_confirmed else 'FALSIFIED'}")
    print()
    print(f"  Struct>wiki: p={mw_sw.pvalue:.4f}  Struct>ctrl: p={mw_sc.pvalue:.4f}")

    # Comparison with exp-127 original
    EXP127_WIKI_TS = 0.507
    EXP127_CTRL_TS = 0.651
    EXP127_EFFECT = 0.144
    EXP127_P1_P = 0.0014
    print()
    print("── Comparison with exp-127 (gauge-dependent) ────────────────────────")
    print(f"  exp-127  Δ-window λ₁/Σλ median: {EXP127_WIKI_TS:.3f} → now: {np.median(wiki_ts):.3f}")
    print(f"  exp-127  Control  λ₁/Σλ median: {EXP127_CTRL_TS:.3f} → now: {np.median(ctrl_ts):.3f}")
    print(f"  exp-127  effect: {EXP127_EFFECT:.3f}  → now: {effect_ts:.3f}")
    print(f"  exp-127  P1 p:   {EXP127_P1_P:.4f}  → now: {mw_ts.pvalue:.4f}")

    return {
        "h1_confirmed": h1_confirmed,
        "k1_fired": k1_fired,
        "k2_fired": k2_fired,
        "wiki_top_share_median": float(np.median(wiki_ts)),
        "ctrl_top_share_median": float(np.median(ctrl_ts)),
        "struct_top_share_median": float(np.median(struct_ts)),
        "effect_top_share": effect_ts,
        "p1_mw_p": float(mw_ts.pvalue),
        "p2_mw_p": float(mw_sm.pvalue),
        "effect_supra_mp": effect_sm,
        "wiki_supra_mp_median": float(np.median(wiki_sm)),
        "ctrl_supra_mp_median": float(np.median(ctrl_sm)),
        "p3_struct_vs_wiki_p": float(mw_sw.pvalue),
        "p3_struct_vs_ctrl_p": float(mw_sc.pvalue),
        "head_metrics": {str(k): v for k, v in head_metrics.items()},
        "exp127_comparison": {
            "original_wiki_median": EXP127_WIKI_TS,
            "original_ctrl_median": EXP127_CTRL_TS,
            "original_effect": EXP127_EFFECT,
            "original_p1_p": EXP127_P1_P,
        },
    }


# ── Part B helpers ─────────────────────────────────────────────────────────────

def canonical_R_from_W_Q_pythia(W_Q_h_np):
    """
    W_Q_h_np: shape (d_k, d_model) — Pythia convention.
    QR of W_Q_h^T (d_model × d_k): W_Q_h^T = Q_R R_R.
    Returns R_R (d_k × d_k, upper triangular).
    Canonical key transform: k^{can} = k @ R_R
       (where k = (W_K_h @ x^T)^T = x @ W_K_h^T)
    """
    W_Q_h_T = W_Q_h_np.T  # (d_model, d_k)
    _, R_R = np.linalg.qr(W_Q_h_T, mode='reduced')  # Q: (d_model, d_k), R: (d_k, d_k)
    signs = np.sign(np.diag(R_R))
    signs[signs == 0] = 1.0
    R_R = signs[:, None] * R_R
    return R_R  # (d_k, d_k)


def compute_jeff_can_squared(model, token_indices):
    """
    Compute gauge-fixed J_eff_can² for every (layer, head) in Pythia-70m.
    """
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    d_model = cfg.hidden_size
    d_k = d_model // n_heads  # 64 for pythia-70m

    jeff_can_sq = np.zeros((n_layers, n_heads))
    sigma_K_can2_all = np.zeros((n_layers, n_heads))

    # Embedding vectors for sampled tokens
    wte = model.gpt_neox.embed_in.weight.detach().float()
    x = wte[token_indices].numpy()  # (N, d_model)
    N = x.shape[0]

    # Embedding scale proxy (same as exp-136, for normalization comparison)
    sigma_E2 = float((wte.numpy() ** 2).mean())

    for ell in range(n_layers):
        layer = model.gpt_neox.layers[ell]
        qkv_weight = layer.attention.query_key_value.weight.detach().float().numpy()
        # W_Q: rows [0 : d_model], W_K: rows [d_model : 2*d_model]
        W_Q_all = qkv_weight[0:d_model, :]          # (d_model, d_model)
        W_K_all = qkv_weight[d_model:2 * d_model, :]  # (d_model, d_model)

        for h in range(n_heads):
            W_Q_h = W_Q_all[h * d_k: (h + 1) * d_k, :]  # (d_k, d_model)
            W_K_h = W_K_all[h * d_k: (h + 1) * d_k, :]  # (d_k, d_model)

            # Canonical gauge transformation
            R_R = canonical_R_from_W_Q_pythia(W_Q_h)  # (d_k, d_k)

            # Canonical key weight: W_K_can = R_R^T @ W_K_h  (d_k × d_model)
            W_K_can = R_R.T @ W_K_h  # (d_k, d_model)

            # Canonical σ_K²
            sigma_K_can2 = float((W_K_can ** 2).mean())
            sigma_K_can2_all[ell, h] = sigma_K_can2

            # Canonical key activations
            k_can = (W_K_can @ x.T).T / (d_k ** 0.5)  # (N, d_k)

            # Raw key Gram (canonical)
            K_can_gram = k_can @ k_can.T  # (N, N)
            K_can_gram_t = torch.tensor(K_can_gram)

            # Double-center
            row_m = K_can_gram_t.mean(dim=1, keepdim=True)
            col_m = K_can_gram_t.mean(dim=0, keepdim=True)
            grand_m = K_can_gram_t.mean()
            dK_can = K_can_gram_t - row_m - col_m + grand_m

            # Ω̂_can
            KdK_can = K_can_gram_t @ dK_can
            Omega_can = (KdK_can ** 2).sum().item() / (N ** 2)

            jeff_can_sq[ell, h] = (sigma_K_can2 ** 2) * Omega_can

    return jeff_can_sq, sigma_K_can2_all, sigma_E2


# ── Part B — Pythia canonical J_eff² ─────────────────────────────────────────

def run_part_b():
    print("=" * 70)
    print("Part B — Pythia-70m: J_eff² with canonical σ_K² (gauge-fixed)")
    print("=" * 70)

    rng_p = np.random.default_rng(PYTHIA_SEED)
    token_indices_np = rng_p.integers(0, 50257, size=PYTHIA_N_TOKENS)
    token_indices = torch.tensor(token_indices_np, dtype=torch.long)

    # Load exp-086 n_syk_near
    n_syk_series = None
    if EXP086_PATH.exists():
        with open(EXP086_PATH) as f:
            d086 = json.load(f)
        results_rand = d086.get("results_rand", [])
        n_syk_series = {r["step"]: r["n_syk_near"] for r in results_rand if "step" in r}

    per_checkpoint = []
    jeff_can_means = []
    steps = []

    for step in PYTHIA_CHECKPOINTS:
        revision = f"step{step}" if step > 0 else "step0"
        print(f"  Loading step={step}...", end=" ", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            PYTHIA_MODEL_ID, revision=revision,
            dtype=torch.float32, trust_remote_code=True
        )
        model.eval()
        print("done.", flush=True)

        with torch.no_grad():
            jeff_can_sq, sigma_K_can2_all, sigma_E2 = compute_jeff_can_squared(
                model, token_indices
            )

        jeff_mean = float(jeff_can_sq.mean())
        jeff_can_means.append(jeff_mean)
        steps.append(step)

        # Load original exp-136 J_eff² for comparison
        entry = {
            "step": step,
            "jeff_can_sq_mean": jeff_mean,
            "jeff_can_sq_by_layer": jeff_can_sq.mean(axis=1).tolist(),
            "sigma_K_can2_mean": float(sigma_K_can2_all.mean()),
            "sigma_E2": sigma_E2,
        }
        per_checkpoint.append(entry)
        print(f"    J_eff_can²_mean={jeff_mean:.4e}  σ_K_can²_mean={entry['sigma_K_can2_mean']:.6f}")

        del model

    # Normalized growth (relative to step 0)
    j0 = jeff_can_means[0] if jeff_can_means[0] > 0 else 1e-30
    normalized = [j / j0 for j in jeff_can_means]
    print()
    print("── Normalized J_eff_can² growth ─────────────────────────────────────")
    print(f"  {'Step':>8}  {'R=J/J_0':>14}  {'n_syk_near':>12}")
    for s, r in zip(steps, normalized):
        ns = n_syk_series.get(s, "?") if n_syk_series else "?"
        print(f"  {s:>8}  {r:>14.4e}  {ns!s:>12}")

    # Spearman ρ
    rho_step, p_step = spearmanr(steps, jeff_can_means)
    k3_fired = rho_step < 0.50

    print()
    print("── Spearman correlation with training step ──────────────────────────")
    print(f"  ρ = {rho_step:.3f}  (p = {p_step:.2e})")
    print(f"  K3 (ρ < 0.50): FIRED={k3_fired}")
    print(f"  H2: {'CONFIRMED' if not k3_fired else 'FALSIFIED'}")

    # Correlation with n_syk_near if available
    rho_nsyk = None
    if n_syk_series:
        matched = [(jeff_can_means[i], n_syk_series[s]) for i, s in enumerate(steps) if s in n_syk_series]
        if matched:
            j_vals, n_vals = zip(*matched)
            rho_nsyk, p_nsyk = spearmanr(j_vals, n_vals)
            print(f"  ρ(J_eff_can², n_syk_near) = {rho_nsyk:.3f}  (p = {p_nsyk:.2e})")

    return {
        "h2_confirmed": not k3_fired,
        "k3_fired": k3_fired,
        "rho_step": float(rho_step),
        "p_step": float(p_step),
        "rho_nsyk": float(rho_nsyk) if rho_nsyk is not None else None,
        "steps": steps,
        "jeff_can_means": jeff_can_means,
        "normalized_growth": normalized,
        "per_checkpoint": per_checkpoint,
        "exp136_comparison": {
            "original_rho_step": 0.934,
            "original_rho_nsyk": 0.888,
        },
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print()
    print("exp-139: Gauge-fixed key Gram and σ_K² (canonical QR gauge)")
    print("Pre-registration: attention-geometry 9ad9b7c")
    print()

    results_a = run_part_a()
    print()
    results_b = run_part_b()

    # Overall verdict
    h1 = results_a["h1_confirmed"]
    h2 = results_b["h2_confirmed"]
    if h1 and h2:
        overall = "confirmed"
        headline = (
            "Both confirmed. exp-127 Δ-window key structure and exp-136 formation gate "
            "are preserved in the canonical QR gauge. The gauge issue is resolved."
        )
    elif h1 and not h2:
        overall = "partial"
        headline = (
            "H1 confirmed (key structure), H2 falsified (formation gate). "
            "exp-127 conclusion stands; exp-136 σ_K² interpretation requires revision."
        )
    elif not h1 and h2:
        overall = "partial"
        headline = (
            "H2 confirmed (formation gate), H1 falsified (key structure). "
            "exp-136 conclusion stands; exp-127 λ₁/Σλ signature requires revision."
        )
    else:
        overall = "falsified"
        headline = (
            "Both falsified. exp-127 key structure difference and exp-136 formation gate "
            "are both gauge-dependent. Both results require revision before being cited."
        )

    print()
    print("=" * 70)
    print("OVERALL VERDICT:", overall.upper())
    print()
    print(headline)
    print("=" * 70)

    # Save results
    results = {
        "experiment": "exp-139",
        "date": "2026-09-10",
        "prereg_commit": "attention-geometry 9ad9b7c",
        "overall_verdict": overall,
        "headline": headline,
        "part_a": results_a,
        "part_b": results_b,
    }

    out_path = Path(__file__).parent / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
