"""
exp-086 — Longitudinal Δ-spectrum on Fixed Input (Pythia-70m Training Trajectory)

Pre-registration: notes.md (committed bdb2e8b8, 2026-07-17).

Tests:
  H_mono_rand: SYK-near head count increases monotonically with training steps
               for fixed random-token inputs. Criterion: Spearman ρ ≥ 0.70.
  H_comparison: n_syk_near(random) ≥ n_syk_near(natural) at step 143000.
  H_mono_nat:  Same monotone test for fixed natural-language input.

Ariel — July 17, 2026.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

# ─── constants (pre-registered in notes.md) ──────────────────────────────────

MODEL_NAME = "EleutherAI/pythia-70m"

CHECKPOINTS = [0, 1, 4, 16, 64, 256, 1000, 4000, 16000, 64000, 143000]

SEQ_LEN = 256
N_RAND = 20      # number of random sequences
RAND_SEED = 86

# Measurement protocol (matching exp-083 / exp-062 census)
FIT_LOW = 3
FIT_HIGH = 50
R2_THRESHOLD = 0.85
SYK_LOW = 0.20
SYK_HIGH = 0.30

# Pre-registered natural language passage (from TinyStories training distribution)
NAT_TEXT = (
    "Once upon a time, there was a little girl named Lily. She lived with her family "
    "in a small house near a big forest. One day, Lily decided to go on an adventure "
    "in the forest. She put on her coat and her boots and walked into the trees. The "
    "forest was very quiet. She could hear the birds singing and the wind blowing "
    "through the leaves."
)

OUT = Path(__file__).resolve().parent


# ─── measurement helpers ─────────────────────────────────────────────────────

def compute_attention_decay(attn_head: np.ndarray, max_dx: int = FIT_HIGH + 5) -> np.ndarray:
    """Average A[i, i-dx] over positions i (position-space attention profile).
    
    Vectorized: uses np.diag(attn_head, -dx) which extracts the lower off-diagonal.
    For dx=0: main diagonal. For dx=k: k-th lower diagonal (i.e., attn[i, i-k]).
    """
    A = np.array([np.diag(attn_head, -dx).mean() if dx < attn_head.shape[0] else 0.0
                  for dx in range(max_dx)])
    return A


def fit_power_law(dx_arr: np.ndarray, y_arr: np.ndarray) -> tuple[float | None, float | None]:
    """Fit power-law A ~ dx^{-2Δ}. Returns (Δ, R²) or (None, None)."""
    mask = (
        (dx_arr >= FIT_LOW)
        & (dx_arr < FIT_HIGH)
        & (y_arr > 1e-20)
    )
    if mask.sum() < 5:
        return None, None
    log_dx = np.log(dx_arr[mask].astype(float))
    log_y = np.log(y_arr[mask])
    A = np.column_stack([np.ones_like(log_dx), log_dx])
    coeffs, _, _, _ = np.linalg.lstsq(A, log_y, rcond=None)
    pred = A @ coeffs
    ss_res = np.sum((log_y - pred) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0
    exponent = -coeffs[1]
    delta = exponent / 2.0
    return delta, R2


def measure_heads(
    model: torch.nn.Module,
    input_batch: list[torch.Tensor],
    n_layers: int,
    n_heads: int,
    dx_arr: np.ndarray,
) -> dict:
    """
    Run model on each input in input_batch, average attention profiles,
    and compute per-head Δ and R².

    Returns dict with:
      n_conformal, n_syk_near, deltas_conformal, delta_median, all_deltas
    """
    max_dx = FIT_HIGH + 5
    # Accumulate attention profiles: [layer][head] → np.ndarray of shape (max_dx,)
    A_sum = [[np.zeros(max_dx) for _ in range(n_heads)] for _ in range(n_layers)]

    for inp in input_batch:
        with torch.no_grad():
            out = model(inp.unsqueeze(0), output_attentions=True)
        for l_idx in range(n_layers):
            attn = out.attentions[l_idx][0].numpy()  # (n_heads, seq, seq)
            for h_idx in range(n_heads):
                A_sum[l_idx][h_idx] += compute_attention_decay(attn[h_idx], max_dx)

    n = len(input_batch)
    all_deltas = []
    deltas_conformal = []
    n_conformal = 0
    n_syk_near = 0

    for l_idx in range(n_layers):
        for h_idx in range(n_heads):
            A_avg = A_sum[l_idx][h_idx] / n
            delta, R2 = fit_power_law(dx_arr, A_avg)
            if delta is None:
                continue
            all_deltas.append(delta)
            if R2 >= R2_THRESHOLD:
                n_conformal += 1
                deltas_conformal.append(delta)
                if SYK_LOW <= delta <= SYK_HIGH:
                    n_syk_near += 1

    return {
        "n_conformal": n_conformal,
        "n_syk_near": n_syk_near,
        "deltas_conformal": deltas_conformal,
        "delta_median": float(np.median(deltas_conformal)) if deltas_conformal else None,
        "all_deltas": all_deltas,
    }


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  exp-086 — Longitudinal Δ-spectrum (Pythia-70m training trajectory)")
    print(f"  Checkpoints: {CHECKPOINTS}")
    print(f"  Random inputs: N={N_RAND}, seed={RAND_SEED}, seq_len={SEQ_LEN}")
    print(f"  SYK-near criterion: Δ ∈ [{SYK_LOW}, {SYK_HIGH}], R² ≥ {R2_THRESHOLD}")
    print("=" * 80)
    print()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ── pre-generate fixed inputs (pre-registration: done before any checkpoint loads) ──
    rng = np.random.default_rng(RAND_SEED)

    # Load tokenizer to get vocab size and tokenize the natural language passage
    print("Loading tokenizer...", end=" ", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    vocab_size = tokenizer.vocab_size
    print(f"vocab={vocab_size}")

    # Random inputs: N_RAND sequences of SEQ_LEN tokens each (fixed, pre-registered seed)
    rand_ids = rng.integers(0, vocab_size, size=(N_RAND, SEQ_LEN))
    rand_inputs = [torch.tensor(ids, dtype=torch.long) for ids in rand_ids]
    print(f"Fixed random inputs: {len(rand_inputs)} sequences, "
          f"first tokens: {rand_ids[0, :5].tolist()}...")

    # Natural language input: tokenize, truncate/pad to SEQ_LEN
    nat_enc = tokenizer.encode(NAT_TEXT)
    if len(nat_enc) > SEQ_LEN:
        nat_enc = nat_enc[:SEQ_LEN]
    else:
        nat_enc = nat_enc + [0] * (SEQ_LEN - len(nat_enc))
    nat_tensor = torch.tensor(nat_enc, dtype=torch.long)
    # Repeat N_RAND times for equivalent averaging
    nat_inputs = [nat_tensor] * N_RAND
    print(f"Fixed natural language input: {len(nat_enc)} tokens (padded), "
          f"first tokens: {nat_enc[:5]}...")
    print()

    dx_arr = np.arange(FIT_HIGH + 5)

    results_rand = []
    results_nat = []

    for step in CHECKPOINTS:
        revision = f"step{step}"
        print(f"─── checkpoint step={step} ({revision}) ───")

        t0 = time.time()
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                revision=revision,
                torch_dtype=torch.float32,
                attn_implementation="eager",
            )
            model.eval()
        except Exception as e:
            print(f"  FAILED to load: {e}")
            results_rand.append({"step": step, "n_syk_near": None, "error": str(e)})
            results_nat.append({"step": step, "n_syk_near": None, "error": str(e)})
            continue

        config = model.config
        n_layers = config.num_hidden_layers
        n_heads = config.num_attention_heads
        load_time = time.time() - t0

        print(f"  Loaded in {load_time:.1f}s  ({n_layers}L × {n_heads}H = {n_layers*n_heads} heads)")

        # Measure: random condition
        t1 = time.time()
        r_rand = measure_heads(model, rand_inputs, n_layers, n_heads, dx_arr)
        r_rand["step"] = step
        results_rand.append(r_rand)
        t_rand = time.time() - t1

        # Measure: natural language condition
        t2 = time.time()
        r_nat = measure_heads(model, nat_inputs, n_layers, n_heads, dx_arr)
        r_nat["step"] = step
        results_nat.append(r_nat)
        t_nat = time.time() - t2

        rand_med = f"{r_rand['delta_median']:.4f}" if r_rand['delta_median'] is not None else "N/A"
        nat_med  = f"{r_nat['delta_median']:.4f}"  if r_nat['delta_median']  is not None else "N/A"
        print(f"  RAND: n_syk_near={r_rand['n_syk_near']:2d}  n_conf={r_rand['n_conformal']:2d}  "
              f"Δ_med={rand_med}  ({t_rand:.1f}s)")
        print(f"  NAT:  n_syk_near={r_nat['n_syk_near']:2d}  n_conf={r_nat['n_conformal']:2d}  "
              f"Δ_med={nat_med}  ({t_nat:.1f}s)")
        print()

        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # ─── compute statistics ───────────────────────────────────────────────────

    print("=" * 80)
    print("  RESULTS SUMMARY")
    print("=" * 80)
    print()
    print(f"  {'Step':>8}  {'RAND n_syk':>12}  {'NAT n_syk':>12}  {'RAND−NAT':>10}")
    print("  " + "─" * 50)

    valid_rand = [(r["step"], r["n_syk_near"]) for r in results_rand if r["n_syk_near"] is not None]
    valid_nat  = [(r["step"], r["n_syk_near"]) for r in results_nat  if r["n_syk_near"] is not None]

    for step, n_r in valid_rand:
        n_n = dict(valid_nat).get(step)
        delta_str = f"{n_r - n_n:+d}" if n_n is not None else "N/A"
        nat_str = str(n_n) if n_n is not None else "N/A"
        print(f"  {step:8d}  {n_r:12d}  {nat_str:>12}  {delta_str:>10}")

    print()

    # H_mono_rand
    steps_r = [s for s, n in valid_rand]
    syk_r   = [n for s, n in valid_rand]
    log_steps = [np.log(s + 1) for s in steps_r]

    rho_rand, p_rand = spearmanr(log_steps, syk_r)
    print(f"  H_mono_rand: Spearman ρ = {rho_rand:.4f}  (p={p_rand:.4f})")
    if rho_rand >= 0.70:
        verdict_rand = "KEEP"
    elif rho_rand < 0.30:
        verdict_rand = "KILL"
    else:
        verdict_rand = "PARTIAL"
    print(f"  → {verdict_rand}")
    print()

    # H_mono_nat
    steps_n = [s for s, n in valid_nat]
    syk_n   = [n for s, n in valid_nat]
    log_steps_n = [np.log(s + 1) for s in steps_n]

    rho_nat, p_nat = spearmanr(log_steps_n, syk_n)
    print(f"  H_mono_nat:  Spearman ρ = {rho_nat:.4f}  (p={p_nat:.4f})")
    if rho_nat >= 0.70:
        verdict_nat = "KEEP"
    elif rho_nat < 0.30:
        verdict_nat = "KILL"
    else:
        verdict_nat = "PARTIAL"
    print(f"  → {verdict_nat}")
    print()

    # H_comparison (final checkpoint)
    n_rand_final = dict(valid_rand).get(143000)
    n_nat_final  = dict(valid_nat).get(143000)
    if n_rand_final is not None and n_nat_final is not None:
        comparison_keep = n_rand_final >= n_nat_final
        print(f"  H_comparison (step 143000): RAND={n_rand_final}, NAT={n_nat_final}")
        print(f"  → {'KEEP (RAND ≥ NAT)' if comparison_keep else 'FAIL (RAND < NAT)'}")
    else:
        comparison_keep = None
        print("  H_comparison: final checkpoint not available")
    print()

    # ─── write results.json ───────────────────────────────────────────────────

    results_out = {
        "experiment": "exp-086",
        "date": "2026-07-17",
        "model": MODEL_NAME,
        "checkpoints": CHECKPOINTS,
        "n_rand": N_RAND,
        "rand_seed": RAND_SEED,
        "seq_len": SEQ_LEN,
        "fit_low": FIT_LOW,
        "fit_high": FIT_HIGH,
        "r2_threshold": R2_THRESHOLD,
        "syk_low": SYK_LOW,
        "syk_high": SYK_HIGH,
        "results_rand": [
            {k: v for k, v in r.items() if k not in ("deltas_conformal", "all_deltas")}
            for r in results_rand
        ],
        "results_nat": [
            {k: v for k, v in r.items() if k not in ("deltas_conformal", "all_deltas")}
            for r in results_nat
        ],
        "statistics": {
            "spearman_rho_rand": float(rho_rand),
            "spearman_p_rand": float(p_rand),
            "spearman_rho_nat": float(rho_nat),
            "spearman_p_nat": float(p_nat),
            "n_syk_near_rand_final": n_rand_final,
            "n_syk_near_nat_final": n_nat_final,
            "comparison_keep": comparison_keep,
        },
        "verdicts": {
            "H_mono_rand": verdict_rand,
            "H_mono_nat": verdict_nat,
            "H_comparison": "KEEP" if comparison_keep else ("FAIL" if comparison_keep is not None else "N/A"),
        },
    }

    out_path = OUT / "results.json"
    with open(out_path, "w") as f:
        json.dump(results_out, f, indent=2)
    print(f"  Wrote {out_path}")
    print()
    print("exp-086 DONE.")


if __name__ == "__main__":
    main()
