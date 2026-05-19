"""
BCFT Direct Comparison: Attention Geometry → Accuracy Prediction → Published Data

Measures Δ and λ from LongChat-13B-16K attention weights on a cloud GPU,
then uses those parameters to predict the LiTM accuracy curve shape and
compares to the Liu et al. (2023) published results.

The chain of reasoning:
  1. Measure Δ and λ from random-token attention patterns (no task involved)
  2. Use measured parameters in the BCFT functional form to predict accuracy shape
  3. Fit only baseline + amplitude (vertical scaling) to match published accuracy
  4. Report R² — how well the attention-derived shape matches task performance

If this works, it means attention geometry alone predicts task-level behavior.

Usage:
  modal run research/physics/bcft_cloud_comparison.py

Ariel — April 15, 2026
"""

import modal

app = modal.App("bcft-litm-comparison")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.40",
        "accelerate",
        "numpy",
        "scipy",
    )
)

vol = modal.Volume.from_name("hf-model-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
    volumes={"/cache": vol},
    memory=65536,
)
def measure_attention_geometry(
    model_name: str = "lmsys/longchat-13b-16k",
    seq_len: int = 512,
    n_inputs: int = 50,
):
    """Load model on A100, extract attention, fit Δ and λ per head."""
    import os
    os.environ["HF_HOME"] = "/cache"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/transformers"

    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoConfig

    print(f"Loading {model_name} (fp16, eager attention)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.eval()

    config = model.config
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    vocab_size = config.vocab_size

    print(f"  {n_layers} layers, {n_heads} heads, vocab {vocab_size}")
    print(f"  seq_len={seq_len}, n_inputs={n_inputs}")

    device = next(model.parameters()).device

    # Accumulate averaged attention per layer
    # Store on CPU to avoid GPU OOM
    attn_sum = [
        np.zeros((n_heads, seq_len, seq_len), dtype=np.float64)
        for _ in range(n_layers)
    ]

    for inp in range(n_inputs):
        input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)
        for l in range(n_layers):
            attn_sum[l] += outputs.attentions[l][0].cpu().float().numpy()
        del outputs
        torch.cuda.empty_cache()
        if (inp + 1) % 10 == 0:
            print(f"  Input {inp + 1}/{n_inputs}")

    for l in range(n_layers):
        attn_sum[l] /= n_inputs

    print("  Fitting per-head Δ and λ...")

    FIT_LOW = 3
    FIT_HIGH = min(120, seq_len // 3)
    DEEP_LO = seq_len // 2

    head_results = []

    for l in range(n_layers):
        for h in range(n_heads):
            A = attn_sum[l][h]  # (seq_len, seq_len)

            # === Δ from bulk power-law ===
            A_dx = np.zeros(FIT_HIGH)
            count = np.zeros(FIT_HIGH)
            for dx in range(FIT_HIGH):
                q_pos = np.arange(max(DEEP_LO, dx), seq_len)
                k_pos = q_pos - dx
                valid = k_pos >= 0
                if np.sum(valid) > 0:
                    vals = A[q_pos[valid], k_pos[valid]]
                    A_dx[dx] = np.mean(vals)
                    count[dx] = np.sum(valid)

            dx_arr = np.arange(FIT_LOW, FIT_HIGH)
            y = A_dx[FIT_LOW:FIT_HIGH]
            valid = y > 1e-15
            if np.sum(valid) < 10:
                continue

            dx_v = dx_arr[valid].astype(float)
            y_v = y[valid]
            log_dx = np.log(dx_v)
            log_y = np.log(y_v)

            A_mat = np.column_stack([np.ones_like(log_dx), log_dx])
            coeffs, _, _, _ = np.linalg.lstsq(A_mat, log_y, rcond=None)
            pred_log = A_mat @ coeffs
            ss_res = np.sum((log_y - pred_log) ** 2)
            ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
            R2 = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0
            delta = -coeffs[1] / 2
            C = np.exp(coeffs[0])

            if R2 < 0.80 or delta < 0.05:
                continue

            # === λ from boundary excess ===
            ratios = []
            etas = []
            for x_k in range(1, min(30, seq_len // 4)):
                q_pos = np.arange(max(DEEP_LO, x_k + FIT_LOW), seq_len)
                dx = q_pos - x_k
                A_meas = A[q_pos, x_k]
                A_pow = C * dx.astype(float) ** (-2 * delta)
                eta = dx.astype(float) ** 2 / (4.0 * q_pos.astype(float) * x_k)
                ok = (A_meas > 1e-15) & (A_pow > 1e-15)
                if np.sum(ok) > 0:
                    ratios.extend((A_meas[ok] / A_pow[ok]).tolist())
                    etas.extend(eta[ok].tolist())

            lambda_fit = 0.0
            lambda_R2 = 0.0
            if len(ratios) > 10:
                ratios_arr = np.array(ratios)
                etas_arr = np.array(etas)
                eta_delta = etas_arr ** delta
                excess = ratios_arr - 1.0
                denom = np.sum(eta_delta ** 2)
                if denom > 1e-30:
                    lambda_fit = np.sum(excess * eta_delta) / denom
                    pred_ex = lambda_fit * eta_delta
                    ss_r = np.sum((excess - pred_ex) ** 2)
                    ss_t = np.sum((excess - np.mean(excess)) ** 2)
                    lambda_R2 = 1 - ss_r / ss_t if ss_t > 1e-30 else 0

            # === U-shape profile ===
            Q_LO = 3 * seq_len // 4
            profile = np.zeros(seq_len)
            p_count = 0
            for x_q in range(Q_LO, seq_len):
                profile[:x_q + 1] += A[x_q, :x_q + 1]
                p_count += 1
            if p_count > 0:
                profile /= p_count

            third = seq_len // 3
            start_attn = np.mean(profile[1:third])
            middle_attn = np.mean(profile[third:2 * third])
            end_attn = np.mean(profile[2 * third:Q_LO])

            if min(start_attn, middle_attn, end_attn) < 1e-15:
                continue

            peak = max(start_attn, end_attn)
            valley_depth = 1 - middle_attn / peak
            u_amplitude = (start_attn + end_attn) / (2 * middle_attn)

            interior_lo = max(1, seq_len // 10)
            interior_hi = Q_LO - seq_len // 10
            if interior_hi > interior_lo:
                valley_pos = interior_lo + np.argmin(profile[interior_lo:interior_hi])
                valley_frac = valley_pos / Q_LO
            else:
                valley_frac = 0.5

            head_results.append({
                'layer': l, 'head': h,
                'delta': float(delta), 'R2': float(R2),
                'C': float(C),
                'lambda': float(lambda_fit), 'lambda_R2': float(lambda_R2),
                'valley_depth': float(valley_depth),
                'u_amplitude': float(u_amplitude),
                'start': float(start_attn),
                'middle': float(middle_attn),
                'end': float(end_attn),
                'valley_frac': float(valley_frac),
                'asymmetry': float(start_attn / end_attn),
            })

    vol.commit()

    print(f"\n  Found {len(head_results)} conformal heads "
          f"(out of {n_layers * n_heads} total)")

    return {
        'model': model_name,
        'seq_len': seq_len,
        'n_inputs': n_inputs,
        'n_layers': n_layers,
        'n_heads': n_heads,
        'heads': head_results,
    }


# ========== Liu et al. (2023) published accuracy data ==========

LIU_DATA = {
    'LongChat-13B-16K': {
        10: {'indices': [0, 4, 9],
             'accuracy': [72.1, 58.9, 58.5]},
        20: {'indices': [0, 4, 9, 14, 19],
             'accuracy': [68.6, 57.4, 55.3, 52.5, 55.0]},
        30: {'indices': [0, 4, 9, 14, 19, 24, 29],
             'accuracy': [57.4, 53.3, 51.4, 49.2, 48.6, 46.2, 46.0]},
    },
    'MPT-30B-Instruct': {
        10: {'indices': [0, 4, 9],
             'accuracy': [60.2, 56.2, 59.7]},
        20: {'indices': [0, 4, 9, 14, 19],
             'accuracy': [53.7, 51.8, 52.2, 52.7, 56.3]},
        30: {'indices': [0, 4, 9, 14, 19, 24, 29],
             'accuracy': [52.2, 51.2, 49.2, 49.8, 50.5, 50.2, 53.0]},
    },
}


def bcft_attention_curve(indices, n_docs, delta, lam):
    """BCFT-predicted attention weights from query to document positions."""
    import numpy as np
    x_q = n_docs + 0.5
    positions = np.array(indices) + 0.5

    attn = np.zeros(len(positions))
    for i, x_k in enumerate(positions):
        dx = x_q - x_k
        if dx <= 0 or x_k <= 0:
            continue
        eta = dx ** 2 / (4.0 * x_q * x_k)
        attn[i] = dx ** (-2 * delta) * (1.0 + lam * eta ** delta)

    return attn


def compare_to_published(delta, lam, model_label, liu_data):
    """
    Compare BCFT prediction (using measured Δ, λ) to published accuracy.
    Fits only baseline and amplitude (2 free params).
    """
    import numpy as np
    from scipy.stats import spearmanr

    print(f"\n  BCFT prediction using Δ={delta:.4f}, λ={lam:.4f}")
    print(f"  (measured from {model_label} attention weights)")
    print()

    for n_docs in sorted(liu_data.keys()):
        entry = liu_data[n_docs]
        indices = np.array(entry['indices'])
        acc = np.array(entry['accuracy'])

        attn = bcft_attention_curve(indices, n_docs, delta, lam)
        if attn.max() > 0:
            attn_norm = attn / attn.max()
        else:
            attn_norm = np.ones(len(indices)) / len(indices)

        # Linear regression: acc = baseline + amplitude * attn_norm
        A_mat = np.column_stack([np.ones(len(indices)), attn_norm])
        params, _, _, _ = np.linalg.lstsq(A_mat, acc, rcond=None)
        baseline, amplitude = params

        predicted = baseline + amplitude * attn_norm
        ss_res = np.sum((acc - predicted) ** 2)
        ss_tot = np.sum((acc - np.mean(acc)) ** 2)
        R2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0

        rho, p = spearmanr(predicted, acc)

        print(f"  {n_docs} documents:")
        print(f"    R² = {R2:.4f}  Spearman ρ = {rho:+.4f} (p={p:.4f})")
        print(f"    baseline = {baseline:.1f}, amplitude = {amplitude:.1f}")
        for i, (idx, m, pr) in enumerate(zip(indices, acc, predicted)):
            print(f"      doc {idx:>2d}: measured={m:5.1f}%  predicted={pr:5.1f}%  "
                  f"residual={m - pr:+5.1f}")
        print()


@app.local_entrypoint()
def main():
    import json
    import numpy as np
    from scipy.stats import spearmanr
    from scipy.optimize import differential_evolution

    # ========== Phase 1: Measure on GPU ==========
    print("=" * 80)
    print("  BCFT DIRECT COMPARISON: ATTENTION → ACCURACY PREDICTION")
    print("=" * 80)
    print()
    print("  Launching A100 job for LongChat-13B-16K...")
    print()

    results = measure_attention_geometry.remote(
        model_name="lmsys/longchat-13b-16k",
        seq_len=512,
        n_inputs=50,
    )

    # Save raw measurements
    out_path = "research/physics/bcft_longchat_measurements.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved raw measurements → {out_path}")

    # ========== Phase 2: Summarize attention geometry ==========
    heads = results['heads']
    if not heads:
        print("\n  ERROR: No conformal heads found! Cannot proceed.")
        return

    deltas = np.array([h['delta'] for h in heads])
    lambdas = np.array([h['lambda'] for h in heads])
    R2s = np.array([h['R2'] for h in heads])
    valleys = np.array([h['valley_depth'] for h in heads])
    vfracs = np.array([h['valley_frac'] for h in heads])

    print(f"\n{'='*80}")
    print(f"  LONGCHAT-13B-16K ATTENTION GEOMETRY")
    print(f"{'='*80}")
    print(f"  Total heads: {results['n_layers'] * results['n_heads']}")
    print(f"  Conformal heads (R² > 0.80): {len(heads)}")
    print()
    print(f"  Δ: median={np.median(deltas):.4f}, "
          f"mean={np.mean(deltas):.4f}, "
          f"IQR=[{np.percentile(deltas, 25):.4f}, {np.percentile(deltas, 75):.4f}]")
    print(f"  λ: median={np.median(lambdas):.4f}, "
          f"mean={np.mean(lambdas):.4f}, "
          f"IQR=[{np.percentile(lambdas, 25):.4f}, {np.percentile(lambdas, 75):.4f}]")
    print(f"  R²: median={np.median(R2s):.4f}")
    print(f"  Valley depth: median={np.median(valleys):.4f}")
    print(f"  Valley position: median={np.median(vfracs):.3f} of context")
    print()

    # Per-head correlation
    rho_dv, p_dv = spearmanr(deltas, valleys)
    print(f"  Per-head Spearman(Δ, valley_depth) = {rho_dv:+.4f} (p={p_dv:.4f})")

    # Per-layer breakdown
    print(f"\n  Per-layer summary:")
    print(f"  {'Layer':>7} {'N':>4} {'Δ_med':>8} {'λ_med':>8} {'Valley':>8}")
    print(f"  {'-'*40}")
    for l in range(results['n_layers']):
        layer_heads = [h for h in heads if h['layer'] == l]
        if layer_heads:
            ld = [h['delta'] for h in layer_heads]
            ll = [h['lambda'] for h in layer_heads]
            lv = [h['valley_depth'] for h in layer_heads]
            print(f"  {l:>7d} {len(layer_heads):>4d} "
                  f"{np.median(ld):>8.4f} {np.median(ll):>8.4f} "
                  f"{np.median(lv):>8.4f}")

    # ========== Phase 3: Compare to Liu et al. ==========
    print(f"\n{'='*80}")
    print(f"  COMPARISON TO LIU ET AL. (2023) PUBLISHED ACCURACY")
    print(f"{'='*80}")

    DELTA_MEAS = float(np.median(deltas))
    LAMBDA_MEAS = float(max(0.1, np.median(lambdas)))

    liu = LIU_DATA['LongChat-13B-16K']

    print(f"\n  A. MEASURED PARAMETERS (Δ={DELTA_MEAS:.4f}, λ={LAMBDA_MEAS:.4f})")
    print(f"     Fitting only baseline + amplitude (2 free params)")
    compare_to_published(DELTA_MEAS, LAMBDA_MEAS, "LongChat-13B-16K", liu)

    # Also try with positive-λ heads only
    pos_lambda = [h for h in heads if h['lambda'] > 0]
    if pos_lambda:
        ld_pos = np.median([h['delta'] for h in pos_lambda])
        ll_pos = np.median([h['lambda'] for h in pos_lambda])
        print(f"  B. POSITIVE-λ HEADS ONLY ({len(pos_lambda)} heads, "
              f"Δ={ld_pos:.4f}, λ={ll_pos:.4f})")
        compare_to_published(ld_pos, ll_pos, "LongChat-13B-16K (λ>0 heads)", liu)

    # ========== Phase 4: Free-parameter fit for comparison ==========
    print(f"\n{'='*80}")
    print(f"  REFERENCE: FULLY FREE FIT (4 params: Δ, λ, baseline, amplitude)")
    print(f"{'='*80}")
    print()

    for n_docs in sorted(liu.keys()):
        entry = liu[n_docs]
        indices = np.array(entry['indices'])
        acc = np.array(entry['accuracy'])

        def loss(params):
            d, l, b, a = params
            if d < 0.05 or d > 2.0 or l < 0 or a < 0:
                return 1e12
            curve = bcft_attention_curve(indices, n_docs, d, l)
            if curve.max() > 0:
                curve_n = curve / curve.max()
            else:
                return 1e12
            pred = b + a * curve_n
            return np.sum((acc - pred) ** 2)

        bounds = [(0.05, 2.0), (0.1, 50.0), (30.0, 80.0), (1.0, 60.0)]
        res = differential_evolution(loss, bounds, seed=42, maxiter=2000, tol=1e-12)
        d, l, b, a = res.x
        curve = bcft_attention_curve(indices, n_docs, d, l)
        curve_n = curve / curve.max() if curve.max() > 0 else curve
        predicted = b + a * curve_n
        ss_res = np.sum((acc - predicted) ** 2)
        ss_tot = np.sum((acc - np.mean(acc)) ** 2)
        R2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0

        print(f"  {n_docs} docs: Δ={d:.4f}, λ={l:.2f}, R²={R2:.4f}")
        for i, (idx, m, pr) in enumerate(zip(indices, acc, predicted)):
            print(f"    doc {idx:>2d}: measured={m:5.1f}%  predicted={pr:5.1f}%")
        print()

    # ========== Phase 5: Prediction summary ==========
    print(f"\n{'='*80}")
    print(f"  PREDICTION SUMMARY")
    print(f"{'='*80}")
    print()
    print(f"  Model: LongChat-13B-16K")
    print(f"  Measured from attention: Δ = {DELTA_MEAS:.4f}, λ = {LAMBDA_MEAS:.4f}")
    print()
    print(f"  Prediction 1 (valley at ~40% of context):")
    print(f"    Attention-level valley: {np.median(vfracs):.3f}")
    print()
    print(f"  Prediction 2 (Δ controls valley depth):")
    print(f"    Per-head ρ(Δ, valley) = {rho_dv:+.4f} (p={p_dv:.4f})")
    print()
    print(f"  Direct comparison (attention params → accuracy prediction):")
    print(f"    See R² values above. If R² > 0.70 with only 2 free params,")
    print(f"    the attention geometry predicts task-level performance.")
