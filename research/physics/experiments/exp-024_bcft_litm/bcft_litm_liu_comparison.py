"""
BCFT vs Liu et al. (2023) Accuracy Data

Fit the BCFT functional form to the actual accuracy-vs-position data
from "Lost in the Middle" and check:
  1. Does the BCFT form fit the data?
  2. Are the fitted Δ and λ physically reasonable?
  3. Do the parameters vary systematically across context lengths?
  4. Where is the fitted valley relative to the BCFT prediction?

Data source: Liu et al. (2023), Tables 5-7 (arXiv:2307.03172v3)
Position indices: 0 = start of context, N-1 = end (nearest to query)

Ariel — April 15, 2026
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution

# ========== Liu et al. data ==========
# 20-document setting (Table 6): indices 0, 4, 9, 14, 19
LIU_20 = {
    'GPT-3.5-Turbo':       [75.8, 57.2, 53.8, 55.4, 63.2],
    'GPT-3.5-Turbo-16K':   [75.7, 57.3, 54.1, 55.4, 63.1],
    'Claude-1.3':           [59.9, 55.9, 56.8, 57.2, 60.1],
    'Claude-1.3-100K':      [59.8, 55.9, 57.0, 57.4, 60.0],
    'MPT-30B-Instruct':     [53.7, 51.8, 52.2, 52.7, 56.3],
    'LongChat-13B-16K':     [68.6, 57.4, 55.3, 52.5, 55.0],
}
LIU_20_INDICES = np.array([0, 4, 9, 14, 19])
N_DOCS_20 = 20

# 10-document setting (Table 5): indices 0, 4, 9
LIU_10 = {
    'GPT-3.5-Turbo':       [76.8, 61.2, 62.4],
    'Claude-1.3':           [62.9, 58.3, 59.7],
    'MPT-30B-Instruct':     [60.2, 56.2, 59.7],
    'LongChat-13B-16K':     [72.1, 58.9, 58.5],
}
LIU_10_INDICES = np.array([0, 4, 9])
N_DOCS_10 = 10

# 30-document setting (Table 7): indices 0, 4, 9, 14, 19, 24, 29
LIU_30 = {
    'GPT-3.5-Turbo':       [72.8, 55.0, 52.0, 52.6, 55.2, 56.4, 62.0],
    'Claude-1.3':           [59.1, 55.1, 54.8, 55.7, 56.4, 56.2, 59.0],
    'MPT-30B-Instruct':     [52.2, 51.2, 49.2, 49.8, 50.5, 50.2, 53.0],
    'LongChat-13B-16K':     [57.4, 53.3, 51.4, 49.2, 48.6, 46.2, 46.0],
}
LIU_30_INDICES = np.array([0, 4, 9, 14, 19, 24, 29])
N_DOCS_30 = 30


def bcft_accuracy_model(indices, n_docs, delta, lam, baseline, amplitude):
    """
    BCFT-predicted relative accuracy at document positions.

    Maps document index k to position x_k = (k + 0.5) in the context.
    Query at x_q = n_docs + 0.5 (just past the last document).

    Accuracy = baseline + amplitude * normalized_bcft_attention(position)
    """
    x_q = n_docs + 0.5
    positions = indices + 0.5  # center of each document

    attn = np.zeros(len(positions))
    for i, x_k in enumerate(positions):
        dx = x_q - x_k
        if dx <= 0 or x_k <= 0:
            attn[i] = 0
            continue
        eta = dx**2 / (4.0 * x_q * x_k)
        power_law = dx ** (-2 * delta)
        correction = 1.0 + lam * eta**delta
        attn[i] = power_law * correction

    # Normalize attention to [0, 1]
    if attn.max() > 0:
        attn_norm = attn / attn.max()
    else:
        attn_norm = np.ones(len(positions)) / len(positions)

    return baseline + amplitude * attn_norm


def fit_bcft_to_accuracy(indices, n_docs, accuracy_data, model_name=""):
    """Fit BCFT form to accuracy data. Returns fitted parameters and diagnostics."""
    acc = np.array(accuracy_data)

    def loss(params):
        delta, lam, baseline, amplitude = params
        if delta < 0.05 or delta > 2.0 or lam < 0 or amplitude < 0:
            return 1e12
        pred = bcft_accuracy_model(indices, n_docs, delta, lam, baseline, amplitude)
        return np.sum((acc - pred) ** 2)

    # Global search first, then local refinement
    bounds = [(0.05, 2.0), (0.1, 50.0), (30.0, 80.0), (1.0, 60.0)]
    result_global = differential_evolution(loss, bounds, seed=42, maxiter=2000,
                                           tol=1e-12, polish=True)

    params = result_global.x
    delta, lam, baseline, amplitude = params

    pred = bcft_accuracy_model(indices, n_docs, delta, lam, baseline, amplitude)
    residuals = acc - pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((acc - np.mean(acc)) ** 2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0

    # Valley position from the fitted model
    all_indices = np.arange(n_docs)
    full_pred = bcft_accuracy_model(all_indices, n_docs, delta, lam, baseline, amplitude)
    valley_idx = np.argmin(full_pred)
    valley_frac = valley_idx / n_docs

    return {
        'delta': delta, 'lambda': lam,
        'baseline': baseline, 'amplitude': amplitude,
        'R2': R2, 'residuals': residuals,
        'predictions': pred,
        'valley_idx': valley_idx, 'valley_frac': valley_frac,
        'full_curve': full_pred,
    }


# ========== Fit all models ==========
print("=" * 90)
print("  BCFT FIT TO LIU ET AL. (2023) ACCURACY DATA")
print("=" * 90)
print()
print("  Fitting: accuracy = baseline + amplitude * BCFT_attention(position; Δ, λ)")
print("  Free parameters: Δ (conformal dimension), λ (boundary strength),")
print("                    baseline (task floor), amplitude (task sensitivity)")
print()

for n_docs, data_dict, indices, label in [
    (10, LIU_10, LIU_10_INDICES, "10 documents (~2K tokens)"),
    (20, LIU_20, LIU_20_INDICES, "20 documents (~4K tokens)"),
    (30, LIU_30, LIU_30_INDICES, "30 documents (~6K tokens)"),
]:
    print(f"\n{'='*90}")
    print(f"  {label}")
    print(f"{'='*90}")
    print()

    print(f"  {'Model':>25} {'Δ':>7} {'λ':>7} {'R²':>7} {'Valley':>8} "
          f"{'V_frac':>7}  Residuals")
    print(f"  {'-'*80}")

    for model_name, acc_data in sorted(data_dict.items()):
        result = fit_bcft_to_accuracy(indices, n_docs, acc_data, model_name)
        res_str = " ".join(f"{r:+.1f}" for r in result['residuals'])
        print(f"  {model_name:>25} {result['delta']:>7.3f} {result['lambda']:>7.2f} "
              f"{result['R2']:>7.4f} {result['valley_idx']:>8d} "
              f"{result['valley_frac']:>7.3f}  [{res_str}]")

    # Detailed output for GPT-3.5-Turbo
    if 'GPT-3.5-Turbo' in data_dict:
        r = fit_bcft_to_accuracy(indices, n_docs, data_dict['GPT-3.5-Turbo'])
        print(f"\n  GPT-3.5-Turbo detailed fit (Δ={r['delta']:.3f}, λ={r['lambda']:.2f}):")
        print(f"  {'Index':>7} {'Measured':>10} {'BCFT pred':>10} {'Residual':>10}")
        for i, (idx, meas, pred) in enumerate(zip(indices, data_dict['GPT-3.5-Turbo'],
                                                    r['predictions'])):
            print(f"  {idx:>7d} {meas:>10.1f} {pred:>10.1f} {meas-pred:>10.1f}")

        # Full predicted curve
        print(f"\n  Full predicted curve (all {n_docs} positions):")
        for idx in range(n_docs):
            acc_val = r['full_curve'][idx]
            bar = '█' * int((acc_val - r['baseline']) / r['amplitude'] * 40)
            measured_mark = " ← measured" if idx in indices else ""
            print(f"    {idx:>3d}: {acc_val:6.1f}%  {bar}{measured_mark}")
        print(f"    Valley at index {r['valley_idx']} "
              f"(fraction: {r['valley_frac']:.3f})")


# ========== Cross-context-length comparison ==========
print("\n\n" + "=" * 90)
print("  CROSS-CONTEXT-LENGTH COMPARISON: GPT-3.5-Turbo")
print("=" * 90)
print()
print("  Does the BCFT predict how the U-shape changes with context length?")
print("  Same model, three context lengths: 10, 20, 30 documents.")
print()

gpt_results = {}
for n_docs, data_dict, indices, label in [
    (10, LIU_10, LIU_10_INDICES, "10 docs"),
    (20, LIU_20, LIU_20_INDICES, "20 docs"),
    (30, LIU_30, LIU_30_INDICES, "30 docs"),
]:
    if 'GPT-3.5-Turbo' in data_dict:
        r = fit_bcft_to_accuracy(indices, n_docs, data_dict['GPT-3.5-Turbo'])
        gpt_results[n_docs] = r
        acc = data_dict['GPT-3.5-Turbo']
        asym = acc[0] / acc[-1]  # start/end accuracy ratio
        depth = 1 - min(acc) / max(acc)  # valley depth
        print(f"  {label:>8}: Δ = {r['delta']:.3f}, λ = {r['lambda']:.2f}, "
              f"R² = {r['R2']:.4f}, valley_frac = {r['valley_frac']:.3f}, "
              f"asym(start/end) = {asym:.3f}, depth = {depth:.3f}")

print()
print("  BCFT predictions for context length scaling:")
print("  → Valley should deepen with more documents (confirmed by data)")
print("  → Valley position should shift (check above)")
print("  → Δ should be CONSISTENT across context lengths (same model)")
print("     (if Δ varies wildly, the BCFT form doesn't generalize)")


# ========== Compare to BCFT predictions from attention weights ==========
print("\n\n" + "=" * 90)
print("  COMPARISON TO ATTENTION-WEIGHT PREDICTIONS")
print("=" * 90)
print()
print("  GPT-2 attention weight measurements:")
print("    Δ = 0.25 (median conformal head)")
print("    λ = 0.3–4.6 (range across heads)")
print()
print("  Fitted from Liu et al. accuracy data:")
for n_docs in sorted(gpt_results.keys()):
    r = gpt_results[n_docs]
    d_match = "✓" if 0.10 < r['delta'] < 0.50 else "✗"
    l_match = "✓" if 0.1 < r['lambda'] < 50 else "✗"
    print(f"    {n_docs} docs: Δ = {r['delta']:.3f} {d_match}  "
          f"λ = {r['lambda']:.2f} {l_match}")

print()
print("  NOTE: GPT-3.5-Turbo is NOT GPT-2. Different model, different training.")
print("  The test is whether Δ is in a physically reasonable range (0.1-0.5)")
print("  and whether λ is consistent with boundary enhancement being real.")
print()
print("  The strongest test would be: measure Δ from a model's attention weights,")
print("  then predict its LiTM accuracy curve using ONLY the measured Δ and λ.")
print("  This requires running the LiTM benchmark on a model where we can also")
print("  measure the attention weights (GPT-2 or Pythia).")


# ========== PURE BCFT PREDICTION (no fitting to accuracy data) ==========
print("\n\n" + "=" * 90)
print("  PURE BCFT PREDICTION vs DATA (no accuracy fitting)")
print("=" * 90)
print()
print("  Using Δ = 0.25, λ = 2.0 (from GPT-2 attention weights)")
print("  Predict the accuracy curve shape (baseline + amplitude fitted only)")
print()

DELTA_PRED = 0.25
LAMBDA_PRED = 2.0

for n_docs, data_dict, indices, label in [
    (20, LIU_20, LIU_20_INDICES, "20 documents"),
    (30, LIU_30, LIU_30_INDICES, "30 documents"),
]:
    for model_name in ['GPT-3.5-Turbo', 'LongChat-13B-16K']:
        if model_name not in data_dict:
            continue
        acc = np.array(data_dict[model_name])

        # Only fit baseline and amplitude (2 params), keep Δ and λ fixed
        def loss_constrained(params):
            baseline, amplitude = params
            if amplitude < 0:
                return 1e12
            pred = bcft_accuracy_model(indices, n_docs, DELTA_PRED, LAMBDA_PRED,
                                       baseline, amplitude)
            return np.sum((acc - pred) ** 2)

        from scipy.optimize import minimize as minimize_opt
        res = minimize_opt(loss_constrained, [50.0, 20.0], method='Nelder-Mead')
        baseline, amplitude = res.x

        pred = bcft_accuracy_model(indices, n_docs, DELTA_PRED, LAMBDA_PRED,
                                   baseline, amplitude)
        ss_res = np.sum((acc - pred) ** 2)
        ss_tot = np.sum((acc - np.mean(acc)) ** 2)
        R2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0

        print(f"  {model_name} ({label}), Δ=0.25 λ=2.0 (fixed from GPT-2 attn):")
        print(f"    R² = {R2:.4f}")
        for i, (idx, m, p) in enumerate(zip(indices, acc, pred)):
            print(f"    Index {idx:>2d}: measured={m:5.1f}%  predicted={p:5.1f}%  "
                  f"residual={m-p:+5.1f}")
        print()

        # Now with λ as free (Δ still fixed at 0.25)
        def loss_delta_fixed(params):
            lam, baseline, amplitude = params
            if amplitude < 0 or lam < 0:
                return 1e12
            pred = bcft_accuracy_model(indices, n_docs, DELTA_PRED, lam,
                                       baseline, amplitude)
            return np.sum((acc - pred) ** 2)

        res2 = differential_evolution(loss_delta_fixed,
                                      [(0.1, 50.0), (30.0, 80.0), (1.0, 60.0)],
                                      seed=42)
        lam_fit, base_fit, amp_fit = res2.x
        pred2 = bcft_accuracy_model(indices, n_docs, DELTA_PRED, lam_fit,
                                    base_fit, amp_fit)
        ss_res2 = np.sum((acc - pred2) ** 2)
        R2_2 = 1 - ss_res2 / ss_tot if ss_tot > 1e-10 else 0
        print(f"    With λ free (Δ=0.25 fixed): λ={lam_fit:.2f}, R²={R2_2:.4f}")
        print()
