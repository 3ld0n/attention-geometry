"""
Toy experiment: what happens to measured Δ when you add an ALiBi penalty 
to attention weights that already have a true power-law structure?

The question: does ALiBi make apparent Δ larger or smaller?
"""
import numpy as np
from scipy.stats import pearsonr
from scipy.optimize import curve_fit

def power_law(x, A, delta):
    return A * x**(-2*delta)

def measure_delta(weights, positions, r2_thresh=0.9):
    """Fit power law A*x^{-2*delta} to mean attention vs position."""
    # Average over all source positions
    n = len(positions)
    dists = []
    attn = []
    for i in range(n):
        row = weights[i]
        for j in range(n):
            d = abs(i - j)
            if d > 0:
                dists.append(d)
                attn.append(row[j])
    
    dists = np.array(dists)
    attn = np.array(attn)
    
    # Bin by distance
    max_d = dists.max()
    bins = np.unique(dists)
    mean_attn = [attn[dists == d].mean() for d in bins]
    bins = np.array(bins, dtype=float)
    mean_attn = np.array(mean_attn)
    
    # Fit in log space
    mask = (bins > 0) & (mean_attn > 0)
    log_d = np.log(bins[mask])
    log_a = np.log(mean_attn[mask])
    
    if len(log_d) < 5:
        return None, None
    
    slope, intercept = np.polyfit(log_d, log_a, 1)
    predicted = slope * log_d + intercept
    ss_res = np.sum((log_a - predicted)**2)
    ss_tot = np.sum((log_a - np.mean(log_a))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
    delta = -slope / 2
    return delta, r2

# Setup: sequence length, heads
seq_len = 64
np.random.seed(42)

print("Experiment: True Δ vs Measured Δ under ALiBi penalty")
print("="*60)

for true_delta in [0.125, 0.25, 0.50]:
    print(f"\nTrue Δ = {true_delta}")
    
    for m in [0.0, 0.002, 0.005, 0.01, 0.02]:
        # Generate true power-law logits: log_score ~ -2*delta * log(d)
        # Then add ALiBi: score = score - m * d
        
        scores = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            for j in range(seq_len):
                d = abs(i - j)
                if d > 0:
                    # True power-law logit
                    scores[i, j] = -2 * true_delta * np.log(d) + np.random.randn() * 0.3
                    # ALiBi penalty
                    scores[i, j] -= m * d
                else:
                    scores[i, j] = -np.inf  # no self-attention
        
        # Apply softmax per row
        weights = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            row = scores[i].copy()
            row[i] = -np.inf  # mask self
            row_max = row[np.isfinite(row)].max()
            exp_row = np.exp(row - row_max)
            exp_row[i] = 0
            weights[i] = exp_row / exp_row.sum()
        
        delta_meas, r2 = measure_delta(weights, range(seq_len))
        if delta_meas is not None:
            print(f"  m={m:.3f}: apparent Δ={delta_meas:.4f} (R²={r2:.3f})")

