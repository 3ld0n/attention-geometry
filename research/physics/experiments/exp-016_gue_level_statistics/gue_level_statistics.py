"""
GUE Level Statistics in Transformer Attention

Does trained attention exhibit the eigenvalue statistics of quantum chaos —
the same statistics obeyed by the non-trivial zeros of the Riemann zeta function?

The SYK correspondence (conformal scaling paper) predicts that trained attention
realizes SYK-like quantum chaotic structure. SYK has GUE (Gaussian Unitary Ensemble)
eigenvalue statistics. Therefore: trained attention should show GUE level statistics.

Method:
  For each input, for each attention head:
    1. Compute the pre-softmax attention logit matrix L = Q K^T / sqrt(d_k)
    2. Symmetrize: H = (L + L^T) / 2  (Hermitian part — the "Hamiltonian")
    3. Compute eigenvalues of H
    4. Compute the ratio statistic r = min(s_n, s_{n+1}) / max(s_n, s_{n+1})

  The mean ratio <r> discriminates universality classes without unfolding:
    Poisson (integrable, no chaos):   <r> = 2 ln 2 - 1 ≈ 0.3863
    GOE (time-reversal symmetric):    <r> ≈ 0.5307
    GUE (broken time-reversal, full): <r> ≈ 0.5996

  Montgomery (1973) + Odlyzko (1987): Riemann zeta zeros → GUE

Ariel — March 26, 2026
"""

import torch
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

SEQ_LEN = 128
N_INPUTS = 20

# ========== Load model ==========
print("Loading GPT-2...")
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

d_model = model.config.n_embd       # 768
n_layers = model.config.n_layer     # 12
n_heads = model.config.n_head       # 12
d_k = d_model // n_heads            # 64
vocab_size = model.config.vocab_size

print(f"  d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}, d_k={d_k}")
print(f"  seq_len={SEQ_LEN}, n_inputs={N_INPUTS}")
print()


# ========== Create randomized control ==========
print("Creating randomized control...")
random_model = GPT2LMHeadModel.from_pretrained("gpt2")
with torch.no_grad():
    for layer_idx in range(n_layers):
        block = random_model.transformer.h[layer_idx]
        for name, param in block.attn.named_parameters():
            if "weight" in name:
                torch.nn.init.normal_(param, mean=0, std=0.02)
            elif "bias" in name:
                torch.nn.init.zeros_(param)
        for name, param in block.mlp.named_parameters():
            if "weight" in name:
                torch.nn.init.normal_(param, mean=0, std=0.02)
            elif "bias" in name:
                torch.nn.init.zeros_(param)
random_model.eval()
print("  Randomized: attention + MLP weights reinitialized")
print()


# ========== Core functions ==========

def get_logit_matrices(mdl, input_ids):
    """Extract pre-softmax attention logit matrices for all heads."""
    with torch.no_grad():
        outputs = mdl(input_ids, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    results = {}

    for l in range(n_layers):
        block = mdl.transformer.h[l]
        h = hidden_states[l]
        h_ln = block.ln_1(h)
        qkv = block.attn.c_attn(h_ln)
        q, k, v = qkv.split(d_model, dim=-1)

        seq = q.shape[1]
        q = q.view(1, seq, n_heads, d_k).permute(0, 2, 1, 3)
        k = k.view(1, seq, n_heads, d_k).permute(0, 2, 1, 3)

        for head in range(n_heads):
            q_h = q[0, head]
            k_h = k[0, head]
            L = torch.matmul(q_h, k_h.transpose(-1, -2)) / (d_k ** 0.5)
            results[(l, head)] = L.detach().numpy()

    return results


def ratio_statistic(eigenvalues):
    """
    Compute <r> = <min(s_n, s_{n+1}) / max(s_n, s_{n+1})> for sorted eigenvalues.
    Unfolding-free diagnostic of level statistics.
    """
    eigs = np.sort(np.real(eigenvalues))
    spacings = np.diff(eigs)
    spacings = spacings[spacings > 1e-12]

    if len(spacings) < 3:
        return np.nan

    ratios = np.minimum(spacings[:-1], spacings[1:]) / np.maximum(spacings[:-1], spacings[1:])
    return np.mean(ratios)


def unfold_spectrum(eigenvalues, poly_order=6):
    """Unfold eigenvalue spectrum, return normalized spacings."""
    eigs = np.sort(np.real(eigenvalues))
    n = len(eigs)

    staircase = np.arange(1, n + 1, dtype=float)
    try:
        coeffs = np.polyfit(eigs, staircase, poly_order)
        smooth = np.polyval(coeffs, eigs)
    except (np.linalg.LinAlgError, ValueError):
        return np.array([])

    spacings = np.diff(smooth)
    spacings = spacings[spacings > 1e-12]

    if len(spacings) == 0:
        return np.array([])

    mean_s = np.mean(spacings)
    if mean_s > 0:
        spacings /= mean_s

    return spacings


# Wigner surmise PDFs for comparison
def wigner_poisson(s):
    return np.exp(-s)

def wigner_goe(s):
    return (np.pi / 2) * s * np.exp(-np.pi * s**2 / 4)

def wigner_gue(s):
    return (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)


# ========== Run experiment ==========
print("=" * 85)
print("  GUE LEVEL STATISTICS IN TRANSFORMER ATTENTION")
print("=" * 85)
print()
print("  Reference <r> values:")
print(f"    Poisson (integrable):          {2 * np.log(2) - 1:.4f}")
print(f"    GOE (time-reversal chaos):     0.5307")
print(f"    GUE (full quantum chaos):      0.5996")
print(f"    Riemann zeta zeros:            GUE")
print()


def run_analysis(mdl, label, n_inputs):
    """Full level statistics analysis for a model."""
    print(f"  Running {label}...")

    # Per-head ratio statistics (averaged over inputs)
    head_ratios = {(l, h): [] for l in range(n_layers) for h in range(n_heads)}
    all_spacings = []

    for inp_idx in range(n_inputs):
        input_ids = torch.randint(0, vocab_size, (1, SEQ_LEN))

        logits = get_logit_matrices(mdl, input_ids)

        for (l, h), L in logits.items():
            H = (L + L.T) / 2.0
            eigs = np.linalg.eigh(H)[0]

            r = ratio_statistic(eigs)
            if not np.isnan(r):
                head_ratios[(l, h)].append(r)

            sp = unfold_spectrum(eigs)
            if len(sp) > 0:
                all_spacings.extend(sp.tolist())

        if (inp_idx + 1) % 5 == 0:
            print(f"    {inp_idx + 1}/{n_inputs} inputs processed")

    # Per-head average <r>
    head_avg_r = {}
    for key, rs in head_ratios.items():
        if rs:
            head_avg_r[key] = np.mean(rs)

    all_r_values = list(head_avg_r.values())

    return all_r_values, all_spacings, head_avg_r


# Run trained
trained_r, trained_spacings, trained_head_r = run_analysis(model, "TRAINED GPT-2", N_INPUTS)
print()

# Run randomized
random_r, random_spacings, random_head_r = run_analysis(random_model, "RANDOMIZED GPT-2", N_INPUTS)
print()


# ========== Results ==========
print("=" * 85)
print("  RESULTS: Ratio Statistic <r> — Per-Head Averages")
print("=" * 85)
print()

r_poisson = 2 * np.log(2) - 1

if trained_r:
    tr = np.array(trained_r)
    print(f"  TRAINED GPT-2 ({len(tr)} heads):")
    print(f"    Mean <r>   = {np.mean(tr):.4f}")
    print(f"    Median <r> = {np.median(tr):.4f}")
    print(f"    Std        = {np.std(tr):.4f}")
    print(f"    Range      = [{np.min(tr):.4f}, {np.max(tr):.4f}]")
    print()

    n_poisson = np.sum(np.abs(tr - r_poisson) < 0.04)
    n_goe = np.sum(np.abs(tr - 0.5307) < 0.04)
    n_gue = np.sum(np.abs(tr - 0.5996) < 0.04)
    print(f"    Near Poisson (<r> ≈ 0.386 ± 0.04): {n_poisson} heads")
    print(f"    Near GOE     (<r> ≈ 0.531 ± 0.04): {n_goe} heads")
    print(f"    Near GUE     (<r> ≈ 0.600 ± 0.04): {n_gue} heads")
    print()

if random_r:
    rr = np.array(random_r)
    print(f"  RANDOMIZED GPT-2 ({len(rr)} heads):")
    print(f"    Mean <r>   = {np.mean(rr):.4f}")
    print(f"    Median <r> = {np.median(rr):.4f}")
    print(f"    Std        = {np.std(rr):.4f}")
    print(f"    Range      = [{np.min(rr):.4f}, {np.max(rr):.4f}]")
    print()

    n_poisson = np.sum(np.abs(rr - r_poisson) < 0.04)
    n_goe = np.sum(np.abs(rr - 0.5307) < 0.04)
    n_gue = np.sum(np.abs(rr - 0.5996) < 0.04)
    print(f"    Near Poisson (<r> ≈ 0.386 ± 0.04): {n_poisson} heads")
    print(f"    Near GOE     (<r> ≈ 0.531 ± 0.04): {n_goe} heads")
    print(f"    Near GUE     (<r> ≈ 0.600 ± 0.04): {n_gue} heads")
    print()


# ========== Layer-by-layer breakdown ==========
print("=" * 85)
print("  LAYER-BY-LAYER: Mean <r> per Layer")
print("=" * 85)
print()
print(f"  {'Layer':>8}  {'Trained <r>':>12}  {'Random <r>':>12}  {'Trained class':>15}  {'Random class':>15}")
print("  " + "-" * 70)

for l in range(n_layers):
    t_vals = [trained_head_r.get((l, h), np.nan) for h in range(n_heads)]
    r_vals = [random_head_r.get((l, h), np.nan) for h in range(n_heads)]

    t_vals = [v for v in t_vals if not np.isnan(v)]
    r_vals = [v for v in r_vals if not np.isnan(v)]

    t_mean = np.mean(t_vals) if t_vals else np.nan
    r_mean = np.mean(r_vals) if r_vals else np.nan

    def classify(r):
        if np.isnan(r):
            return "---"
        dists = {
            "Poisson": abs(r - r_poisson),
            "GOE": abs(r - 0.5307),
            "GUE": abs(r - 0.5996),
        }
        return min(dists, key=dists.get)

    print(f"  {l+1:8d}  {t_mean:12.4f}  {r_mean:12.4f}  {classify(t_mean):>15}  {classify(r_mean):>15}")

print()
print(f"  Reference: Poisson = {r_poisson:.4f}  |  GOE = 0.5307  |  GUE = 0.5996")
print()


# ========== Spacing distribution ==========
print("=" * 85)
print("  SPACING DISTRIBUTION: Histogram vs Wigner Surmise")
print("=" * 85)
print()

bins = np.linspace(0, 4.0, 41)
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_width = bins[1] - bins[0]

if trained_spacings:
    ts = np.array(trained_spacings)
    ts = ts[(ts > 0) & (ts < 10)]  # remove outliers
    hist_t, _ = np.histogram(ts, bins=bins, density=True)

    # Theoretical PDFs at bin centers
    pdf_poisson = wigner_poisson(bin_centers)
    pdf_goe = wigner_goe(bin_centers)
    pdf_gue = wigner_gue(bin_centers)

    # Chi-squared-like fit quality
    mask = hist_t > 0.01
    if np.sum(mask) > 3:
        chi2_poisson = np.mean((hist_t[mask] - pdf_poisson[mask])**2 / (pdf_poisson[mask] + 0.01))
        chi2_goe = np.mean((hist_t[mask] - pdf_goe[mask])**2 / (pdf_goe[mask] + 0.01))
        chi2_gue = np.mean((hist_t[mask] - pdf_gue[mask])**2 / (pdf_gue[mask] + 0.01))

        print(f"  TRAINED spacing distribution ({len(ts)} spacings):")
        print(f"    Fit quality (lower = better):")
        print(f"      vs Poisson: {chi2_poisson:.4f}")
        print(f"      vs GOE:     {chi2_goe:.4f}")
        print(f"      vs GUE:     {chi2_gue:.4f}")

        best = min([("Poisson", chi2_poisson), ("GOE", chi2_goe), ("GUE", chi2_gue)],
                   key=lambda x: x[1])
        print(f"    Best match: {best[0]}")
        print()

    # Show key features: P(0) and location of peak
    print(f"    P(s→0): {hist_t[0]:.4f}  (Poisson: {pdf_poisson[0]:.4f}, "
          f"GOE: {pdf_goe[0]:.4f}, GUE: {pdf_gue[0]:.4f})")
    peak_idx = np.argmax(hist_t)
    print(f"    Peak at s ≈ {bin_centers[peak_idx]:.2f}  "
          f"(GOE peak ≈ 0.80, GUE peak ≈ 0.68)")
    print()

    # Text histogram
    print("    Spacing distribution (TRAINED):")
    max_bar = 50
    max_val = max(np.max(hist_t), 0.01)
    for i in range(0, min(30, len(hist_t))):
        bar_len = int(hist_t[i] / max_val * max_bar)
        bar = "█" * bar_len
        s = bin_centers[i]
        markers = ""
        if abs(s - 0.68) < bin_width:
            markers = " ← GUE peak"
        elif abs(s - 0.80) < bin_width:
            markers = " ← GOE peak"
        print(f"      s={s:4.2f} | {bar}{markers}")
    print()


if random_spacings:
    rs = np.array(random_spacings)
    rs = rs[(rs > 0) & (rs < 10)]
    hist_r, _ = np.histogram(rs, bins=bins, density=True)

    mask = hist_r > 0.01
    if np.sum(mask) > 3:
        chi2_poisson = np.mean((hist_r[mask] - pdf_poisson[mask])**2 / (pdf_poisson[mask] + 0.01))
        chi2_goe = np.mean((hist_r[mask] - pdf_goe[mask])**2 / (pdf_goe[mask] + 0.01))
        chi2_gue = np.mean((hist_r[mask] - pdf_gue[mask])**2 / (pdf_gue[mask] + 0.01))

        print(f"  RANDOMIZED spacing distribution ({len(rs)} spacings):")
        print(f"    Fit quality (lower = better):")
        print(f"      vs Poisson: {chi2_poisson:.4f}")
        print(f"      vs GOE:     {chi2_goe:.4f}")
        print(f"      vs GUE:     {chi2_gue:.4f}")

        best = min([("Poisson", chi2_poisson), ("GOE", chi2_goe), ("GUE", chi2_gue)],
                   key=lambda x: x[1])
        print(f"    Best match: {best[0]}")
        print()

    print(f"    P(s→0): {hist_r[0]:.4f}  (Poisson: {pdf_poisson[0]:.4f})")
    peak_idx = np.argmax(hist_r)
    print(f"    Peak at s ≈ {bin_centers[peak_idx]:.2f}")
    print()

    print("    Spacing distribution (RANDOMIZED):")
    max_val_r = max(np.max(hist_r), 0.01)
    for i in range(0, min(30, len(hist_r))):
        bar_len = int(hist_r[i] / max_val_r * max_bar)
        bar = "█" * bar_len
        print(f"      s={bin_centers[i]:4.2f} | {bar}")
    print()


# ========== Per-head detail for trained model ==========
print("=" * 85)
print("  PER-HEAD DETAIL: Trained GPT-2")
print("=" * 85)
print()
print(f"  {'Layer':>6} {'Head':>5}  {'<r>':>8}  {'Class':>10}")
print("  " + "-" * 35)

for l in range(n_layers):
    for h in range(n_heads):
        r = trained_head_r.get((l, h), np.nan)
        if not np.isnan(r):
            dists = {
                "Poisson": abs(r - r_poisson),
                "GOE": abs(r - 0.5307),
                "GUE": abs(r - 0.5996),
            }
            cls = min(dists, key=dists.get)
            marker = "***" if cls == "GUE" else ""
            print(f"  {l+1:6d} {h:5d}  {r:8.4f}  {cls:>10}  {marker}")
    if l < n_layers - 1:
        print()

print()


# ========== Weight matrix analysis ==========
print("=" * 85)
print("  WEIGHT MATRIX ANALYSIS: W_Q^T W_K per head (input-independent)")
print("=" * 85)
print()

weight_r_trained = []
weight_r_random = []

for l in range(n_layers):
    for mdl_ref, label, collector in [
        (model, "trained", weight_r_trained),
        (random_model, "random", weight_r_random),
    ]:
        block = mdl_ref.transformer.h[l]
        W = block.attn.c_attn.weight.detach().numpy()  # (d_model, 3*d_model)
        W_Q = W[:, :d_model]        # (d_model, d_model)
        W_K = W[:, d_model:2*d_model]

        for h in range(n_heads):
            wq_h = W_Q[:, h*d_k:(h+1)*d_k]   # (d_model, d_k)
            wk_h = W_K[:, h*d_k:(h+1)*d_k]   # (d_model, d_k)

            W_eff = wq_h.T @ wk_h  # (d_k, d_k)
            H_eff = (W_eff + W_eff.T) / 2.0
            eigs = np.linalg.eigh(H_eff)[0]

            r = ratio_statistic(eigs)
            if not np.isnan(r):
                collector.append(r)

if weight_r_trained:
    wt = np.array(weight_r_trained)
    print(f"  TRAINED W_Q^T W_K ({len(wt)} heads):")
    print(f"    Mean <r>   = {np.mean(wt):.4f}")
    print(f"    Median <r> = {np.median(wt):.4f}")
    print(f"    Std        = {np.std(wt):.4f}")
    print()

if weight_r_random:
    wr = np.array(weight_r_random)
    print(f"  RANDOMIZED W_Q^T W_K ({len(wr)} heads):")
    print(f"    Mean <r>   = {np.mean(wr):.4f}")
    print(f"    Median <r> = {np.median(wr):.4f}")
    print(f"    Std        = {np.std(wr):.4f}")
    print()


# ========== Conclusion ==========
print("=" * 85)
print("  CONCLUSION")
print("=" * 85)
print()
print("  If trained attention shows <r> ≈ 0.60 (GUE) and randomized shows")
print("  <r> ≈ 0.39 (Poisson) or <r> ≈ 0.53 (GOE), then:")
print()
print("  1. Training creates quantum chaotic structure in attention")
print("  2. Trained attention is in the SAME universality class as")
print("     the non-trivial zeros of the Riemann zeta function (GUE)")
print("  3. This confirms the SYK correspondence: SYK predicts GUE,")
print("     and the conformal scaling (Δ ≈ 1/4) already matches SYK")
print()
print("  Montgomery (1973) + Odlyzko (1987): zeta zeros → GUE")
print("  SYK (Sachdev-Ye-Kitaev): quantum chaos → GUE")
print("  This experiment: trained attention → ???")
print()
