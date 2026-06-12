# Attention exponent census — replication package

One script, public checkpoints, minutes of runtime, no training.

## The claim

In trained decoder-only transformers with softmax attention, a sub-population
of heads shows power-law decay of the lag-averaged attention profile,
A(Δx) ∝ |Δx|^(−2Δ), and the median exponent of the high-quality subset
clusters near **Δ = 0.25** across architectures and positional-encoding
schemes (GPT-2: 0.249; GPT-2-medium: 0.259; OLMo-7B: 0.265; GALA-7B softmax:
0.260). Re-initialized weights of the same architectures contain **zero**
such heads, and shuffled-profile nulls put zero mass in the window
(permutation p = 0.001). The structure is created by training.

The prediction you can test on a model of your choosing: a trained softmax
LM will show a conformal sub-population with median Δ in [0.20, 0.30] on the
high-R² subset; its randomized-weight control will show ~none.

## Run it

```bash
pip install torch transformers numpy
python measure_conformal_heads.py gpt2                  # ~2 min on a laptop GPU
python measure_conformal_heads.py gpt2 --randomized     # control: fresh init
python measure_conformal_heads.py EleutherAI/pythia-410m
```

Output: one JSON per run with per-head (Δ, R²) and the summary line
(`n_conformal`, `delta_median_conformal`, `n_syk_near`).

## Protocol notes (the two that matter)

1. **fp32, asserted at extraction.** fp16/bf16 attention underflows in the
   profile tail and silently corrupts deep layers (we found Pythia-410m
   layers 15+ returning all-NaN in fp16). The script asserts dtype per layer.
2. **The protocol is frozen.** 50 random-token sequences (seed 42), L = 512,
   queries i ≥ max(256, Δx), OLS over lags [8, 256], conformal = R² ≥ 0.90 ∧
   Δ ≥ 0.05. Random-token inputs isolate positional structure from content;
   natural-text inputs shift exponents (known single point: GPT-2
   0.25 → 0.37) — use the frozen protocol for comparison with our numbers.

## Provenance and track record

This protocol comes from a research program that pre-registers its
predictions and publishes its kills:

- Pre-registered behavioral test (Δ rank-predicts the lost-in-the-middle
  attention valley): confirmed 6/7 named models, **falsified on the 7th
  (Pythia-2.8B) and published as falsified** (Zenodo DOI
  10.5281/zenodo.19629862). Split-half stability subsequently confirmed on
  the falsified model itself (no shared-noise inflation).
- Pre-registered adversarial test of our own BCFT boundary interpretation:
  **lost on both committed legs and withdrawn** — the boundary correction
  carries an absolute length scale (ξ ≈ 20 tokens in GPT-2), which a
  boundary CFT forbids. The phenomenology stands; the identification died.
- Causal follow-up: low-rank edits to a head's QK positional subspace move
  its measured Δ and its valley statistic in the quantitatively predicted
  direction (ρ = +0.82, 24/24 signs, sham-controlled).

Census preprint: Zenodo DOI 10.5281/zenodo.19225996.

## Contact

If you run this on a model family we have not measured, we would like to see
the JSON either way — especially if it disagrees with the prediction.
