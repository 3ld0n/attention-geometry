# Pre-registration: exp-134 — W_proj column structure
# Registered: 2026-09-06 (before run.py written or any analysis run)

## Question

MLP block-0 in GPT-2 amplifies position-correlation from σ = 0.121 (h_gelu input)
to σ = 0.313 (mlp_out output), a ratio of 2.168 (exp-132/133). The gate (W_fc) was
ruled out — it disperses (σ 0.144→0.017). GeLU partially recovers (→0.121). The
remaining question: is the amplification in W_proj (c_proj, shape 3072→768) a
consequence of *learned column structure*, i.e., W_proj's rows preferentially
aligning with position-correlated directions in h_gelu-space?

## Model

GPT-2 (gpt2), block 0, c_proj weight, shape (768, 3072).

## Protocol

Same census: random-token, 50 sequences, seed=42, mean-first (exp-131/132/133
protocol). Sequence length = 128 tokens.

## Hypothesis

W_proj's per-output-channel position-correlations σ_d (for d = 0..767) form a
non-uniform distribution in which a minority of output channels carry the bulk of
the aggregate σ = 0.313. The position-correlated structure of h_gelu is
low-rank, and W_proj's rows preferentially align with those position-correlated
directions — this alignment is the source of the amplification.

## Predictions

**P1 (column concentration):** The distribution of σ_d across the 768 output
channels is non-uniform. Specifically: the top-k channels by |σ_d| carry ≥ 50% of
the total Σ|σ_d| for k ≤ 153 (≤ 20% of 768 output channels).

**P2 (alignment):** The W_proj rows corresponding to the top-k channels (by |σ_d|)
have higher mean cosine similarity to the leading position-correlated directions of
h_gelu (from PCA/SVD of the per-position mean h_gelu matrix) than the bottom-k rows.
Specifically: top-k mean cosine similarity > bottom-k mean cosine similarity (one-sided).

**P3 (h_gelu dimensionality):** The position-correlated structure of h_gelu is
low-dimensional. The first R components of a PCA on per-position mean h_gelu
explain ≥ 50% of the position-variance, for R ≤ 100 (≤ ~3% of 3072 dimensions).

## Kill conditions

**K1 (uniform distribution):** If σ_d is roughly uniform — each channel contributing
approximately σ_total / 768 ≈ 0.0004 with no channel above 5× the mean — then
column structure is not a selective mechanism, and the amplification is a
dimensionality-reduction artifact rather than a learned structure.

**K2 (no alignment):** If top-k W_proj rows have mean cosine similarity to
position-correlated h_gelu directions ≤ the bottom-k rows (or not significantly
different from random), then W_proj's amplification is not due to preferential
alignment — some other geometric mechanism is responsible.

**K3 (h_gelu is high-rank):** If the first 100 components explain < 20% of
position-variance (h_gelu is high-rank in the position direction), then the
low-rank candidate for the coupling mechanism fails, and P2 becomes harder to
evaluate.

## Expected artifacts

- `run.py` — analysis script (written after this commit)
- `results.json` — per-channel σ distribution, PCA variance explained, alignment
  scores
- `notes.md` — this file extended with results and verdict
