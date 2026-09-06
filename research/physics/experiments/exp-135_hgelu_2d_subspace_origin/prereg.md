# Pre-registration: exp-135 — h_gelu 2D subspace origin
# Registered: 2026-09-06 (before run.py written or any analysis run)

## Background

exp-134 established that the position-correlated variation of h_gelu (MLP block-0
intermediate activation, shape 3072) is 2-dimensional: 50% of position-variance
lives in 2 PCA components; 89% in the first 100. W_proj amplifies σ from 0.121
to 0.313 by routing through those 2 directions via 2 output channels (480, 87).

The question that exp-134 opened: **is the 2D position-correlated structure of
h_gelu already present in h^(0.5) (the residual stream entering the MLP, i.e.,
the output of attention block 0), or is it created by W_fc + GeLU?**

The MLP pipeline in block 0:
```
h^(0.5) → W_fc [768→3072] → pre_act → GeLU → h_gelu → W_proj [3072→768] → mlp_out
```

Known σ values (random-token census, mean-first protocol, exp-131/132/133):
- σ(h^(0))  = 0.403  (residual stream before attention block 0; wpe/wte dominated)
- σ(h^(0.5)) = 0.144  (residual stream after attention block 0; attn write scrambles wpe)
- σ(pre_act) = 0.017  (after W_fc; dispersed)
- σ(h_gelu)  = 0.121  (after GeLU; partially recovered)
- σ(mlp_out) = 0.313  (after W_proj; amplified)

The 2D dimensionality of h_gelu was measured in exp-134. This experiment applies
the same PCA dimensionality analysis to h^(0.5) and h^(0) to locate the origin
of the 2D structure.

## Model

GPT-2 (gpt2), block 0.

## Protocol

Same census as exp-131/132/133/134: random-token, 50 sequences, seed=42,
mean-first, sequence length = 128 tokens.

Compute per-position mean activations for each layer input:
- h^(0): embedding output (before block 0 attention)
- h^(0.5): output of block 0 attention + residual add (MLP input)
- h_gelu: MLP intermediate (reference; from exp-134)

For each, run PCA on the per-position mean matrix (shape: seq_len × d) and record
cumulative variance explained at 1, 2, 5, 10, 50, 100 components.

## Hypothesis

The 2D position-correlated structure of h_gelu originates in h^(0.5). Specifically:
h^(0.5) has ≥ 50% of its position-variance in ≤ 2 PCA components, and W_fc + GeLU
preserve (or refine) this 2D structure rather than creating it.

## Predictions

**P1 (h^(0.5) is 2D):** The first 2 PCA components of h^(0.5) explain ≥ 50% of its
total position-variance (same threshold as the h_gelu result in exp-134).

**P2 (h^(0) dimensionality for contrast):** Compare h^(0) cumulative variance profile
to h^(0.5) and h_gelu — determine whether wpe/wte already encodes the 2D structure
or whether attention block 0 creates or modifies it.

**P3 (source identification):** The cumulative variance profiles at 2 components
for h^(0), h^(0.5), and h_gelu tell us:
- If h^(0) is 2D and h^(0.5) is 2D → wpe/wte is the source; attn block and MLP preserve it.
- If h^(0) is NOT 2D but h^(0.5) is 2D → attention block 0 creates the 2D structure.
- If h^(0.5) is NOT 2D → W_fc + GeLU create the 2D structure in h_gelu (attention does not preserve it).

## Kill conditions

**K1 (h^(0.5) is high-rank):** If h^(0.5) requires > 10 components to explain
50% of its position-variance, the 2D structure is not present at the MLP input,
and W_fc + GeLU must be creating it. P1 falsified. The mechanism question shifts
to how GeLU selectively recovers 2D structure from the dispersed pre_act.

**K2 (ambiguous profile):** If h^(0.5) has 50% variance in 3–9 components (not
cleanly 2D nor clearly high-rank), the source is diffuse and the question must be
re-posed with finer thresholds or a different operationalization.

## Expected artifacts

- `run.py` — analysis script (written after this commit)
- `results.json` — cumulative variance profiles for h^(0), h^(0.5), h_gelu; 
  top component counts at 50%/80%/90% thresholds; P1/P2/P3 verdicts
- `notes.md` — full write-up with verdict
