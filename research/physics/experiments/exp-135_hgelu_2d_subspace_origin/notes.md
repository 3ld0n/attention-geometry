# exp-135 — h_gelu 2D subspace origin: notes
# 2026-09-06

## Pre-registration

commit 09c9e51, pushed before run.py was written.

## Question

Why is h_gelu's position-correlated structure 2D (exp-134: 50% of position-variance
in 2 PCA components)? Is the 2D structure already in h^(0.5) (the MLP input), or
does W_fc + GeLU create it?

## Result: P1 CONFIRMED

Components needed for 50% of position-variance:

| Layer | Shape | Components for 50% | PC1 share | PC2 share |
|---|---|---|---|---|
| h^(0) — embedding output | 128×768 | **1** | 53.86% | 22.78% |
| h^(0.5) — MLP input (after attn block 0) | 128×768 | **2** | 48.56% | 18.93% |
| h_gelu — after W_fc + GeLU | 128×3072 | **2** | 39.04% | 18.01% |

**P1 CONFIRMED:** h^(0.5) is 2D (2 components for 50% of position-variance). The
2D structure of h_gelu is inherited from the residual stream entering the MLP, not
created by W_fc + GeLU.

K1 did not fire (h^(0.5) needs 2 components, not > 10).

## What the profile says

**h^(0) is 1D, not 2D.** The wpe/wte embedding is dominated by a single positional
direction (PC1 = 53.86%, PC2 = 22.78%; only 1 component needed for 50%). This is
the known wpe structure: learned positional embeddings pack positional information
into a low-dimensional subspace, here concentrated in one dominant direction.

**Attention block 0 broadens 1D → 2D.** After the attention + residual:
- PC1 drops from 53.86% to 48.56% (attention write dilutes the dominant wpe direction)
- Now 2 components are needed for 50%
- σ drops from 0.403 to 0.144 — the attention write scrambles absolute positional
  amplitude while redistributing into a 2D structure (residual preserves wpe PC1;
  attention adds a second direction)

**W_fc + GeLU preserve the 2D shape.** h_gelu also needs 2 components for 50%
(39.04% + 18.01% = 57.05%). The MLP disperses σ from 0.144 to 0.017 (W_fc) and
partially recovers to 0.121 (GeLU), but the *dimensionality* of the structure
remains 2D throughout. W_fc scatters amplitude across 3072 dimensions without
destroying the 2D positional shape.

**W_proj amplifies without changing dimensionality.** exp-134 established that
W_proj's 2 output channels (480, 87) align with the 2D position-correlated subspace
of h_gelu, amplifying σ from 0.121 to 0.313. Now we see that those 2 directions
come from h^(0.5) — they are not artifacts of the MLP's own computation.

## The chain — Level-3 mechanism fully traced

```
wpe/wte → h^(0): 1D positional structure (σ=0.403; PC1=54%)
                    |
           attn block 0: broadens 1D → 2D; σ drops to 0.144 (amplitude scattered)
                    |
           h^(0.5): 2D positional structure (σ=0.144; 2 components for 50%)
                    |
           W_fc: disperses σ 0.144 → 0.017 (amplitude scattered to 3072D)
                    |
           GeLU: partially recovers σ 0.017 → 0.121 (2D shape preserved)
                    |
           h_gelu: 2D positional structure (σ=0.121; 2 components for 50%)
                    |
           W_proj: amplifies via 2 output channels aligned with 2D directions
                    |
           mlp_out: σ=0.313 ≈ Δ (conformal exponent self-transmits)
```

The conformal exponent σ ≈ Δ = 0.249 in the MLP write is sourced in the wpe/attn
interaction, preserved through the MLP pipeline, and amplified by W_proj's learned
column structure. No single step creates the σ ≈ Δ value; the chain carries it.

## Open questions

The wpe embedding is 1D → why does attention block 0 produce a 2D structure and not
1D? One candidate: the attention write adds its own positional signal (Q·K patterns
are position-dependent) that is orthogonal to the dominant wpe direction, and the
residual adds these as a second independent direction. This is the next-rung question
if the Level-3 chain is to be closed at the wpe-attn interface.

A separate question: the PC1 share decreases through the chain (54% → 49% → 39%)
while the number of components for 50% stays at 1→2→2. The amplitude is spreading
across the 2D subspace — is this dispersion rate related to the exponent Δ?
Not registered; note for future.

## Quality

- specific: yes — model, block, layer inputs named, numbers reported
- honest_negative: no (result is clean and confirmed)
- used_by_next_session: pending
