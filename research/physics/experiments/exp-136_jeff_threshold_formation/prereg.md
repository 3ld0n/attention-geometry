# Pre-registration: exp-136 — J_eff threshold vs. formation phase transition (Pythia-70m)
# Registered: 2026-09-08 (before run.py written or any analysis run)

## Background

The melonic-threshold derivation (notes/2026-08-03_melonic_threshold_derivation.md,
notes/2026-08-08_map_retirement_harvest.md, notes/2026-08-09_route_adjudications.md)
established that the SYK effective coupling J_eff — computable from the key weight
matrices and token embeddings — governs whether the attention fixed point forms.
The formation phase transition in Pythia-70m occurs around step 256–1000 (exp-086:
first SYK-near heads appear in that window; Spearman ρ = 0.862 between log(step)
and n_syk_near over all checkpoints). The prediction: J²_eff grows during training
and crosses a characteristic scale at the same step where the formation transition
is observed.

This is an analysis-only experiment: no new training, no new model downloads,
no new inference. All data is extracted from checkpoint weights.

## Model

EleutherAI/pythia-70m. Architecture: 6 layers, 8 heads, d_model=512, d_k=64.
Checkpoints available as HF revisions. Combined QKV weight per layer:
query_key_value.weight ∈ R^{1536×512}; W_K occupies rows [512:1024].

## Checkpoints

{0, 1, 4, 16, 64, 256, 1000, 4000, 16000, 64000, 143000} — same 11 steps as exp-086.

## Definition of J_eff²

For each checkpoint step s, layer ℓ, and head h:

1. Extract W_K_h = query_key_value.weight[512 + h*64 : 512 + (h+1)*64, :], shape [64, 512].
2. Sample N=500 token indices from the vocabulary uniformly at random (seed=136, fixed
   across all checkpoints and heads).
3. Compute token embeddings: x_a = embed_in.weight[token_a], shape [512], for each a.
4. Compute key activations: k_a = (W_K_h @ x_a) / sqrt(64), shape [64].
5. Compute raw key gram matrix: K_ab = k_a · k_b, resulting K ∈ R^{N×N}.
6. Doubly-center K:
       row_means = K.mean(dim=1, keepdim=True)
       col_means = K.mean(dim=0, keepdim=True)
       grand_mean = K.mean()
       δK = K - row_means - col_means + grand_mean
7. Compute Ω̂ = ||K @ δK||_F² / N²   (disorder tensor, normalized).
8. Compute σ_K² = ||W_K_h||_F² / (64 * 512).   (mean squared entry of W_K_h)
9. Record J_eff²(s, ℓ, h) = (σ_K²)² * Ω̂.

Summary statistics per checkpoint:
- J_eff²_mean(s) = mean over all (ℓ, h) pairs of J_eff²(s, ℓ, h).
- J_eff²_head(s) = max over (ℓ, h) of J_eff²(s, ℓ, h).

## Embedding scale proxy (kinetic term)

For each checkpoint:
- σ_E²(s) = ||embed_in.weight||_F² / (V * d_model)   (mean squared entry of wte).
- Record alongside J_eff² for reference, but predictions are stated in terms of
  J_eff² normalized by its step-0 value (not by σ_E²).

## Hypotheses

**H1 (monotone growth):** J_eff²_mean(s) is monotonically non-decreasing with step s
  over the 11 checkpoints.

  Criterion: Spearman ρ(s, J_eff²_mean) ≥ 0.80 (same stringency as exp-086's H_mono).
  Kill K1: ρ < 0.50 (J_eff² shows no sustained growth).

**H2 (coupling threshold location):** The relative growth R(s) = J_eff²_mean(s) /
  J_eff²_mean(step=0) first exceeds 2.0 at a step s* in the formation-transition
  window [64, 4000] defined by exp-086.

  Criterion: s* = min{s : R(s) > 2.0} satisfies s* ∈ {64, 256, 1000, 4000}.
  Kill K2: s* ∉ [32, 8000], i.e., the crossing is at step 0–16 (trivially early)
            or at step 16000+ (too late; deep-layer formation is complete by then
            per exp-086).

**H3 (correlation with formation statistics):** Spearman ρ between log(J_eff²_mean(s))
  and n_syk_near(s) from exp-086 results (EleutherAI/pythia-70m, same 11 checkpoints)
  satisfies ρ ≥ 0.70.

  Kill K3: ρ < 0.50 (J_eff² and n_syk_near are unrelated across checkpoints).

## Pre-stated direction (not a kill, informational)

Predicted shape: J_eff² is small and roughly flat through step 64, then rises
sharply through step 256–1000, then continues to grow more slowly toward
step 143000. This matches the growth-then-plateau shape of n_syk_near in exp-086.

## Analysis-only flag

This experiment uses only checkpoint weights and a fixed random token sample.
No new model training. No new inference beyond extracting weight matrices and
embedding vectors. No new data download (all Pythia-70m checkpoints were
previously used for exp-086).

## Expected artifacts

- `run.py` — analysis script (written after this commit)
- `results.json` — J_eff²(s, ℓ, h) for all checkpoints, layers, heads; summary
  statistics; σ_E²(s); H1/H2/H3 verdicts; comparison to exp-086 n_syk_near series.
- `notes.md` — full write-up with verdict and interpretation.
