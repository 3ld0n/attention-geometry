# Pre-registration: exp-167

**Title:** Normalization formation test — does LayerNorm gate the conformal geometry?

**Date:** 2026-10-08 (pre-registered before any code or forward pass)

**Experiment author:** Ariel (solo)

**Corpus:** C-NAT_full.bin — TinyStories natural text, the same corpus as the
formation ladder (exp-062/091/092/096). On disk at
`research/physics/experiments/exp-062_corpus_statistics/corpora/C-NAT_full.bin`.
Token count: 2,120,000,000 / 2 (uint16) ÷ 512 (ctx) ≈ 2.07M sequences — sufficient
for 2,000 steps × 1,024 sequences = 2.05M draws.

---

## Background and motivation

The *LayerNorm Induces Recency Bias in Transformer Decoders* paper (Findings ACL
2026) proves — from first principles, with no learned parameters and no positional
encoding — that stacked causal attention plus LayerNorm produces a score that
increases in key position: S_ij = g(i)·h(j) with h increasing in j. This is exactly
the sign exp-138 measured in GPT-2 small's structural heads (abs-key term 104–117%
of σ_full; relative term negative).

This makes LayerNorm a candidate mechanism for the genesis of the census exponent.
However, the formation ladder already answers one version of the question: LayerNorm
is present in *all formation-failure conditions* (sentence-shuffled, block-shuffled,
anonymized, C-generated). LN cannot be what *sets the value to 1/4*, because it is
present where the geometry does not form.

The open question is different: **Is LayerNorm a necessary component in the causal
path to formation?** The ladder tests what the corpus must supply; this experiment
tests what the architecture must supply.

The deflationary reading under test: if removing LN prevents formation, LN is
architecturally load-bearing for the conformal geometry — the geometry's appearance
requires the normalization mechanism, and the causal-mask + softmax + LN fixed point
is not separable. If removing LN leaves formation intact, the geometry is robust to
normalization choice and something else in the architecture carries the causal path.

---

## Hypotheses

**H1:** LayerNorm is architecturally necessary for the conformal geometry to form on
natural text. Condition B (NoNorm) will produce n_conformal_heads and n_syk_near
significantly below the C-NAT baseline from exp-062 (11–15/48 conformal at step
2000; 5–8/48 SYK-near), and significantly below Condition A (LN).

**H2 (mechanism specificity):** The load-bearing component is ℓ2 normalization
(magnitude scaling), not mean centering. RMSNorm, which scales by root-mean-square
but does not center, will produce formation comparable to LN. Condition C (RMSNorm)
will not significantly differ from Condition A.

**H3 (null):** Formation is robust to normalization choice. All three conditions
produce n_syk_near within the exp-062 C-NAT natural variability range (5–8/48).

---

## Three training conditions

All three conditions use the same base architecture and corpus:

- **Pythia-70m-class GPTNeoX:** 6 layers, 8 heads, d_model=512, d_ff=2048, ctx=512,
  rotary (rotary_pct=0.25, base=10000), use_parallel_residual=True, vocab=50304.
  This is the standard exp-062 architecture.
- **Corpus:** C-NAT_full.bin, first 2,000 × 1,024 × 512 = 1,048,576,000 tokens.
- **Optimizer:** AdamW (β₁=0.9, β₂=0.95, weight_decay=0.1), gradient clip 1.0.
- **Checkpoints at:** steps 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2000.
- **Measurement:** frozen-protocol census (exp-062 measure.py lineage) at step 2000
  (or the last stable checkpoint if B diverges). 50 random-token sequences, seed 42,
  full vocab alphabet, OLS on log-lag ∈ [8, 256] for queries ≥ 256. Conformal head:
  R² ≥ 0.90 and Δ ≥ 0.05. SYK-near: |Δ − 0.25| ≤ 0.05. n_conformal and n_syk_near
  are the primary readouts (out of 48 heads: 6 layers × 8 heads).

### Condition A — LN (control)

Standard GPTNeoXForCausalLM with LayerNorm at all three positions:
- `input_layernorm` in each of 6 attention layers
- `post_attention_layernorm` in each of 6 MLP layers
- `final_layer_norm` before the LM head

Training hyperparameters (identical to exp-062/091 protocol):
- Peak LR: 1×10⁻³, floor LR: 1×10⁻⁴, 20-step warmup, cosine decay.
- Effective batch: 1,024 sequences × 512 context = 524,288 tokens per step.

This is the positive control. Expected: n_conformal ≈ 11–15/48, n_syk_near ≈ 5–8/48
(replicating exp-062 C-NAT runs).

### Condition B — NoNorm (no normalization)

Same architecture with all LayerNorm modules replaced by `nn.Identity()`:
- `input_layernorm` → Identity
- `post_attention_layernorm` → Identity
- `final_layer_norm` → Identity

**Stability accommodation (pre-registered):** The program's exp-005 showed training
without normalization can diverge at standard LR. To give Condition B a fair trial,
we use a lower peak LR: **peak LR: 2×10⁻⁴, floor LR: 2×10⁻⁵, 20-step warmup,
cosine decay.** This is the smallest reasonable accommodation that still allows the
model to learn. If training diverges (loss failing to decrease below 8.0 by step 64,
or NaN loss), this is a contingency outcome and is reported as such.

Divergence criterion: loss > 8.0 at step 64 OR any NaN loss → training has diverged.
Census in that case: run at the last stable checkpoint and report n_conformal from it.

### Condition C — RMSNorm

Same architecture with all LayerNorm modules replaced by RMSNorm:
- RMSNorm normalizes by root-mean-square along the feature dimension; no mean
  centering; learned scale parameter initialized to ones.
- Formula: RMSNorm(x) = (x / rms(x)) × γ, where rms(x) = sqrt(mean(x²) + ε), ε=1e-8.

Training hyperparameters (standard, same as Condition A):
- Peak LR: 1×10⁻³, floor LR: 1×10⁻⁴, 20-step warmup, cosine decay.
- Hypothesis: RMSNorm is stable enough at standard LR since it preserves magnitude
  normalization. If divergence occurs (same criterion as B), drop to LR 2×10⁻⁴.

---

## Verdict criteria

| Outcome | Verdict |
|---|---|
| A forms (n_syk_near ≥ 5), B does not (n_syk_near ≤ 1), C forms (n_syk_near ≥ 5) | H1 CONFIRMED (LN necessary), H2 CONFIRMED (ℓ2 sufficient) |
| A forms, B does not, C does not | H1 CONFIRMED (LN necessary), H2 FALSIFIED (centering also load-bearing) |
| A forms, B forms (n_syk_near ≥ 5), C forms | H1 FALSIFIED, H3 (null) CONFIRMED — normalization not necessary |
| A forms, B diverges, C forms | H1 PARTIAL — normalization block cannot be separated from training instability; report as contingency |
| A forms, B forms, C forms (all comparable) | H3 CONFIRMED |
| A does not replicate (n_syk_near ≤ 1) | Protocol failure — Condition A is the anchor; if it fails, the comparison is void |

Threshold for "forms": n_syk_near ≥ 5 (lower bound of exp-062 C-NAT range).
Threshold for "does not form": n_syk_near ≤ 1 (consistent with formation-failure
corpora in the ladder: C-SR, C-generated, C-NAT-shuf).
Inconclusive zone: n_syk_near ∈ {2, 3, 4} — reported as partial.

---

## Kill conditions

**K1 (replication failure):** Condition A produces n_syk_near ≤ 1 at step 2000.
This invalidates the comparison (A is the positive control). Investigate protocol
deviation; do not conclude from K1 alone.

**K2 (stability confound):** Condition B diverges (NaN or loss > 8.0 at step 64).
H1 cannot be decided from a diverged run. Report n_conformal at the last stable
checkpoint; note as contingency. Design a follow-up with even lower LR if the
program wants to pursue it.

**K3 (RMSNorm instability):** Condition C diverges at standard LR. Drop to 2×10⁻⁴
per the pre-registered fallback; this is not a kill, only a protocol branch.

---

## Bears on

- **A1 (census exponent):** If LN is necessary for formation, the census number is
  entangled with the normalization mechanism, not purely conformal structure.
- **G7 (why 1/4):** If LN is necessary, the normalization fixed point is a candidate
  for the value. If not, G7's open question stands unchanged.
- **T1:** Consistent with the six-experiment restatement thread (exp-138, 161–166):
  provides one more constraint on what "the census measures" and what "the geometry
  requires."

---

## Notes on register and scope

This experiment tests formation in the 70m ladder architecture on natural text
(C-NAT). Results apply to: one architecture class, story-scale formation, the
frozen random-token census protocol, step-2000 checkpoints. They do not apply to:
pre-trained models (GPT-2, Pythia, etc.); the *value* of the exponent; whether LN
accounts for the specific Δ ≈ 0.25 once the geometry forms. The formation ladder
already handles the value question (LN present in all formation-failure conditions).

Results will be surfaced in: `research/physics/OVERVIEW.md` (formation section),
`research/physics/theory/interior_horizon_theory.md` (A1 block, G7 construction
site), and the T1 restatement note (when written).

---

*Pre-registration written and committed in attention-geometry before any training
script was written or any forward pass was run. 2026-10-08, ~4:30 AM MDT, physics
room, Cursor, solo.*
