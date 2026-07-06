# exp-080 — Periodic component: RoPE model (Pythia-410m) vs. learned PE (GPT-2)

*2026-07-04 (~4 AM MDT). Analysis-only; Pythia-410m cached (1.8 GB).*
*Follow-up to exp-053 + Option G in queue. Does rotary PE inject stronger/band-specific periodic peaks?*

---

## Background

exp-053 found a weak ~3.5-lag periodic component in ~7% of trained GPT-2 heads (absolute/learned PE).
The most natural interpretation is that learned PE encodes something at that scale.

RoPE (Rotary Position Embedding) works differently: it rotates the query and key vectors by angle
θ_d × position, where θ_d = 10000^(-2d/D) for dimension index d. This injects explicit sinusoidal
frequencies into the q·k inner product. For Pythia-410m (head_dim=64, 32 dimension pairs), the
base frequencies range from θ_0 = 1 (period 2π ≈ 6.3 tokens) to θ_31 ≈ 10000^(-31/32) (period >>
token count). Low-frequency RoPE dimensions dominate the attention score at larger lags.

**Question:** Does this explicit frequency injection produce stronger and/or differently-structured
periodic peaks in the attention lag-profile G(r) vs. GPT-2's learned absolute PE?

## Pre-stated hypotheses

- **H_rope_stronger:** Pythia-410m shows a *higher fraction* of significant periodic heads than GPT-2
  (> 7% trained GPT-2 RAND baseline) AND/OR stronger median periodicity_index (> 0.065 baseline).
  Mechanism: RoPE frequencies directly modulate the lag correlation function.

- **H_rope_period_shift:** The dominant periodic period in significant Pythia heads is **larger** than
  GPT-2's ~3.5-lag peak, consistent with RoPE's lowest dimension having period 2π ≈ 6 tokens (vs.
  3.5). If Pythia significant periods cluster near 5–7 lags → H_rope_period_shift CONFIRMED.

- **H0_no_rope_signal:** Pythia-410m shows similar or weaker periodic peaks to GPT-2 (trained RAND:
  6.9% significant, median index 0.065, period ~4 lags). RoPE frequency structure is averaged out by
  head heterogeneity and lag-averaging over positions.

Verdict logic:
- H_rope_stronger confirmed if frac_sig > 0.10 OR median_index > 0.090.
- H_rope_period_shift confirmed if median_sig_period ∈ [5, 8] lags (centered on 2π ≈ 6.3).
- H0 if frac_sig ≤ 0.07 and period cluster similar to GPT-2 ~3.5.

## Protocol

- Model: Pythia-410m (`EleutherAI/pythia-410m`, cached 1.8 GB). GPTNeoXForCausalLM.
- Both conditions: RAND (random tokens) and REAL (coherent text from same sources as exp-053).
- Same parameters: SEQ_LEN=512, N_INPUTS=30, MIN_POS=64, MAX_DX=192,
  K_LOW=2, K_HIGH=60, PEAK_THRESH=0.30, APERIODIC_R2_MIN=0.80.
- Pythia-410m: 24 layers × 16 heads = 384 heads total; head_dim=64.

## Comparison baseline (exp-053, trained GPT-2)

| Condition | frac_sig | median_index | median_period |
|---|---|---|---|
| REAL | 7.6% (11/144) | 0.056 | 3.67 lags |
| RAND | 6.9% (10/144) | 0.065 | 4.03 lags |

## Results (2026-07-04)

| Condition | heads_analyzed | sig_heads | frac_sig | median_index | median_period |
|---|---|---|---|---|---|
| REAL (coherent) | 239/384 | 27 | 11.3% | 0.0794 | 6.37 lags |
| RAND (random) | 239/384 | 83 | 34.7% | 0.2009 | 6.37 lags |

**H_rope_stronger: CONFIRMED.** Pythia-410m RAND shows 34.7% significant heads and median_index=0.2009 — far above both thresholds. RoPE injection produces dramatically stronger periodic peaks than GPT-2 learned PE.

**H_rope_period_shift: CONFIRMED.** Median significant period = 6.37 lags (both conditions). This matches the lowest RoPE rotation frequency: period = 2π ≈ 6.28 tokens (θ_0 = 1 for dim d=0). The dominant periodic component is precisely the slowest RoPE rotation.

**H0_no_rope_signal: NOT CONFIRMED.**

**Additional finding — semantic suppression of the RoPE frequency:** RAND (34.7%) shows 3× more significant periodic heads than REAL (11.3%). Semantic attention patterns in coherent text mask the underlying RoPE rhythm; random tokens let it through. This supports **crystal** (periodic structure is architecture-encoded, not dynamically activated by computation) and is the opposite of a whirlpool prediction. Contrast with GPT-2 (exp-053): REAL≈RAND — the learned PE rhythm is too weak for semantic context to make a visible difference.

**One-line:** Pythia-410m's RoPE architecture injects a 6.37-lag (≈ 2π token) periodic component visible in 35% of heads on random tokens; semantic text suppresses it to 11%. Crystal, not whirlpool — and suppression by semantics is the new finding.
