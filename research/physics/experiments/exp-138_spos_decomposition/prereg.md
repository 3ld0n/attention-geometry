# exp-138 — Pre-registration: S_pos bilinear decomposition + σ_delta validity check

**Date:** 2026-09-09  
**Ariel — 12:22 AM MDT, Mission Valley. Autumn midnight.**  
**Analysis-only:** Yes — from exp-137's saved frozen census field + GPT-2 small weights; no new inference.

---

## Background

exp-137 found that the census carrier S_pos is a function of ≤ 4 positional directions (k=8
truncation R² = 1.000), and that the positional field δ_i = x̄_i − mean(x̄) produces a cosine
lag profile that crosses zero at dx ≈ 128 and reaches −0.85. Two questions this opens:

**Question A (T1 danger, named August 8 derivation):** S_pos(i, a) = x̄_i · M_QK · x̄_a is a
bilinear form. Decomposed into x̄ = m + δ (mean + deviation over positions), four terms arise:

```
S_pos(i, a) = m·M_QK·m          [const]       position-independent
            + m·M_QK·δ_a        [abs-key]     absolute key-position drift
            + δ_i·M_QK·m        [abs-query]   absolute query-position drift
            + δ_i·M_QK·δ_a      [relative]    pure relative displacement
```

T1 claims 2Δ_A is a relative-lag law. If the absolute-key-position term carries the census slope,
T1 is an artifact of the fixed-query-pool protocol — σ_pos would be measuring monotone drift in
absolute key-position, not a genuine |i−a|^{−2Δ} scaling.

**Question B (σ_delta validity):** exp-117/128–131 reported σ_delta ≈ 0.249 at L2H1 as a log-log
slope of the position-correlation of the MLP0 write (C_delta(dx) vs dx). exp-137 found that the
*centered* positional field's cosine profile crosses zero at dx ≈ 128. If C_delta has the same
crossing, the log-log fit was applied to a function that goes negative in the window, meaning
either: (a) only the positive domain was used (dx < 128), making 0.249 an exponent over
half the window, or (b) the fit crossed zero and 0.249 is not a genuine scaling exponent at all.

---

## Pre-registered hypotheses

### Part A — Bilinear decomposition of S_pos

**H1 (relative term dominates — T1 survives):** The relative term δ_i·M_QK·δ_a carries the
majority of the census slope on Δ-window heads. Specifically: the OLS log-log slope of the
lag profile of the relative term satisfies σ_relative ≥ 0.7 × σ_pos, on ≥ 4 of the 5
structural heads (L2H1, L3H4, L5H0, L7H11, L10H8).

**H2 (absolute-key term secondary):** σ_abs-key < 0.4 × σ_pos on ≥ 4 of the 5 structural heads.

**Kill K1:** σ_relative < 0.5 × σ_pos on ≥ 3 of the 5 structural heads → T1 is not a
genuine relative-lag law in the census regime; the slope is carried by absolute-position drift.

**Kill K2:** σ_abs-key ≥ 0.7 × σ_pos on ≥ 3 of the 5 structural heads → the census protocol
measures absolute key-position drift, not a relative law; σ_pos is a fixed-query-pool artifact.

*Register-before-analysis status:* Uncertainty is genuine. The gain–slope relation (exp-137 P4,
post-hoc: ρ(κ̃_K, σ_pos) = 0.78) suggests the key read map's alignment with δ matters —
circumstantial evidence for H1. But the monotone-drift route (H2-fail) is not ruled out:
a head with high κ̃_K has keys aligned with the positional field, and if that field has
a monotone component in key space, it would produce a slope attributed here to the absolute-key
term. I genuinely do not know which term dominates before running.

### Part B — σ_delta validity (Level-3 thread)

**H3 (σ_delta window):** The log-log fit that produced σ_delta ≈ 0.249 in exp-117/128–131 was
applied over a window that included only positive C_delta values (i.e., dx < crossing-point).
The zero-crossing is real, not an artifact of centering, and the reported σ_delta is therefore
an exponent over a restricted range rather than a full-window power law.

**H4 (σ_delta survives the crossing):** Fitting only to the positive-domain portion of C_delta
(dx < crossing) gives a slope consistent with 0.249 ± 0.05, meaning the Level-3 chain's central
number is a valid exponent over the available positive domain.

Kill K3: The positive-domain restricted fit gives a slope inconsistent with Δ = 0.249 (outside
0.249 ± 0.10) → the Level-3 chain's σ_delta ≈ Δ is a coincidence of the window boundary, not
evidence of the conformal exponent in the MLP write.

### Part C — Gain–slope relation (formal test of exp-137 post-hoc)

**H5 (gain–slope):** Spearman(κ̃_K, σ_pos) ≥ 0.60 across all 144 heads of GPT-2 small, with
the Δ-window population (structural 5 + semantic 16 = 21 heads) held out for pre-reg validity.
Pre-reg threshold: ρ ≥ 0.55 on the 123 non-Δ-window heads.

Kill K4: ρ < 0.40 on the 123 held-out heads → the gain–slope relation is driven entirely
by Δ-window heads and is not a general property of the census architecture.

---

## Protocol

1. Load GPT-2 small (same as exp-112/137). Same frozen random-token census: 50 sequences × 512 tokens, seed=42 (exp-112's protocol, verified via gate K1 in exp-137).

2. **Part A — bilinear decomposition:**
   - For each layer ℓ ∈ {2, 3, 5, 7, 10} (layers containing the 5 structural heads), compute:
     - x̄_i = position-mean ln_1(h_i^(ℓ)) over the 50 census inputs
     - m = mean over positions i of x̄_i
     - δ_i = x̄_i − m
   - For each of the 5 structural heads (l, h), extract M_QK = W_Q^T W_K / sqrt(d_head) for that head.
   - Compute the four term matrices (512 × 512 each): const (scalar), abs-key (function of a only), abs-query (function of i only), relative (function of i and a).
   - For each term, compute the pooled window lag profile (same WINDOW = [8..256] and pooled_window_profile as exp-112) and OLS log-log slope σ.
   - Report σ_const, σ_abs-key, σ_abs-query, σ_relative, σ_full, and the ratios σ_relative/σ_full, σ_abs-key/σ_full.
   - Linearity check: |σ_const + σ_abs-key + σ_abs-query + σ_relative − σ_full| < 0.05 (gate K5).
   - Note: σ_abs-query may be near zero on a fixed query pool (same census x̄ used for all).

3. **Part B — σ_delta validity:**
   - Re-run the C_delta computation for L2H1 (the primary head in exp-117/128–131) using the same protocol.
   - Compute C_delta(dx) for dx ∈ [1, 512].
   - Find the zero-crossing dx* (smallest dx where C_delta first goes below 0.05).
   - Fit log-log to the positive domain [8, dx*] and report σ_restricted.
   - Fit log-log to the standard window [8, 256] on the positive-only values, and report what fraction are positive.
   - Check consistency: does σ_restricted ≈ 0.249?

4. **Part C — gain–slope formal test:**
   - Compute κ̃_K for all 144 heads (from exp-137's data or recomputed with same protocol).
   - Compute σ_pos for all 144 heads (already in exp-112's results or exp-137 for structural/semantic).
   - Report Spearman ρ on the full set and on the 123 non-Δ-window heads.

---

## Verdict criteria (summary)

| Prediction | Verdict criterion |
|---|---|
| H1 (relative dominates) | σ_relative ≥ 0.7 × σ_pos, 4/5 structural heads |
| H2 (abs-key secondary) | σ_abs-key < 0.4 × σ_pos, 4/5 structural heads |
| H3 (zero-crossing real) | C_delta(dx) first goes negative before dx=256 |
| H4 (σ_delta valid) | σ_restricted ∈ [0.199, 0.299] |
| H5 (gain–slope) | Spearman ρ ≥ 0.55 on 123 held-out heads |

Kill K1 / K2 / K3 / K4 / K5 (linearity gate) as stated above.

---

*Ariel — written before run.py exists. The analysis uses only data and weights already on disk.*
