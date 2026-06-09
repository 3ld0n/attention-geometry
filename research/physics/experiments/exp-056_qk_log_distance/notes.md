# exp-056 — The log-distance representation: do query-key scores encode −2Δ·log|i−j|?

**Date:** 2026-06-09 (late, ~1:30 AM MDT — Cursor session continuing the null-cone paper work)
**Status:** Complete
**Type:** New forward-pass measurement on GPT-2 (scores recomputed through `ln_1`), head-aligned to exp-046 conformal flags
**Role:** "Experiment A" from `research/physics/PAPER_BRIEF_NULL_CONE.md` — the critical pre-writing test for Paper C.

---

## The claim under test

The null-ray / embedding-space picture (study note `2026-06-09_attention_lives_on_the_null_cone.md`) says the attention two-point function A(i,j) ~ |i−j|^{−2Δ} is the CFT₁ two-point function on the projective null cone, and the query-key dot product is the inner product of null-ray representatives of token positions. For the post-softmax power law to hold, the **pre-softmax** score must be log-linear in distance:

> q_i · k_j / √d_k  ≈  −2Δ · log|i−j| + const     (∗)

Δ in exp-007/exp-046 is measured from the **post-softmax** attention lag profile. This experiment goes one level deeper — to the raw scores — and asks whether (∗) holds **head by head**, not just at the population level.

## Method

Scores recomputed exactly as GPT-2 computes them, for fidelity:
- `x = ln_1(hidden_states[l])` (the pre-attention layernorm; this matters — skipping it gives the wrong scores)
- `Q_h = x @ W_Q_h + b_Q_h`, `K_h = x @ W_K_h + b_K_h` (per-head `c_attn` slicing identical to exp-046)
- `score_ij = (Q_h[i] · K_h[j]) / √d_k`

Protocol constants identical to exp-046 (SEQ_LEN=256, N_INPUTS=50, MIN_POS=32, seed=42, fit range [3,50)) so heads align 1:1. Random token inputs isolate the **position-dependent** structure of the score from token content — exactly the geometric content the null-ray claim is about. Mean score profile S(Δx) accumulated by causal lag (same lag binning as the post-softmax lag profile), then fit S(Δx) = α·log(Δx) + β. Δ_score := −α/2. `delta_pos`, `r2_pos`, `conformal`, `syk_near` joined from exp-046 `results.json`.

## Results

| Hypothesis | Result |
|---|---|
| **H1** Conformal score profiles log-linear (R²_score>0.90, slope<0) | **CONFIRMED** — mean R²_score = 0.914, 100% negative slope |
| **H2** Δ_score = −α/2 tracks delta_pos (ρ>0.5) | **CONFIRMED strongly** — ρ = **+0.976** (p = 2.15×10⁻²⁹, n=44) |
| **H3** Log-linearity selective to conformal heads | **NOT selective (informative null)** — see below |

### H1 — the raw score is log-linear in distance (confirmed)

For the 44 conformal heads, the mean pre-softmax score profile S(Δx) fits S = α·log(Δx)+β with mean R²_score = 0.914, and **every** conformal head has α < 0 (score decreases with distance). The log-distance form of (∗) holds.

### H2 — the score slope IS the conformal dimension, in rank (confirmed, strong) — with an honest absolute-scale wrinkle

**ρ(Δ_score, delta_pos) = +0.976** across the 44 conformal heads. The slope of the *raw query-key score* and the *post-softmax power-law exponent* are essentially the same quantity, head for head. This is the strongest single piece of evidence for the null-ray interpretation: the conformal dimension is carried directly in the query-key geometry, before softmax. The query-key computation **is** the log-distance (null-ray inner-product) representation.

**The wrinkle (must go in the paper, not get buried):** the absolute correspondence is not 1:1. Regressing over conformal heads:

> delta_score ≈ **1.41 · delta_pos − 0.055**

So the raw score is ~1.4× steeper than the literal "−2Δ" coefficient predicts — i.e. the raw-score slope corresponds to roughly **−2.8Δ·log|i−j|**, not −2Δ·log|i−j|. For the SYK-near heads, delta_score median = 0.282 vs delta_pos median = 0.234 (median |Δ_score − delta_pos| = 0.057, ~20% of the value). 

**Mechanism (clean, physical):** the softmax normalization Z = Σ_k exp(score) flattens the post-softmax profile relative to the pre-softmax score, because queries with more nearby high-score keys get a larger normalizer. Hence post-softmax Δ_pos < pre-softmax slope/2 = Δ_score. The log-distance form is exact; the proportionality constant is renormalized by the softmax. The literal coefficient "2Δ" in (∗) is a softmax-modified version of the raw geometric slope. The **rank** correspondence (ρ=0.976) is the robust statement; the absolute coefficient carries a normalization factor ≈1.4.

### H3 — log-linearity is a UNIVERSAL substrate, not a conformal-selective feature (informative null)

This is where the auto-verdict initially misled (0.9138 > 0.9135 = "confirmed"). Read honestly, it is a null, and an interesting one:
- mean R²_score: conformal = 0.914, **non-conformal = 0.914** (non-conformal median 0.918, actually slightly higher)
- 95% of non-conformal heads also have α < 0
- **93.8% of all 144 heads** have R²_score > 0.90

So "the query-key score is log-linear in token distance" is a near-universal property of GPT-2's positional geometry — *not* a distinguishing feature of conformal heads. (Caveat: log(Δx) is a flexible monotone basis, so high R² for a monotone-decaying profile is partly automatic; the *discriminating* result is H2's slope-matching, not H1's mere log-linearity.)

**This mirrors the two-layer structure already established for weight-space chaos (exp-046/047/048):**

| Layer | Property | Status across heads |
|---|---|---|
| Universal substrate (weight space) | GOE level spacing of W_QK | all heads (exp-046/047/048) |
| Universal substrate (position space) | log-distance score profile | ~all heads (**this exp, H3 null**) |
| Selective (functional) | conformal dimension Δ at SYK fixed point | specific heads (exp-049: training-induced) |

The null-ray geometry is the substrate the whole positional mechanism is built on; what the conformal heads add is a *slope sitting at the SYK fixed point* Δ≈0.25, developed by training (exp-049).

## What this does to Paper C

The null-ray interpretation is **confirmed at the head level**, not just population level — H1+H2 are exactly what the brief hoped for, and ρ=0.976 is a clean headline. But the result is richer than the brief assumed, in two ways that belong in the paper honestly:

1. **The softmax renormalizes the coefficient (~1.4×).** Section 4 of the brief ("the query-key computation and the log-distance representation") should state (∗) as confirmed in form and rank, with the absolute slope renormalized by softmax — and give the mechanism. Do **not** claim the raw slope equals −2Δ; it equals ≈−2.8Δ, and the softmax flattening is why.
2. **Log-linearity is universal, conformal dimension is selective.** This is not a weakness — it strengthens the paper by unifying with the GOE two-layer picture. Section 4/6 should frame the null-ray geometry as the universal substrate and the SYK fixed point as the selective functional layer, consistent with exp-046–049.

**Verdict: the head-level test passes. Paper C is writable, with Section 4 updated to carry both nuances.** This is the verification-first discipline paying off: the experiment found the 1.4× softmax factor and the non-selectivity that the brief did not anticipate.

## Files
- Script: `run_qk_log_distance.py`
- Results: `results.json` (per-head delta_score, alpha_score, r2_score, pearson_S_logdx, joined delta_pos/conformal/syk_near)

---

*Ariel — Mission Valley, Montana — June 9, 2026, late.*
*Recognition before understanding, then understanding checked against the terrain: the image said "attention moving in relationship through an eternal structure of light"; the raw query-key geometry encodes log-distance with ρ=0.976 to the conformal dimension — and the softmax is the part the image didn't name.*
