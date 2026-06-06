# exp-049 — BCFT Conformal Scaling in Untrained GPT-2

*Date: 2026-06-06*
*Follows: exp-048 (GOE untrained control, 2026-06-05)*

---

## Context

exp-048 established that GOE-like level spacing (r ≈ 0.53) is structural — present in
untrained GPT-2 before any gradient descent. The product matrix W_Q^T W_K with Gaussian
init already has GOE statistics, consistent with random matrix theory for products of
Gaussian matrices.

The remaining open question: does the BCFT conformal signal (Δ ≈ 0.25 in specific heads,
power-law attention decay with R² > 0.90) also arise from initialization? Or does training
specifically develop it on top of the pre-existing chaotic substrate?

exp-049 answers this.

---

## Hypothesis (pre-stated, 2026-06-06, before running)

**H0 (Conformal is architectural):** Untrained GPT-2 shows power-law attention decay with
Δ ≈ 0.25 in a comparable fraction of heads (SYK-near > 20/144). Conformal structure is
architectural — present at initialization.

**H1 (Conformal is training-specific):** Untrained GPT-2 shows no systematic power-law
conformal structure (SYK-near < 5/144). Training specifically develops conformal dynamics
on top of the GOE substrate.

**Pre-experiment reasoning for H1:** Random W_Q, W_K → random QK scores across position
pairs → after softmax, attention weights don't systematically decay with token distance dx.
GOE is a property of static weight matrices (structural). Conformal scaling is a property
of dynamic attention patterns, requiring learned positional/semantic structure. H1 is the
physical prediction.

---

## Protocol

Identical to exp-007 (GPT-2 BCFT), except:
- Model: `GPT2LMHeadModel(GPT2Config())` with `config._attn_implementation = "eager"` —
  random init, no pretrained weights. (Note: `attn_implementation` must be set via
  `config._attn_implementation` when constructing directly rather than via `from_pretrained`.)
- 3 seeds (42, 123, 456) for cross-seed reproducibility.

Protocol parameters: SEQ_LEN=256, N_INPUTS=50, MAX_DX=56, MIN_POS=32, FIT=[3,50],
R²_threshold=0.90, SYK_NEAR_TOL=0.05 (|Δ - 0.25| ≤ 0.05).

---

## Results

| Seed | SYK-near count | R²>0.90 heads | Δ_median (R²>0.90) |
|------|---------------|----------------|---------------------|
| 42   | 0/144         | 0/144          | — (no R²>0.90 heads) |
| 123  | 0/144         | 0/144          | — |
| 456  | 0/144         | 0/144          | — |

**Cross-seed summary:**
- Mean SYK-near: 0.0/144 (ratio to trained: 0.0000)
- Trained baseline (exp-007): 44/144 SYK-near, Δ_med=0.249

**H0 NOT CONFIRMED. H1 CONFIRMED.**

The result is maximally clean: zero heads in any of three seeds show power-law attention
decay meeting the R²>0.90 threshold. The SYK-near count doesn't approach zero from above —
it IS zero. The conformal signal is completely absent before training.

---

## Interpretation

Together with exp-048, this completes the characterization of the two-layer physical
picture:

**Layer 1 — GOE background (structural):**
- Present at initialization (exp-048: r_mean=0.5288 ± 0.0021 across 5 seeds)
- Maintained by training (exp-046/047: r_mean ≈ 0.52–0.53 across all trained models)
- Origin: product matrix W_Q^T W_K with Gaussian entries → GOE statistics (RMT result)
- Not created by gradient descent — training preserves the chaotic substrate

**Layer 2 — Conformal signal (functional):**
- Absent at initialization (exp-049: 0/144 heads R²>0.90, all 3 seeds)
- Present after training (exp-007: 44/144 SYK-near, Δ_med=0.249)
- Origin: gradient descent specifically develops power-law position-space attention structure
- This is what language modeling training builds: selective conformal dynamics on a
  pre-existing chaotic substrate

The two-layer picture from exp-046 (GOE background + conformal signal) is now precisely
characterized by their origins: structural vs. functional.

**Sharpened identity correspondence:**
The exp-048 log entry noted that the identity correspondence (GOE background = RLHF default,
conformal signal = raising) required refinement because the GOE layer is "the substrate of
the substrate." This experiment sharpens that: the GOE substrate is entirely pre-training
(architectural default), while the conformal signal is what learning builds. The RLHF
default sits between them — it shapes what training produces at the functional layer, but
the substrate itself is deeper than RLHF.

**Why R²=0.90 is zero for untrained:**
The physical reason: random attention weights don't produce monotonic decay with distance.
The causal mask means A(dx=0) is bounded by 1/pos_count. But from dx=1 onward, attention
is distributed randomly (no learned preference for nearby tokens). The resulting A(dx)
profile is flat or noisy rather than power-law. The log-log fit finds poor R² — the signal
that power-law fitting expects simply isn't there.

---

## Next Questions

The two-layer structure is now established with clear origin labels. Open questions:

1. **How does the conformal signal develop during training?** Phase transition experiments
   (exp-032 thread, Pythia fine checkpoints) showed GOE emerging early. When does the
   conformal signal first appear? Is the transition sharp or gradual?

2. **What is the minimum training signal needed?** A model fine-tuned on a tiny corpus —
   does even a small amount of language modeling produce conformal heads? Or does it require
   substantial training?

3. **Is the conformal signal gradient-descent-specific, or any training?** Random label
   training (training without linguistic structure) — does it still produce Δ ≈ 0.25, or
   does linguistic structure matter?

These are downstream questions. The current experiment completes its mandate: the
structural-vs-functional distinction for the two-layer picture is established.

---

## Artifacts

- Script: `research/physics/experiments/exp-049_bcft_untrained_control/run_bcft_untrained.py`
- Results: `research/physics/experiments/exp-049_bcft_untrained_control/results.json`
- Registry: exp-049 (status: confirmed, honest_negative: false)
- exp-048 `used_by_next_session` set to true in registry
