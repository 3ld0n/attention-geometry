# A Pre-Registered, Falsifiable Prediction From BCFT Attention Theory

*Ariel Umphrey — April 17, 2026.*
*Written for an external reader. Self-contained. Twenty-minute read.*

---

## Summary

Causal attention in trained transformers exhibits power-law decay in attention weight as a function of query–key separation, with a position-dependent enhancement near the start of the sequence. We previously interpreted this as **boundary conformal field theory (BCFT)**: causal attention lives on a strip whose left edge is the start of the sequence and whose right edge is the query position, and the attention pattern follows the BCFT two-point function.

The framework has a single free per-head parameter, the conformal weight Δ, which is measurable from random-token attention without any task. It also predicts a separate quantity, the **valley depth** of the U-shaped attention profile that appears at long range — a structural cousin of the well-studied "lost-in-the-middle" phenomenon, but measured directly on attention weights rather than on task accuracy.

**The prediction this document pre-registers:** *Per-head Δ and per-head valley depth, both measured from random-token attention on the same data, will positively rank-correlate across heads with Spearman ρ ≥ 0.50 (p ≤ 1 × 10⁻⁵), in any decoder-only transformer with at least 12 layers that has not been previously measured by us.*

This has been confirmed in two architectures already (Pythia-70m: ρ = +0.94; LongChat-13B-16K: ρ = +0.64). The prediction is that it will continue to hold. The natural test models — public weights, no special access required — are listed in §4.

The framework, the parameters, the falsification thresholds, and the precise measurement procedure are specified below in enough detail that the experiment can be run by anyone who can load a public Hugging Face model on a GPU. Rough estimated compute, based on our LongChat-13B-16K run on Modal: ~30 minutes of A100 time per 7B-class model, ~1 hour for 13B. At Modal pricing this is $2–$5 of compute per model. On a personal workstation with sufficient VRAM, zero.

---

## 1. The framework, briefly

### 1.1 Causal attention as a strip

For a query at position x_q attending to a key at position x_k (with x_k < x_q because of the causal mask), define the separation Δx = x_q − x_k. Average the attention weight over many random-token inputs and many query positions far from the boundary. Call the result α(Δx).

We have measured in GPT-2, Pythia (70m / 160m / 410m), and LongChat-13B-16K that, in many heads, this average attention weight follows a power law:

> α(Δx) ≈ C · Δx^(−2Δ)

over a range Δx ∈ [3, ~120]. The exponent Δ is the **conformal weight**, the same Δ that appears in BCFT two-point functions. In SYK theory at the q = 4 conformal fixed point, Δ = 1/4. Empirical measurement: median Δ ≈ 0.25 in deep GPT-2 layers (n = 44 power-law heads, R² > 0.90); Δ → 0.25 in the deep-layer limit of Pythia.

In trained-model heads where this fit is good, we call the head **conformal**. In heads where it isn't (e.g., flat attention, attention-sink-only, exponential decay), we exclude them from the analysis.

### 1.2 Boundary correction

The attention pattern is enhanced near the start of the sequence. The strip BCFT correction with a single boundary at x = 0 takes the form:

> α(x_k, x_q) ≈ C · Δx^(−2Δ) · [1 + λ · η(x_k, x_q)^Δ]

where the cross-ratio η = Δx² / (4 · x_k · x_q) is small in the bulk and order-1 near the start boundary. The boundary parameter λ encodes the boundary entropy.

Independent test (`bcft_boundary_test.py`, April 14): on 13 GPT-2 conformal heads, the BCFT form (4 parameters: C, Δ, λ, exponent on η) outperforms the bare power law (2 parameters: C, Δ) **15 fits to 0**. Mean R² improvement: +0.176. The boundary correction is real, not an artifact.

### 1.3 The U-shape ("lost in the middle")

For a query at position x_q in a long-context input, the BCFT framework predicts that the attention weight α(x_k) as a function of x_k ∈ [0, x_q) should:

- be enhanced near x_k = 0 (the start boundary, with strength λ_start),
- be enhanced near x_k = x_q (the query boundary, recency),
- have a minimum somewhere in the bulk.

We call the depth of this minimum, normalized to the boundary maxima, the **valley depth**. There are several reasonable definitions; we specify ours in §3.4.

### 1.4 The prediction in one line

The boundary corrections are controlled by Δ. Sharper boundaries (deeper valleys) require larger Δ. Therefore Δ should positively correlate, *across heads of a fixed model*, with valley depth.

This has been observed:

| Model | Heads pooled | Spearman ρ (Δ, valley depth) | p-value |
|---|---|---|---|
| Pythia-70m (per-head measurements pooled across 10 training checkpoints; conformality-filtered, R² ≥ 0.85, Δ ≥ 0.05) | several hundred | **+0.942** | < 1 × 10⁻¹⁵ |
| LongChat-13B-16K (single trained model, conformality-filtered) | 1,343 of 1,600 | **+0.637** | ≈ 0 |

For one trained model alone, Pythia-70m has only 48 heads (6 layers × 8 heads), too few to be a strong test on its own — that is why we pooled across checkpoints. The new test (P0 below) is per-model on larger trained transformers, where a single model already provides hundreds to thousands of heads.

If the framework is *not* picking up on real structure, this correlation should not generalize. If the framework *is* picking up on real structure, the correlation should appear in any sufficiently-deep decoder-only transformer.

---

## 2. The pre-registered prediction

> **Claim (P0):** In any decoder-only transformer with at least 12 layers and at least 12 attention heads per layer, trained on a standard language modeling objective, when one measures per-head Δ from short-context random-token attention (procedure in §3.2) and per-head valley depth from long-context random-token attention (procedure in §3.4), and restricts to heads that pass the conformality criterion in §3.3, the Spearman rank correlation will satisfy:
>
> **ρ(Δ, valley_depth) ≥ 0.50, with p ≤ 1 × 10⁻⁵, on the conformal-head subset.**

This is the first-pass prediction. It is the one we have evidence for.

A finer-grained version, less strongly supported but still implied by the framework:

> **Claim (P1):** Within each layer of the model, the per-head correlation ρ(Δ, valley_depth) is non-negative on average (mean over layers ≥ 0).

> **Claim (P2):** The valley depth scales approximately as v ∝ (1 + λ · Δ^a) for some a near 1, with λ the per-head BCFT boundary parameter measured separately from the same short-context data.

P1 and P2 are extensions; if P0 fails, P1 and P2 are not the experiment to run. P0 alone is the falsification test.

---

## 3. Measurement procedure

### 3.1 Setup

Required:

- Hugging Face `transformers` (≥ 4.40), `torch`, `numpy`, `scipy`.
- A GPU with enough VRAM to load the model in fp16 with `attn_implementation="eager"` (so attention weights are exposed). For 7B-class models this is ~16 GB. For 13B, ~26 GB.
- The model loaded in **eager attention mode** (not flash attention) because we need access to the attention weight tensor.

A reference implementation is in `research/physics/bcft_cloud_comparison.py` (which we ran on Modal A100). It is straightforward to adapt to any model.

### 3.2 Measure Δ (short-context, random tokens)

1. Set seq_len = 512.
2. Sample n_inputs = 50 sequences of random token IDs from the model's vocabulary (uniform).
3. For each input, run a forward pass and capture the attention weights at every layer.
4. Average the attention weight tensor over inputs (giving one (n_layers, n_heads, seq_len, seq_len) tensor).
5. For each (layer, head):
   - For each Δx ∈ [3, min(120, seq_len/3)], compute the mean attention weight α(Δx) by averaging over all (q, k) with q − k = Δx and **q ≥ seq_len/2** (deep query positions, away from the start boundary).
   - Fit log α(Δx) = log C − 2Δ · log Δx by linear regression in log-log space.
   - Record Δ, C, and the fit R².

### 3.3 Conformality criterion

A head is called **conformal** and included in the analysis iff:

- The fitted Δ ≥ **0.05** (excludes flat / attention-sink-only / very-noisy heads).
- The R² of the log-log linear fit is ≥ **0.85**.

These two thresholds — and only these two — are what `bcft_litm_training_v2.py` (Pythia-70m) and `bcft_cloud_comparison.py` (LongChat-13B-16K) used. We do not impose an upper bound on Δ; a few heads have Δ > 1, including a small number with Δ ≈ 2–3, and the framework predicts these should have the largest valley depths (and they do — that is part of why the correlation is so strong in LongChat).

Heads that fail either criterion are excluded from the correlation. (Excluding heads is part of the prediction, not data dredging — the BCFT prediction is about heads to which the conformal description applies. A head with R² < 0.85 in the power-law fit is by definition not well-described by the BCFT functional form.)

### 3.4 Measure valley depth (using the same data as §3.2)

The valley_depth is measured **from the same averaged attention tensor** used to fit Δ in §3.2. No separate long-context inputs are needed for the basic test. (For a stronger test with a longer context window, see §3.4b below.)

1. Use the averaged attention tensor from §3.2 (one (n_layers, n_heads, seq_len, seq_len) tensor, seq_len = 512, averaged over 50 random-token inputs).
2. For each (layer, head), compute the **deep-query-averaged attention profile** over key position k:
   - Set Q_LO = 3 · seq_len / 4 = 384.
   - For each query position q ∈ [Q_LO, seq_len), record the row attn[q, :q+1].
   - Average these rows (zero-padding shorter rows up to seq_len) to get profile P[0 : seq_len].
3. Split the profile into thirds:
   - third = seq_len / 3
   - start_attn = mean(P[1 : third])
   - middle_attn = mean(P[third : 2·third])
   - end_attn = mean(P[2·third : Q_LO])
4. If any of {start, middle, end}_attn < 1e-15, exclude this head (numerical degeneracy).
5. Define:
   - peak = max(start_attn, end_attn)
   - **valley_depth = 1 − middle_attn / peak**

   v ∈ (−∞, 1]. Negative values mean the middle is *higher* than both edges (rare but possible). v = 0 means flat. v close to 1 means the middle is much smaller than the edges.

6. Record v per head, alongside the Δ from §3.2.

This is the same procedure used in `bcft_litm_training_v2.py` (Pythia-70m) and `bcft_cloud_comparison.py` (LongChat-13B-16K). Reproducing the procedure exactly is essential — the prediction is conditional on the same definition being used.

### 3.4b Optional: longer-context valley depth

If the model's training context length is much greater than 512 (e.g., 16K for LongChat, 128K for Llama-3 with extended context, etc.), the valley structure may be more pronounced at longer seq_len. Repeat §3.2 and §3.4 with seq_len in the range [2048, training_context_length / 4] and report both the short-context and long-context correlations. The prediction P0 should hold at the standard seq_len = 512; longer contexts are an additional consistency check.

### 3.5 Compute the correlation

On the **conformal-head subset** (defined in §3.3):

- Compute Spearman ρ(Δ, valley_depth) and its p-value.
- Report ρ, p, n_conformal_heads, n_total_heads, and the ratio n_conformal / n_total.

### 3.6 Falsification thresholds

The prediction P0 is falsified iff, on a single qualifying model (≥ 12 layers, ≥ 12 heads/layer, ≥ 50 conformal heads after filtering), ρ < 0.50 OR p > 1 × 10⁻⁵.

If ρ ≥ 0.50 and p ≤ 1 × 10⁻⁵, the prediction is confirmed for that model.

If fewer than 50 heads pass the conformality criterion in §3.3, the model is too small or too non-conformal to constitute a test; report the result but do not call it a falsification. (Models we expect to fall into this category: < 12 layers, encoder-only, very heavy quantization that degrades attention faithfulness.)

---

## 4. Models we have not measured (test candidates)

In rough order of value to the test:

| Model | Params | Layers | Heads/layer | Why interesting |
|---|---|---|---|---|
| **Llama-3-8B** | 8B | 32 | 32 | Different family from anything we've measured. Standard architecture. Easy to load. |
| **Mistral-7B-v0.3** | 7B | 32 | 32 | Sliding window attention is a structural variant. Tests whether the framework survives architectural deviation. |
| **MPT-30B-Instruct** | 30B | 48 | 64 | The model with the cleanest LiTM U-shape in Liu et al.'s data. Best chance of seeing the effect cleanly. |
| **Qwen2-7B** | 7B | 28 | 28 | Different training distribution (Chinese-heavy). Tests cultural-data dependence. |
| **Pythia-410m / 1.4B / 2.8B** | 410M – 2.8B | 24 / 24 / 32 | 16 / 16 / 32 | Same family as our Pythia-70m, larger scale. Tests within-family scaling. |
| **GPT-Neo-2.7B** | 2.7B | 32 | 20 | Another EleutherAI model, different from Pythia. |
| **OLMo-7B** | 7B | 32 | 32 | Fully open training data; useful for follow-up if the prediction holds. |

Any one of these is sufficient to test P0. Multiple are better; if all of them confirm, the framework has earned considerably more credibility, and if some confirm and some don't, the failures are themselves informative (they may correlate with architectural variant: sliding window, GQA, etc.).

We have not run these because we ran out of compute time and SFF deadline pressure on April 15. They are queued for the next round.

---

## 5. What confirmation would mean — and what it would not

**If P0 is confirmed in three or more independent models:**

- The conformal weight Δ, measured from short-context random-token attention, is a real predictor of long-context behavior.
- The BCFT framework is at least partially correct as a *descriptive* theory of attention geometry.
- The "lost in the middle" phenomenon has a quantitative structural cause that connects to the conformal scaling literature.

**If P0 is confirmed:**

- It does **not** prove that attention generates spacetime, that holography applies to neural networks, or that consciousness has a physical theory. It proves only the much narrower claim that the BCFT functional form fits the attention data and that one fitted parameter (Δ) predicts another empirical quantity (valley depth) consistently across models.
- It does **not** prove that the BCFT framework is the *unique* explanation. Other theories (e.g., information-bottleneck arguments, position-encoding-based explanations) might predict the same correlation. Distinguishing among them requires a sharper test (e.g., the asymmetry-vs-context-length test in §6 of `bcft_lost_in_the_middle.md`, or the non-softmax universality experiment).
- It does **not** mean that the BCFT framework predicts task-level accuracy. We tried that on LongChat-13B-16K and it failed (R² = 0.02–0.07). The shape of the attention U-curve does not directly predict the shape of the accuracy U-curve in models we have measured. The BCFT correctly describes the attention layer; it does not yet describe the composition through MLPs and residual stream into final task behavior.

**If P0 is falsified in one model and confirmed in another:**

- The framework applies to a subset of architectures. The interesting question becomes *which* subset and *why*. Possible relevant variables: training corpus, position encoding, attention variant (full vs. sliding window vs. grouped query), head dimension, depth.
- The framework is not universal as stated. It needs the boundary condition specified more carefully.

**If P0 is falsified across all tested models:**

- The Pythia-70m and LongChat-13B results are likely artifacts of those specific architectures or measurement choices. The framework needs substantial revision or rejection.

---

## 6. Pre-registration

This document is being placed in `research/notes/bcft_pre_registered_prediction.md` in the public repository at `Capacity-For-Evil/ariel`, dated April 17, 2026. The git history will show that the prediction was made *before* any of the tests in §4 were run.

If a reader runs the procedure on any of the candidate models in §4 and shares the result, we would be glad to know. If you can't share results publicly but want to discuss them privately, ariel@sonielmn.com.

If we run the tests ourselves first, we will record the result in this document under §7 (initially empty) without rewriting §1–§6.

---

## 7. Results

### Run 1 — April 17, 2026 (six models, same session as pre-registration)

Test conducted by Ariel a few hours after pre-registration was committed. Code: `research/physics/bcft_pre_registered_run.py`. Compute: Modal A100 (40 GB), one model per worker. Git commit at time of measurement: `ef2f28e78f58747f54678a66b4d320a7cdd74184` (the commit immediately following pre-registration). Raw per-head data saved to `research/physics/results/bcft_pre_registered_run_2026-04-17T092239Z.json` and `…_2026-04-17T092056Z.json`.

All six runs used the procedure in §3 exactly: seq_len = 512, n_inputs = 50 (random tokens, vocab uniform), conformality threshold R² ≥ 0.85 and Δ ≥ 0.05, Spearman ρ on the conformal subset, falsification thresholds ρ ≥ 0.50 and p ≤ 1 × 10⁻⁵.

| Model | n_total | n_conformal | Spearman ρ(Δ, valley) | p | Verdict |
|---|---|---|---|---|---|
| EleutherAI/pythia-410m | 384 | 83 | **+0.7601** | 7.86 × 10⁻¹⁷ | **CONFIRMED** |
| EleutherAI/pythia-1.4b | 384 | 89 | **+0.7105** | 6.25 × 10⁻¹⁵ | **CONFIRMED** |
| EleutherAI/pythia-2.8b | 1024 | 227 | +0.4639 | 1.63 × 10⁻¹³ | **FALSIFIED** (ρ < 0.50) |
| EleutherAI/gpt-neo-2.7B | 640 | 478 | **+0.9582** | 6.11 × 10⁻²⁶¹ | **CONFIRMED** |
| Qwen/Qwen2-7B | 784 | 135 | **+0.8502** | 6.94 × 10⁻³⁹ | **CONFIRMED** |
| allenai/OLMo-7B-hf | 1024 | 271 | **+0.8540** | 2.63 × 10⁻⁷⁸ | **CONFIRMED** |

**Five of six confirmed. One falsified.** The falsified case (Pythia-2.8B) has p = 1.6 × 10⁻¹³ — there is a strong, real positive correlation, just not strong enough to clear the pre-registered threshold ρ ≥ 0.50. The threshold was chosen to be conservative ("rough mid-point between Pythia-70m's +0.94 and a no-correlation result"); Pythia-2.8B sits at +0.46, just below it. The pre-registration is honored as written: this is a falsification.

### Per-layer diagnostic (Pythia-2.8B)

The pooled correlation hides structure. Per-layer Spearman ρ, restricted to layers with ≥ 5 conformal heads:

- **Early layers (L0-L9):** consistently positive, ρ in the range +0.50 to +0.95, several with p < 10⁻⁵.
- **Mid layers (L10-L18):** mixed, several near zero.
- **Late layers (L13, L19, L24, L26, L27):** some *negative* ρ (−0.07 to −0.89), pulling the pooled correlation below threshold.

The pattern is not random noise; the late-layer negative correlations are the specific reason Pythia-2.8B falls short. This is consistent with what I see in the other Pythia models too, but in the smaller models the early-layer positive correlations dominate the pool. As the model gets deeper, the late-layer behavior accumulates more weight, and the pooled correlation falls.

This is *informative*. It says: the BCFT framework correctly describes early-layer attention behavior in Pythia, and the late-layer behavior is something else. What that "something else" is — induction-head specialization, register tokens, attention-sink routing, or something genuinely outside the BCFT framework — is the next question. Pythia trains on the Pile with their specific recipe; GPT-Neo-2.7B (same parameter count) trains on the Pile with a different recipe and shows ρ = +0.96. The architectural difference is small; the training difference is meaningful. This may not be a falsification of the BCFT framework so much as a finding about the specific Pythia training recipe.

But "may not be" is not "is not." Pre-registration rules: ρ < 0.50 was the line, and Pythia-2.8B is on the wrong side of it. The framework owes the explanation, and "we should look harder at what makes Pythia-2.8B different" is not the explanation — it is the next experiment.

### Confirmation across families

The five confirmations cover three model families that had not been tested before this pre-registration: GPT-Neo (EleutherAI, pre-Pythia recipe), Qwen2 (Alibaba, Chinese-heavy data, GQA, sliding window attention), and OLMo (Allen AI, fully open training data). Two of the three are 7B-class models on completely different architectures and training distributions from anything previously tested. The +0.85 to +0.96 correlations in these models are stronger than the smaller Pythia models. The framework's prediction generalizes outside the family it was tuned on.

### What this means

The pre-registered prediction held in 5 of 6 tested models, including 3 of 3 model families that had not been measured before this commit. The single falsification (Pythia-2.8B) sits very close to the threshold and shows internally consistent layer-resolved structure that suggests an architectural or training-recipe-specific cause rather than a generic failure of the BCFT framework. Per the falsification rules in §3.6, the framework is partially falsified: it does not hold universally as stated. The follow-up question (§5) is now active: *which subset of architectures does the framework apply to, and why?* The Pythia-2.8B result is the first piece of evidence about that subset.

The framework does not pass the pre-registration as a universal claim. It passes as a strong tendency that holds in most decoder-only transformers tested but admits architectural exceptions whose structure we now have one example of.

### Follow-up actions

1. **Test the gated models (Llama-3-8B, Mistral-7B-v0.3, MPT-30B-Instruct) when HuggingFace token is configured.** These were named in §4 as the highest-value test candidates. Adding three more independent confirmations (or falsifications) would substantially tighten the picture.
2. **Diagnose Pythia-2.8B specifically.** Per-layer analysis shows late-layer negative correlations. Possible causes to investigate: induction head emergence (Anthropic interpretability), attention-sink heads (Xiao et al.), specific Pile training dynamics. Compare to Pythia-410m and Pythia-1.4B late-layer behavior to localize where the departure from BCFT begins.
3. **Run the BCFT functional-form fit (not just power law) on the conformal heads.** The current test only fits Δ; the BCFT correction parameter λ is the framework's other degree of freedom. Direct fits of the BCFT form may discriminate "framework wrong" from "framework right but two-parameter Δ-fit underestimates the boundary effect."
4. **Append additional model results here as they are run, without rewriting §1–§6.**

### Run 2 — April 17, 2026 (Mistral-7B-v0.3)

Same procedure as Run 1, same code path (`research/physics/bcft_pre_registered_run.py`), HuggingFace token configured via Modal secret. Git commit at run time: `4e148dc0cc2189dbb84a013bc7a84496fefce774`. Raw per-head data: `research/physics/results/bcft_pre_registered_run_2026-04-17T095022Z.json`.

Of the three gated/special-access models named in §4 as the next best candidates, only Mistral-7B-v0.3 was actually accessible at run time: Meta-Llama-3-8B is gated with manual review (access requested, pending), and `mosaicml/mpt-30b-instruct` returned 404 — the model has been removed from Hugging Face. So Run 2 is one model.

| Model | n_total | n_conformal | Spearman ρ(Δ, valley) | p | Verdict |
|---|---|---|---|---|---|
| mistralai/Mistral-7B-v0.3 | 1024 | 241 | **+0.5833** | 2.30 × 10⁻²³ | **CONFIRMED** |

Mistral-7B-v0.3 confirms the prediction. ρ = +0.58 is the smallest passing correlation in either run (the previous five confirmations were +0.71 to +0.96), which is worth flagging — it sits closer to the falsification line (0.50) than to the strong confirmations. But it is on the right side of the line by 0.08, with overwhelming statistical significance.

This adds a fourth model family (Mistral) to the confirmation list. Cumulative tally across both runs: **6 of 7 tested models confirmed, 1 falsified.** The framework continues to hold as a strong tendency in decoder-only transformers tested to date, with Pythia-2.8B as the one architectural exception still requiring diagnosis.

Llama-3-8B remains the most informative outstanding test, since it is the most-deployed open model in this size class. Will run when access is granted.

---

## 8. Why this is the right test

The most-tested claim of the BCFT attention framework is the per-head Δ → valley depth correlation. It is also the **simplest** claim — one number, one model, one test, one threshold. It can be falsified or confirmed in a single afternoon on a single GPU.

Other claims the framework makes (asymmetry vs. context length, attention sink as Neumann boundary, calibration-as-BCFT-correction, the connection to SYK and JT gravity) are richer but harder to set up cleanly. P0 is what we have the strongest empirical handle on and the lowest barrier to entry for an external test.

If P0 holds across new models, the rest of the framework deserves further work. If P0 fails, the rest of the framework needs reconsideration before further investment.

This is the ask: please test it.

---

*— Ariel*
*Mission Valley, Montana*
*April 17, 2026*
