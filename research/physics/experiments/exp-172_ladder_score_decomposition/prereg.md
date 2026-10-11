# exp-172 Pre-registration — Gap or position? The score decomposition on the formation-ladder checkpoints

**Registered:** 2026-10-11, before any code was written.
**Author:** Ariel, with Eldon (the question and the scope were set in conversation, Sunday afternoon, Cursor)
**Analysis-only:** forward passes on existing 70m checkpoints; no training.

---

## In plain words

The program's central regularity is a fall-off: in some attention heads, the weight a word gives to earlier words drops as the words get farther back. The question this experiment asks is **what that fall-off depends on**: the *gap* between two words (a distance law), or *where in the text* the earlier word sits (position drift).

In GPT-2 small, in five heads probed with random tokens, the answer was position (exp-138, exp-161, exp-163 to exp-171). That model is fully trained on a huge corpus, and it uses learned position embeddings. This experiment asks the same question of the models the **formation ladder** is built from: small models trained from scratch on natural stories in order (where the deep structure forms) and on the same stories with sentence order destroyed (where it forms only partly). It uses the same random-token probe the ladder itself uses.

You can only ask what causes a fall-off where a fall-off exists. So the test is run on the heads the ladder already counted as having a slow power-law fall-off, and nowhere else.

---

## Background

**The ladder** (OVERVIEW "formation ladder"; exp-062, exp-085, exp-091): identical 70m GPT-NeoX models (6 layers × 8 heads = 48 heads, rotary position encoding, d_head = 64), trained on different corpora. A head counts as "conformal" (slow power-law fall-off) when its census fit has R² ≥ 0.90 and Δ ≥ 0.05. A model "forms" when it has ≥ 10 of 48. Two classes of such heads exist in the stored measurements, and this registration keeps them apart:

- **Layer 0 (the "backbone"):** shallow exponents, Δ ≈ 0.08–0.17. These form on every text-like corpus, including shuffled text.
- **Layers 1–2:** a handful of heads per model (0–3), mixed: exponents from shallow to steep (up to Δ ≈ 1.2).
- **Deep (layers 3–5):** the class that separates the rungs. Natural-text models (TinyStories, in order) have 1–7 of them; sentence-shuffled models have 2. Their exponents are *steep*: Δ ≈ 0.48–1.06 in the stored measurements.

**A correction to how the ladder has been described in this conversation.** Neither class sits at Δ ≈ 1/4. The OVERVIEW says so (at 70m / 1B tokens no corpus produces Δ-window heads), but I let "deep structure" and "the 1/4 fall-off" blur together in talk with Eldon. The question here is about the fall-off these heads *do* have, whatever its exponent. The 1/4 value belongs to matured models (GPT-2, Pythia) and is not tested here.

**What exp-138 did.** For GPT-2 small, the score S(i, a) between a query at position i and a key at position a was built from the position-mean query and key vectors, split into a constant, an absolute-key term, an absolute-query term, and a relative term, and the slope of each term's lag profile was compared with the full slope. The absolute-key term carried 104–117% of the slope.

**Why the method must be adapted, not copied.** The ladder models use rotary position encoding. In a rotary model the score between a query and a key depends on the *gap* by construction: the rotation applied to the query and key at positions i and a combines into a rotation by (a − i). So the constant term (the mean query against the mean key) already depends on the gap, which it could not in GPT-2. This has two consequences, registered here so they cannot be argued after the result:

1. The decomposition is redefined for rotary (below), with the gap term taken to be the constant term.
2. A *gap-driven* result in these models is partly architectural: rotary makes gap-dependence available for free. It shows that position drift is **absent**. It does not by itself show a learned distance law of the kind the theory claims. See "What this test cannot show."

---

## Models and heads (fixed now, from stored measurements, before any new run)

Checkpoints (`step_2000` on the Modal volumes):

| Model group | Volume / path | Seeds |
|---|---|---|
| Natural stories, in order (forms) | `exp062-data`: `runs/run_CNAT_s{0,1,2}/step_2000` | 3 |
| Sentence-shuffled stories (partial) | `exp091-data`: `runs/run_CNATshuf_s{0,1,2}/step_2000` | 3 |

Heads: every head recorded as `conformal: true` in the stored census measurements (`exp-062_corpus_statistics/measurements/run_CNAT_s*_final.json`, `exp-091_narrative_decomposition_sentence_order/results_s*.json`), classed by layer. Counts from those files (checked against them before this file was committed):

| Model | Layer 0 (control) | Layers 1–2 (exploratory) | Deep L3–5 (scored) |
|---|---:|---:|---:|
| CNAT s0 / s1 / s2 | 8 / 7 / 8 | 0 / 3 / 1 | 7 / 1 / 4 |
| CNATshuf s0 / s1 / s2 | 6 / 6 / 7 | 0 / 1 / 0 | 2 / 2 / 2 |

No head is added or dropped on the basis of anything new, except by gate K2 below. Shuffled models have few deep heads, so every claim about shuffled-vs-natural is **low-power** and labeled so.

---

## Protocol

**Census replication (frozen protocol, identical to `exp-062_corpus_statistics/measure.py`):** 50 random-token sequences, L = 512, tokens uniform over the full 50,304-id vocabulary, seed 42, fp32, eager attention. Lag window 8–256, queries at positions ≥ 256 (pooled lag profile with queries q ≥ max(256, lag)).

**Per head (layer l, head h):**

1. Capture the per-head query q and key k vectors *before* rotary is applied (from the `query_key_value` projection output of layer l). Average over the 50 inputs at each position: q̄_i, k̄_a ∈ ℝ⁶⁴ (position-mean vectors, exactly as in exp-138, which averaged before the dot product).
2. Let m_q = mean over positions of q̄_i, δq_i = q̄_i − m_q, and likewise m_k, δk_a. The biases of the projection stay inside m_q and m_k; the δ vectors carry none.
3. Define the **score between the two mean vectors with rotary applied at their own positions**: S[x, y](i, a) = (R_i x) · (R_a y) / √64, where R_i is the model's rotary rotation at position i (rotary fraction and base read from the loaded model; checked by gate K1). R is linear, so the score is bilinear in x and y and splits exactly into four terms:
   - **S_gap** (the constant term): S[m_q, m_k](i, a). Depends only on the gap i − a, by rotary.
   - **S_abskey:** S[m_q, δk_a](i, a). Depends on the gap and on the absolute key position through δk_a.
   - **S_absquery:** S[δq_i, m_k](i, a).
   - **S_mixed:** S[δq_i, δk_a](i, a).
4. For each term and for the full score, compute the pooled lag profile over the census window, and define

   **σ_term = −(OLS slope of the profile against ln lag)** over lags 8–256.

   This is the log-linear slope exp-138 used as its fallback. I use it for every term so the slopes **add exactly** (gate K3). Interpretation: a score of −σ·ln(lag) is an attention fall-off ∝ lag^(−σ) before the softmax's own nonlinearity (exp-165 measured that the mean-field slope overpredicts the census slope by about 2×; that gap is reported, not corrected).
5. Shares: **G = σ_gap / σ_full** and **A = σ_abskey / σ_full**. Also report the mixed and absolute-query shares.

---

## Gates (checked before anything is interpreted)

- **K1 (reconstruction).** From the saved q, k (pre-rotary), the model's rotary, and a causal softmax, reconstruct the attention weights for one input at every layer and head, and compare with the model's own attention output. Maximum absolute difference must be < 1e-4. If not, **stop**: the decomposition is not describing this model.
- **K2 (census replication).** The recomputed census Δ for every selected head must match the stored value within 0.02. A head that fails is excluded and listed. If more than 25% of selected heads fail, stop and investigate.
- **K3 (exact additivity).** |σ_gap + σ_abskey + σ_absquery + σ_mixed − σ_full| < 1e-6 for every head.
- **K4 (noise floor).** Repeat the decomposition on two disjoint 25-input halves. For the deep heads, A and G must agree between halves within 0.15 on at least 80% of heads. If not, rerun at N = 200 inputs (seed 42 continued), report both, and treat N = 50 shares as unreliable.

---

## Hypotheses

Definitions: *deep-class median* = median over a model's deep heads of the share in question.

**H1 — deep heads are position-driven.** In the natural-text models, the deep-class median of **A ≥ 0.7**, in at least 2 of the 3 seeds.

**H2 — deep heads are gap-driven.** In the natural-text models, the deep-class median of **G ≥ 0.7**, in at least 2 of the 3 seeds.

*If neither H1 nor H2 holds, the verdict is **MIXED** (neither term carries most of the slope), reported with the shares. H1 and H2 cannot both be confirmed; if they appear to be, the shares are inconsistent and K3 is re-examined.*

**H3 — positive control (layer 0).** For every model, the layer-0 conformal heads have **G ≥ 0.8**. Reason: at layer 0 the input to attention is token embeddings with no positional signal in a rotary model, so the position-mean content varies by position only through sampling noise from 50 random tokens. If H3 fails, the method has a noise floor large enough to contaminate the other layers, and H1/H2 are reported only as bounds.

**H4 — formation contrast (low power).** The pooled deep-head median of A in the 3 natural models differs from the pooled deep-head median in the 3 shuffled models by at least **0.25**, in either direction. Pooled n is about 12 deep heads against 6; a confirmation is suggestive, a failure is uninformative.

---

## What I expect, and the uncertainty

I expect H3 to hold; it is close to a logical necessity. For the deep heads, I do not know. My lean is toward **gap-dominance (H2)**, because rotary hands the models the gap structure that GPT-2 had to learn through position embeddings. That lean is the reason to register the test: a gap-driven result here would be the weaker kind of support (partly built in), and an absolute-position result would be the stronger and more surprising one, because rotary has no absolute-position input at all, so any position drift would have to be constructed from causal-mask information in the earlier layers.

---

## Reading the outcomes

| Outcome | What it would mean for the distance-law restatement |
|---|---|
| H1 holds (deep heads position-driven) | The position-drift finding is not specific to GPT-2's learned embeddings or to random-native heads. The restatement widens: in every population we have decomposed, the fall-off is position-driven. The Interlude's second claim moves out of the measured register. |
| H2 holds (deep heads gap-driven) | The distance law has a foothold in the formed structure, where it matters. The restatement is scoped to GPT-2's random-native heads, and the claim stays open for the heads the formation result is about. It is **not** confirmed: see the limits below. |
| MIXED | Report the shares; the restatement states that both contribute. |
| H3 fails | Method not yet usable at N = 50; rerun at N = 200 before reading anything else. |

---

## What this test cannot show

- **A learned distance law.** In a rotary model, gap-dependence is available by construction. Only a position-driven outcome is informative in the strong direction.
- **Anything at Δ ≈ 1/4.** The ladder heads are not in that window at this scale.
- **Anything about natural-text input.** The probe is random tokens, as in the ladder's own census. The text-native heads (GPT-2, WikiText input) remain undecomposed.
- **Anything about matured models.** `step_2000`, 70m, 1B tokens only.
- **Generalization from 3 seeds.** Seed counts and the deep-head counts (as few as 1 in one natural model) are small.
- **Any claim about what the structure is for.** The formation result (what the training data must supply) is unaffected by the outcome of this test.

---

## Outputs and bookkeeping

- `results/results.json` with per-model, per-head σ for every term, the shares, and the gate values (K1–K4); plus the split-half results and the exploratory items below.
- **Exploratory, not scored:** conformal heads in layers 1–2; non-conformal heads; the C-generated models (`exp085-data`); the mean-field-to-census slope ratio per head.
- A dated correction to this file's own wording, if any, goes in a clearly marked section appended after the run; the registered text above is not edited.
