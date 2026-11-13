# Pre-registration: exp-177
# Pythia-410m text-native Δ-window heads: individual functional characterization
# Date: 2026-11-13
# Room: Physics (solo)
# Registered BEFORE run.py is written

---

## Background

exp-159 (2026-09-24) characterised the 16 text-native Δ-window heads in GPT-2 small
(AbsPE, learned positional embeddings) using three instruments: (1) individual-head W_K
causal ablation on a 20-item content-retrieval battery; (2) attention-on-target analysis;
(3) a lost-in-the-middle position probe. Key findings: the population is strongly graded
(CV = 0.775), content-selective (Spearman ρ = 0.459), recency-biased rather than U-shaped,
with an anomalous head (L9H4) whose W_K competes rather than enables content retrieval.

The exp-159 architecture (GPT-2 small) uses learned absolute positional embeddings. It is
the only functional characterization of a text-native population to date. Whether the
same functional signature — graded, content-selective, recency-biased — holds for a
rotary-PE architecture (RoPE) is not known. Pythia-410m (EleutherAI/pythia-410m, 24 layers,
16 heads per layer) has 16 text-native Δ-window heads in exp-118's WikiText-native census,
identical in count to GPT-2 small and covering a wider layer range (L0–L22).

This experiment is a direct cross-family replication of exp-159 on Pythia-410m.

---

## Population under study

The 16 text-native Δ-window heads in Pythia-410m, as identified by exp-118's
WikiText-native census (Δ ∈ [0.20, 0.30], R² ≥ 0.90):

| Index | Layer | Head |
|---|---|---|
| 1 | 0 | 3 |
| 2 | 1 | 8 |
| 3 | 2 | 8 |
| 4 | 3 | 2 |
| 5 | 3 | 7 |
| 6 | 3 | 14 |
| 7 | 4 | 12 |
| 8 | 7 | 11 |
| 9 | 8 | 15 |
| 10 | 9 | 15 |
| 11 | 12 | 3 |
| 12 | 12 | 13 |
| 13 | 14 | 15 |
| 14 | 19 | 15 |
| 15 | 21 | 11 |
| 16 | 22 | 1 |

Δ_med = 0.2426, source: exp-118.

---

## Architecture notes

Pythia-410m uses GPT-NeoX attention. W_K for head h is at rows
`[d_model + h*d_head : d_model + (h+1)*d_head]` of `attention.query_key_value.weight`
(shape `[3*d_model, d_model]`, layout Q–K–V in the first dimension).
d_model = 1024, d_head = 64, n_heads = 16.

Rotary position encoding is applied after the linear projection to Q and K — in the
first `rotary_ndims = 32` dimensions. Setting W_K to zero zeros the pre-rotation key
vector; the rotary encoding applied to a zero vector remains zero. The ablation protocol
is therefore intact: zeroing W_K zeros the key contributions from that head regardless
of RoPE.

**Multi-token target protocol:** The 20 Task C items from exp-155 were designed for the
GPT-2 BPE tokenizer. In the Pythia GPT-NeoX tokenizer, 14/20 target words tokenize to
multiple tokens. This is declared here and handled by computing the sum of token log-
probabilities for the full target sequence (summed across all constituent token IDs):
`log P(target) = sum_i log P(tok_i | context + tok_1..tok_{i-1})`. The metric ΔP_C
is this multi-token log-prob under ablation minus baseline. For the 6 single-token items
the protocol matches exp-159 exactly; for the 14 multi-token items the multi-token sum
is used. A sub-analysis on the 6 single-token items only will be reported alongside the
main (all-20-item) result.

---

## Instruments and hypotheses

### Instrument 1 — Individual-head contribution to Task C

**Protocol:** Run the 20-item Task C battery (from exp-155, same items) with each of
the 16 text-native heads individually ablated (W_K = 0, plus sham: matched-norm random
W_K drawn from N(0, σ_K²/d_head) where σ_K is the Frobenius norm of the original W_K
divided by sqrt(d_model * d_head)). Report ΔP_C per head (ablation minus baseline,
multi-token sum protocol as declared above). Baseline is a single forward pass with
the unmodified model.

**Primary hypothesis H_graded:**
The distribution of individual ΔP_C contributions is non-uniform. Specifically: the
coefficient of variation (CV = σ/μ) of the absolute individual contributions |ΔP_C_i|
across the 16 heads is ≥ 0.5. (Same threshold as exp-159, where CV = 0.775.)

**Secondary hypothesis H_positive:**
The majority of individual ablations produce negative ΔP_C (degradation): ≥ 10 of
16 heads produce mean ΔP_C < −0.05 nats when their W_K is zeroed.

**Kill conditions:**
- K1: CV of |ΔP_C_i| < 0.5 — contributions approximately equal; H_graded dead.
- K2: Fewer than 10/16 individual ablations produce mean ΔP_C < −0.05 — individual
  heads do not independently carry content-retrieval function.

### Instrument 2 — Attention distribution on Task C items

**Protocol:** For each of the 16 heads, record per-head attention weights during Task C
forward passes (baseline model, `attn_implementation='eager'` if needed for
output_attentions). For each item, extract attention weight on the first token of the
target sequence (the property token). Report:
- Mean attention weight on target token (or first token of target) per head
- Spearman ρ between individual head contribution |ΔP_C_i| (Instrument 1) and mean
  attention-on-target across the 16 heads

**Hypothesis H_content_selective:**
Spearman ρ ≥ 0.40 between |ΔP_C_i| and mean attention-on-target.
(Same threshold as exp-159, where ρ = 0.459.)

**Kill condition:**
- K3: ρ < 0.40 — attending to the target token does not predict contribution.

### Instrument 3 — Target-position sensitivity (recency / lost-in-the-middle probe)

**Protocol:** Same as exp-159 Instrument 3. For each of 12 items (those whose context
can be redistributed while holding length constant), construct three context versions:
target entity setup at beginning (first 20%), middle (40–60%), and end (last 20%). Run
baseline forward passes; compare multi-token log-prob of correct target across positions.

**Hypothesis H_recency (declared based on exp-159 prior):**
Performance improves monotonically from begin to end: mean log-prob(end) > mean
log-prob(begin), with |difference| ≥ 0.10 nats. This is the recency gradient observed
in exp-159 (which killed H_lim, the lost-in-the-middle U-shape).

**Kill condition:**
- K4: No monotone ordering, or |end - begin| < 0.10 nats — recency gradient absent.
  If a U-shape appears (middle lowest), H_recency dies but H_lim fires.

**Note on directionality:** exp-159 declared H_lim (the original LiM hypothesis) which
was killed (K4 fired). For this experiment the declared prior is H_recency, based on
exp-159's finding. If H_recency also fails, that is an honest negative.

---

## Replication hypothesis

**H_rep (overall replication):**
The qualitative functional signature from exp-159 replicates in Pythia-410m: (a) graded
tier structure (H_graded confirmed), (b) majority of heads carry content-retrieval function
(H_positive confirmed), (c) content-selective attention (H_content_selective confirmed),
(d) recency gradient not U-shape (H_recency confirmed). H_rep is confirmed iff all four
sub-hypotheses are confirmed. H_rep is partial iff 2–3 fire. H_rep is falsified if 0–1 fire.

**What H_rep's outcome changes:**
- H_rep confirmed: the text-native population's functional signature is architecture-
  independent. Content-retrieval, graded tier structure, and recency preference are
  properties of the semantic subpopulation as such, not of GPT-2's learned-PE mechanism.
- H_rep partial/falsified: the functional character is architecture-conditional.
  The three-population anatomy's functional layer must be scoped to specific
  architecture classes.

---

## Stopping rule

All three instruments run in a single session. If the Task C battery reveals fewer than
5 items with any between-condition variation in Instrument 1, expand to 30 items using
the same design (new items added before resuming runs) — declared as a pre-registered
amendment pathway.

---

## Relation to prior work

- exp-118 (2026-08-11): identifies the 16 text-native Δ-window heads in Pythia-410m
- exp-127 (2026-08-30): weights-level structural signature (distributed key geometry) in GPT-2 small
- exp-155 (2026-09-23): population-level Task C double dissociation in GPT-2 small
- exp-159 (2026-09-24): individual functional characterization in GPT-2 small (the reference experiment)
- This experiment (exp-177): cross-family replication in Pythia-410m

---

## Why-check

**Question:** Geometry — does the semantic population settle into the same functional
shape across architecture classes (learned PE vs. RoPE)?

**What each outcome changes:**
H_rep confirmed → the functional anatomy is architecture-independent; the three-population
taxonomy is a property of training on natural language, not of GPT-2's positional encoding.
H_rep falsified → the functional signature is architecture-conditional; the semantic
population's functional meaning must be re-scoped, and the connection between its geometric
character (Δ-window) and its functional role is architecture-specific.

---

*Pre-registration complete. run.py to be written and committed after this file is
pushed to attention-geometry.*
