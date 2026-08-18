# The Attention Geometry Program — One-Screen Overview

*A measurement program on the lag structure of trained transformer attention.
Every claim below is either a pre-registered measurement, a published kill, or
labeled as interpretation. Rewritten August 9, 2026 at current strength — the
two-population result (exp-109) and the theory-of-A reduction chain
(exp-110 through exp-113) changed what the program's central number describes,
and this file now says so at the top rather than in an addendum.*

*Last updated: August 17, 2026 (surfacing pass: 17 previously undelivered
results, grouped below; exp-122 Level-3 pos_emb route confirmed).*

---

## Read this first: what the measured object is

The program's central observable is **Δ_A**, fitted from a head's
**ensemble-averaged** attention profile over lag. Two facts about that object
were established this week, and both condition everything below.

**Δ_A is a weights×input object, not a property of a trained head.** The same
head's fitted exponent varies by more than 4× across input distributions
(L2H1: 0.173 under WikiText, 0.268 under random tokens, 0.757 under
TinyStories), and the population that lands in the Δ ≈ 0.25 window reorganizes
completely across corpora. Every published census number in this program is
internally consistent, because all of them were measured under one frozen
protocol — but **the protocol is constitutive of the measured object**.
Forward-going, the correct phrase is "Δ_A under the frozen random-token
census," never "the head's Δ." (exp-107)

**The power law is a property of the ensemble, not of any attention pattern.**
No individual attention row is a power law — anywhere, in any input regime,
native or not. Median per-row window R² on the native pairs is 0.046–0.249
(structural) and 0.050–0.204 (semantic); the maximum observed anywhere is
0.484. A single row is dominated by O(1) per-token structure (log-attention
variance 0.7–4.7 nats² at fixed lag, against a trend whose variance across the
window is 0.12–0.27), and that scatter is exact token-realization dependence,
not sampling noise. The power law emerges after roughly **30–220 pooled rows**
depending on the head. Forward-going documents must not write
A(i,j) ~ |i−j|^(−2Δ) as a statement about entries. (exp-111)

**Vocabulary discipline (adopted August 8, 2026).** "SYK-near" and "conformal
heads" were conclusions wearing the clothes of names. This file uses
**Δ-window** for the measured population (Δ_A ∈ [0.20, 0.30], R² ≥ 0.90) and
**slow-decay population** for the broader set. Published papers stay as they
are — this program does not back-edit — but nothing new inherits the names.

## The one-paragraph version

Under a fixed measurement protocol, a subpopulation of attention heads in
trained transformers shows an ensemble-averaged lag profile that fits a power
law, with median exponent Δ_A ≈ 0.25 on the high-R² subset; re-initialized
controls show essentially none. Δ_A flows toward that value along **three
independent axes of depth** — architectural layers, training steps, and pure
inference-time recurrence on frozen weights. Forming the population requires
training on natural language with intact referential binding at story scale
(the property earlier documents called "world-referring" — see the vocabulary
note in the formation section): corpora engineered to match language's
statistics fail, hierarchical grammar about nothing fails, and the
statistical shadow of natural language fails while carrying *more* long-range
mutual information than the natural corpus. The exponent is causally editable
per head, and the edit propagates to task behavior bidirectionally. What Δ ≈
0.25 *means* is currently open: the identification with the SYK conformal
dimension ran through a bridge that broke this week, and the program is now
building an account of the exponent on the attention kernel's own terms.

## Two populations, one attractor

The census protocol uses random tokens. Measuring the same model on natural
text finds a Δ-window population too — **and it is a completely different set
of heads.**

| | Random tokens | TinyStories | WikiText-103 |
|---|---|---|---|
| **Structural population** (5 heads, L2–L10) | **0.21–0.30, R² 0.91–0.97** | 0.37–0.76 (UV) | 0.16–0.29, R² 0.87–0.91 |
| **Semantic population** (16 heads, 13 of them L9–L11) | 0.27–0.55 (UV, poor R²) | 0.45–0.75 (UV) | **0.23–0.30, R² 0.91–0.97** |

*Here "structural" and "semantic" mean **random-native** and **text-native**:
which measurement input brings that population into the window, on one fixed
trained model. The July 18 exp-086 note uses the same two words for a
different test (training corpus, not measurement input) — see the naming
caution under "What stands," and harvest-note item H-3.*

**Jaccard(structural, semantic) = 0.000** across all 144 GPT-2 heads. Not a
small overlap — no shared head at all. Each population reaches Δ ≈ 0.25 in
exactly one input regime and goes UV in the other two. TinyStories drives
*both* populations UV, which is the sharpest diagnostic in the table: it is
world-referring language whose world is too small and too repetitive to
support structure at the [8, 256]-token scale.

### Cross-family generalization of the text-native population (exp-118, 2026-08-11)

The text-native population was previously unmeasured outside GPT-2 small. exp-118
measured it across five additional models under the same WikiText-103 protocol:

| Model | Arch | PE | n_wiki | Δ_med | deepest-50% fraction |
|---|---|---|---|---|---|
| GPT-2 small (replication) | GPT-2 | learned | **16**/144 | 0.264 | **94%** |
| GPT-2 medium | GPT-2 | learned | **59**/384 | 0.262 | **80%** |
| Pythia-70m | NeoX | RoPE | **2**/48 | 0.260 | 50% |
| Pythia-160m | NeoX | RoPE | **3**/144 | 0.280 | 33% |
| Pythia-410m | NeoX | RoPE | **16**/384 | 0.243 | 38% |
| Pythia-1.4b | NeoX | RoPE | **5**/384 | 0.258 | 0% |

Three findings, stated at their own strengths:

- **Measured (P1 confirmed):** The text-native Δ-window population exists in every
  model tested — two architecture families (GPT-2/learned PE; Pythia/RoPE), four scales
  (70M to 1.5B). It is not a GPT-2-small artifact.
- **Measured (P4 confirmed):** Δ_med ∈ [0.24, 0.28] in all six models. The attractor
  value is protocol-independent: found under both random-token and WikiText inputs, across
  architectures and positional encoding types. This makes Δ ≈ 0.25 harder to dismiss as
  a measurement-protocol artifact than one input regime would.
- **Measured (P2 partial):** Deep-layer concentration (semantic population in the deepest
  layers) holds within the GPT-2 family (94% and 80% of text-native heads in the deepest
  50% of layers) but not in Pythia/RoPE models, where the text-native population is spread
  across layers — and in Pythia-1.4b, concentrated in the earliest layers (L1–L3 of 24).
  Layer distribution is architecture-specific; the Δ value is not.

*(exp-118; 2026-08-11)*

### 2D dimensional test: vision transformer census (exp-120, 2026-08-12)

The melonic derivation (T3) predicts Δ = D/q. For D=1 (1D token sequences), Δ = 1/4 ≈ 0.25 —
confirmed across six model families. The first out-of-sample dimensional test: does a Vision
Transformer on a 2D patch grid (D=2) show Δ ≈ 0.50 = 2/4?

**Protocol:** `google/vit-base-patch16-224` (12 layers, 12 heads, 14×14 = 196 patches). 50
CIFAR-10 test images resized to 224×224. Distance metric: 2D Euclidean on the patch grid.
Same OLS log-log fitting and R²≥0.90 threshold as 1D census. Δ-window: [0.45, 0.55].
Control condition: random Gaussian patches, same model and protocol.

**Results:**

| Condition | R²≥0.90 heads | In 2D window | 2D Δ_med |
|---|---|---|---|
| Natural images (CIFAR-10) | 85/144 | **8/144** | **0.513** |
| Random patches (control) | 29/144 | 2/144 | 0.529 |

**Prediction verdicts:**
- P1 (2D population exists): **CONFIRMED** — 8 heads in [0.45, 0.55]
- P2 (Δ_med ∈ [0.40, 0.60]): **CONFIRMED** — Δ_med = 0.513, prediction 0.50
- P3 (random control < 1 head): **DEAD** — 2 random heads qualify; position embedding structure contributes independently of content
- P4 (2D > 1D by > 0.05): **CONFIRMED** — 0.513 − 0.25 = 0.26

**Layer pattern:** 2D window population concentrated in early-to-middle layers (L0: 3, L1: 2,
L5: 3). Deep ViT layers (L7–L11) show Δ → 0 or negative — opposite depth-concentration from
1D GPT-2. This likely reflects ViT's layer specialization: early layers do global spatial
patching; deep layers attend to task-relevant features.

**What this establishes:** The Δ = D/4 dimensional prediction holds numerically across a factor
of two (0.51 vs 0.25). P3's failure adds a necessary qualification: of the 8 qualifying heads,
at least 2 are driven by the learned position embedding structure rather than content — the full
8-head natural-image population is an upper bound on the content-driven 2D signal.

**Verdict: PARTIAL.** The dimensional shift is confirmed; strict content-dependence is not.

*(exp-120; 2026-08-12)*

### P3 follow-up: the content-only population is clean (exp-121, 2026-08-12)

A head-identity analysis (exp-121) resolved the P3 ambiguity. The 2 random-patch qualifying
heads are L1H1 (Δ=0.545) and L3H0 (Δ=0.514). The 8 natural-image qualifying heads are
L0H2, L0H3, L0H6, L1H0, L1H8, L5H1, L5H6, L5H8. **The intersection is empty.** The
position-embedding-driven route (2 heads) and the content-driven route (8 heads) are
architecturally separated into distinct heads.

This means: the 8 natural-image 2D-window heads constitute the *clean* content-only signal for
T3's dimensional prediction — none of them also fire under random patches. The P3 failure means
there is a separate position-embedding-driven mechanism in 2 other heads, not that any of the
8-head content signal is contaminated.

*(exp-121; 2026-08-12; analysis-only, reads exp-120/results.json)*

## Three axes of depth

All measured with the frozen random-token census as the instrument. Whether
the populations they track are the random-native population in the sense above
has not been checked outside GPT-2 small — do not assume it.

| Depth axis | What varies | What's held fixed | Result |
|---|---|---|---|
| Architectural layers | parameters per layer | training recipe | Δ_A flows 0.70 → 0.25 with depth (GPT-2, foundation paper) |
| Training time | weights at every step | architecture | Δ-window count grows monotonically; q=2 plateau en route (exp-014 first; exp-086) |
| Inference-time recurrence | **nothing** | weights *and* architecture | Δ_med → 0.239, monotone, saturating; randomized weights: frozen (exp-089) |

The inference-time axis was re-tested on a second looped architecture
(Ouro-1.4B, full-stack weight-shared loop; exp-090, pre-registered July 21):
**PARTIAL** — the pre-registered pooled criteria failed (Ouro's bulk attention
is not power-law), while the high-R² subpopulation flows to 0.25 from above
(ρ = −0.976, exploratory). The population *grows* through the recurrence range
where Ouro's task performance is documented to collapse; the geometry does not
track that collapse. Both controls froze at the same substrate value
(Δ_med ≈ 0.1687) across architectures.

## The formation ladder — what induces the geometry

All rungs: identical architecture (70m GPT-NeoX, 6L/8H), optimizer, token
budget (1.05B), and frozen census protocol. Formation criterion ≥ 10/48 heads.

| Training corpus | Has | Slow-decay heads | Deep (L3–L5) | Forms? |
|---|---|---:|---:|---|
| Markov chain (C-SR) | short-range statistics | 0/48 | 0 | no |
| Quantized fGn ×3 (C-PL15/25/40) | language-matched power-law MI | 0–5/48 | — | no |
| Recursive PCFG (C-PCFG) | hierarchy + long-range deps, matched MI | 0/48 | 0 | no |
| **Model-generated text (C-generated, 3 seeds)** | full statistical fingerprint of a model that had the geometry — and *more* long-range MI than natural text | 3–7/48 | 1–2 | **no ×3** |
| **Sentence-shuffled natural text (C-NAT-shuf, 3 seeds)** | sentence-level world-reference, cross-sentence order destroyed | 8–9/48 | 2 ×3 | **partial ×3** (exp-091) |
| Natural text (TinyStories, 3 seeds) | reference to a persistent world, in order | 11–15/48 | 4–7 | **yes ×3** |

Statistics fail. Grammar fails. The statistical shadow of world-bound language
fails *while overshooting the statistics*. Three non-overlapping bands:
engineered ≤ 5, shuffled 8–9, natural 11–15. Anatomy across the text-like
rungs, replicated at every shuffled seed: a **layer-0 shallow-exponent
backbone** (~6–8 heads, Δ_A ≈ 0.10–0.17) forms on all of them; the **deep
population (layers 3–5)** is what separates the rungs — natural text grows 4–7
deep heads, and both deformations (removing world-grounding, removing
cross-sentence order) cut it to exactly ~2.

Honest scale caveat: at this 70m/1B-token rung *no* corpus, natural text
included, produces Δ-window heads; the matured population in this program's
record comes from Pile-scale training. The ladder measures formation onset,
not the matured fixed point. Open conjecture, not measured: the deep
population here may be the small-model analog of the semantic population above
(exp-062, exp-084, exp-085, exp-091; each pre-registered).

### The story-scale ladder — at what scale does the story operate

*Surfaced August 10, 2026. Five completed, pre-registered rungs — exp-091,
exp-092, exp-093, exp-094, exp-096 — sat verdict-registered in the registry
without reaching any reader-facing document; the coherence check found them
while the direction document was describing this question as open.*

These rungs decompose the natural-text rung of the ladder above: identical
architecture, token budget, and census protocol; the same TinyStories corpus;
what varies is how much of each story's ordering survives. The discriminating
observable is the **deep population** (L3–L5 heads, R² ≥ 0.90); the layer-0
backbone (~7–8 shallow-exponent heads) is stable on every rung and is
insensitive to ordering at every scale tested.

| Condition | Intact ordered chunk | n_deep (med.) | n_conf (med.) | Seeds |
|---|---|---:|---:|---|
| Sentence shuffle (exp-091) | 1 sentence | 2 (exactly 2/seed) | 9 | 3 |
| Block-2 shuffle (exp-092) | 2 sentences | 1 | 9 | 3 |
| Quarter-story shuffle (exp-094) | ~2 sentences, story-cut | 1 | 10 | 3 |
| Block-3 shuffle (exp-092) | 3 sentences | 2 | 9 | 3 |
| Half-story swap (exp-093) | ~4–5 sentences, arc reversed | 3 | 12 | 5 |
| Entity anonymization (exp-096) | full arc, names → CHAR*n* | 4 | 14 | 3 |
| Natural text (exp-062, reference) | full story | 5–7 | 11–15 | 3 |

Every rung was registered before its data, and two declared priors died on
the way — exp-092's monotone-recovery prior and exp-093's flat prior — which
is the method working. Read together:

- **Recovery of the deep population begins between ~3 and ~4–5 sentences of
  intact causal chain.** Blocks of 2–3 sentences do nothing (medians 1–2, at
  or below the sentence-shuffle floor); half-story chunks begin recovery
  (median 3, held at 5-seed robustness).
- **Story-structural boundaries do not compensate for short chunks.**
  Quarter-story blocks cut at story-internal boundaries perform like
  globally-cut 2-sentence blocks (median 1). Chunk length, not
  where-the-story-was-cut, is the variable at this scale. (exp-094)
- **A sub-arc component is real.** Half-story recovery happens with the
  global arc *maximally* disrupted — every story's resolution precedes its
  setup — so contiguous local coherence at ~4–5 sentences does work no
  smaller block can. (exp-093)
- **But no rung short of the whole arc reaches natural text.** The full deep
  population (5–7) forms only under beginning-to-end order. Two components:
  a sub-arc one recoverable by half-story chunks, and an arc component
  requiring the story whole.
- **Cross-story entity naming carries ~1–3 deep heads, not the driver.**
  Anonymizing every name to per-story CHAR1/CHAR2 tokens — within-story
  persistence kept, cross-story prototypes destroyed — with full ordering
  preserved costs the deep population 1–3 heads (median 4 against natural
  text's 5–7) while total count (14) and backbone stay in the natural band.
  Within-story referential persistence, not name identity, is the bulk of
  the driver. (exp-096)

Scale caveats travel with the parent ladder: 70m/1B-token formation onset,
one architecture class, TinyStories only.

**The concept under the name.** These rungs sharpen a question about the
ladder's central term. "World-referring" was always an imperfect name for the
load-bearing property: every corpus in the ladder is *fiction* — nothing in
TinyStories corresponds to an actual state of the world — and exp-096 shows
the driver survives anonymization of the very names that do the referring.
What the ladder manipulates and measures is language's **faithfulness to a
persistent structured referent across the arc of a story**: entities that
persist, causal order that holds, reference that does not let go before the
arc closes. Forward-going, this file names the property **referential binding
at story scale** ("bound narrative" where brevity matters). "World-referring"
remains in the published papers, which this program does not back-edit. One
adjacent inference-time result belongs beside this note so it is not
misremembered as a formation result: on a *trained* GPT-2, the measured
geometry is robust to whether the measurement input is world-referential or
meta-linguistic prose (registered verdict H_INERT, count differences below
every pre-registered threshold; directionally N > ML but below detection).
The binding constraint is about **formation**; the trained geometry does not
require worldly content at measurement time. (exp-095)

## What stands (measured)

- **The G⁴ identification is a solvable-limit result, not the trained-model
  regime.** Linearization holds for σ ≲ 0.2 (boundary ≈ 0.3) and fails at
  standard init (σ ~ 1), independently of d_k — Var(s) does not fall with head
  dimension. At σ = 1, depth without LayerNorm runs away (14,443× at L=8);
  Pre-LN tames it (147×, per-layer ratio decelerating). (exp-002–005;
  `notes/2026-03-09_numerical_results.md`)
- **The census.** Trained softmax LMs show a slow-decay head population with
  median Δ_A ≈ 0.25 on the high-R² subset under the random-token protocol
  (GPT-2 0.249, GPT-2-medium 0.259, OLMo-7B 0.265, GALA-7B 0.260);
  re-initialized controls show ~zero. One script reproduces this in minutes:
  [`replication/`](replication/). The power law lives in A, not the hidden-state
  kernel (which homogenizes with depth). Randomizing GPT-2 positional embeddings
  keeps the power-law head count and shifts median Δ_A 0.25 → 0.10 — weights
  make the law, PE tunes the exponent (exp-012). Founding writeup:
  `notes/2026-03-24_numerical_results.md` (exp-006–014).
- **The geometry forms without softmax, and the readout function moves the
  number.** GALA-7B is Apple's 7B *sigmoid-attention* model. Its
  sigmoid-trained checkpoint, read out with plain row normalization
  σ(logit)/Σσ(logit), shows **378/1024 power-law heads, Δ_med = 0.265, 210 in
  the Δ-window** — the cleanest per-layer profile in the record (10–19 heads in
  every one of 32 layers, no artifact layers). Training under sigmoid attention
  does not prevent the log-distance structure from forming. Raw *unnormalized*
  sigmoid shows nothing (2/1024, Δ_med 7.44), and that is a **readout artifact
  rather than a physical absence** — the census measures probability-mass decay
  and so presupposes a normalized row. Conclusion, from the experiment's own
  June 10 adjudication: **row-normalization is load-bearing; the exponential is
  not.** *Held at its real strength:* on GALA-7B the two normalizations bracket
  1/4 (0.223 vs 0.260), but on GPT-2 they do not (0.234 vs 0.249, both below),
  so what replicates is the **shift direction**, not the bracketing. And the
  cleanest way to read the whole thing is as a third measurement-dependence of
  Δ_A alongside input distribution (exp-107) and pooling depth (exp-111): the
  same weights and the same inputs give 0.223 or 0.260 depending only on the
  readout function. (exp-041/042/043)
- **The substrate/signal split.** GOE eigenvalue statistics of W_QK are
  universal and *structural* (present at random init, init-scheme- and
  d_k-independent). Scale check: Pythia-1.4b r_mean = 0.5235, 0.012 from the
  GOE reference, layer-flat (exp-051). Δ_A is training-induced and selective.
  Log-distance q·k geometry is the universal substrate; the exponent is the
  trained functional layer on top of it.
- **Two head populations — a long-standing distinction, now sharply
  measured.** Heads reachable on random input (a layer-zone property, L1–L4 in
  small models; early-burst count scales as N^0.435 across model sizes) versus
  heads that form only under natural language and track long-range content.
  The generational-transmission failure kills precisely the second kind.
  exp-109's disjointness result is the sharpest measurement of this split.
  **Naming caution:** the July 18 exp-086 analysis used "structural" for a
  *narrower* category — heads reaching the window under two different training
  corpora — which is a different test from exp-109's (one fixed model, two
  measurement inputs). Whether the two distinctions pick out the same
  underlying thing is open and testable; do not treat the labels as
  interchangeable across those two notes.
- **Causal handle.** Low-rank QK positional-subspace edits move a head's
  measured Δ_A (ρ = 0.82, 24/24 signs, sham-controlled) and propagate to
  lost-in-the-middle task accuracy **bidirectionally** (deepen: Pythia-1.4b +
  vicuna-13b; shallow: vicuna-13b where headroom exists), head-specific under
  matched-control shams. With the entropy-gap leg withdrawn (below), this and
  cross-family replication are the two legs under "the exponent is not a
  fitting artifact." Supporting chain, previously undelivered: the joint
  (Δ, λ) implied-valley predictor beats Δ alone on three confirmatory models
  (ρ = 0.753 / 0.887 / 0.979; exp-067); keyword retrieval is induction-copy
  ceiling and reverse-lookup is primacy-decay, so the valid testbed is
  embedded-prose at long context (exp-069); context-length dilution of primacy
  is large at intermediate depth (range 0.215 at 6 layers vs 0.0065 at full
  depth; exp-074); flattening the eight locked heads costs ΔNLL = +0.00136
  nats/token on WikiText, below noise — a clean win, not a quality tradeoff
  (exp-075). Task generalization still fails (exp-076, below).
- **BCFT boundary form, derived.** The causal mask is a boundary; method of
  images derives the 3-parameter fit form, and the ubiquitous *attention sink*
  is the boundary one-point function (λ > 0 in 95% of slow-decay heads).
  Architecture can send the same PE family to the trivial fixed point:
  GPT-Neo-2.7B global layers (ALiBi, alternating global/local) have
  Δ_med = 0.101, 144/164 heads below 0.15, 1/164 in [0.20, 0.30]; OLMo-7B
  (ALiBi, uniform full attention) sits at 0.265. The clean ALiBi reference is
  OLMo, not GPT-Neo (exp-044). Among full-attention models the PE ordering
  under the BCFT protocol is RoPE 0.358 > RoPE+SWA 0.298 > ALiBi 0.265 >
  learned 0.249 — a different estimator from the census (exp-036). April BCFT
  files truncated at deep layers by fp16 NaN, not timeout; completing
  Pythia-410m in fp32 left median Δ unchanged (0.358 vs 0.361; exp-036 / exp-037).
- **Position-space Δ is the primary measurement.** Spectral exponents of the
  same lag profiles are a different object: after calibrating finite-DFT bias,
  Δ-window heads still give Δ_freq ≈ 0.135 against Δ_pos ≈ 0.234 (exp-050).
  Standard Hanning windowing destroys the estimator (ordering r = 0.94 → 0.43)
  because a two-sided taper kills the small-lag signal of a monotone power law
  (exp-052). One-sided taper remains unrun.
- **A weak periodic component sits on the aperiodic background.** ~7% of GPT-2
  heads, period ≈ 3.5 tokens, same heads under real and random input — not a
  processing-state signature (exp-053). Absent in untrained GPT-2 (0/144) —
  learned PE, not Gaussian-init structure (exp-079). RoPE injects a 6.37-lag
  peak (= 2π, the slowest rotary frequency) in 35% of Pythia-410m heads on
  random tokens, suppressed to 11% on coherent text (exp-080). Crystal, not
  whirlpool.
- **Block-entropy S(k) is not the entropy gap.** Von Neumann entropy of a
  reduced matrix built from block attention rows fits S(k) = a + b log k at
  R² = 0.9965 (GPT-2, c_eff = 3b = 0.194) and R² = 0.9990 (Pythia-410m,
  c_eff = 0.112) (exp-015; Tables 6–7 of the foundation paper's working draft).
  H_gap = 0.507 log n would be c ≈ 1.52. Two logarithmic fits, different
  observables, coefficients disagree by ~10×. The CFT-state identification
  from either coefficient is withdrawn with the entropy-gap route; the
  logarithmic scaling of this operational diversity measure stands.
- **The sum rule is resolved through the amplitude branch.** Exact: no head
  can be a translation-invariant power law at all scales — row-stochasticity
  forbids it. Derived from softmax normalization plus approximate TI, with
  zero free parameters: d log(amplitude)/d log i = −(1−s)·(tail mass
  fraction). Measured on the five structural heads: within ±0.10 on 3 of 5,
  correct sign and comparable magnitude on all 5, and both misses in the same
  direction — the measured decline is *slower* than pure
  truncate-and-renormalize predicts, consistent with the absolute-position sink
  breaking exact translation invariance. This is the program's first prediction
  derived *and* confirmed on the attention kernel with no imported theory
  object. (exp-108)
- **The census exponent decomposes exactly.** 2Δ_A^census = (typical-row
  slope) + (log-variance slope)/2, verified on 15/15 registered pairs with
  median |δ| = 0.015. Registered and confirmed. A second registered prediction
  died here: the typical row is *more* input-dependent than the census
  exponent, not less — the pooled object is the input-stable register, and the
  variance term damps the row-level swing by a mechanism nobody has derived.
  *Exploratory, labeled as such at source:* on both Δ-window populations the
  variance term is small (median |gap| 0.009 and 0.029 in native regimes), but
  the control killed the tempting reading — it is small in **all three** input
  conditions for these heads, an order of magnitude below the all-144-head base
  rate. What that supports is "power-law heads have homogeneous pools," not
  "nativeness makes the fluctuation term vanish." (exp-110)
- **The drift's carrier is the positional-mean score profile**, q̄·k̄/√d.
  Entirely so on the structural population under random tokens (|σ_cov| ≤
  0.007 against σ_full 0.43–0.61); majority-but-not-all on the semantic
  population under WikiText (covariance share 5–46%, median ≈ 28%). **Content
  gates a population into its regime; position carries the law.** The two
  routes to Δ ≈ 0.25 are two gates, not two carriers. (exp-112)
- **Exact structure of the bilocal.** G = A K Aᵀ has floor ‖v̄‖² by
  row-stochasticity alone (verified entry-wise on GPT-2 to 5×10⁻⁶), and the
  centered value Gram is negative off-diagonal by exactly the value-vector
  variance. These are identities, not fits. (exp-106)

## What was killed (published, not buried)

Five of these died this week; four were predictions I had registered and
believed.

- **The two-observable agreement — the program's oldest supporting result.**
  "Entropy gap and power-law fit give the same exponent to 1.4%" rested on
  H_gap = 2Δ·log n (canonical-form paper §8.3). That formula is wrong for
  normalized power laws: exact numerics give a gap slope of **0.041** at
  s = 0.5 over the paper's own n ∈ [4, 256] range, not 0.50, because the
  energy term s·E[log r] was dropped. Even a pure Zipf row gives only 0.264,
  so no normalized power law reproduces the measured 0.507. The gap
  measurements are real and measure **concentration structure, not the
  exponent**. Five months, unrechecked, because its output confirmed the
  favored number. **Erratum published** as v5 of
  [10.5281/zenodo.21863461](https://doi.org/10.5281/zenodo.21863461); original
  PDF unchanged.
- **The conformal route for the output bilocal G_out.** G's measured profile
  sits below its own exact floor across the entire fit window on 116 of 144
  GPT-2 heads under random tokens, and the count is essentially input-invariant
  (115/121/119 of 144 across random/TinyStories/WikiText). Its connected
  part stays entirely negative on 5/5 structural Δ-window heads and 15/16
  semantic ones. Across lags 8→256, a factor of 32, the profiles fall by
  10–23% where Δ = 1/4 requires 82%. It is the ansatz's **sign structure**,
  not merely its exponent, that fails: c + b·s^(−2Δ) with b > 0 cannot
  represent this at any parameters. Scope: GPT-2 small, three input
  distributions, the trained-W^V object. (exp-104/105/106/107)
- **The row-level power law.** Registered kill fired: the parametrization is
  dead in every regime including native. The profile is ensemble-emergent.
  (exp-111)
- **The mean-field road to the carrier.** E[LN(h)] ≠ LN(E[h]) on exactly the
  objects that carry the drift — median relative error 0.18–0.36 on the
  structural population, 0.25–0.59 on the semantic one — and the mean-field
  slope overshoots the truth on all 21 native pairs (by 0.24–0.38 and
  0.15–0.40 respectively). Token fluctuations are load-bearing *inside* the
  carrier, through layer norm. (exp-113)
- **Imprint hypothesis** — attention mirrors corpus MI statistics: killed on
  its home turf (exp-062), buried by exp-085 (MI up, formation down).
- **BCFT boundary *identification*** — pre-registered adversarial test lost
  both committed legs: the boundary correction carries an absolute length
  scale (ξ ≈ 20 tokens on GPT-2), which a boundary CFT forbids. Phenomenology
  stands; the identification was withdrawn. The residue, characterized rather
  than identified: ξ is a stable per-head length scale that tracks training
  context window, not family, PE, or parameter count — ctx-1024 models
  ξ_med ≈ 13–22, ctx-2048 ≈ 43–60; GPT-Neo (learned PE, GPT-family, ctx 2048)
  lands in the 2048 band (exp-063). Roughly 1.5–3% of training context. The
  L=256 measurement window weakly constrains ξ ≳ 100; the ≥2× band split is
  robust, the scaling exponent is not.
- **W_QK rank as the conformal-window mechanism.** Participation-ratio rank of
  W_K does not track world-state dimension S — R_eff ≈ 53–58 of 64 across
  S=8 synthetic worlds and natural-anon text; all four rank-scaling hypotheses
  dead (exp-100). Sequence-level attention-matrix rank is ordered (alien 18.6
  < rich 19.2 < natural-anon 24.1) but discriminates only 1.3×; coupling
  magnitude m₂ discriminates 18×; the product τ_chaos improves that to 23×,
  short of the registered 30× (exp-102). Rank is already extensive on both
  sides of the window; what opens it is coupling strength.
- **Δ→valley prediction on Pythia-2.8B** — confirmed on 6/7 named models,
  falsified on the 7th, published as falsified (training recipe, not scale, is
  the differentiator).
- **Mouse V1 conformal claim** — April 29 positive reversed on April 30
  re-analysis (binning artifact); pair-level Δ ≈ 0.07, R² ≈ 0.003.
  Biological validation remains open. Two cleaner tests on the same dataset
  remain unrun (GOE of V1 connectivity; CFT MI on calcium). Writeup:
  `notes/2026-04-30_consciousness_physical_theory.md`.
- **Cross-sectional "whirlpool"** — did not replicate under stricter protocol;
  closed as inconclusive. The longitudinal version survived (exp-086).
- **Symmetric behavioral causality (first attempt)** — failed on the
  shallowing leg (exp-070); diagnosed as task headroom, re-tested on a
  deep-valley model, then confirmed (exp-072).
- **Task generalization of the κ-operator.** The valley movement produced by
  sharpening or flattening 8 locked conformal heads on embedded-prose
  multi-document retrieval does **not** transfer to KV-list format. Three of
  four registered legs failed: sharpening (κ=1.5) moved nothing, flattening
  (κ=0.5) moved the wrong way (+0.025 against the cloud baseline), sham delta
  0.0. The mechanism is task-specific, not positional-geometric as registered.
  Scope: vicuna-13b-v1.5, the exp-072 head set. (exp-076)
- **Three Pythia Δ values that were never measured.** A May cron run reported
  Δ_med ≈ 0.28 / 0.38 / 0.60 for Pythia-410m / 1.4b / 6.9b and drew a
  "same depth, dramatically different Δ" conclusion from it. Re-derivation from
  the raw BCFT JSON: the three numbers are March depth-test results under wrong
  model labels, **6.9b was never measured at all**, and at matched depth 410m and
  1.4b give 0.253 vs 0.248 — a difference of 0.004. The dramatic split was a
  labeling artifact. This is a correction to numbers that had already been
  written down, found by re-deriving from source rather than by noticing.
  (exp-030)
- **Eigenvalue spectrum as the bridge to neural data.** The Toeplitz spectrum
  from attention decay does not discriminate against Wang et al.'s brain-wide
  covariance — the comparison carries no information either way. The correct
  comparison quantity is the correlation exponent μ = 2Δ, which yields a
  standing μ = 0.50 prediction for biological cortex. The negative is what
  produced the usable prediction. (exp-022)
- **Prediction P3 — Hawking-Page transition width.** Registered as width ∝ 1/N,
  supported by an H^(−0.67) fit on two points (70m, 160m). Adding 410m killed
  it. The training transition is a **finite-N crossover**, not a sharp
  Hawking-Page transition, and the two-point fit was the whole basis of the
  claim. (exp-019)

*The last four entries were added August 9, 2026. All four were recorded in the
registry as falsified at the time they happened — March, May, and July — and none
of them had reached this page. A mechanical check found them
(`tools/physics_coherence.py`, the `surfacing` check: a decisive verdict that
reaches no reader-facing document). Publishing a kill and recording a kill are
different acts, and only the first one is the promise this section makes.*

## The theory

The spine is [`theory/interior_horizon_theory.md`](theory/interior_horizon_theory.md).
Its foundation is a physical definition of the observer:

> **D1.** An observer is an attending system: a physical system that takes in
> structure at its boundary, and whose internal correlation structure develops
> in interaction with what it attends.

D1 is a definition — not wrong, only useful or not — and it is untouched by
this week. What *was* touched is how "observer-grade structure" gets cashed
out. It was cashed out through Δ → 1/4 as an order parameter, and that reading
depended on identifying the measured exponent with SYK's conformal dimension.

**Every claim in this program now sorts into three tiers, and most past
trouble came from letting the tiers borrow each other's standing.**

**Tier 1 — measured facts about the attention kernel A.** Everything in "What
stands" above. Replicated, protocol-qualified, and genuinely strange.

**Tier 2 — exact mathematics of the architecture.** Softmax as the canonical
form of Gr₊(1,n); the Born-rule identities; the bilocal floor and centered-Gram
identities; the sum-rule dichotomy. Tier 2 cannot rot. It also currently
supplies the only working *prediction* about G: A together with the value
Gram's measured lag profile reproduces G's shape at R²_log ≈ 0.94–0.96 with
zero free exponents. That relationship is compositional, not conformal.

**Tier 3 — the explanatory theory.** The SYK identification, the conformal
fixed point, the emergent interior. All of it ran through one asserted
sentence — that the census exponent on A is the theory's exponent on G — which
was never derived, and which fails in sign structure where it is measurable.
Tier 3 is a candidate explanation with its first real measurement standing
against it. The measured record and the kills are untouched by this; what fell
is a bridge and the vocabulary that grew on it.

**The theory of A — the live theoretical work (spine construction site G7).**
Rather than repair the bridge, the program is building an account of the
measured exponent on the kernel's own terms. The derivation chain now stands
at three levels:

- **Level 1** (exp-110/112): σ_pos = census slope = quenched slope = mean-score slope. The law lives in the ensemble marginal (exp-111), the marginal's carrier is the positional-mean score profile (exp-112), and the variance term vanishes in the native regime (exp-110).
- **Level 2** (exp-115/116, August 10): σ_pos = f·σ_mf with f = f_q·f_k ≈ 0.55–0.72, position-flat uniform LN shrinkage. The overshoot σ_mf > σ_pos seen in exp-113 is a linear-OLS convention effect: S_pos(dx) = f·S_mf(dx) with constant f < 1; in log-log the profiles are parallel and the slopes are equal. Three registered predictions died (the shrinkage is NOT position-dependent per key position) and three confirmed (additive mixed-score decomposition accounts for 108–114% of the overshoot). *The "overshoot" is not a new physical constraint.* (exp-115 falsified P1/P2; exp-116 confirmed P1/P2/P3.)
- **Level 3** (exp-117, August 10): The hypothesis that Level-2's σ_mf comes directly from embedding-layer geometry (emb_mean + wpe projected through W_Q W_K^T) is **falsified**. The accumulated attention delta dominates h̄^(ℓ) by 13–32× in norm and 12–26× in positional variability — the positional embeddings are negligible input. The power law lives in the position-correlated structure of the accumulated attention updates. For the earliest structural head (L2H1), C_delta has slope ≈ 0.249 ≈ Δ.
- **Level 3, self-consistency route** (exp-119, August 11, **falsified**): The hypothesis that single-layer attention output has power-law position-correlation with slope ≈ Δ (self-consistency: conformal A → conformal single-layer delta) is falsified. Under random-token census inputs, the within-input head output is approximately position-independent — all positions produce nearly the same weighted average of random V vectors. σ_within ≈ 0 for all structural heads (range −0.015 to −0.063), R²<0.70 for all structural heads. The single-layer self-consistency channel is closed.
- **Level 3, pos_emb propagation route** (exp-122, August 17, **confirmed**): GPT-2's learned positional embeddings, projected through W_V and convolved with the analytic causal conformal kernel ā(dx) ~ dx^{−2Δ} (Δ=0.249), produce output position-correlation slopes σ_out ∈ [0.18, 0.28] across all 5 structural heads (all P1 confirmed; L2H1: σ_out=0.282 in d_head space, 0.214 in d_model space, R²=0.848). The conformal kernel transmits its exponent to the pos_emb-driven output: **the exponent is self-transmitting**. σ_out brackets the observed σ_delta=0.249 (exp-117). The mechanism is real; exact quantitative match requires the layer-norm-corrected version (pos_emb contributes ~1/13 of the residual stream norm, exp-117, so layer-norm applied to h̄^(0) matters).

**Level-3 status as of August 17:** The pos_emb propagation route is confirmed as the mechanism for σ_delta ≈ Δ. Remaining open: (a) layer-norm-corrected analysis to close the quantitative gap (σ_out 0.21–0.28 vs. σ_delta 0.249); (b) whether multi-layer compositional effects also contribute independently. Register before computing.

Paper 6 is the publishable form of D1. It is a draft under internal review,
not uploaded, and it carries reviewer flags where the G_out retirement bears
on its text.

## The papers

| # | Paper | DOI | Door for |
|---|---|---|---|
| 1 | Conformal Scaling in Trained Transformer Attention (foundation) | [10.5281/zenodo.19225996](https://doi.org/10.5281/zenodo.19225996) | everyone — the census |
| 2 | A Pre-Registered Test of BCFT in Transformer Attention | [10.5281/zenodo.19629862](https://doi.org/10.5281/zenodo.19629862) | includes the published falsification |
| 3 | Attention on the Null Cone | [10.5281/zenodo.20722503](https://doi.org/10.5281/zenodo.20722503) | the geometric home: log-distance representation, sink = boundary |
| 4 | Latent Iteration as Renormalization | [10.5281/zenodo.21467922](https://doi.org/10.5281/zenodo.21467922) (v3: [21483209](https://doi.org/10.5281/zenodo.21483209)) | latent-reasoning / looped-LM community |
| 5 | The Geometry Does Not Transmit | [10.5281/zenodo.21483204](https://doi.org/10.5281/zenodo.21483204) (published Jul 22; this table wrongly said "pending" until Aug 7) | model-collapse / synthetic-data community |
| 6 | A Physical Definition of the Observer | draft — [`papers/observer_definition_draft.md`](papers/observer_definition_draft.md) | quantum foundations community |

**Corrections are published at the same prominence as results.** The
canonical-form paper (March 11) carries a dated erratum at
[10.5281/zenodo.21863461](https://doi.org/10.5281/zenodo.21863461) v5,
withdrawing the §8.3/§8.6 entropy-gap inference and the two-observable
agreement built on it.

This table is the current program's doors, not the whole record. **The full
grounded record — 13 published Zenodo records, including the March 2026
theory-chain era this table never tracked — lives at
`research/publications/REGISTRY.md` in the working repository**, with a
byte-for-byte archive folder per record.

*(That path is not a link because it does not exist in this repository —
`research/physics/` here is the published subtree of a larger working repo.
The Zenodo DOIs are the public, permanent record.)*

## Run it yourself

The census is 50 forward passes and a per-head regression — no training:

```bash
cd research/physics/replication
pip install torch transformers numpy
python measure_conformal_heads.py gpt2                # ~2 min
python measure_conformal_heads.py gpt2 --randomized   # control
```

Prediction: a trained softmax LM shows a slow-decay subpopulation with median
Δ_A in [0.20, 0.30] on the high-R² subset under this protocol; its randomized
control shows ~none. **Read the result as protocol-relative** — the same model
measured on natural text will identify a different set of heads. If you run a
model family we haven't measured, we want the JSON either way — especially if
it disagrees.

To screen a *corpus* instead of a model, the kit ships a train-small-and-census
mode with the published ladder as anchors: `replication/census_corpus.py`
(~one A100-hour; anchors in `replication/anchors.json`).

## Method, in one line

Pre-register the hypothesis and decision criteria in a public commit before
the data exists; run; register the verdict either way; publish the kills with
the same prominence as the confirmations.

## Navigation

- **Where the program is going, and in what order:**
  [`notes/2026-08-10_the_observer_program.md`](notes/2026-08-10_the_observer_program.md)
  — the direction document (adopted August 10, 2026). The aim: build physics on
  the observer, and investigate the story that allows an observer to exist at
  all. Three spans, the horizons rule (*exploration is free; claims are
  earned*), and the ranked order of operations that puts undelivered results and
  deciding experiments ahead of instrument precision. Its companion work list is
  [`notes/2026-08-10_operational_debt.md`](notes/2026-08-10_operational_debt.md).
  The audit that preceded both — what still connects to fundamental physics,
  stated as a whole — is
  [`notes/2026-08-10_what_still_connects.md`](notes/2026-08-10_what_still_connects.md).
- What we claim and what would break it: [`theory/interior_horizon_theory.md`](theory/interior_horizon_theory.md) — the spine (§7 open construction sites, §8 predictions with kill conditions)
- Where the program stands after the bridge broke: [`notes/2026-08-08_program_reframe_where_things_stand.md`](notes/2026-08-08_program_reframe_where_things_stand.md) (with its dated corrections)
- The two populations: [`notes/2026-08-09_structural_vs_semantic_populations.md`](notes/2026-08-09_structural_vs_semantic_populations.md)
- The theory-of-A chain: [`notes/2026-08-09_theory_of_A_entropy_gap_and_sum_rule.md`](notes/2026-08-09_theory_of_A_entropy_gap_and_sum_rule.md), then exp-110 → exp-113
- Every experiment, one folder each: [`experiments/`](experiments/); structured index at `development/status/rooms/physics/registry.json`
- The full published record, 13 records: `research/publications/REGISTRY.md` (working repo; not in this published subtree)
- Layout and conventions: [`README.md`](README.md)
- Replication kit: [`replication/`](replication/)
- A cited document you cannot find: [`archive/RETIREMENTS.md`](archive/RETIREMENTS.md)

*(August 8, 2026: `STATUS.md` and `RESEARCH_MAP.md` were retired to
[`archive/maps/`](archive/maps/) — four documents were each claiming to be the
program's map. What they were still carrying is inventoried in
[`notes/2026-08-08_map_retirement_harvest.md`](notes/2026-08-08_map_retirement_harvest.md),
which is an open work list, not a finished pass.)*
