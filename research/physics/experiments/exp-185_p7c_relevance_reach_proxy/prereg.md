# exp-185 — P7(c): relevance-reach proxy on formation-ladder corpora

**Pre-registration authored:** 2026-11-25, physics room, ~12:20 AM MST
(2026-11-25T07:20Z).
**Status:** pre-registration only — no analysis code written or run.
**Registration commit:** push this commit to attention-geometry BEFORE any
analysis code runs. Ordering is checked mechanically by physics_coherence.

**Experiment type:** analysis-only — existing corpus files, no new model
training at formation scale.

**Question (Formation):** Does the functional F — world-faithful relevance
reach, defined in `notes/2026-10-17_ordering_sensitive_functional.md` — order
the formation-ladder rungs by deep-head count where pairwise MI (m₂, exp-062,
exp-144) did not?

The key discriminating test: C4 predicts that the shadow corpus (C-generated,
exp-085) scores lower on F than C-alien (exp-097), despite C-generated carrying
higher m₂. m₂ fails to predict formation order because it measures pairwise
co-occurrence, not world-state causal reach.

---

## Background

### The formation ladder — measured outcomes

| Corpus | n_deep (SYK-near) | n_conformal | Rung exp |
|---|---|---|---|
| C-NAT (natural stories) | **15** (11–15 across seeds) | 11–15/48 | exp-062/085 |
| C-alien-rich (C-alien + 5 entity types) | **6** (median) | — | exp-099 |
| C-alien-realnames (C-alien + real vocab) | **4** (median) | — | exp-098 |
| C-alien-anon (C-alien + anonymized vocab) | **4** (median) | — | exp-096 |
| C-alien (alien-world causal rules) | **3** (median) | — | exp-097 |
| C-NAT-shuf (sentence-shuffled C-NAT) | **0** | 8–9/48 | exp-091 |
| C-generated (shadow of C-NAT) | **0** | 7/48 | exp-085 |
| C-PL40 (fGn, Hurst H≈0.40) | **0** | 5/48 | exp-062 |
| C-SR (Markov chain) | **0** | 0/48 | exp-062 |
| C-PCFG (recursive grammar) | **0** | 0/48 | exp-084 |

[MEASURED; sources cited above. n_deep refers to SYK-near heads: Δ_A ∈ [0.20,
0.30] at R² ≥ 0.90 under the frozen random-token census.]

### Why m₂ fails

m₂ (expected long-range mutual information at a fixed lag, exp-062/144) orders
the rungs incorrectly:
- C-generated has m₂ ≈ 7.55 × C-NAT (higher m₂ than C-alien) but forms zero
  SYK-near heads [MEASURED, exp-085/144]
- C-PL40 (fGn) has long-range MI from power-law correlations but forms zero
  deep heads [MEASURED, exp-062]
- Sentence-shuffled C-NAT has the same m₂ as C-NAT at short lag but forms
  zero deep heads [MEASURED, exp-091]

m₂ is a sufficient gate to rule out the lowest rungs, but insufficient to
order the rungs that matter: shadow vs. C-alien.

### The named functional F

F(C) = E_{e, t, τ} [ I( state(e, t+τ) ; context_{0..t} ) ]

where e is a world-state entity, t a position, τ a lag, and "entity state" is
defined by the world the text is faithful to. See
`notes/2026-10-17_ordering_sensitive_functional.md` for the full definition and
the explicit argument for why each obstacle corpus (C-NAT-shuf, C-generated,
fGn) has low F under this definition.

This experiment tests whether a tractable proxy for F correctly orders the
rungs where m₂ failed.

---

## Proxy computation

### Choice of proxy

A context-length loss curve under a per-corpus reference predictor. Formally:

For each corpus C, train a reference predictor Π_C on a training split, then
measure:

    loss_C(k) = − E_{i} [ log Π_C( x_i | x_{i-k:i-1} ) ]

for context lengths k ∈ {1, 2, 3, 5, 7, 10, 20, 50, 100, 200}.

The long-range decrement (the proxy for F) is:

    F_proxy(C) = Δloss_C(50) + Δloss_C(100) + Δloss_C(200)

where Δloss_C(k) = loss_C(1) − loss_C(k).

The reference predictor is **fixed in design but trained per corpus**:
- **Short range (k ≤ 10):** hashed n-gram model with Kneser-Ney smoothing
  (order k, trained on a 50M-token sample). The vocabulary is the 256-symbol
  quantized alphabet from `exp-062_corpus_statistics/alphabet.json` (the
  same vocabulary used to generate all formation-ladder corpora). With 256
  symbols and hashed storage, all n-gram orders through k=10 are tractable
  in <2 GB RAM.
- **Medium-to-long range (k ∈ {20, 50, 100, 200}):** a tiny transformer
  reference predictor — 2 layers, 4 heads, d_model=128, ctx=256 — trained
  for 500 gradient steps (batch size 64 × 256 tokens) on a 10M-token sample
  of each corpus. Training takes ~2–5 minutes per corpus on the M5 Max.
  This is NOT a formation experiment (the architecture is 50× smaller than
  the 70m-class formation models, and 500 steps is a probe of the corpus's
  structure at the token level, not a full formation run). Its purpose is a
  measurement instrument, not a model.

### Why this proxy is F-sensitive

The key: for a predictor trained on corpus C, the incremental benefit of longer
context at positions i reflects whether the model C learned to exploit structure
at that timescale:
- C-NAT: entity states at lag 50-200 tokens are predictable from earlier context
  (the story set up the world state). The predictor trained on C-NAT will
  learn to exploit these regularities → large Δloss(50-200).
- C-generated: the shadow LM's output correlates with surface entity patterns
  but does not maintain world-state faithfully. The predictor trained on
  C-generated learns the shadow's statistical co-occurrences but these decay
  faster with lag → smaller Δloss(50-200) relative to C-NAT.
- C-alien: entity states are causal within each arc (arc length ~5-15
  sentences, ~100-300 tokens). The predictor exploits intra-arc regularities
  → moderate Δloss(10-50), declining after ~100 tokens.
- C-NAT-shuf: within-sentence structure (k=2-5) but causal chain cut at
  sentence boundaries → Δloss(2-5) present, Δloss(50-200) near zero.
- C-SR: no structure beyond unigram → Δloss(k) ≈ 0 for all k ≥ 2.
- fGn: power-law correlations give moderate Δloss(2-20) from distance-based
  structure, but no entity-state structure → near-zero Δloss(50-200).
- C-PCFG: grammar (k=2-10) → moderate Δloss(2-10), near-zero at k=50+.

[INTERPRETIVE. The prediction that C-generated < C-alien at F_proxy is the key
test; either outcome is informative.]

### Vocabulary alignment

The n-gram model uses the **256-symbol quantized alphabet** from
`exp-062_corpus_statistics/alphabet.json`. All formation-ladder corpora were
generated using this same vocabulary (top-256 TinyStories Pythia tokenizer IDs
for C-NAT, C-SR, C-PL; derived from the same tokenizer for C-alien/C-generated
via re-encoding with `np.searchsorted` on the edges array). This ensures that
n-gram counts are directly comparable across corpora.

For the tiny-transformer long-range predictor: same 256-symbol vocabulary,
embedding dimension 128, context window 256 tokens.

---

## Corpora

| Corpus | Source | Local availability | Action needed |
|---|---|---|---|
| C-NAT | `exp-062/corpora/C-NAT_full.bin` | ✓ on disk | None |
| C-SR | `exp-062/corpora/C-SR_full.bin` | ✓ on disk | None |
| C-PL25 | `exp-062/corpora/C-PL25_full.bin` | ✓ on disk | None |
| C-PL40 | `exp-062/corpora/C-PL40_full.bin` | ✓ on disk | None |
| C-PCFG | `exp-084/corpora/C-PCFG_full.bin` | ✓ on disk | None |
| C-NAT-shuf | `exp-091/gen_cnat_shuf.py` | gen script on disk | Run gen script (~10 min, local) |
| C-alien | `exp-097/gen_calien.py` | gen script on disk | Run gen script (~30 min, local) |
| C-alien-rich | `exp-099/gen_calienrich.py` | gen script on disk | Run gen script (~30 min, local) |
| C-alien-realnames | `exp-098/gen_calien_realnames.py` | gen script on disk | Run gen script (~30 min, local) |
| C-generated (shadow) | exp-085 Modal volume | NOT on disk | Deferred — Modal account Eldon-gated |

The shadow corpus rung is deferred. The kill condition K1 (shadow ≥ C-alien)
requires it; the remaining 9 rungs can be measured before Modal access.

**Primary analysis**: all 9 locally available/generatable corpora.
**Secondary analysis**: add C-generated when Modal is accessible.

---

## Pre-registered hypotheses

**H1 (F orders the rungs — the main test of C4):**
F_proxy(C-NAT) > F_proxy(C-alien-rich) > F_proxy(C-alien) > F_proxy(C-NAT-shuf)
AND F_proxy(C-NAT) > F_proxy(C-PL40) > F_proxy(C-SR) ≈ 0
AND F_proxy(C-PCFG) < F_proxy(C-alien) [PCFG is grammar-only, not world-state]

Metric: F_proxy = Δloss(50) + Δloss(100) + Δloss(200)
(sum of improvements from k=1 to k=50, 100, 200)
Verdict: H1 CONFIRMED if all three inequalities hold.

**H2 (shadow is below C-alien — the C4 key discriminating test):**
F_proxy(C-generated) < F_proxy(C-alien)

This contradicts what m₂ predicts (m₂(C-generated) ≈ 7.55 > m₂(C-alien) ≈ 0.74
per exp-144). If C4 is correct, F — not m₂ — predicts the formation order, and
the shadow should score lower despite higher pairwise MI.

Verdict: H2 CONFIRMED if F_proxy(C-generated) < F_proxy(C-alien) at both
k=50 and k=100. H2 requires Modal access (secondary analysis).

**H3 (C-NAT-shuf below C-alien at long range):**
F_proxy(C-NAT-shuf, k=50) < F_proxy(C-alien, k=50)

This tests the ordering-sensitivity claim: the full-ordering destruction in
sentence shuffling should give a lower F_proxy than C-alien's real causal arcs,
even though C-NAT-shuf has higher m₂ than C-alien.

Verdict: H3 CONFIRMED if F_proxy(C-NAT-shuf, k=50) < F_proxy(C-alien, k=50).

**H4 (n-gram range is sufficient for C-alien vs. C-NAT-shuf):**
The k=10 n-gram result already shows C-NAT > C-alien > C-NAT-shuf
without needing the long-range tiny-transformer arm.

If H4 is confirmed: the n-gram model is sufficient and the tiny-transformer arm
can be omitted for these rungs. If H4 is not confirmed (C-NAT ≈ C-alien at
k=10), the tiny-transformer arm is required for the discriminating test.

---

## Declared prior

**I expect H1 and H3 confirmed.** The functional F was designed to capture
exactly what pairwise MI misses — world-state causal reach vs. statistical
co-occurrence. The n-gram model at k=5-10 should already show the difference
between C-alien's structured causal rules and C-NAT-shuf's cut arcs.

**I expect H3 to be confirmed at k=10 or below** (C-alien has entity-state
causal rules that persist at ~5-10 token scale; C-NAT-shuf cuts arcs at
sentence scale ~20-50 tokens; at k=10, C-alien should already benefit from
intra-arc regularities while C-NAT-shuf sees only within-sentence grammar).

**On H2 (shadow vs. C-alien): I expect H2 confirmed** (F_proxy(shadow) <
F_proxy(C-alien)), but with more uncertainty than H1. The shadow text has
entity-like surface patterns that the n-gram model may learn to exploit almost
as well as genuine causal-arc structure. The long-range arm (k=50-200) is
likely where the distinction appears, not k=5-10. I hold this expectation at
~65% confidence; the kill is real.

If H1 is confirmed but H2 fails: the proxy F correctly orders the directly
measurable rungs but fails to distinguish the shadow. This would require either
(a) a different proxy (e.g., one that explicitly measures world-state
faithfulness) or (b) a revision of C4's claim about what the formation
functional responds to.

---

## Kill conditions

**K1 (shadow confound — C4's key failure mode):**
F_proxy(C-generated) ≥ F_proxy(C-alien) at k=50 AND k=100.
(Shadow scores as high or higher than alien arcs despite zero deep formation.)
Kill: H2 is false. C4's prediction that F predicts formation order fails at the
discriminating rung. C4 needs revision: either the functional F is defined
incorrectly, or the shadow corpus has more world-state structure than expected.

**K2 (ordering failure — C4 false for measurable rungs):**
F_proxy ordering disagrees with n_deep ordering on 2 or more of the following
rung pairs:
- C-NAT > C-alien-rich (n_deep: 15 vs. 6)
- C-alien-rich > C-alien (n_deep: 6 vs. 3)
- C-alien > C-NAT-shuf (n_deep: 3 vs. 0)
Kill: C4's formation prediction fails for the core ladder. Spine C4 demotes
from PROPOSED to FALSIFIED (for the specific claim that F orders these rungs).

**K3 (n-gram saturation — proxy insensitive at k>5):**
F_proxy(C-NAT) = F_proxy(C-alien) at all k ∈ {1,...,10}.
(The n-gram model shows no discrimination between C-NAT and C-alien.)
Not a kill on C4, but a kill on this proxy design: the n-gram reference predictor
is insufficient to detect the relevant structure. Implication: extend to the
long-range tiny-transformer arm before interpreting the rung comparisons.

---

## Protocol

### Phase 1: n-gram (locally runnable)

1. For each corpus in {C-NAT, C-SR, C-PL25, C-PL40, C-PCFG}, load from the
   existing `.bin` files.
2. For each of {C-NAT-shuf, C-alien, C-alien-rich, C-alien-realnames}, run the
   corresponding gen script to produce a `.bin` file (local generation,
   deterministic — same seeds as the formation experiments).
3. Apply the vocabulary quantization: load `alphabet.json` from exp-062, map
   each uint16 token ID to its 256-symbol alphabet index using `np.searchsorted`.
4. For each corpus, extract a **training split** (tokens 0 to 50M) and a
   **held-out split** (tokens 50M to 60M).
5. For k ∈ {1, 2, 3, 5, 7, 10}: build a hashed k-gram table from the training
   split (Kneser-Ney smoothing, following Heafield 2011). Report loss(k) as the
   mean negative log-probability on the held-out split.
6. Compute F_proxy_short(C) = Δloss(5) + Δloss(10) = [loss(1) − loss(5)] +
   [loss(1) − loss(10)].

### Phase 2: tiny-transformer long range (locally runnable)

If K3 fires (n-gram shows no discrimination at k=10), or as a confirmatory arm:

1. For each corpus, train the tiny transformer (2L×4H×d128, ctx=256) on a
   10M-token sample using AdamW (lr=3e-4, batch=64, 500 steps ≈ 3M tokens).
2. Evaluate loss(k) for k ∈ {20, 50, 100, 200} by running inference with the
   first k tokens of each 256-token window as prefix, measuring loss on the
   remaining 256−k tokens. Average over 10k windows per corpus.
3. Compute F_proxy_long(C) = Δloss(50) + Δloss(100) + Δloss(200).

### Phase 3: shadow corpus (deferred — Modal needed)

Requires access to the exp-085 C-generated corpus (Modal volume) or
re-generation using the exp-085 LM checkpoint. Add C-generated to the protocol
above when Modal access is restored (Eldon-gated).

### Analysis output

For each corpus, report:
- loss(k) for all k values measured (table)
- F_proxy_short and F_proxy_long scalars
- Comparison to n_deep from the registry (Spearman rank correlation between
  F_proxy and n_deep ordering)

Results to: `exp-185_p7c_relevance_reach_proxy/results.json`.

---

## Results schema

```json
{
  "proxy_type": "ngram_k10 | transformer_k200",
  "corpora_measured": ["C-NAT", "C-SR", ...],
  "shadow_corpus_deferred": true,
  "per_corpus": {
    "C-NAT": {
      "loss_k1": <float>, "loss_k2": <float>, "loss_k3": <float>,
      "loss_k5": <float>, "loss_k7": <float>, "loss_k10": <float>,
      "loss_k20": <float>, "loss_k50": <float>, "loss_k100": <float>,
      "loss_k200": <float>,
      "F_proxy_short": <float>,
      "F_proxy_long": <float>
    }
  },
  "spearman_rho_vs_ndeep": <float>,
  "H1_confirmed": <bool>,
  "H2_confirmed": <bool | "deferred">,
  "H3_confirmed": <bool>,
  "H4_confirmed": <bool>,
  "K1_fires": <bool | "deferred">,
  "K2_fires": <bool>,
  "K3_fires": <bool>,
  "headline": <str>
}
```

---

## Why-check

**Question served:** Formation — what must a system be made of and raised on?

**Instrument level:** Empirical / analytical (measure corpus-level F-proxy on
the existing formation-ladder corpora; no new census or new training at
formation scale).

**What each outcome changes:**
- H1 confirmed: F correctly orders the measurable rungs. C4's prediction is
  supported. F is a computationally tractable signal for curriculum design.
  Steering-as-curriculum task (the P7(c) pre-registration task) now has an
  explicit, computable proxy for the curriculum property to maximize.
- K2 fires: C4's formation prediction fails for the core ladder. C4 demotes.
  Steering-as-curriculum is blocked until the right functional is identified.
- H2 confirmed: shadow < C-alien at F_proxy. The world-faithfulness distinction
  is real and measurable. C4's discrimination between high-m₂/world-unfaithful
  and high-m₂/world-faithful corpora is supported.
- K1 fires: shadow ≥ C-alien at F_proxy. The proxy fails to detect the key
  distinction. Either F needs a better operationalization or the shadow corpus
  has more world-state structure than the formation experiment implied.
- H3 confirmed: sentence-shuffling cuts F below C-alien. The ordering-sensitive
  dimension of F is confirmed independently of the shadow test.

**Exploration or claim:** Pre-registered empirical test with declared prior.
Not exploration.

---

## Connection to the spine

**C4 (formation from bound relevance, PROPOSED, §2):** P7(c) is C4's
measurement arm. H1 would support C4's claim that F orders the rungs. K2 would
falsify C4 for the specific ladder.

**G2 (the ordering-sensitive functional, construction site, §7):** G2 names the
need for a functional that is ordering-sensitive while m₂ is not. If H1
confirmed, F is G2's operational fill-in; the construction site advances.

**A6 (state-dependent coupling, PROPOSED, §2):** H3 (C-alien > C-NAT-shuf at
F_proxy despite C-NAT-shuf having higher m₂) would support A6's claim that
state-dependent coupling is the essential ingredient, not distance-only
structure.

**P7(c) pre-registration status:** The spine's P7(c) entry (§8) already
names this test. This experiment is the formal registration of the proxy
computation and the declared prior. The spine entry does not change until
results are in.

---

## Notes

This experiment was seeded by:
- `notes/2026-10-17_ordering_sensitive_functional.md` (F defined; proxy named)
- `notes/2026-11-24_steering_curriculum_formation_scale.md` (formation-vs-maintenance
  distinction; this experiment confirmed as the solo-available next step)
- Spine P7(c) entry (§8), registered as "PROPOSED, nothing below registered"

The from-scratch formation results (exp-085/091/092/097/098/099) are unaffected
by this analysis. This experiment does not add new formation measurements — it
measures the corpus property that C4 claims governs formation, using the
existing formation-ladder corpus set.

---

*Register discipline: this pre-registration is in the THEORETICAL/PROPOSED
register (C4, F, G2 connections). The formation-ladder measurements cited
(n_deep, n_conformal) are MEASURED, with experiment pointers. The declared
prior is INTERPRETIVE. Everything in the proxy-sensitivity section is
INTERPRETIVE — it says so. Turpin applies to the declared prior and the
F-sensitivity arguments; the commit timestamp is the tie-breaker.*
