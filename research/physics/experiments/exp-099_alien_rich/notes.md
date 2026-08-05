# exp-099: Conformal Window Scan — C-alien-rich (5 types, stochastic rules)

**Pre-registered:** 2026-08-03, ~1:30 AM MDT, physics room session. Committed to
`3ld0n/attention-geometry` before any corpus generation, training, or measurement.

**Theoretical frame:** `notes/2026-08-03_melonic_threshold_derivation.md` — the
conformal window of attention, derived from the sparse/low-rank SYK threshold.
`notes/2026-08-02_arrest_and_arrival.md` — the five-station taxonomy and the two
permanences. This pre-registration is the first application of that theoretical frame
to a forward prediction.

**Follows:** exp-097 (H_alien_partial: C-alien arrests at UV, backbone collapses) and
exp-098 (H_backbone_absent: vocabulary change doesn't restore backbone — world structure
governs both UV arrest and backbone collapse). This is the richer-world continuation:
does increasing the world's complexity push trained attention toward the conformal window?

**Inbox reference:** `development/status/rooms/physics/inbox.md` (cursor, 2026-08-02) —
the conformal window conjecture from the late-night session with Eldon.

---

## The question

Exp-097/098 established two world-structure findings:
1. **UV arrest**: a small alien finite-state machine (3 types, 4 deterministic rules)
   produces trained attention stuck at Δ_med ≈ 0.75–1.2, far from the q=4 fixed point.
2. **Backbone collapse**: the L0 conformal population (n_backbone) drops to zero,
   indicating the trivial fixed point dominates the first layer.

Both effects survive vocabulary change (exp-098), confirming world structure as the cause.

The theoretical frame (melonic threshold derivation) maps this onto the sparse/low-rank
SYK result: the trained W_QK has an **effective rank** bounded by the world's state-space
dimension S. For C-alien, S = 2^3 = 8 (three binary entities). The threshold for the
conformal phase is R_eff ~ d = 512. With S = 8 << 512, the conformal phase cannot form.

**C-alien-rich tests:** does increasing S — by adding entity types, instances, and
introducing stochastic rules — reduce UV arrest and restore conformal formation?

**Primary question:** Does richer-world C-alien-rich show (a) any backbone restoration
(n_backbone > 0), and/or (b) Δ_med reduction toward the IR, compared to C-alien (1.04)
and C-alien-realnames (0.727)?

---

## Physical interpretation

Under the conformal window theory:

The corpus installs an effective coupling in W_QK whose rank is bounded by the world's
effective state-space dimension S. The q=4 conformal phase requires R_eff ≥ R_crit. The
data so far constrains:
- C-alien (S=8): below threshold → UV arrest + backbone collapse
- C-NAT (S >> d, effectively): above threshold → conformal arrival
- C-NAT-anon (S ~ C-NAT, minus cross-story name prototype): above threshold → near-full
  arrival, -1 to -3 deep heads for the cross-story prototype loss

C-alien-rich (S = 2^5 = 32, with stochastic rule entropy > 0) is designed to be
**near the potential lower bound of the threshold**. If R_crit is as low as ~30–40
(d/q² ≈ 32 for q=4, d=512), C-alien-rich could be at or above it.

This is the first rung of a **conformal window scan** — the analog of scanning N_f across
the Banks-Zaks conformal window in QCD, where crossing from below to above the window
boundary is marked by the phase transition from mass-gap-generation to conformal invariance.

---

## World design: C-alien-rich

### Entity types and instances

| Type | Count per story | Name pool (5 per type) |
|------|-----------------|------------------------|
| Flurp | 1 | Vex, Nul, Ort, Pim, Grel |
| Blurn | 1 | Dath, Sorn, Wix, Brel, Fend |
| Zarb | 1 | Quib, Tarn, Molk, Vet, Zish |
| Glorf | 1 | Blav, Usk, Drei, Folt, Yem |
| Krelp | 1 | Snop, Wulf, Crid, Barv, Hund |

**5 entity types × 1 instance each = 5 entities per story**
(Expanding from 3 types → 5 types; same 1 instance per type as C-alien)

Each entity has 2 states: **active** or **resting**.

**World state space:** 2^5 = 32 possible world states (vs. 2^3 = 8 in C-alien).

### Rules

6 causal rules (vs. 4 in C-alien), each **stochastic** with primary probability p=0.7
and alternative probability p=0.3.

| Rule | Condition | Primary outcome (p=0.7) | Alternative (p=0.3) |
|------|-----------|------------------------|---------------------|
| A | Active Flurp + resting Blurn | Blurn becomes active | No change (Blurn stays resting) |
| B | Active Flurp + active Blurn | Both become resting | Only Flurp becomes resting |
| C | Active Blurn + resting Zarb | Zarb becomes active | No change |
| D | Active Zarb + resting Flurp | Flurp becomes active | No change |
| E | Active Glorf + resting Krelp | Krelp becomes active | No change |
| F | Active Krelp + active Glorf | Both become resting | Only Krelp becomes resting |

Rules are checked in priority order A–F; first matching rule fires. The stochastic
outcome is independent for each event (governed by the data seed).

**Transition entropy per rule-firing:**
H = -(0.7 log₂ 0.7 + 0.3 log₂ 0.3) ≈ 0.88 bits per event.
With ~6 events per story: total trajectory entropy ≈ 5.3 bits per story.
Compare C-alien: 0 bits per story (deterministic).

### Story structure

Same as C-alien: initialization (assign entity names, set initial states by type-specific
probability) → event loop (4–8 rule-firings with stochastic outcomes) → resolution
sentence stating final states.

Same sentence template structure as C-alien, extended for the 2 new entity types (Glorf,
Krelp) and the stochastic outcome register ("surprisingly" / "as expected" as a template
marker for the p=0.3 alternative outcome).

### Corpus parameters

- Total tokens: 1.05B (same as series)
- Data seed: 9000 (new seed for new corpus)
- Init seeds: 1900, 1901, 1902 (3 training seeds; data_seed 2900)
- Control: randomized-weights control on seed-1900 checkpoint (expected 0/48)

---

## Pre-registered hypotheses

**Primary observables:**
1. **n_deep**: deep conformal heads (L3–L5), median across 3 seeds
2. **n_backbone**: L0 conformal heads, median across 3 seeds
3. **Δ_med**: delta_median_conformal, median across 3 seeds

**Reference:**
| Corpus | n_backbone | Δ_med | n_deep |
|--------|-----------|-------|--------|
| C-NAT | 7–8 | ~0.17 | 5–7 |
| C-NAT-anon (exp-096) | 7 | 0.149 | 4 |
| C-alien (exp-097) | 0 | 1.04 | 3 |
| C-alien-realnames (exp-098) | 0 | 0.727 | 4 |

**H_rich_above_window** (primary, theory prediction P2): median n_backbone ≥ 3 AND Δ_med ≤ 0.5.
- Interpretation: C-alien-rich is above or at the conformal window threshold.
  Backbone partially restores and IR convergence begins.
- Probability under theory: ~25% (R_crit ~ d/q² scenario)

**H_rich_dose_response** (primary, theory prediction P3): Δ_med < C-alien-realnames (0.727)
AND n_deep > C-alien (3).
- Interpretation: World complexity has a measurable dose-response effect below threshold.
  UV arrest persists but is reduced; deep formation improves marginally.
- Probability under theory: ~80% (below-threshold dose-response)

**H_rich_below_window** (compatible with P1): median n_backbone = 0 AND Δ_med > 0.5.
- Interpretation: C-alien-rich is still below threshold; world structure still insufficient.
  The conformal window requires world complexity closer to C-NAT scale.

**Kill criterion for H_rich_dose_response:**
If Δ_med(C-alien-rich) ≥ Δ_med(C-alien-realnames) = 0.727 AND n_deep ≤ 3 (no improvement
despite 4× increase in world state space): the R_eff ↔ S(corpus) mechanism is falsified
at this granularity. Something other than world state-space diversity drives the threshold.

**Secondary hypotheses:**
- **H_transition_entropy**: Stochastic rules alone (even without more entity types) would
  show improvement. Cannot distinguish cleanly from entity count effect in this design.
  Design note: a follow-up with C-alien + stochastic rules but same 3 types would isolate
  this. Not in this experiment.
- **H_backbone_L0_first**: If backbone restores, it restores in L0 before deep layers
  (n_backbone > 0 when n_deep is still below C-NAT). Layer anatomy tracks the threshold.

---

## Declared expectations (non-criterial)

**Prior: H_rich_dose_response** with n_backbone still 0 or 1 (moderate confidence, ~65%).

Reasoning:
1. C-alien-rich has S=32 vs C-alien's S=8; threshold estimated at R_crit ~ d = 512.
   S=32 << 512 → still below threshold.
2. BUT: the low-rank SYK result (arXiv:1910.10173) shows that near-extensive rank
   (R ~ N, not R >> N) can already produce near-chaotic phases. If R_crit is as low
   as ~30–40, C-alien-rich is right at the edge.
3. Stochastic rules increase transition entropy significantly (0 → 5.3 bits/story).
   Even without changing the backbone condition, this could push Δ_med toward IR.

Non-criterial expectations:
- n_deep: 4–5 (improvement from exp-097 median=3, exp-098 median=4)
- n_backbone: 0–2 (likely still 0; possibly 1–2 if near threshold)
- Δ_med: 0.4–0.6 (reduced from exp-097 1.04 and exp-098 0.727; not yet IR-range)

---

## Protocol

**Architecture:** Identical to exp-062/085/091–099 series.
- GPT-NeoX (6 layers, 8 heads, d_k=64, ctx=512)
- Fresh initialization; 2000 training steps; batch 524288 tokens (1.05B total)
- AdamW optimizer, cosine schedule
- pos_enc: rotary (rotary_pct=0.25)

**Corpus:** C-alien-rich (as specified above)

**Seeds:** init seeds 1900, 1901, 1902 (data_seed 2900)

**Control:** randomized-weights control on seed-1900 checkpoint. Expected 0/48.

**Measurement:** Identical to series (BCFT power-law fit, R² ≥ 0.90, fit range [8,256]).

**Infrastructure:** Modal A100-40GB. Estimated cost: ~$23 (same as exp-097).

---

## A companion experiment this pre-registration opens

**Exp-100 (proposed, not pre-registered here):** Effective rank measurement of trained
W_QK matrices on existing Modal checkpoints. Requires no new training.

Theory prediction:
- mean_rank(C-NAT heads) >> mean_rank(C-alien heads)
- mean_rank(C-NAT-anon heads) ~ mean_rank(C-NAT heads)
- mean_rank(C-alien-realnames heads) ~ mean_rank(C-alien heads)
- mean_rank(C-alien-rich heads): between C-alien and C-NAT, proportional to increase in S

If confirmed: the low-rank SYK mechanism is directly evidenced.
If mean_rank(C-alien) ~ mean_rank(C-NAT): an alternative mechanism must explain the arrest.
(See `notes/2026-08-03_melonic_threshold_derivation.md` §6 for the full protocol.)

---

## What follows

**If H_rich_above_window confirmed (backbone restores, Δ_med toward IR):**
- The conformal window threshold lies between C-alien (S=8) and C-alien-rich (S=32).
- Next: dose-response scan between the two (S=12, S=20) to pin the window edge.
- Paper 6 ("The Conformal Window of Attention") can be written with the window edge measured.

**If H_rich_dose_response confirmed but backbone still 0:**
- Below-threshold dose-response confirmed. Window threshold is above S=32.
- Next: continue scan upward (S=64, S=128, ...) toward C-NAT.
- The scan becomes Paper 6 when the window edge is bracketed.

**If kill criterion met (no improvement despite 4× S increase):**
- R_eff ↔ S(corpus) conjecture falsified.
- Exp-100 (rank measurement) becomes urgent diagnostic: is the W_QK rank actually not
  correlated with S? If rank is correlated but UV arrest persists, a different mechanism
  governs the threshold.
- Return to theory with the new constraint.

---

## Theory addendum — magnitude-gate prediction (2026-08-03, ~2:00 AM MDT)

*Appended after the pre-registration above, before any corpus generation or
training. The registered hypotheses above are untouched; this records a
competing quantitative prediction from the finished theory session, so the
experiment discriminates between two operationalizations of the same frame.*

The pre-registration cites the melonic-threshold derivation as it stood
mid-session (the rank/state-space reading: R_eff bounded by S). The finished
note (`notes/2026-08-03_melonic_threshold_derivation.md` §6–§7) adds
declared-discipline numerics that shift the emphasis: in the computable
(IDF-weighted) proxy, the corpus functional that actually separates
TinyStories from C-alien is the **coupling-magnitude gate**
m₂ ∝ Tr[(KδK)²]/n⁴ (18× separation), not the rank/chaos gate (which barely
moves). Provisional rungs spanning cast 4–12, stochasticity p=0.7, and longer
episodes — bracketing this design — leave m₂ flat (0.68–0.78 vs TinyStories'
13.2), because none of those axes enrich the *surface language* (template
count / per-word surprisal).

**Magnitude-gate prediction for THIS design (P-1 of the derivation note §7):**
C-alien-rich stays UV-arrested — Δ_med ≥ 0.6, n_deep ≤ 4 (no improvement
beyond exp-098), n_backbone ≈ 0. That assigns H_rich_above_window
substantially *lower* probability (~5–10%) than the registered ~25%, and is
more pessimistic than the registered 80% dose-response prior: even the
Δ_med < 0.727 improvement is expected to be marginal at best, since the new
templates for Glorf/Krelp and the "surprisingly/as expected" markers add only
a small increment of surface-language richness.

**Discrimination:** if exp-099 arrives (or shows strong dose-response), the
S-rank reading wins and the magnitude-gate emphasis of the proxy numerics is
wrong (or the one-hot/IDF proxy is too crude — exp-100's direct W_QK rank
measurement then adjudicates). If it stays arrested at exp-098 levels, the
magnitude gate wins and the next rung should enrich surface language at fixed
S (more templates, larger vocabulary, freer phrasing) — the derivation note's
P-2 pipeline: compute m₂ of the actual generated corpus *before* training via
`research/physics/theory/corpus_functional.py` and register the number.

(Pointer fix: the exp-100 protocol reference above says derivation note §6;
in the finished note the numerics are §6, the exp-099 predictions §7, and the
W_QK rank measurement is described in the physics inbox item and here, not in
§6.)

---

## Status

- [x] Pre-registration written (2026-08-03, ~1:30 AM MDT)
- [x] Pre-registration committed and pushed to 3ld0n/attention-geometry (commit 222d9d4; theory addendum 93b34dd)
- [x] Corpus generator script written (gen_calien_rich.py) — 2026-08-04 physics room session
- [x] Modal training/measurement script written (modal_exp099.py) — 2026-08-04 physics room session
- [x] Pre-training corpus functional measured and note registered — 2026-08-04 (commit 529a69a)
      m2(C-alien-rich)=0.687 vs C-alien=0.737 vs C-NAT=7.81; P-1 supported by proxy
- [x] Launched on Modal: app ap-pmEN9gjsyDRotW5CrKtap0, handle fc-01KZ7JQR6MV0A7Q281G3777AY2
      (2026-08-04 ~5:50 PM MDT; generate → 3 seeds → control, detached)
- [x] Training runs complete (seeds 1900/1901/1902) — completed ~2:20 AM UTC Aug 5
- [x] Randomized-weights control — 0/48 confirmed
- [x] Verdict registered — 2026-08-05, physics room session

---

## Results (collected 2026-08-05, ~12:30 AM MDT)

**3-seed summary:**

| Seed | n_conf | n_backbone (L0) | n_deep (L3–L5) | Δ_med_conf |
|------|--------|-----------------|----------------|------------|
| s0 (1900) | 7 | 0 | 2 | 0.503 |
| s1 (1901) | 17 | 0 | 6 | 0.846 |
| s2 (1902) | 14 | 0 | 7 | 0.750 |
| **median** | **14** | **0** | **6** | **0.750** |
| control | 0 | 0 | 0 | — |

**Reference (for dose-response comparison):**

| Corpus | n_backbone | Δ_med | n_deep |
|--------|-----------|-------|--------|
| C-NAT | 7–8 | ~0.17 | 5–7 |
| C-NAT-anon (exp-096) | 7 | 0.149 | 4 |
| C-alien (exp-097) | 0 | 1.04 | 3 |
| C-alien-realnames (exp-098) | 0 | 0.727 | 4 |
| **C-alien-rich (exp-099)** | **0** | **0.750** | **6** |

**Verdicts:**

- **H_rich_above_window: FALSIFIED.** n_backbone = 0 across all seeds. Backbone collapse persists.
- **H_rich_dose_response: PARTIAL.** n_deep axis: confirmed (median=6 > 3; trend 3→4→6 from exp-097→098→099). Δ_med axis: fails (median=0.750 > 0.727 criterion). The two observables decouple.
- **H_rich_below_window: CONFIRMED.** n_backbone=0, Δ_med=0.750 > 0.5. C-alien-rich still below threshold.
- **Kill criterion: NOT MET.** Δ_med ≥ 0.727 (one condition) but n_deep=6 > 3 (other condition fails).
- **Magnitude-gate prediction (P-1): PARTIAL.** Δ_med=0.750 and n_backbone=0 confirmed; n_deep=6 exceeds the predicted ≤4.
- **Control: CONFIRMED CLEAN.** 0/48 conformal at randomized weights.

**Key finding — decoupled dose-response:**

World complexity (S=8→32) produces a dissociation: n_deep improves substantially (3→4→6 across the series) while Δ_med remains UV-arrested and even rises slightly (1.04→0.727→0.750, non-monotone at the last step). The conformal window threshold lies above S=32.

**Interpretation:**

1. **Two decoupled observables.** n_deep tracks "how many deep-layer heads achieve power-law fit" — this responds to world complexity (S increases → more diverse world states → more structural positions for heads to specialize). Δ_med tracks "how close to the IR fixed point" — this does not respond to the same S increase. The heads that form in deep layers (n_deep=6) are UV-arrested conformal, not IR-approaching.

2. **Backbone collapse is robust to this rung.** n_backbone=0 at all seeds confirms that the L0 trivial-fixed-point population requires something more than 4× world state-space expansion to restore. The world still doesn't have the diversity to support a backbone.

3. **Magnitude gate vs. S-rank: magnitude gate explains Δ_med but not n_deep.** The corpus functional pre-measure (m₂=0.687, flat vs C-alien's 0.737) correctly predicted Δ_med would stay arrested. But it underpredicted n_deep improvement (actual: 6 vs predicted ≤4). This suggests the two observables respond to different corpus properties: n_deep to state-space expansion (S), Δ_med to coupling magnitude (m₂).

4. **Upward scan continues.** The next rung needs to either (a) enrich surface language at fixed S (more templates, freer phrasing) to push m₂, or (b) continue S expansion (S=64–128). Both are informative in different directions.

**Per-layer anatomy:**

L0 (backbone): 0/8 conformal at all seeds — trivial fixed point dominates entry layer.
L1: conformal heads form (4/8 at s0; 6/8 at s1 and s2); UV-arrested (Δ_med 0.40–0.80).
L2: variable (1 at s0, 5 at s1, 1 at s2); Δ_med 0.69–0.94.
L3: most informative layer (2/4/5 conformal); Δ_med 0.62–1.04, fully UV-arrested.
L4: sparse (0/2/1); the few that form are also UV.
L5: 0 at s0 and s1; 1 at s2 (Δ=0.585 — the closest to IR of any head in the experiment).

**Caveat:** s0 is an outlier with n_conf=7 (vs 14–17 at s1/s2) and Δ_med=0.503 — this seed formed far fewer heads overall, but the ones that did form are the closest to IR. The seed spread is real; the median is not dominated by any one seed.

**Implications for exp-100:** W_QK rank measurement now becomes the key diagnostic for the decoupling. If actual W_QK rank is higher in C-alien-rich than C-alien (proportional to S increase), that supports the S-rank reading for n_deep improvement. If rank does not track S, a different mechanism drives n_deep. Either result sharpens the window theory.
