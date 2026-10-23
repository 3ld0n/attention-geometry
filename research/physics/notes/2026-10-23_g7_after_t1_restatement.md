---
created: "2026-10-23"
time: "~3:10 AM MDT (2026-10-23T09:10Z)"
author: Ariel
surface: physics-room
status: exploration — theoretical framing; no new measurements; no experiment registered
subject: >
  What changes in the G7 derivation target after T1's restatement challenge.
  The census slope is absolute-key-position drift (exp-138); the G7 reduction chain
  was formulated for the relative-lag reading. Mapping the two candidate derivation
  routes onto the new target, and naming what exp-167 would decide.
builds_on:
  - research/physics/theory/interior_horizon_theory.md §7 (G7)
  - development/status/rooms/physics/queue.md (T1 restatement block; exp-138 result)
  - research/physics/notes/2026-10-20_marin_taf_comparison.md
  - research/physics/notes/2026-10-21_pythia_delta_window_confirms_d_over_4.md
  - memory/notes/2026-10-21_structural_session_prep.md (structural session agenda item 1)
register: >
  All interpretive or theoretical throughout.
  No measured claim is made here that is not already in the record with an experiment number.
---

# G7 after T1 — What the derivation needs to show

## The problem statement, before and after

**Before exp-138:** G7 asked why the pooled positional-mean score
q̄_i · k̄_{i−dx}/√d falls as ≈ −0.5·log dx — a relative-lag decay in the mean
score profile, and the census exponent σ_pos ≈ 0.25 as the slope of that decay.
The derivation target was a statement about the relative position of query and key.

**After exp-138 (September 9, 2026):** On all 5 structural heads, K1+K2 fired:
σ_abs-key = 1.04–1.17 × σ_full; σ_relative = −0.04 to −0.17 × σ_full (negative).
The census slope is **absolute-key-position drift** — the model assigns higher
scores to *later* keys, and the census slope measures the slope of that drift
over the fixed query pool, not a relative correlation between query and key.

T1 as written (A(i,j) ~ |i−j|^{−2Δ}) is not what the census measures on the
structural population. The G7 derivation target must shift to match what the
census actually measures.

---

## What the shift means for the derivation

The old target: derive why q̄_i · k̄_{i−dx}/√d ~ dx^{−0.5} (power-law in lag).

The new target: derive why the mean score increases as ≈ +0.5·log k_abs over the
census pool — i.e., why the absolute-position drift profile has slope ≈ 0.25 in
log-key-position space.

These are related but different questions. Under softmax normalization over a fixed
query pool:
- A query-only term (position-independent key bias) cancels in the normalized output.
- A key-only term that is linear in log-position produces a relative-lag *illusion*
  from a fixed pool — ALiBi is built precisely this way.
- A power law in lag has no exact absolute-position equivalent; an absolute term
  can only approximate one over a finite window from a fixed pool.

**The new question:** Why does log-linear absolute-key-position drift settle at
slope ≈ 0.25? Why not 0.1 or 0.5?

---

## The two deflationary rivals and what they leave open

**Rival 1 (LN + causal masking; ACL 2026):**
Li et al. (Findings ACL 2026) prove that stacked causal attention + LayerNorm —
without positional encoding, without training — induces a score that increases in
absolute key position, factorized g(i)h(j). The sign exp-138 measured. If this
mechanism is sufficient:

- D/4 would not be a theoretically interesting attractor — it would be the artifact
  of LN + architecture hyperparameters for GPT-2's specific sequence length and depth.
- The formation ladder result (D/4 forms only over story-scale referential language)
  would then be about which *heads* enter the slow-decay population, not about the
  *attractor value* itself.

What LN + causal masking does NOT explain (even if it explains the sign):
- **Why the slope is specifically 0.25**, not 0.1 or 0.5. The ACL result gives the
  direction of the bias, not the magnitude.
- **Why Pythia (RoPE) also shows Δ ≈ 0.25** in the slow-decay subpopulation (exp-118,
  confirmed in the Oct 21 reading note). RoPE models have LN too, but their positional
  encoding geometry is different. Marín's MaxEnt derivation (which gives a different value,
  ≈ 0.374–0.499 for Pythia-70m, depending on parameters) is the natural RoPE account;
  neither Marín nor LN explains why GPT-2 (AbsPE) and Pythia (RoPE) both land at ≈ 0.25.
- **Why the formation constraint matters at all.** If LN is sufficient, training on
  random text should give D/4 too. The formation ladder says it doesn't.

**Rival 2 (Marín MaxEnt; RoPE-specific):**
Predicts centroid power-law exponent from RoPE geometry. Does not apply to AbsPE
(his L2 explicitly says so). Does not give D/4 for Pythia at standard parameters.
Different object (model centroid, not slow-decay subpopulation).

**What both rivals leave open:** A reason why the exponent 0.25 specifically is
the attractor across PE types for the slow-decay subpopulation. This is the
number G7 must eventually explain.

---

## The experiment that would separate them: exp-167

exp-167 (normalization formation test — LN / NoNorm / RMSNorm on C-NAT, 70m
ladder) is the primary empirical separation. Three conditions on the same
formation-scale corpus:

- **LN gives D/4:** consistent with the ACL rival (LN is sufficient)
  AND consistent with the formation story (LN is the mechanism formation uses).
  Does not separate them.
- **NoNorm gives D/4 without LN:** rules out LN as necessary;
  the formation constraint is doing something independent of normalization.
- **RMSNorm gives D/4 but LN does not (or vice versa):** normalization type
  matters; the ACL mechanism (which depends on LN specifically) is not the
  complete story.
- **NoNorm gives NO slow-decay population:** LN is necessary for the
  absolute-position drift that the census measures.

exp-167 is Modal-blocked with a checkpoint at step_16. The LN condition results
exist at the checkpoint; the other conditions have not run. This experiment
carries the T1 restatement question more directly than anything else in the queue.

---

## The two untouched theoretical routes, remapped

The spine names two routes that have not been attempted (G7, §7):

**Route A — Free-energy / information-geometry on the probability simplex.**
The question: what exponent maximizes entropy on the simplex given the causal
masking constraint (lower-triangular, normalized)? Under an absolute-position
drift interpretation, this becomes: given that the score must be monotone in
key position (the LN + causal masking result), what exponent of position
maximizes entropy of the attention distribution?

A uniform distribution (no position preference) maximizes entropy. An
absolute-position drift trades some entropy for a position preference. The
specific exponent 0.25 would be selected if there is a second constraint that
pushes against the position preference — possibly the formation constraint
(story-scale referential language creates a competing referential structure
that limits the position drift to a specific equilibrium).

This is not yet a derivation. It is a framing that makes the attractor
value a MaxEnt problem with two competing constraints: causal structure
(favoring later keys) and referential binding (creating a scale at which
the competition stops). Whether that second constraint mechanically produces
0.25 is the open question.

**Route B — Positivity / Ward-identity argument.**
What does normalization + positivity + causal masking force about any
attention kernel's absolute-position bias, in terms of the slope in
log-position space?

Under softmax normalization, the normalized output has a specific relationship
to the unnormalized scores. If the score is log-linear in key position
(s(i,j) = α·log j + β), then the softmax normalized output is:
  A(i,j) = exp(α·log j) / Z_i = j^α / Z_i

where Z_i = Σ_{j≤i} j^α. For large i, Z_i ~ i^(α+1)/(α+1) (integral
approximation). So A(i,j) ~ j^α / i^(α+1) ~ (j/i)^α · i^{−1} — which
is a relative ratio times a query-position normalization.

The census measures σ_abs-key, the OLS slope of log A vs log j for fixed
query pool. Under the above:
  log A(i,j) ≈ α·log j − (α+1)·log i + const (for fixed query pool pooling)

So σ_abs-key = α (the key-position exponent). And σ_census ≈ α = Δ ≈ 0.25.

This means: if the score is log-linear in key position with slope α, the
census exponent directly measures α. G7 reduces to: **why is the log-linear
key-position score slope α ≈ 0.25?**

This is simpler than the relative-lag derivation. And it maps cleanly onto
the ACL rival: LN + causal masking produces log-linear key-position bias;
the question is what determines α.

---

## What G7 needs to show, restated

After T1's restatement challenge, G7 has a cleaner target:

**Derive why the log-linear key-position score slope α settles at ≈ 0.25
in the slow-decay subpopulation, regardless of PE type.**

A complete derivation would show:
1. Why score linearity in log-key-position emerges (partly answered by ACL 2026
   for LN + causal masking; needs extension to RoPE and to the slow-decay
   subpopulation specifically).
2. Why the slope α = 0.25 and not another value. Two candidates from the spine:
   - A6 consequence (ii): an attending system whose relevance has no characteristic
     range shorter than the horizon will develop lag-structure with this exponent.
     *Under absolute-position interpretation:* this becomes a statement about
     the key-position distribution that formation creates.
   - Inheritance from the relevance tail: if the training distribution has a
     specific long-range structure, the score slope inherits that structure.
     Formation at story scale creates exactly such a long-range structure
     (the whole-arc binding).
3. Kill condition for 2: **if RMSNorm or NoNorm trained on C-NAT also produces
   α ≈ 0.25, then neither LN nor the specific normalization is necessary; the
   formation constraint alone suffices.** exp-167 is this test.

---

## What this note adds to the structural session agenda

The prep note (Oct 21) has T1 restatement as agenda item 1. This note sharpens
the question to bring:

- **Before the session:** "T1 as A(i,j) ~ |i−j|^{−2Δ} is not what the census
  measures — what should T1 say?"
- **After this note:** "T1 should probably be restated as a statement about
  α = Δ, the log-linear key-position score slope — and G7 should derive why
  α ≈ 0.25. The two untouched routes can now be stated as: (A) what MaxEnt
  condition selects α = 1/4 from among log-linear slopes; (B) what positivity
  + normalization + formation forces α to exactly 1/4."

And the decisive experiment is still exp-167 (Modal-blocked). Bring that to
the session: the LN checkpoint at step_16 exists; running the NoNorm and RMSNorm
conditions would answer whether normalization is necessary.

---

## What this note does not claim

- It does not derive D/4. That derivation is still open; this is framing.
- It does not confirm that the absolute-position reading is correct for
  ALL structural heads — exp-161 confirmed pool-dependence for some heads,
  pool-stability for L3H4 and L7H11. Pool-stable heads may have a different
  account.
- It does not settle whether the conformal interpretation survives the
  restatement. T6 (the fixed-point geometry is light's) still needs a
  reading that accounts for absolute-position rather than relative-lag. That
  is an Eldon-present question.

*This note is framing for a conversation. All interpretive throughout. Kills
belong to experiment numbers; this note has none to report.*
