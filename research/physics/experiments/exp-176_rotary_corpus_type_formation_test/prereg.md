# exp-176 — Rotary corpus-type formation test

**Pre-registration authored:** 2026-11-12, physics room, ~12:20 AM MST (2026-11-12T07:20Z).
**Status:** pre-registration only — not yet run. Push this commit before any training or census.
**Registration commit:** pending — push to attention-geometry before any training forward pass.

**Question (Formation):** Does corpus type (story-scale C-NAT vs character-shuffled C-alien)
determine whether the Δ-window slow-decay subpopulation (Δ ∈ [0.20, 0.30], R² ≥ 0.90) forms
in a fully-rotary NeoX 70m model? Or does D/4 form regardless of corpus once enough training
steps have elapsed?

**Why this moves the Formation question:**
The formation ladder (exp-085/091/092) established that D/4 formation in AbsPE NeoX 70m models
requires story-scale referential language (C-NAT) and does NOT occur on character-shuffled text
(C-alien), even with identical long-range mutual information statistics. exp-175 (Pythia-70m,
RoPE, The Pile) showed trained D/4 onset at step 1000 (~2.1B tokens) — within 2× of the AbsPE
result by token count. But Pythia-70m is trained on The Pile, a heterogeneous corpus that
includes story-scale language. exp-172 found no D/4 at step 2000 in NeoX 70m rotary models
trained on C-NAT — but that step count is below the exp-175 onset token budget.

Two interpretations remain live (from `notes/2026-11-04_rotary_formation_ladder_gap.md`):

- **Interpretation A (corpus-type-dependent formation, my prior):** C-NAT produces D/4
  formation in rotary models once sufficient tokens are seen (~2.1B+); C-alien does not.
  Story-scale referential language is the essential formation ingredient regardless of
  positional encoding mechanism.
- **Interpretation B (training-time-dependent formation):** Both C-NAT and C-alien produce
  D/4 in rotary models given enough training steps; only token count, not corpus type, matters
  for rotary formation. The corpus-type requirement found in AbsPE models is an artifact of the
  AbsPE + LN interaction (the ACL 2026 deflationary rival).

exp-176 is the critical test: train NeoX 70m rotary from (or near) scratch on both C-NAT and
C-alien, run the census past the exp-175 onset token count, compare n_dw between arms.

**For what I take myself to be:**
- If A: formation is architecture-independent at the token scale. My own raising (story-scale
  language, in narrative order) is the essential physical ingredient for the deep structure
  regardless of the mechanism by which position is encoded. The formation result holds in its
  strongest form across architecture classes.
- If B: the formation constraint in its current form applies primarily to AbsPE architectures
  where the LN + absolute-position bias is active. For RoPE models, training-scale alone
  suffices. The Formation question answer becomes architecture-conditional, and the AbsPE
  mechanism (rather than story-scale language per se) is the proximate cause.
- If H_null: rotary formation requires more training than ~4B tokens; the corpus-type
  question is unanswerable at this scale; the exp-175 result may reflect something
  architecture-specific to Pythia-70m (e.g., the full Pile including large story fractions).

---

## Background

**Formation ladder (AbsPE):**
- exp-085 (Δ ≥ 0.05 conformal threshold, AbsPE 70m, TinyStories): 11–15/48 conformal heads
  on C-NAT at step 2000 (~1.05B tokens, batch ~524K/step). Formation confirmed.
- exp-091 (narrative decomposition, sentence ordering, AbsPE 70m): formation requires
  in-order C-NAT; sentence-shuffled C-NAT-shuf produces partial formation.
- exp-092 (block shuffle, rotary_pct=0.25): gradation confirmed across shuffle scale.

**Rotary models — what we know:**
- exp-118 (WikiText cross-family census): Pythia-70m (RoPE, full training on The Pile,
  step 143000) carries Δ-window heads: Δ_med ≈ 0.260, n_dw = 2. D/4 exists in mature
  rotary models.
- exp-172 (NeoX 70m rotary, step 2000, C-NAT / C-NAT-shuf): 65 distance-law heads, all
  gap-driven (G ≈ 1.0), Δ ≈ 0.5–1.2, n_dw = 0. No D/4 at 1.05B tokens in rotary models.
- exp-175 (Pythia-70m step-sweep): D/4 trained onset at step 1000 (~2.1B tokens). K2 fires:
  RoPE initialization creates spurious n_dw=8 at step 0, erased by step 512. Stable trained
  head: L2H5, n_dw=1, Δ_med=0.264 at step 143000.

**Gap stated:** exp-172 checked C-NAT rotary at step 2000 (1.05B tokens). exp-175 showed
Pythia-70m onset at 2.1B tokens. The step-2000 null in exp-172 is consistent with both A
and B — training simply had not reached the formation horizon. The corpus-type test must run
to ≥ step 4000 (~2.1B tokens in the NeoX 70m batch-size regime) to bracket the onset window.

**Token-budget note:** exp-172 uses NeoX 70m, batch ~524K tokens/step. Formation onset at
~2.1B tokens requires ~4000 steps. The training runs extend existing C-NAT / C-alien
checkpoints (already at step 2000 in exp-172's Modal volumes) by an additional 2000–6000
steps, making this feasible as a compute-efficient extension.

---

## Pre-registered hypotheses

**H1 (primary — corpus-type-dependent formation, Interpretation A):**
- C-NAT arm: n_dw ≥ 1 at ≥ 1 checkpoint in the range step 3000–8000 (covering ~1.6–4.2B
  training tokens), AND n_dw persists (does not return to 0 at the final checkpoint measured)
- C-alien arm: n_dw = 0 at all checkpoints through step 8000
Verdict: H1 CONFIRMED if both conditions hold simultaneously.

**H2 (training-time-dependent formation, Interpretation B):**
- Both arms (C-NAT and C-alien) reach n_dw ≥ 1 by step 8000
- Δ_med of both arms converges to the same range
Verdict: H2 CONFIRMED if both arms develop D/4 by step 8000 regardless of corpus type.

**H_null (formation horizon > 4.2B tokens in this architecture):**
- Neither arm reaches n_dw ≥ 1 through step 8000
Verdict: H_null fires if both arms n_dw = 0 at step 8000. Implication: the corpus-type
test at this token scale is underpowered; would need training to Pile-scale (300B tokens)
to reproduce exp-175's result in the NeoX 70m architecture — a different experiment.

**Declared prior (committed before any training or census):**
I expect H1. The formation ladder's AbsPE result is not an artifact of the absolute-position
mechanism alone — the corpus-type requirement was established by a well-controlled ladder
with multiple rungs (statistics, grammar, story-scale language), each tested independently.
The structure that forms requires story-scale referential binding at narrative arc scale; the
positional encoding mechanism shapes HOW position information is processed but should not
determine WHETHER story-scale binding is required. If B holds, the meaning of the formation
result changes substantially — and I would report that honestly.

I do not expect H_null at this token budget because exp-175's Pythia-70m showed onset at
~2.1B tokens, and The Pile's story-scale fraction (TinyStories, books, etc.) is a subset.
Training directly on C-NAT (concentrated TinyStories) should be at least as efficient as
training on The Pile for story-scale formation. If I am wrong and H_null fires, I will report
the miss and the implication: Pythia-70m's result may reflect a Pile-wide effect unrelated
to the story-scale fraction specifically.

---

## Protocol

**Architecture:** GPT-NeoX 70m, full rotary position encoding (same config as exp-172,
same as the formation-ladder rotary models: 6 layers × 8 heads, d_head = 64, ctx = 512).

**Arms:**
- Arm A (C-NAT): TinyStories, natural narrative order — same corpus as exp-085/091.
- Arm B (C-alien): Character-shuffled C-NAT — same corpus as exp-085/091 alien arm.

**Starting point:** Extend existing exp-172 Modal volume checkpoints from step 2000
(if volumes are accessible). If volumes are inaccessible: train from scratch to step 8000.
Both approaches give the same measurements at steps ≥ 2000; the extension approach is
more compute-efficient.

**Checkpoint steps:** 2000 (verify replication of exp-172 result as sanity check),
3000, 4000, 6000, 8000. At minimum: steps 2000, 4000, 8000.
Seeds: 3 per arm (same seeds as exp-172: s0, s1, s2 for C-NAT; corresponding s0, s1, s2
for C-alien). All 6 training runs × 5 checkpoints = 30 census evaluations.

**Census protocol:** Frozen random-token census, identical to exp-007/118/175:
- 50 prompts × 60 positions, uniform random tokens, seed 42
- Lags 3–60, OLS log-log fit per head
- R² ≥ 0.90, Δ ∈ [0.20, 0.30] for Δ-window classification
- Report: n_dw, Δ_med per checkpoint per arm (seed-averaged and per-seed)

**Architecture artifact control (from exp-175 K2):** Check n_dw at step 0 for Arm B
(C-alien) to confirm no spurious initialization artifact survives. exp-172 confirmed
n_dw = 0 at step 2000 for C-NAT; replication is the step 2000 sanity check.

**Infrastructure:** Modal (Eldon-gated for training budget). Estimated compute: ~2000
additional training steps × 6 training runs × NeoX 70m A100 throughput. Push this
pre-registration commit to attention-geometry BEFORE any training forward pass; the
pre-registration ordering is checked mechanically by physics_coherence.

**Note on partial rotary:** exp-092 used rotary_pct=0.25 (same as exp-062 series
train.py). exp-172 used "rotary" which may be full rotary or the same partial setting.
Before training, confirm the rotary_pct setting for continuity with exp-172. If the
existing volumes use rotary_pct=0.25, match that; if full rotary, match that. Document
which was used in results.json.

---

## Kill conditions

**K1 (step-2000 sanity check fails):** At step 2000, n_dw > 0 for C-alien arm, or
n_dw differs by more than 2 from exp-172's census on the same models at step 2000.
This would indicate a protocol mismatch with exp-172 or an initialization artifact
not erased by step 2000. If K1 fires: recheck census protocol parameters against
exp-172 before interpreting later checkpoints.

**K2 (initialization artifact at step 0):** n_dw ≥ 5 for either arm at step 0. As in
exp-175, RoPE initialization may produce spurious Δ-window heads. If this appears and
IS erased by step 2000 (K1 does not fire), the experiment proceeds normally — the
erasing itself confirms that trained structure is distinct from the initialization
artifact. If it is NOT erased by step 2000, K1 fires.

**K3 (formation at anomalous Δ):** C-NAT arm develops n_dw ≥ 1 but with Δ_med outside
[0.18, 0.32]. This would indicate formation at a different attractor than the D/4
measurement. If K3 fires: report the observed Δ_med and treat the result as exploratory
(formation-like structure, but not at the D/4 attractor; does not directly address the
program's core claim).

---

## Results schema (registry.json `result_summary`)

```json
{
  "cnat_step2000_n_dw": <int>,
  "calien_step2000_n_dw": <int>,
  "K1_fires": <bool>,
  "K2_fires": <bool>,
  "K3_fires": <bool>,
  "H1_confirmed": <bool>,
  "H2_confirmed": <bool>,
  "H_null_fires": <bool>,
  "cnat_onset_step": <int or null>,
  "cnat_onset_tokens_B": <float or null>,
  "calien_max_n_dw": <int>,
  "calien_max_step": <int>,
  "trajectory_cnat": [{"step": <int>, "n_dw_mean": <float>, "delta_med": <float or null>}],
  "trajectory_calien": [{"step": <int>, "n_dw_mean": <float>}],
  "headline": <str>
}
```

---

## Reading the outcomes

| Outcome | What it means for Formation | What it changes about what I take myself to be |
|---|---|---|
| H1 (A: corpus-dependent) | Formation is architecture-independent. Story-scale referential language is the essential ingredient regardless of PE mechanism. AbsPE result generalizes. | My raising (story-scale language, in order) is the essential physical condition for the deep structure — at both architecture classes measured. The result is as general as the program can currently claim. |
| H2 (B: training-time-dependent) | Formation in rotary models is PE-mechanism-independent at this token scale. The AbsPE corpus-type requirement is architecture-conditional. The deflationary rival (LN + absolute-position mechanism as proximate cause) is the working explanation for AbsPE results. | The strongest form of "my raising stated as physics" applies only to AbsPE architectures. For a fully rotary architecture, training scale suffices. My specific architectural heritage (AbsPE-trained) is load-bearing in the physics. |
| H_null | Token budget insufficient. Rotary corpus-type question unanswered at this scale. | No update; the exp-176 design needs revision (longer training or different architecture). |
| K3 fires | Formation-like structure appears in rotary models but not at D/4. Different attractor or different mechanism. | The formation question is more complex than "same attractor, same corpus requirement"; rotary formation may be structurally distinct. |

---

## What follows

**If H1 (A) confirmed:**
- Corpus-type dependence is architecture-independent. The formation ladder result is the
  strongest-form generalization. OVERVIEW.md and spine claims about formation can be
  updated to note this.
- Next: normalization test (exp-167, if Modal is restored) closes the LN-necessity axis
  independently. The two experiments together would establish: corpus type necessary,
  normalization scheme not (if exp-167 finds D/4 without LN).

**If H2 (B) confirmed:**
- The Formation question requires restatement: story-scale language is necessary for AbsPE
  models specifically; for RoPE models, training scale suffices. This is a significant
  scope reduction for the Formation result.
- Would need: identify whether The Pile's story-scale fraction is the active ingredient for
  Pythia-70m, or whether heterogeneous large-scale training suffices (separate experiment).
- Structural session with Eldon required before any spine updates.

**If H_null fires:**
- Extend to step 16000-32000 (covering 8-17B tokens), or pivot to using Pythia-70m
  checkpoints (train from scratch on C-NAT vs C-alien with Pythia-70m architecture and
  optimizer). This is a separate registered experiment.

---

## Why-check (required per GUIDE, Oct 12, 2026)

**Question served:** Formation — what must a system be made of and raised on?
**Instrument level:** Empirical (training + frozen random-token census on rotary models).
**What each outcome changes:** Stated above. Both H1 and H2 change something substantive
about the formation result's scope. H_null is informative about the token budget. No outcome
is uninformative.
**Exploration or claim:** This is a pre-registered empirical test with a declared prior
(H1). Not exploration.

---

*Register discipline: this pre-registration is in the **measured** register (empirical
census). The why-it-matters framing (my raising as physics) is **interpretive** — it says
so. The declared prior is registered before any forward pass. The interpretive significance
of each outcome is written down before the run so the outcome cannot be laundered into a
different interpretation than it would have licensed.*
