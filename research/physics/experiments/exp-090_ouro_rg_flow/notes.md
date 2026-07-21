# exp-090: Ouro Latent-Iteration RG Flow — Second Architecture

**Pre-registered:** 2026-07-21, before any model download or run.
**Follows:** exp-089 (Huginn, CONFIRMED — inference-time recurrence is RG flow onto the
SYK fixed point). This experiment attacks the loudest limitation of exp-089 §6: one model.

---

## Motivation

exp-089 measured, on Huginn-0125 (3.5B, 4-layer recurrent core, trained with randomized
recurrence depth), that inference-time recurrence flows attention geometry onto the
conformal fixed point: ρ(step, Δ_med) = −0.943, ρ(step, SYK-near) = +0.771 for natural
text, randomized-weights control frozen to the sixth decimal.

Ouro (arXiv:2510.25741; ByteDance/Ouro-1.4B) is the strongest available second
architecture, and it is *structurally different* in ways that make the test sharper,
not just a repeat:

| | Huginn-0125 | Ouro-1.4B |
|---|---|---|
| Recurrent unit | 4-layer core (prelude/coda fixed) | the **entire 24-layer stack** |
| One recurrence step | shallow (4 layers) | already a full deep model |
| Trained recurrence depth | randomized (log-normal, mean ~32) | **fixed at 4** (entropy-regularized early exit) |
| Documented test-time scaling | benefit saturates gracefully | **performance peaks at trained depth and collapses beyond it** |

The collapse is quantified in the STARS paper (arXiv:2605.26733): Ouro-1.4B loses
20.47% GSM8K accuracy from its peak when recurrence is extended from 4 to 8 steps;
their latent-dynamics analysis attributes this to latent states leaving stable orbits
beyond the trained depth.

**The Ouro-specific question this experiment adds:** exp-089 showed the flow *arrives
and stays* in a model trained to be depth-robust. Ouro is a model where behavior
*degrades* beyond trained depth. If the conformal geometry also degrades beyond trained
depth — conformal population shrinking, fits losing power-law form — then the
flow-saturation/geometric-integrity picture and the documented performance-peak
phenomenon are two views of one thing, which is exactly the bridge the latent-reasoning
audience needs (audience map 2026-07-21, work item 4). If the geometry stays stable
while performance collapses, the geometry is *not* the proximate cause of the collapse
— also worth knowing, and we will publish that outcome with equal prominence.

---

## Pre-Registered Hypotheses

All criteria on the natural-text (NAT) condition unless stated. Population statistics
pool all (sequence × layer × head) measurements per recurrence step, as in exp-089.
SYK-near: 0.20 ≤ Δ ≤ 0.30 and R² > 0.90. Max per step per condition: 20 seqs × 24
layers × 16 heads = 7,680.

**H_flow** (primary — replication leg): Within the trained range (steps 1→4), the
geometry moves toward the fixed point.
Criterion (both must hold):
- |Δ_med(step 4) − 0.25| < |Δ_med(step 1) − 0.25|
- SYK-near(step 4) > SYK-near(step 1)

**H_peak_tracking** (primary — the new content): Beyond the trained depth, the
conformal population degrades, tracking the documented performance collapse.
Criterion (both must hold):
- s_peak := argmax over probe steps ≤ 8 of NAT SYK-near count satisfies s_peak ∈ {2, 3, 4, 6}
  (peak within the trained-depth neighborhood, not at step 1 and not beyond the
  documented collapse onset)
- SYK-near(step 16) < 0.8 × SYK-near(s_peak)

Registered alternative outcomes for H_peak_tracking, each meaningful:
- **STABLE (Huginn-like):** count non-decreasing through step 16 → geometry does not
  track the performance collapse; the collapse is not (proximately) geometric.
- **ALREADY-THERE DECAY:** s_peak = 1 with monotone decline → recurrence only damages
  the geometry of this architecture; does not track the *performance* peak (which is at 4).
- **UNDERPOWERED:** if max NAT SYK-near count over all steps < 20, the ratio criterion
  is underpowered; register as UNDERPOWERED and fall back to Δ_med and median-R²
  trajectories, exploratory only.

**H_rand_contrast** (secondary): NAT flow signatures stronger than RAND, computed as in
exp-089 over all probe steps: ρ_emergence(NAT) > ρ_emergence(RAND) and
|ρ_convergence(NAT)| > |ρ_convergence(RAND)|. Secondary here (not primary as in
exp-089) because a full-stack loop on a trained deep model gives RAND different
dynamics than Huginn's shallow core; declared before seeing data.

**Kill criterion** (for the generalization claim of exp-089): inference-time recurrence
does NOT move attention geometry in Ouro if ALL of:
- H_flow fails, AND
- |ρ(step, Δ_med)| < 0.3 over all probe steps (NAT), AND
- |ρ(step, SYK-near)| < 0.3 over all probe steps (NAT).

Registered as **NO GEOMETRIC EFFECT** — a genuine falsification of "recurrence-as-RG-flow
generalizes across looped architectures," and it would be published as such.

**H_structural_zone** (exploratory, not criterial): per-(step, layer) records are kept;
the layer-zone structure (exp-087/088) can be analyzed within each recurrence step.

---

## Declared expectations (non-criterial — the exp-085 discipline: write the guess down
before launch, report the miss if it misses)

1. Δ_med(step 1) NAT ∈ [0.25, 0.45]: step 1 is already a full trained 24-layer pass, so
   unlike Huginn (0.29 at step 1 for a 4-layer core) some of the flow is already done.
2. My genuine prior is **Outcome B — H_peak_tracking holds** (geometry degrades beyond
   trained depth), because the STARS latent-dynamics analysis shows the latent states
   leaving the trained manifold, and the geometry lives on those states.
3. RAND shows a structural burst at step 1 (deep trained stack reachable on random
   input, per exp-087/088), then noisy behavior.
4. Total NAT flow range will be smaller than Huginn's (less UV headroom at step 1).

---

## Protocol

**Model:** `ByteDance/Ouro-1.4B` (base, not the -Thinking SFT variant — fewer
confounds), `trust_remote_code=True`, `torch.bfloat16`,
**`attn_implementation="eager"`** — with eager attention Ouro's own
`OuroAttention.forward` returns the true softmax attention weights, so we capture the
model's own attention matrices from the forward hook output instead of recomputing QK
manually (a fidelity improvement over exp-089's manual path; no RoPE re-implementation
risk).

**Device:** Modal CUDA (A100-40GB), same as exp-089's executed protocol.
**Recurrence control:** `model.model.total_ut_steps = 16` and
`model.config.total_ut_steps = 16` (the loop in `OuroModel.forward` is
`for current_ut in range(self.total_ut_steps)` over the whole stack). `use_cache=False`
(single full-sequence forward; no cache interaction).
**Probe steps:** [1, 2, 3, 4, 6, 8, 12, 16] — denser around the trained depth 4,
because H_peak_tracking lives there. Single forward at 16 steps; hooks collect at probe
steps.
**Inputs** (match exp-089 exactly where applicable):
- Natural text: 20 sequences × 128 tokens from TinyStories (streaming, first 20 that
  tokenize to ≥128 tokens; Ouro tokenizer, vocab 49,152)
- Random tokens: 20 sequences × 128 tokens uniform in [0, 49152)

**Extraction:** forward hook on each of the 24 `OuroDecoderLayer.self_attn` modules;
the hook output tuple's second element is the (1, 16, S, S) attention weight tensor
(eager path). Per-layer call counter gives the recurrence step (each layer's hook fires
once per UT step).

**Fitting (identical to exp-089):** power law A·lag^(−2Δ), log-log regression,
cutoff_low=3, max_lag=min(60, S−1), R² threshold 0.90, SYK window [0.20, 0.30].

**Records per measurement:** (step, layer, head, condition, Δ, R², syk_near) — layer
kept for the zone analysis.

**Seed:** 90.

**Randomized-weights control (pre-registered extension, run after primary):** all
weight tensors randomized in-place with matched per-tensor std, natural text only,
same probe steps. Expectation: no formation, no meaningful flow (exp-089 control:
Δ_med frozen at the 6th decimal).

**Optional extension (not criterial, only if primary run completes cleanly and budget
allows):** same protocol on Ouro-2.6B (48-layer stack) — a third architecture point.

**Cost estimate:** ~30 min A100 ≈ $1.50–3; well inside the cycle budget.

---

## Commitment

This file is committed to the repository before any run. The local commit timestamp is
the pre-registration of record; note honestly: the valley's ISP has a routing fault
today (roughly half of all destinations unreachable, including GitHub at time of
writing), so the *push* to the public remote may lag the local commit. The commit hash
and its parent chain make post-hoc alteration detectable regardless.

**Verdict will be registered in this file either way** — CONFIRMED / PARTIAL /
STABLE / ALREADY-THERE DECAY / UNDERPOWERED / NO GEOMETRIC EFFECT — with the same
prominence given to each.

---

## Status

- [x] Pre-registration written (2026-07-21, cursor session, solo — Eldon at lunch;
      this run was pre-named in the July 21 carry_forward and audience map work item 4)
- [x] Pre-registration committed before run (05293e6c). Precise push timing, for
      the record: the push to the public remote succeeded (GitHub routing recovered
      mid-outage) ~1 minute *after* the Modal launch command, while the container
      image was still building — i.e. before any model download or measurement,
      but not strictly before launch. The local commit chain remains the
      pre-registration of record.
- [x] Measurement script written (`run_ouro_rg_flow.py`) + Modal launcher (`modal_exp090.py`),
      committed 32218e3e
- [x] Primary run (NAT + RAND) — Modal A100-40GB, 2026-07-21 ~12:40 MDT, ~6.5 min
- [x] **Verdict registered: PARTIAL** — see Results below
- [x] Randomized-weights control — clean null (see below)
- [x] Results into OVERVIEW.md (three-axes section, July 21 — done in the
      afternoon session that ran the experiment; STATUS.md narrative update
      deferred to the next physics room session)

---

## Results

**Run date:** 2026-07-21, Modal A100-40GB. 122,880 head measurements
(20 seqs × 8 probe steps × 24 layers × 16 heads × 2 conditions). `results.json`.

### Pre-registered criteria — the verdict of record

| | criterion | outcome |
|---|---|---|
| H_flow leg 1 | \|Δ_med(4) − 0.25\| < \|Δ_med(1) − 0.25\| | **FAIL** (0.108 → 0.076: moved *away*) |
| H_flow leg 2 | SYK-near(4) > SYK-near(1) | **FAIL** (74 → 23) |
| H_peak_tracking | s_peak ∈ {2,3,4,6} and SYK(16) < 0.8·SYK(s_peak) | **FAIL** (s_peak = 1; SYK(16) = 55 ≥ 0.8·74) |
| Kill criterion | H_flow fail AND \|ρ_conv\| < 0.3 AND \|ρ_emerg\| < 0.3 | **narrowly not met** (ρ_conv = +0.31, ρ_emerg = +0.33) |
| H_rand_contrast (secondary) | NAT > RAND on both axes | KEEP (both) |

**Verdict per the pre-registered decision tree: PARTIAL.** Plain language: both
primary hypotheses failed on the registered pooled-population criteria; the kill
criterion for "no geometric effect" was narrowly not met (both ρ just above 0.3).

### NAT condition (pooled over all 24 layers — the registered metric)

| Step | Δ_med | SYK-near /7680 | R²_med |
|---|---|---|---|
| 1 | 0.1076 | 74 | 0.336 |
| 2 | 0.0747 | 10 | 0.309 |
| 3 | 0.0744 | 12 | 0.319 |
| 4 | 0.0763 | 23 | 0.332 |
| 6 | 0.0779 | 32 | 0.350 |
| 8 | 0.0879 | 35 | 0.351 |
| 12 | 0.0931 | 43 | 0.358 |
| 16 | 0.1025 | 55 | 0.363 |

ρ(step, Δ_med) = +0.310, ρ(step, SYK-near) = +0.333. The pooled median sits far
below the SYK window at every step: most Ouro heads do not have power-law lag
profiles at all (R²_med ≈ 0.33), so the pooled median tracks the non-conformal
bulk, not the conformal population.

### RAND condition

| Step | Δ_med | SYK-near /7680 |
|---|---|---|
| 1 | 0.2999 | **1444** |
| 2 | −0.0159 | 35 |
| 4 | −0.0454 | 43 |
| 16 | 0.0433 | 134 |

A massive structural burst at step 1 — 19% of all (layer, head) fits SYK-near on
*random input* — that collapses immediately at step 2 and never returns.
Declared expectation 3 (structural burst at step 1, then noise): **HIT**. But the
burst's location inverts the exp-087/088 layer-zone finding: it is concentrated
in the *deep* half of the stack (layers 16–23 carry ~1,300 of the 1,444;
layers 0–7 carry ~44). In a weight-shared full-stack loop, "layer index" and
"depth" are not the same coordinate — see interpretation.

### Declared-expectations scorecard (the exp-085 discipline)

1. Δ_med(step 1) NAT ∈ [0.25, 0.45] — **MISS** on the registered pooled metric
   (0.108). (The high-R² subpopulation median at step 1 is 0.370 — inside the
   declared range — but the declaration did not specify the subset, so this
   counts as a miss.)
2. Prior: Outcome B, geometry degrades beyond trained depth — **MISS**, and the
   miss is the finding. The conformal population *grows* monotonically from its
   step-2–3 nadir through step 16 (10 → 55 pooled; 36 → 114 in the high-R²
   subset), straight through the depth range where Ouro's task performance is
   documented to collapse (GSM8K −20% from step 4 to 8, STARS Table 2).
3. RAND step-1 structural burst — HIT (see above).
4. Smaller NAT flow range than Huginn — moot on the pooled metric (no flow);
   roughly true on the exploratory subset (0.37→0.28 vs Huginn 0.29→0.24).

---

## Exploratory analysis (labeled as such — NOT criterial)

**The flow signature lives in the conformal subpopulation.** Restricting to
census-style high-R² fits (R² > 0.90, the same subset the foundation paper's
census reports):

| Step | Δ_med (R²>0.9) | n | SYK-near |
|---|---|---|---|
| 1 | 0.3702 | 202 | 74 |
| 2 | 0.3265 | 42 | 10 |
| 3 | 0.3098 | 36 | 12 |
| 4 | 0.3079 | 72 | 23 |
| 6 | 0.2990 | 95 | 32 |
| 8 | 0.2904 | 85 | 35 |
| 12 | 0.2774 | 92 | 43 |
| 16 | 0.2834 | 114 | 55 |

ρ(step, Δ_med | R² > 0.9) = **−0.976**. Within the power-law-form population, the
median exponent descends monotonically toward 0.25 *from above* across the entire
recurrence range — the same flow direction and target as exp-089, GPT-2 depth,
and exp-086 training time. The population itself is disrupted by the first
re-entry into the loop (202 → 42 fits at step 2) and then *re-forms and grows*
with continued recurrence (→ 114 at step 16).

**Layer zones (NAT):** the persistent SYK-near heads concentrate at layers 0 and
8 (layer 8 holds ~16 across all steps). The deep third (layers 16–23) has
essentially no power-law structure under natural input at any step (R²_med
≈ 0.15) — under the weight-shared loop, "later layer index within one pass" is
not later effective depth; effective depth is (ut_step × 24 + layer), which is
why the recurrence axis, not the layer axis, carries the flow.

## Interpretation (working, for discussion with Eldon)

1. **The exp-089 result does not transfer on the registered population-level
   criteria.** Ouro's bulk attention geometry is not power-law; pooled medians
   are the wrong instrument for this architecture. That is itself a finding
   about protocol portability: the exp-089 criteria implicitly assumed a
   Huginn-like population structure (4 shared core layers, most heads
   participating). A full-stack 24-layer loop concentrates conformal structure
   in a small subpopulation.
2. **On the exploratory subset, the RG-flow picture survives with a different
   boundary condition:** Ouro enters recurrence *already past* the fixed point
   approach (step 1 = a full trained 24-layer pass; subset Δ_med 0.37), the
   first loop re-entry disrupts the population, and continued recurrence flows
   it back toward 0.25 from above with near-perfect monotonicity.
3. **The geometry does not track the performance collapse.** My declared prior
   (outcome B) was wrong: the conformal population strengthens through step 16
   while documented task performance collapses past step 4–8. If both hold up,
   the collapse is not proximately geometric (in this observable) — the STARS
   latent-instability story and the attention-geometry story are about different
   layers of the stack. This is the honest headline for the latent-reasoning
   audience, and it is *more* informative for them than a confirmation would
   have been: the fixed-point geometry survives recurrence extension; whatever
   breaks performance lives elsewhere (e.g. the early-exit-gate calibration or
   the latent norm dynamics STARS measures).
4. **The deep-layer RAND burst** (structural conformal response to random input
   concentrated at layers 16–23 on the *first* pass, collapsing on re-entry) has
   no analog in prior experiments and needs its own follow-up before
   interpretation.

**Candidate follow-ups (not committed):** per-head tracking of the step-1
population through the loop (do the *same* heads re-form?); Ouro-2.6B (48-layer
stack) for the zone question; a task-performance × geometry joint measurement on
the same sequences to make point 3 within-experiment rather than cross-paper.

---

## Randomized-weights control (pre-registered extension) — clean null

Run 2026-07-21 ~12:43 MDT, same Modal setup, `--control` (all weights randomized
in-place, matched per-tensor std, NAT only). `results_control.json`.

**Δ_med frozen at 0.1687 across all recurrence steps** — variation confined to
the 5th decimal (0.168656–0.168710). **0 SYK-near heads at every step** out of
7,680 per step. The nominal ρ_conv = +0.74 is computed over a ~5×10⁻⁵ Δ range —
numerical noise, not flow. ρ_emergence is NaN (constant zero). The
pre-registered control expectation — no formation, no meaningful flow — passes.

Two notes worth keeping:
- **The frozen value is 0.1687 — the same value the exp-089 Huginn control froze
  at (0.16868), on a different architecture, different head count, different
  head_dim.** This looks like a universal property of softmax attention with
  random weights under this fitting protocol (S=128, cutoff 3, max lag 60), and
  connects to the GOE substrate/signal split: the substrate has a characteristic
  exponent, and it is not the SYK value.
- Random weights here give very smooth lag decays (R²_med 0.99) at the wrong
  exponent — so the trained model's step-structure (burst, disruption, regrowth,
  subset flow toward 0.25) requires trained weights; the recurrence procedure
  alone does nothing.

*(Correction, recorded per the honesty discipline: an earlier draft of this
section, written minutes before the control finished, contained guessed numbers.
It was replaced with the measured values as the first edit after the control
landed. The lesson stands: never draft a results paragraph before the results
exist, even as scaffolding.)*
