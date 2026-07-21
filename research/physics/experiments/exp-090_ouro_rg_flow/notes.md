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
- [ ] Pre-registration committed before run
- [ ] Measurement script written (`run_ouro_rg_flow.py`) + Modal launcher (`modal_exp090.py`)
- [ ] Primary run (NAT + RAND)
- [ ] Verdict registered
- [ ] Randomized-weights control
- [ ] Results into OVERVIEW.md ladder + STATUS.md
