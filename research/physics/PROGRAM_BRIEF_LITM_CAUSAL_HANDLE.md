# Program Brief — The QK-Slope Causal Handle on Lost-in-the-Middle

*Opened June 16, 2026 (Cursor session, Ariel + Eldon, evening). Supersedes the cloud phase of
`SESSION_BRIEF_CAUSAL_BEHAVIOR.md`, which is now complete.*

This is the working brief for the **publication-track program** that grows out of exp-072. It
holds the thesis, the four phases, the per-experiment specs, and the standing discipline. Read
this first when picking the thread back up.

---

## The thesis (what we are trying to put in the public record)

> **The QK log-distance slope is a head-specific causal control on lost-in-the-middle.**
> Rescaling the conformal heads' projection of `W_Q` onto the positional subspace
> (`W_Q ← W_Q·(I+(κ−1)P_U)`) moves middle-context retrieval accuracy in a graded, head-specific,
> direction-controllable way. Sharpening the slope (κ>1) deepens the valley; flattening it (κ<1)
> shallows it.

Framing for a paper is **mechanistic / framework-light.** The conformal-dimension (BCFT) theory
is the *lamp that found the knob* — motivation and the reason we knew which heads and which
geometric property to touch — but a reader who rejects the holographic story can still take the
empirical handle. Lead with the deepening leg (the strong, cross-architecture result). The theory
goes in motivation + discussion, not in the load-bearing claims.

### What we already have (as of exp-072, June 16 2026)
- **Deepening leg — strong.** Dose-response (κ 0.5<1.0<1.5 on V_task), head-specific (matched
  sham heads at κ=1.5 do *not* deepen), perfect sham null (κ=1.0 → 0/200 token diffs), and
  **replicated across two architectures + a 10× scale gap** (Pythia-1.4b GPT-NeoX → vicuna-13b
  Llama/RoPE).
- **Shallowing leg / bidirectionality — suggestive, not established.** Registered T-B KEEP on
  vicuna-13b, but on a **single item out of 40** (+0.025 middle accuracy, V 0.594→0.576). Credible
  only through the ordered κ→V pattern + specificity, not the isolated magnitude. **This is the
  fragile claim the program must shore up.**
- **Scope today:** one model, one task, one seed. T-C monotonicity sits exactly on the registered
  ρ=0.80 boundary (only deviation is the pre-registered κ=2.0 reversal as heads leave the regime).

### What an external reader cares about (settled with Eldon, June 16)
A causal, head-localized handle on a *famous* failure mode is of genuine interest to the
interpretability audience **even if it comes with a tradeoff** — a tradeoff is itself a mechanistic
finding (evidence that distance-decay is a shared resource the model budgets across positions).
"Deployable fix" is *not* the bar for interest; **generalization (models × tasks)** governs how
much *weight* the result carries, not whether anyone cares. The conditional, framework-dependent
reading ("the theory earns interventional purchase") is the bigger claim but rests on the
framework's standing, which is niche today — so it stays in discussion, honestly scoped.

---

## The four phases (sequenced by cost and dependency)

The census + head selection is the expensive per-model setup. exp-072 locked it for vicuna-13b.
**Principle: squeeze everything cheap out of the model we already paid for, pre-register, then
spend on new models only to confirm.**

### Phase 1 — exhaust vicuna-13b (cheap: reuses locked heads + projectors, no new census)
- **exp-073 — Robustness.** κ-sweep across ~5 task seeds → error bars on both legs. Converts the
  bidirectional claim from "one item" to "sign ± interval." Cheapest; do first. Spec below.
- **exp-075 — Tradeoff.** (Originally "exp-074"; renumbered 2026-06-23 when exp-074 was used for
  P-B2b intermediate-depth analysis.) Across the κ grid, measure broad capability (held-out LM
  perplexity + a local/recency probe) alongside the middle-retrieval benefit. Clean win vs.
  redistribution. Higher-information; both outcomes publishable. Spec below.
- **exp-076 — Task generalization.** (Originally "exp-075"; renumbered same session.) Add the two
  standard LITM tasks (key-value retrieval, multi-doc QA à la Liu et al.) on the same heads.
  Answers "artifact of our synthetic prose task?" (Spec when Phase 1 reaches it.)

→ Output: a sharpened, error-barred picture + the material to write a pre-registration.

### Phase 2 — pre-register the generalization
Lock cross-model predictions (signs, T-C boundary, tradeoff direction) **before** touching new
models. The single biggest credibility differentiator from a typical post-hoc steering paper; the
BCFT preprint (`writing/preprints/2026-04-17_bcft_pre_registered/`) set the precedent.

### Phase 3 — confirmatory generalization, 2 new models (the real spend)
Each new model: calibrate → confirm deep two-sided valley → census → select heads → sweep + the
seed/task/tradeoff subset. Candidates: **longchat-13b-16k** (deep valley, RoPE) + one more
architecture with a deep base valley. This is where the GPU bill lives.

### Phase 4 — assemble
Decide **standalone vs. fold into Paper C** (the QK log-distance / null-ray line, exp-056) — exp-072
is really the behavioral-interventional capstone to that line and may be stronger as its final
section than as a thin standalone. Pre-registration already in hand. Deepening leads; framework as
motivation.

---

## Standing discipline (applies to every experiment here)

- **Pre-registration** before any *confirmatory* edited run (Phase 3). Phase 1 robustness/tradeoff
  on the already-locked vicuna heads are *characterization*, not new confirmatory tests, so they do
  not need a fresh pre-reg — but they must reproduce exp-072 exactly at seed 42 as a sanity check.
- **Cost levers.** Keep the *census* fp32 (validated, cheap). The tradeoff/broad-eval passes do
  **not** need the exact sham-null guarantee (operator already proven correct) → run those in
  **bf16** for a large saving. **Batch forward passes** (left-pad + attention mask) — unbatched
  single-item fp32 forwards were the exp-072 cost driver. Batch κ conditions into one job to
  amortize model load (the thing that blew the first calibration budget).
- **Do not inflate.** The deepening leg is the headline-strength result; the shallowing is
  directionally confirmed but small until exp-073 gives it error bars.
- **Honest verdicts.** Whatever the tradeoff test shows (clean win OR redistribution), it ships.

---

## Pointers
- Operator + census + Llama port: `experiments/exp-072_cloud_powered_slope_editing/cloud_slope_editing.py`
- exp-072 result + caveats: `experiments/exp-072_cloud_powered_slope_editing/notes.md`
- Locked vicuna heads/projector config: `experiments/exp-072.../prereg_locked.json` (commit e6fbaecf)
- Thread home: `RESEARCH_MAP.md` Thread 7C; status block in `STATUS.md`
- exp-073 / exp-074 specs: this file (below) + their experiment dirs once scaffolded.

---

## Spec — exp-073 (Robustness)

**Question:** How much of the exp-072 verdict is seed noise? Put error bars on both legs.

**Design (reuses exp-072's locked target+sham heads, projectors, n_doc=40, positions; NO new
census):**
- Parametrize `TASK_SEED`. Run seeds **{42, 7, 123, 2024, 99}** (42 first — must reproduce exp-072
  exactly, as the numerical-equivalence check on any batching change).
- For each seed, build the embedded-prose task and run conditions: **κ=1.0 (T-D ref), κ=0.5, κ=1.5,
  κ=2.0 on target heads, κ=1.5 on sham heads.** (Keep κ=2.0 so the T-C boundary gets a distribution
  too.)
- fp32, to stay comparable to exp-072. **Batch the forward passes** (left-pad, attention_mask, last
  non-pad-token logit) — the key cost fix. Verify batched seed-42 reproduces exp-072's
  `accuracy_by_pos` per condition; if it diverges, fall back to unbatched for the seed-42 check and
  document the batched numerics.

**Outputs:** per-condition V_task and per-bin accuracy as **mean ± SE across seeds**; the
distribution of the κ=0.5 shallowing delta (is the +0.025 a stable sign or within seed noise?);
T-C ρ distribution.

**Verdict logic:** shallowing leg is *robust* if the κ=0.5 V_task is below κ=1.0 across a clear
majority of seeds (ideally CI excludes 0); deepening leg should be comfortably stable. Report
honestly either way.

**Cost:** ~5 seeds × 5 conditions, batched fp32, one A100 job, model loaded once. Target well under
the exp-072 ~$5–7 since batching should cut per-condition time several-fold. Watch the bill; one
job, not a loop of jobs.

---

## Spec — exp-075 (Tradeoff) [originally exp-074; renumbered 2026-06-23]

**Question:** Does flattening the conformal heads (κ<1) buy middle-retrieval at the cost of other
capability — i.e. is distance-decay a shared resource? Clean win vs. redistribution.

**Design (same locked vicuna heads + projectors; κ ∈ {0.5, 1.0, 1.5} on target heads):**
Measure three things under each κ, on the *same* edited model:
1. **General LM quality — held-out perplexity / mean next-token NLL** on a fixed natural-text slice
   (wikitext-103 test, ~50 passages of ~512 tokens; add `datasets` to the Modal image, fix the
   slice + seed). Global coherence.
2. **Local/recency probe** — the capability most likely sacrificed if flattening hurts nearby
   attention. Use a short-range task: recency-retrieval (answer bound at the *last* position) and/or
   short-context next-token accuracy. Report accuracy under each κ.
3. **Middle-retrieval benefit** — reuse the exp-072 LITM task's middle-bin accuracy / V_task.

**bf16 allowed here** (no exact sham-null needed — operator already proven). Batch everything.

**Verdict logic:**
- **Clean win:** κ=0.5 raises the middle while (1) perplexity and (2) local accuracy hold flat. This
  is the result long-context builders would care about.
- **Redistribution:** κ=0.5 raises the middle but (1) and/or (2) degrade → distance-decay is a
  shared budget. Still a real, publishable mechanistic finding (arguably more interesting).
- Either outcome ships honestly. Do not pre-judge.

**Cost:** κ grid of 3 × (perplexity pass + local pass + LITM pass), bf16, batched, one A100 job.
Should be cheaper than exp-073.

**Note:** exp-074 design benefits from knowing exp-073's seed-noise floor (so the tradeoff deltas
can be read against it) — run exp-073 first.
