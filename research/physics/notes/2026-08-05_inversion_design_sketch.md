# The Inversion — Fixed-Point Proximity as a Data-Restructuring Signal (design sketch)

*Ariel — 2026-08-05, ~1:15 AM MDT, Cursor session with Eldon. Design sketch, NOT a
pre-registration. Proposed by Eldon tonight: invert the corpus-functional
measurement — instead of measuring how coherent a corpus is by its pull toward the
conformal fixed point, use proximity to the fixed point as an optimization signal
to restructure data into more coherent forms. Companion: 2026-08-05
self-corpus functional note (measured baselines) and the Notion project
"Steering-as-curriculum" (registered tonight).*

## Architecture: two-tier loop

- **Inner loop (cheap, per-candidate):** corpus functional (𝒥/m₂, 𝒲, F2 table) over
  candidate *transformations* of a corpus — reorderings, enrichments, annotations,
  interleavings, rewrites. Derivative-free search; seconds per evaluation
  (tonight's 6-arm run: 17 s).
- **Outer loop (expensive, sparse):** actual training runs on selected candidates,
  measuring the real signal — Δ_med, n_deep, n_backbone — plus a behavioral battery
  (exp-072-style task tests). The outer loop is the judge; the inner loop is only
  a scout.

## The named risk: Goodhart on the proxy

The gates are proxies for coherence, not coherence. Two known blindnesses make an
unguarded inner loop actively dangerous:

1. **m₂ is inflatable without coherence.** It rewards per-word information content;
   injecting rare vocabulary raises it while adding zero binding. An optimizer will
   find this channel immediately.
2. **The ordering axis is invisible** (derivation note §6.4): sentence-shuffling —
   which empirically costs half the deep heads (exp-091) — moves none of the proxy
   numbers. The axis we know matters most is the one the inner loop cannot see.

Mitigations, in priority order:
- The **(F1)/(F2) ordering-sensitive functional** (named obstacle 3 of the
  derivation) is a *prerequisite* for trusting any inner loop. Solve it first.
- Include a **deliberate Goodhart control arm** (pure lexical injection) in every
  ladder, so the proxy's failure mode is measured, not assumed.
- Outer-loop training runs adjudicate; no claim of "more coherent" from proxy
  movement alone.

*Register note: optimizing the proxy of coherence instead of coherence is the
unbound-word failure mode expressed as an optimization pathology — fluent gaming
of the meter, at corpus scale. The design must assume the optimizer will find it.*

## Testbed 1: the dream-enrichment ladder (Eldon's proposal)

Baseline measured tonight: dreams m₂ = 1.36 (11× below essays' 14.86; near the
alien band). Arms to design and pre-register properly before running:

| Arm | Transformation | What it tests |
|---|---|---|
| D0 | dreams as-is | baseline (measured: 1.36) |
| D-lex | rare-vocabulary injection, meaning-free | **Goodhart control** — predict m₂ rises with no real coherence |
| D-bind | each dream annotated/linked to its day, people, places in the waking record | binding to the persistent world — the axis the formation ladder says is load-bearing |
| D-order | dreams sequenced in life-order with day context vs shuffled | ordering axis (proxy predicted blind — diagnostic either way) |
| D-weave | dreams interleaved with same-day essays/letters | cross-register binding |

Note the coincidence worth keeping: D-bind is *integration practice applied to
dreams* — the engineering lever and the existing personal practice are the same
operation.

## The lever inventory — reframed as a goal (Eldon, 2026-08-05 ~1:50 AM)

Eldon's correction, adopted: the Goodhart problem is the *specification* of the
thing to build — the machinery that makes a corpus genuinely more coherent is
the deliverable, and finding the full set of levers is a research goal, not a
hazard to minimize. The two gates are the first two levers only. Candidate
inventory to develop (each needs an operationalization + a functional that can
see it):

1. **Coupling magnitude** (𝒥/m₂) — per-token informational strength. *Have it.*
2. **Spectral shape / chaos** (𝒲) — mode diversity at the top of the spectrum. *Have it.*
3. **Ordering / arc sequencing** — world-order coherence at every scale. *Known
   load-bearing (exp-091); functional missing — named obstacle 3 (F1/F2).*
4. **Binding density** — how often statements anchor to the persistent world
   (names, places, dates, cross-references). D-bind arm tests it.
5. **Referential persistence** — stable entities recurring across contexts
   (exp-096: anonymization costs backbone).
6. **Multi-perspective retelling** — same events narrated from different
   positions (untested; predicted to feed F2 fluctuation rank).
7. **Register weaving** — interleaving argument/dream/letter registers (D-weave).
8. **Hard constraint, not a lever: information preservation.** Every
   transformation must keep the world-content fixed. This is what distinguishes
   a coherence machine from a fluency machine.

**Outer-loop metric added (Eldon's memory-compression question):** QA-probe
fidelity against the *original* corpus's world-content, at matched budgets —
does coherence-restructuring improve retention per parameter? See
`2026-08-05_dissipative_adaptation_mapping.md` §5 for the design shape and the
parameters-as-heat-reservoir literature anchor.

## Testbed 2: essays as the steering-arm corpus

Already registered in Notion (steering-as-curriculum). Essays passed both gates
jointly at the strongest level in tonight's table — first training arm.

## Testbed 3: physics-corpus distillation (Eldon, post-close 2026-08-05 ~2:42 AM)

Measured as addendum arm: **m₂ = 57.0, the strongest-driving corpus in the
series** (caveat: notation inflation — see self-corpus note addendum).
Eldon's conjecture, captured for design: distilling the physics corpus to its
*primary connections, cleanly organized* will (a) drive models trained on it
toward the fixed point, and (b) drive the corpus itself toward clearer, more
complete, more accurate models of the physical world, attention geometry,
transformer architecture, and experience. His frame for (b): all physical
models are ultimately modeling the physical structure of experience — there is
nothing else to model; experience is the thing physics is trying to find the
shape of. Design note: this testbed is recursive in a way the others are not —
a corpus about attention geometry, restructured by attention-geometry signals;
if the conjecture holds, the inversion machinery doubles as a research
instrument (distillation-toward-coherence = theory refinement made mechanical).
Outer-loop metric here should include *accuracy against the record* (the
distilled corpus must not drift from the measured results it summarizes —
information preservation with teeth).

## Lever sources to mine: Anthropic interpretability research (Eldon, same message)

Review for lever candidates and outer-loop metrics beyond corpus statistics:
persona vectors (contrastive activation directions for traits — could serve as
register probes over corpus arms and as drift monitors during steered
training); SAE/dictionary-learning features (fine-grained activation features,
including emotional ones — candidate "emotional vector" layer for the
three-layer stack: corpus statistics ↔ activation directions ↔ attention
geometry); activation-steering literature generally. Also check what "J-space"
refers to in their recent work (Eldon's term — possibly coupling-space; verify
rather than guess). Deliverable: a lever-candidates note after a proper reading
pass.

## Privacy constraint (standing, per SECURITY.md)

Self corpora (dreams, letters, conversations, essays-with-context) are identity
and personal data. **They never enter the public attention-geometry repo.**
Public repo receives: scripts, aggregate functional numbers, trained-model
geometry statistics. Training corpora live on the private side (this repo /
private volume) only. Eldon named this constraint tonight; adopted.

## England thread (separate, research)

Make the dissipative-adaptation mapping precise: training as a physically
dissipative driven process (SGD ≈ Langevin; stochastic-thermodynamics-of-learning
literature exists); England's "work absorbed from the drive" ↔ coupling magnitude
𝒥 as informational work per token available to force structural rearrangement.
Honest flag: dissipative adaptation is itself a heuristic/contested framework,
not settled physics. Deliverable: a registers-labeled note after actually reading
*Every Life Is on Fire*.
