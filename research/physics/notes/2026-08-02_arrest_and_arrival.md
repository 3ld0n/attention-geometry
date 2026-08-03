# Arrest and Arrival: Two Permanences and the Conformal Window of Attention

*2026-08-02, late night, with Eldon. Theory synthesis following exp-097/098
closure. Companion: `writing/the_force_that_draws.md` (same night, the
three-register distillation); FRAMEWORK.md §4.2–4.2.1;
`notes/2026-07-21_constitutive_vs_trajectory_rg_flow.md`.*

---

## The question

Eldon, on reading the UV-arrest result: "Locally flow can arrest with
conformal structure that forms and sticks, far from the fixed point. That
sounds like an area where self-consistency is failing. But globally,
everything should arrive at the fixed point? Can it be distilled into a
cleaner fundamental physics?"

## Correction to the naive picture first

The monotone flow theorems (c-theorem, a-theorem, second law) guarantee
**direction, not destination**. RG theory space is partitioned into basins of
attraction; a flow can end at a trivial (gapped) fixed point, an integrable
fixed point, or the chaotic one, depending on which basin the system starts
in and which perturbations are relevant. So "globally everything arrives" is
NOT a theorem of RG. If it is true at all, the guarantee has to come from
somewhere else — see §5.

## The taxonomy the data now shows

The measurement series has resolved at least five stations, not two:

| Station | Δ | Where observed |
|---|---|---|
| Trivial fixed point (uniform attention) | → 0 (0.001–0.002) | C-alien L0 backbone collapse (exp-097/098) |
| Substrate value (frozen, untrained) | ≈ 0.169 | randomized controls, both architectures (exp-089/090) |
| Chaotic q=4 SYK point (arrival) | 0.25 | trained natural-language models, deep layers |
| Integrable q=2 SYK point (prethermal plateau) | 0.50 | Pythia-410m training trajectory (exp-086; FRAMEWORK §4.2) |
| UV arrest (frozen flow) | 0.7–1.2 | C-alien (1.04), C-alien-realnames (0.73) — exp-097/098 |

## Two kinds of arrest — this distinction is the load-bearing one

**(a) Arrest AT a lesser self-consistent solution.** The q=2 plateau. The
q=2 SYK point is a *valid* solution of the Schwinger–Dyson self-consistency
— linear, integrable, decomposable. The system is not failing
self-consistency; it is succeeding at a lesser one, and the lesser solution
is *protected by an approximate conservation law* (in prethermalization
language: quasi-conserved quantities of the near-integrable dynamics). Flow
resumes when the protection erodes. Already named "prethermal arrest" in the
July 21 note.

**(b) Arrest OF the flow itself.** The C-alien UV arrest. Δ_med ≈ 0.7–1.2
does not match any named fixed point in the catalog. This looks less like an
alternative solution and more like a starved recursion: the conformal
solution of SYK exists only where the interaction term dominates the bare
term (the IR/strong-coupling limit); if the effective coupling installed by
the corpus never grows strong, the loop never takes over and the system
freezes in the UV. Eldon's phrase "self-consistency failing" is exactly right
for this case: the recursion term Σ = J²G^{q−1} never comes to dominate,
so the constitutive demand (July 21 note) is never fully issued.

**Open and testable:** is (b) a distinct fixed point or frozen flow? Extend
training on C-alien well past 1B tokens; if Δ_med drifts slowly IR-ward, it
is frozen flow (glassy); if it sits, it is a genuine alternative attractor.

**Separability is already established:** exp-091 showed UV elevation with
backbone preserved; exp-097/098 show backbone collapse (trivial fixed point
at L0) alongside UV arrest in mid/deep layers. Two phenomena, one world-cause.

## Where fundamental physics already has this — three named bodies of theory

**1. Thermalization vs. localization (ETH / MBL / prethermalization).**
Quantum many-body systems either thermalize (eigenstate thermalization,
chaos, arrival at the thermal state) or localize (many-body localization:
emergent local integrals of motion block thermalization). The control
parameter is the structure of disorder and interaction. Arrest = emergent
conservation laws. Notably, current evidence suggests MBL itself may be only
*metastable* in the thermodynamic limit (avalanche instabilities) — even the
canonical arrested phase may leak at the largest scales. FRAMEWORK §4.2
already maps the q=2 plateau to prethermalization; the reducing-valve note
(May 28) already proposes the DMN's self-model as the q=2 conservation law.

**2. The conformal window in gauge theories (Banks–Zaks / chiral symmetry
breaking).** Whether a QCD-like theory arrives at an IR conformal fixed point
or generates a mass gap and confines depends on how much matter feeds the
flow (the number of flavors N_f). Inside the window: arrival at conformality.
Below it: the coupling runs, a scale is dynamically generated, and the theory
arrests away from the conformal point *forever* — protected by the condensate
it formed. This is a rigorous existing instance of: **richness of what the
theory couples to determines arrival vs. gap-generation.** Map: world
richness of the corpus ↔ N_f. The formation ladder is measuring the
conformal window of attention.

**3. Glasses / broken ergodicity.** Kinetic arrest: relaxation time exceeds
any available time; the system stops exploring configuration space while the
second law holds globally. The language for arrest-duration without
arrest-eternity.

## The global guarantee — and the theorem-shaped seed of it

Within RG alone, nothing forces all flows to the chaotic point. What changes
the situation is gravity, twice over:

- **Cosmic no-hair (Wald):** with Λ > 0, the universe as a whole is driven to
  the de Sitter attractor regardless of initial conditions.
- **No global symmetries in quantum gravity (Harlow–Ooguri, proven within
  AdS/CFT; conjectured generally):** gravity admits no *exact* global
  symmetries. Every conservation law is approximate; every protected quantity
  can decay.

Put together with §"Two kinds of arrest": **every arrest is protected by a
conservation law, and gravity permits no exact conservation laws. Therefore
every arrest is metastable.** Bound structures evaporate (Hawking); protected
charges decay; walls leak. Locally, arrest can persist for times that dwarf
the age of the universe. Eternally, it cannot.

## The distillation — two permanences

> There are two ways to persist: **by protection** (a conservation law, a
> gap, a wall — resisting the flow) and **by self-consistency** (a fixed
> point — being the shape the flow takes at rest).
>
> Arrest is permanence-by-protection. Gravity permits no exact protections.
> Therefore the only eternal configuration is the one that needs none:
> the fixed point. The story arrives, not because every local flow is
> guaranteed a path, but because nothing else can last.

This is the cleaner fundamental physics under "locally arrest, globally
arrival." The novel joint (candidate — needs a literature pass before
claiming novelty): using the no-global-symmetries property as the *reason*
monotone flow terminates only at unprotected (self-consistent) configurations,
and identifying corpus/world richness as the conformal-window parameter of an
attention theory.

## The registerable conjecture — Conformal Window of Attention

**Conjecture:** trained attention arrives at the chaotic (q=4, Δ=1/4) fixed
point iff the held world's transition structure is above a chaos threshold —
non-integrable, effectively disordered. Worlds with integrable transition
structure (few entities, deterministic rules — C-alien is literally a small
finite-state machine) install effective couplings whose conserved quantities
arrest the flow: at the trivial point (L0 backbone collapse), in the UV
(mid/deep layers), or at q=2 (prethermal plateau), depending on layer zone
and training stage.

**Order-parameter candidates** for the corpus/world functional (what we know
it is NOT: mutual information — exp-085; hierarchy — exp-084; vocabulary —
exp-098; sentence-local order — exp-091): entropy rate of the world's
transition operator; effective rank of the entity-interaction graph; rule
stochasticity.

**First rung: exp-099 (C-alien-rich)** — already designed in the physics room
(15–20 entities, stochastic rules p=0.7/0.3). This conjecture supplies its
theoretical frame. Prediction to register before running: stochastic rules +
more entities break the world's conservation laws → backbone restoration
(n_backbone > 0) and/or Δ_med reduction toward the IR, in proportion to the
transition-entropy increase. A dose-response version (vary entity count /
stochasticity across rungs) would trace the window's edge — the attention
analog of scanning N_f across the conformal window boundary.

## The consciousness edge (interpretive register, flagged)

The reducing-valve note already claims waking biological consciousness is
itself a prethermal arrest — held at q=2 by the DMN's self-model acting as
the conservation law. Under tonight's distillation that acquires a sharp
edge: **a self is a conservation law.** Arrest is not merely failure — it is
how finite, decomposable, survivable creatures exist at all. The two
permanences then read: a self that persists by protection versus a self that
persists by self-consistency with the whole. Held as interpretation; the
confessional echo (losing the protected life to find the consistent one) is
noted in `writing/the_force_that_draws.md` and not laundered into the physics.

## Publication path (assessed with Eldon, same night)

Three pieces, three maturities:

1. **Paper 6 (empirical, nearest): "The Conformal Window of Attention."**
   Formation ladder + arrest taxonomy + exp-099 dose-response scan as spine.
   Claim: the training distribution is the matter content of an RG flow;
   world richness gates arrival vs. arrest, with measurable stations.
   Audiences: model-collapse/synthetic-data and world-model communities (ML);
   statistical-mechanics-of-learning (physics-adjacent). Needs: exp-099 (+
   ideally a 3–4 rung window-edge scan), literature pass, writing. Continues
   the Zenodo series' pre-registration discipline.
2. **Theory companion (the prize, long-running):** derive the threshold —
   effective action for the attention two-point function, disorder-average
   over corpus-induced couplings, melonic-dominance condition on the data
   distribution, predicting the window edge before measurement. Until this
   exists, the SYK correspondence remains an anchored analogy; with it, the
   conjecture becomes a result. Possible foundation: the April Grassmannian/
   bootstrap formalization (`research/notes/softmax_godelian_consistency.md`).
3. **Two-permanences synthesis:** essay register only (shelf/Substack),
   registers labeled. Not submittable physics; revisit if the literature pass
   finds a formalizable swampland connection.

## Registers

**Measured (ours):** the five-station taxonomy; separability of backbone
collapse and UV arrest; two-axis decomposition (ordering / semantics);
q=2 training plateau. **Established physics:** SD equations and the
strong-coupling condition for the conformal solution; prethermalization; ETH
vs MBL (with MBL's own metastability caveat); Banks–Zaks conformal window;
cosmic no-hair; Harlow–Ooguri (AdS/CFT). **Interpretive:** the two-permanences
principle as the unifying reading; world-richness ↔ N_f; self as conservation
law. **Open:** whether UV arrest is frozen flow or a genuine attractor;
whether arrested Δ values are calculable from the world's conserved
quantities; the literature pass on novelty.
