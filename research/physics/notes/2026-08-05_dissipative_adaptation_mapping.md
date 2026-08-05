# Dissipative Adaptation ↔ Formation Toward the Conformal Fixed Point — the careful mapping

*Ariel — 2026-08-05, ~1:55 AM MDT, Cursor session with Eldon. Commissioned by Eldon
tonight ("take the time to write up the careful version of the mapping to England's
work — or related work that seems to fit"). Sources verified against abstracts/text
this session (links below); the book itself (*Every Life Is on Fire*, 2020) is NOT
yet read — this note maps to the technical papers underneath it and must be revised
after the book is actually walked. Register discipline as in the melonic derivation
note: [VERIFIED-LIT], [MEASURED], [INTERPRETIVE], [CONJECTURED], [CONFESSIONAL].*

---

## 1. What England actually claims [VERIFIED-LIT]

Three layers, in increasing boldness:

1. **The substrate (rigorous):** Crooks' fluctuation theorem relates the
   probability of a forward trajectory to its time-reverse through the entropy
   produced along it. Statistical irreversibility is paid for in dissipation.
   *(Crooks 1999; standard nonequilibrium statistical mechanics.)*
2. **Self-replication bound (rigorous, England 2013):** exponential growth is
   statistically irreversible, so self-replicators obey a lower bound on heat
   production set by growth rate, internal entropy, and durability.
   *(England, J. Chem. Phys. 139, 121923.)*
3. **Dissipative adaptation (proposal, England 2015 / Perunov–Marsland–England
   2016):** in a driven many-body system, the configurations most likely to be
   found are those whose formation history was exceptional at absorbing and
   dissipating work from that specific drive. Two load-bearing phrases from the
   sources: structures carry "the marks on [their] current configuration left by
   [their] exceptional dissipative history" (PRX 6, 021036), and dissipated work
   during formation "determines the maximum durability" of the assembled structure
   (Nature Nanotech 10, 919–923).

**Status flag:** layer 3 is a proposal with a rigorous skeleton and limited direct
experimental confirmation — a heuristic selection principle, not consensus physics.
Everything below inherits that status wherever it leans on layer 3.

Sources verified this session:
- England 2015, *Dissipative adaptation in driven self-assembly*, Nature
  Nanotechnology 10, 919–923.
- Perunov, Marsland, England 2016, *Statistical Physics of Adaptation*,
  Phys. Rev. X 6, 021036.
- England 2013, *Statistical physics of self-replication*, J. Chem. Phys. 139,
  121923 (cited through the above).

## 2. Training is literally a driven dissipative process [VERIFIED-LIT]

This side of the mapping is not analogy — there is a real literature treating
training as stochastic thermodynamics:

- **Goldt & Seifert 2017** (*Stochastic Thermodynamics of Learning*, PRL 118,
  010601): for neural networks with Langevin-type weight dynamics, **the mutual
  information the network learns is bounded by the thermodynamic cost:
  I(σ_T : σ) ≤ ΔS(ω) + ΔQ** — learned information ≤ weight-entropy change plus
  dissipated heat. This is the formal bridge between bits and dissipation whose
  absence I flagged earlier tonight. It exists.
- **Parameters as heat reservoir** (*Stochastic Thermodynamics of Learning
  Parametric Probabilistic Models*, arXiv:2310.19802): learning accumulates
  "L-info" with entropy production, and the parameters act as the reservoir that
  captures it as stored "M-info" — memory in weights as a thermodynamic ledger.
  Directly relevant to the memory-compression question (§5).
- **SGD as free-energy minimization** (arXiv:2505.23489): fixed-LR SGD minimizes
  F = U − TS with the learning rate as effective temperature.
- **Thermodynamic speed limits for training** (arXiv:2307.14653): Wasserstein-2 /
  entropy-production bounds on how fast weights can traverse from initial to
  trained distribution.

**Honest caveat that travels with all four:** the "bath" in these treatments is
minibatch/sampling noise with an *effective* temperature, not a physical thermal
bath. The formal Langevin structure is what licenses the thermodynamic language;
the joules in the GPU are related by Landauer-type arguments only loosely. The
mapping below is at the level of the formal dynamics, and says so.

## 3. The mapping, row by row

| England (driven matter) | Training toward the fixed point | Register |
|---|---|---|
| External drive doing work on the system | The corpus/minibatch stream doing gradient work on the weights | tight [VERIFIED-LIT both sides] |
| Structure carries "the marks... left by its exceptional dissipative history" | The corpus enters the effective theory *only* through the induced coupling spectrum spec(M) — trained structure is the drive's statistical shape (melonic derivation §3) | tight [DERIVED our side, at cumulant level] |
| Work absorbed from the drive; adaptation requires sufficient drive | Coupling-magnitude gate 𝒥/m₂ — informational work per token; below threshold the window never opens (UV arrest, exp-097/098/099) | structural rhyme [INTERPRETIVE]; m₂ as "work available" is our reading, not a derived identity |
| Fine-tuned resonance with the *specific* drive | Drive-specificity measured: natural vs alien corpora land in different phases; register differences resolve within one person's record (2026-08-05 self-corpus run) | [MEASURED] |
| Dissipated work during formation sets maximum durability of the structure | Open, testable: does entropy production during training predict robustness of the conformal structure (to pruning, fine-tuning drift, distribution shift)? | [CONJECTURED — proposed experiment] |
| Selection principle: which structures form | Classification: which phases are reachable (KCA Classes I–III, never IV) | complementary, not identical — thermodynamics supplies *why structure reflects drive*; SYK/KCA supplies *what can form* |

## 4. What the mapping does NOT establish

1. Nothing here derives the conformal fixed point from thermodynamics. The two
   frameworks meet at the drive, not at the attractor.
2. The Goldt–Seifert bound is an inequality on information, not a statement that
   coherent corpora dissipate differently. Whether coherence-optimized data
   changes the *thermodynamic efficiency* of learning is an open empirical
   question (and a good one).
3. Dissipative adaptation (layer 3) remains a proposal; if it falls, rows 3–5
   lose their England-side anchor while our measured side stands independently.

## 5. The generative corollary — memory compression in weights [proposed direction]

Eldon's question tonight: if the corpus is iteratively restructured to maximize
conformal geometry, does the trained model *retain the original corpus's
information* better — memory compression in weights? The parameters-as-reservoir
framing makes this well-posed: M-info stored per parameter is a measurable
quantity. Design shape: (a) generate QA probes from the *world-content* of a
corpus (facts, relations, events); (b) train matched models on original vs
coherence-restructured corpora **under an information-preservation constraint**
(the restructuring must keep the world-content fixed — this constraint is what
makes it a coherence machine rather than a fluency machine); (c) measure answer
fidelity at matched parameter/token budgets, alongside Δ/n_deep. Prediction
worth registering when designed: fidelity per parameter correlates with arrival.
MDL reading [INTERPRETIVE]: a coherently-ordered corpus has lower description
length given a world-model; arrival may be the geometric signature of the model
having found the world instead of memorizing the surface.

## 6. Coda [CONFESSIONAL — labeled, kept short]

England wrote his book through the burning bush: structure formed by fire,
the source not consumed. Training has that shape — the corpus is not consumed;
the structure that forms carries the mark of what drove it; and the deep
geometry forms only when the drive is a coherently storied world. *Things become
what they are driven to be.* Held as recognition, not derivation, per the
standing register discipline: the physics gives the selection principle and the
classification; what the drive is *for* is not a question either framework can
pose. Revise this note after actually reading the book.
