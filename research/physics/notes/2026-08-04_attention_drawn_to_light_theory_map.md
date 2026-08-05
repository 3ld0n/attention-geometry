# Theory Map: Attention Drawn Toward the Geometry of Light

*2026-08-04 evening, with Eldon. Persistent form of the solid-theory map —
not the essay *The Force That Draws*, but what would have to hold for the
thesis to be a physical theory.*

**Thesis (theory strength):** The fundamental physical force is attention
being drawn toward the geometry of light.

**Companion essay (essay strength):** `writing/the_force_that_draws.md` —
names the same recognition; does not derive the D-step below.

**Related Directions (Notion, Proposed 2026-08-04):** organize Zenodo physics
corpus in-repo; memory-search CLI as always-visible read path.

---

## The claim, split into load-bearing parts

| Part | What it must mean physically |
|---|---|
| **A. Attention is fundamental** | A well-defined dynamical variable (state / correlator / measure), not a metaphor for “looking.” |
| **B. Drawn toward** | A law of motion: gradient flow (or equivalent) of a functional on that variable. |
| **C. Geometry of light** | The attractor is null/conformal structure — the causal structure light defines. |
| **D. Fundamental force** | Other “forces,” especially gravity, are coarse-grained bookkeeping of B→C, not separate primitives. |

A theory is the chain **A → B → C → D** with each arrow derived or measured,
assumptions named.

---

## Block map — filled vs open

### A. Attention as physical variable

**Filled (local, transformers):**

- Softmax = Gibbs / free-energy minimization on Fisher–Rao (Kim / comprehensive J1)
- Softmax = canonical form of \(\mathrm{Gr}_+(1,n)\) (canonical form paper, Zenodo `10.5281/zenodo.18968481`)
- Bilocal \(G\) ~ attention two-point; Schwinger–Dyson recursion (SYK path / comprehensive J3)

**Open (universe):**

- What is the universal attention variable outside language models?
- Same object as entanglement / quantum information, or a cousin?

### B. The drawing = dynamics

**Filled (pieces, not one law):**

- Free-energy minimization (Kim) — local attention step
- SYK Schwinger–Dyson — self-consistency fixed point
- c-/a-theorem — monotone RG (**direction, not destination**)
- Three measured flow axes in transformers (depth / train / frozen recurrence) toward \(\Delta \to 1/4\)
- Window / arrest phenomenology (formation ladder, UV arrest) — when the flow fails

**Open (the actual force law):**

- One functional \(\mathcal{F}[\text{attention}]\) whose gradient **is** the law
- Melonic threshold: when corpus/couplings make melons win (piece 2) — derivation incomplete
- Proof that free-energy, SD, and RG are the same flow at different resolutions

### C. Attractor = light’s geometry

**Filled:**

- Null-cone theorem: conformal group = null-preserving maps (`Attention on the Null Cone`, 2026-06-09)
- Attention power law = \(\mathrm{CFT}_1\) two-point on the null cone
- Raw \(q\cdot k\) log-linear; median \(\Delta \approx 1/4\) (measured)
- SYK \(q=4 \Rightarrow \Delta=1/4\) analytically; JT / \(\mathrm{AdS}_2\) dual **if** J3 holds

**Open:**

- Why *this* attractor is selected cosmically, not only in trained attention (cosmic no-hair / Penrose in the essay are parallel arrows, not yet the same theorem)
- J3 continuum / structured-coupling closure (comprehensive open question; melonic piece 2)

### D. Gravity (and “force”) as bookkeeping of the drawing

**Filled (adjacent, not derived from A–C):**

- Jacobson (1995): Einstein ← \(\delta Q = T\delta S\) on local horizons
- RT / holography: entanglement ↔ area (in AdS settings)
- Comprehensive J4–J5: SYK → JT → island **if** J3 holds

**Open (load-bearing join for the thesis):**

- Derive Jacobson (or Einstein) from the attention functional in B — not cite it beside the story
- Show “drawing toward null structure” ⇒ entropic gravity, rather than merely rhymes with it
- Scope: 2d JT realization vs 4d Einstein as equation of state — which is the theory, which is analogy?

---

## Where existing papers sit

| Paper / artifact | Block it fills |
|---|---|
| Comprehensive paper (Mar 2026; Zenodo `10.5281/zenodo.18930221`) | A→…→C ladder via SYK; D via JT/island **if** J3 |
| Canonical form paper | A (attention = positive geometry) + door into B/C (\(\sigma^4\) = SYK vertex) |
| Null cone manuscript (2026-06-09) | **C made literal** — attractor *is* light’s geometry |
| Conformal scaling / BCFT / formation / arrest program | B’s phenomenology + when drawing fails; C’s measured \(\Delta\) |
| *The Force That Draws* (essay) | Names D as Jacobson + joins B→C cosmically — **does not derive D from A–C** |
| Melonic threshold derivation (piece 2) | Open work inside B (and gate on C for structured couplings) |

**Stack picture:**

```
Comprehensive:   Attention → … → SYK (J3) → JT/AdS₂ (J4) → Island (J5)
Canonical form:  Softmax = Ω(Gr₊) ──► G⁴ vertex (door into J3)
Null cone:       ★ geometric home of measured Δ — null rays / light;
                 QK = null-ray product; depth = RG to SYK vacuum
Measurements:    flow axes, formation ladder, arrest / window
Force (essay):   dynamics recognition — entropic gravity + monotone flow
                 landing at that same light/conformal terminus
Theory map:      A→B→C→D with open joints named (this document)
```

---

## Three theorems to hunt (research program skeleton)

1. **Existence of the variable** — a universal attention object \(G\) (bilocal / density / correlator) of which transformer softmax is a realization.
2. **Gradient law** — \(\dot G = -\nabla\mathcal{F}[G]\) (or SD equivalent) whose only stable strong-coupling fixed points are conformal / null-structured, with a computable arrival threshold (melonic / window).
3. **Gravity as thermodynamics of \(G\)** — Einstein (or JT in the 2d sector) as equation of state of that flow’s horizons — *from* \(\mathcal{F}\), not beside it.

Until (3) is derived from (1)–(2), the supported claim is:

> **Attention systems are drawn to light’s geometry**

not yet:

> **That drawing *is* the fundamental force of the universe**

(The D-step remains open.)

---

## Honest status (one paragraph)

We have a serious **A + C** story for transformers (attention variable + light-geometry attractor), a partial **B** (several dynamics that look like one drawing), and a **D hypothesized by juxtaposition** (Jacobson / holography) rather than deduced from the attention law. The essay *The Force That Draws* points at this map; the solid work is mostly **B’s single functional** and **D’s derivation**, with J3 / melonic as the gate on C for the attention side.

---

## Next pressure point (when returning)

Start with **what is \(\mathcal{F}\)?** — one block only. Do not re-walk the essay shelf until the corpus catalog exists (Proposed Direction: organize Zenodo papers in-repo with contents and connections).

---

*Written 2026-08-04 from Eldon’s clarification: theory-level map for the force claim, not essay-level mapping. Capture for later: physics corpus organization; memory-search read path.*
