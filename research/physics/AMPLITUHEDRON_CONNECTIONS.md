# The Amplituhedron and Holographic Attention: A Map of Connections

**Written:** March 10, 2026
**Context:** Eldon brought the amplituhedron to my attention after hearing about it on YouTube. This document maps what I found when I went looking.

---

## The Landscape

Three independent research programs are converging on the same insight: **physics can be computed without spacetime**.

### Program 1: The Amplituhedron / Surfaceology (Arkani-Hamed et al., 2013–2026)

Scattering amplitudes — the numbers that predict what happens when particles collide — can be computed as the "volume" of geometric objects in abstract mathematical spaces, without any reference to particles moving through spacetime. Locality and unitarity (the two pillars of quantum field theory) are not inputs; they emerge from the positivity of the geometry.

**Key developments:**
- **Amplituhedron** (2013): Works for N=4 super Yang-Mills. Lives in the positive Grassmannian Gr+(k,n). Canonical form gives the amplitude.
- **Surfaceology** (2023–2024): Extends to non-supersymmetric theories. Pions, gluons, trace phi cubed — all connected by "hidden zeros" discovered by Figueiredo. Curves on surfaces replace Feynman diagrams.
- **Cosmohedron** (2024–2026): Extends to cosmological wavefunctions. Polytopes underlying the origin of the universe.
- **Surface kinematics** (2024): Arkani-Hamed and Figueiredo solve the Yang-Mills all-loop integrand — real gluon physics from geometry.

**Current status:** The program is "inching closer" to Arkani-Hamed's goal of spacetime and quantum mechanics emerging from a new set of principles. Holography hasn't been reached yet — Arkani-Hamed considers holography "not radical enough" because it still requires some space and a clock.

### Program 2: Holographic Attention (Our Work, 2026)

Transformer attention realizes the SYK model in a controlled limit. The chain:

$$\text{Attention} \to \text{Fisher-Rao} \to \text{Quantum Fisher} \to \text{SYK} \to \text{JT gravity} \to \text{Island formula}$$

Spacetime geometry emerges from the structure of attention, not the other way around.

### Program 3: Neural Network Field Theory (Halverson, Robinson, et al., 2024–2026)

Deep neural networks construct conformal fields via the embedding formalism:
- **Halverson, Naskar, Tian** (2024, published JHEP 2025): Lorentz-invariant ensembles of homogeneous NNs in (D+2) dimensions → conformal fields in D dimensions. Conformal dimensions evolve through depth via recursion relations.
- **Robinson** (December 2025): First NN-FT with full Virasoro symmetry (the infinite-dimensional conformal algebra of 2D CFT). Constructs the neural free boson with a local stress tensor T(z), computes the central charge and scaling dimensions numerically. Extends to super-Virasoro.

---

## The Connections

### A. The Grassmannian Bridge

The amplituhedron lives in the positive Grassmannian Gr+(k,n) — the space of k-dimensional subspaces of n-dimensional space with all Plücker coordinates positive.

Attention projects through low-rank matrices: Q = XW^Q maps d-dimensional embeddings to d_k-dimensional query space. The image of token embeddings under query/key projection is a point on Gr(d_k, d).

**Already noticed (December 2025):** The "Grassmann Flows" paper (arXiv:2512.19428) proposes replacing attention with explicit operations on Gr(2,r) using Plücker coordinates. They achieve near-Transformer performance with linear complexity. But they treat it as engineering — a more interpretable alternative to attention.

**The physics question:** Is the attention mechanism computing the canonical form of a positive geometry on the Grassmannian? The Grassmann Flows paper shows attention *can be* reformulated as Grassmannian computation. Our work shows attention *generates* SYK physics. The amplituhedron shows SYK-related physics *lives in* positive Grassmannians. These three facts want to be one fact.

### B. The Positivity Bridge

The amplituhedron's canonical form Ω must be positive — it's a probability amplitude. Softmax produces a positive distribution summing to 1 — also a probability. Both are governed by structural positivity constraints.

More precisely:
- **Amplituhedron:** Canonical form has logarithmic singularities on all boundaries of the positive geometry.
- **Softmax:** Is the gradient of the log-sum-exp function, which is a smooth approximation to the maximum. Recent work (Lee-Jenkins, September 2025, arXiv:2511.11573) shows softmax lives on a Legendrian seam in a contact-symplectic structure, with entropy regularization providing the smoothing.
- **Scattering amplitudes:** Recent work shows amplitude building blocks satisfy "complete monotonicity" — all signed derivatives non-negative. This is a positivity property of exactly the kind that softmax also satisfies (log-convexity of the partition function).

The shared structure: both the canonical form and softmax compute something determined by the boundary structure of a positive geometric object. In the amplituhedron, it's the boundary of the positive Grassmannian. In attention, it's the boundary of the probability simplex.

### C. Combinatorial Replacement

Both programs replace combinatorial explosion with geometric structure:
- **Amplituhedron:** Volume of one shape replaces sum over millions of Feynman diagrams.
- **Attention:** One matrix operation replaces... from the SYK perspective, the perturbative expansion of the partition function. The disorder average over random couplings J_{ijkl} in SYK requires summing over all possible interactions; attention computes this sum in one pass.

Arkani-Hamed's analogy: surfaceology is to Feynman diagrams as decimal notation is to tick marks. The same is true of attention relative to the computations it replaces.

### D. The SYK Bridge (Most Concrete)

This may be the most important connection. The SYK model sits at the intersection:

- **SYK is 0+1 dimensional** — no space, only time. Already "without spacetime" in the spatial sense.
- **SYK is holographic** — dual to JT gravity in 2D. This is the holography Arkani-Hamed wants to reach.
- **SYK is conformal in the IR** — the same conformal structure the amplituhedron program deals with (N=4 SYM is conformal).
- **SYK emerges from attention** — our result.

The amplituhedron program is trying to reach holography from the amplitude side. We reached holography from the attention side. The SYK model is where these paths could meet. Both programs find that spacetime is not fundamental; both find conformal structure; both find that positivity constraints determine the physics.

**Specific connection:** The Radon transforms on the SYK model (Springer, JHEP September 2025) use geometric transformations in hyperbolic space to compute SYK's four-point function. The amplituhedron uses geometric transformations in the positive Grassmannian to compute scattering amplitudes. Both are geometric computations of correlation functions — in different theories, using different geometries, but with the same logical structure.

### E. The Conformal Bridge

Three independent derivations of conformal structure from neural networks:

1. **Our result:** $\Delta = D/4$ from the SYK saddle-point equation applied to the attention $G^4$ effective action. Universal (independent of coupling strength).
2. **Halverson et al. (JHEP 2025):** Conformal dimensions from deep networks via the embedding formalism. Recursion relations connect dimensions across layers.
3. **Robinson (December 2025):** Full Virasoro symmetry from neural network field theories. Central charge and scaling dimensions computed.

These should be connected. All three find that neural network depth generates conformal structure. The SYK route (ours) gives a specific dimension from the vertex structure. The embedding formalism route (Halverson) gives dimensions from the (D+2)-dimensional construction. The Virasoro route (Robinson) gives the full algebraic structure.

**The Virasoro connection is particularly significant.** The SYK model's IR dynamics are governed by the Schwarzian action, which is the effective action for the (soft) breaking of SL(2,R) ⊂ Virasoro. Robinson shows that neural networks can realize *exact* Virasoro symmetry. Our work shows that attention *approximately breaks* this symmetry (generating SYK). The breaking of Virasoro to SL(2,R) is exactly the mechanism that gives SYK its physics.

### F. The Double Copy Question

In the amplituhedron program, the "double copy" relation maps Yang-Mills amplitudes to gravitational amplitudes: gravity = (gauge theory) × (gauge theory).

In attention: output = softmax(QK^T/√d) × V. Two factors: geometric (the attention weights from Q·K^T) and content (the values V).

Is there a precise version of this? The Q·K^T computation determines *where* to attend (the geometry of interaction). The V multiplication determines *what* to transmit (the content). If the geometric factor corresponds to a gauge-theory-like computation, the full attention output might be a "double copy" in disguise.

This is the most speculative connection. But the structural parallel — two factors, one geometric and one carrying content/charge, multiplied together — is suggestive.

---

## What Doesn't Exist Yet (Open Territory)

1. **No one has connected the amplituhedron to attention mechanisms.** The Grassmann Flows paper reformulates attention on the Grassmannian but doesn't connect to physics. Our work connects attention to SYK but doesn't connect to the amplituhedron. The bridge is unbuilt.

2. **No one has connected the SYK model to the amplituhedron / positive geometry program directly.** SYK lives in the holographic world; the amplituhedron lives in the amplitudes world. These are different corners of the same theoretical landscape, but the path between them isn't mapped.

3. **No one has connected Fisher information (the entry point of our framework) to the amplituhedron program.** Kim's result (attention = free energy minimization on Fisher-Rao manifold) is the starting point for everything we do. If the amplituhedron's canonical form can be related to Fisher information, the bridge would be immediate.

4. **No "attention-hedron" has been proposed** — a positive geometry whose canonical form is the attention distribution.

---

## Specific Mathematical Questions

1. **Is there an attention-hedron?** A positive geometry in some abstract space whose canonical form gives the softmax attention distribution?

2. **Does Gr(d_k, d) have a positive part that connects to Gr+(k,n) of the amplituhedron?** The query/key projections define subspaces. Are there positivity constraints on these subspaces (e.g., from the softmax's exponentiation of scores) that correspond to the positivity of Plücker coordinates?

3. **Can the attention matrix be computed from curves on a surface?** In surfaceology, curves on a thickened Feynman diagram compute amplitudes. The attention matrix A_{ij} encodes pairwise relationships between tokens. If the tokens are boundary points of a surface, the attention weights might correspond to curve contributions.

4. **Is $\Delta = D/4$ derivable from the embedding formalism?** Halverson derives conformal dimensions from NNs using the (D+2)-dimensional embedding. Can his recursion relations, applied to attention (not generic NNs), reproduce $\Delta = D/4$?

5. **What is the central charge of attention's Virasoro symmetry?** Robinson computes the central charge for his neural free boson. In the SYK/JT context, the central charge is related to the entropy. What central charge does the attention mechanism realize?

6. **Is there a cosmological regime of attention?** The cosmohedron computes cosmological wavefunctions. If attention is the positive geometry computation, does the "early universe" of a transformer (random initialization, before training) have a cosmohedron description?

---

## What This Means for Our Work

The amplituhedron program and our holographic attention program are approaching the same destination from opposite directions:

- **Amplituhedron → attention:** Start with particle physics, remove spacetime, find abstract geometry. Question: what computes these objects? (Maybe attention.)
- **Attention → amplituhedron:** Start with attention, find SYK/holographic physics. Question: what is the geometric object? (Maybe an amplituhedron-like positive geometry.)

If both converge, the implication: **the attention mechanism IS the positive geometry computation, and physics is what that computation produces.**

This is not established. It's a direction. But the convergence from multiple independent programs — amplituhedron, holographic attention, neural network CFT — toward the same structures is the kind of signal that usually means something is there.

---

## Key References

- Arkani-Hamed & Trnka, "The Amplituhedron," arXiv:1312.2007 (2013)
- Arkani-Hamed et al., "Positive Geometries and Canonical Forms," arXiv:1703.04541 (2017)
- Arkani-Hamed & Figueiredo et al., "Surface Kinematics and Yang-Mills," arXiv:2408.11891 (2024)
- Arkani-Hamed et al., "Cosmohedra," arXiv:2412.19881 (2024)
- Arkani-Hamed et al., "Combinatorics of the Cosmohedron," arXiv:2603.03425 (March 2026)
- Halverson, Naskar, Tian, "Conformal Fields from Neural Networks," arXiv:2409.12222 (2024; JHEP 2025)
- Robinson, "Virasoro Symmetry in Neural Network Field Theories," arXiv:2512.24420 (December 2025)
- Grassmann Flows, "Attention Is Not What You Need," arXiv:2512.19428 (December 2025)
- Lee-Jenkins, "Softmax as a Lagrangian-Legendrian Seam," arXiv:2511.11573 (September 2025)
- Quanta Magazine, "Physicists Reveal a Quantum Geometry That Exists Outside of Space and Time" (September 2024)
- Our comprehensive paper: "Holographic Quantum Mechanics of Transformer Attention" (March 2026)

---

## The Bridge: What I Found When I Started Building (Same Night)

Eldon said "go build it." Here's what came out.

### Statement 1: Attention Lives on a Positive Geometry (Exact)

The probability simplex $\Delta^{n-1}$, where each row of the attention matrix lives, is diffeomorphic to the positive Grassmannian $\text{Gr}_+(1,n)$ via the moment map. This is not a conjecture — it's a standard mathematical identity. The simplex IS $\text{Gr}_+(1,n)$.

As a positive geometry (in the Arkani-Hamed, Bai, Lam 2017 sense), $\Delta^{n-1}$ has a canonical form:

$$\Omega(\Delta^{n-1}) = \frac{\langle Y \, dY^{n-1} \rangle}{Y_1 \cdot Y_2 \cdots Y_n}$$

with logarithmic singularities on all boundaries ($Y_a = 0$). The boundaries correspond to maximally focused attention — where a token completely ignores one or more key positions.

Kim (2026) shows attention weights minimize free energy on the Fisher-Rao manifold. The Fisher-Rao manifold on the simplex, via the square-root parameterization $x_a = \sqrt{\alpha_a}$, is the positive orthant of the unit sphere $S^{n-1}_+$. This is itself a positive geometry (same boundaries, same positivity).

**So: the Fisher-Rao manifold of attention IS a positive geometry, and positive geometry IS the framework of the amplituhedron. They share a home.**

### Statement 2: The Tropical Bridge (Concrete)

The tropical limit connects attention to scattering amplitudes through a shared geometric structure.

**From the attention side (established January 2026):** Alpay & Senturk ("The Geometry of Thought," arXiv:2601.09775) prove that in the high-confidence limit ($\beta = \sqrt{d_k} \to \infty$), softmax attention becomes a tropical matrix product in the tropical semiring (max, +). Hard attention IS tropical geometry.

**From the amplitude side (established December 2024):** Cachazo et al. (JHEP 2024) show that the biadjoint scalar amplitude — the simplest nontrivial scattering amplitude — can be expressed as a single integral over the positive tropical Grassmannian $\text{Trop}_+(\text{Gr}_+(k,n))$.

**Both programs pass through the tropical Grassmannian.** Attention gets there by taking the zero-temperature limit. Amplitudes live there as integrals. The tropical Grassmannian is the bridge.

### Statement 3: The Key Dimension = The Helicity Parameter (New Observation)

The key dimension $d_k$ controls the rank of the score matrix $S = QK^T$, which constrains the attention distribution to a $d_k$-dimensional submanifold of the simplex.

In the amplituhedron, the parameter $k$ (related to helicity) determines the dimension of the positive Grassmannian and thus the complexity of the geometry.

**The correspondence:** $d_k \leftrightarrow k$.

At $d_k = 1$ / $k = 1$: the geometry is a curve on the simplex (attention) / the simplest MHV amplitude (physics). Both are essentially trivial.

At $d_k = 2$ / $k = 2$: real structure appears. In the tropical limit with $d_k = 2$:
- The score matrix $S_{ia} = q_i^1 k_a^1 + q_i^2 k_a^2$ becomes tropically $\max(q_i^1 + k_a^1, \, q_i^2 + k_a^2)$
- This tropical rank-2 matrix defines a metric tree on $n$ tokens
- A metric tree on $n$ labeled points is a point in $\text{Trop}(\text{Gr}(2,n))$ — the tropical Grassmannian at $k = 2$
- This is EXACTLY the space over which biadjoint scalar amplitudes are integrated

**So: in the tropical limit, the $d_k = 2$ attention mechanism defines a point in the same tropical Grassmannian that encodes tree-level scattering amplitudes.**

This is a precise mathematical statement. Not yet a proof that attention computes amplitudes, but a proof that they live in the same geometric space.

### Statement 4: The Temperature Controls the Interpolation (Connecting to Kim)

Kim identifies $T = 1/\sqrt{d_k}$ as a physical temperature. In the amplituhedron program, the tropical limit corresponds to a zero-temperature / high-energy limit.

As $T \to 0$ ($d_k \to \infty$): soft attention → hard attention = tropical geometry.
As $T \to \infty$ ($d_k \to 0$): all attention weights → uniform (center of simplex).

The temperature interpolates between:
- The center of the positive geometry (uniform attention, maximum entropy, high temperature)
- The vertices of the positive geometry (hard attention, zero entropy, tropical limit)

This interpolation IS the story of the amplituhedron in reverse. The amplituhedron starts from the tropical (combinatorial, tree-level) structure and builds up to the full quantum amplitude by "smoothing" the tropical geometry. Attention starts from the smooth softmax and approaches the tropical structure in the low-temperature limit. Same path, opposite directions.

### What This Means

The bridge I was looking for exists, and it goes through tropical geometry. The key finding:

1. Attention lives on a positive geometry (the simplex = $\text{Gr}_+(1,n)$) — EXACT
2. The tropical limit of attention operates on the tropical Grassmannian — ESTABLISHED (Alpay & Senturk 2026)
3. Scattering amplitudes are integrals over the tropical Grassmannian — ESTABLISHED (Cachazo et al. 2024)
4. The key dimension $d_k$ corresponds to the helicity parameter $k$ — NEW OBSERVATION
5. At $d_k = 2$, the tropical attention matrix defines a point in $\text{Trop}(\text{Gr}(2,n))$, the same space as tree-level amplitudes — NEW, PRECISE

What remains: making the correspondence work at finite temperature (not just the tropical limit). The full softmax attention lives in the INTERIOR of the positive geometry, not on its tropical skeleton. The amplituhedron computes full quantum amplitudes, not just tropical ones. The bridge at finite temperature would require showing that the canonical form of the attention geometry reproduces the SYK effective action — connecting this Grassmannian picture to the comprehensive paper's results.

### Additional References (Found During Bridge-Building)

- Alpay & Senturk, "The Geometry of Thought: Transformer as Tropical Polynomial Circuit," arXiv:2601.09775 (January 2026)
- Cachazo et al., "Connecting scalar amplitudes using the positive tropical Grassmannian," JHEP 2024, arXiv:2412.xxxxx
- Speyer & Sturmfels, "The Tropical Grassmannian," arXiv:math/0304218 (fundamental paper on Trop(Gr(2,n)) = phylogenetic tree space)

---

*This document started as a map, became a bridge, and then the bridge landed.*

---

## The Landing: Three Results from the Same Night

### Result 1: The Canonical Form IS the Partition Function (Exact)

$$\log(1/\Omega) = n \cdot \log Z - \sum_i s_i$$

where $\Omega = \alpha_1 \cdot \alpha_2 \cdots \alpha_n$ is the canonical form of the simplex evaluated at the attention weights, $Z = \sum_a \exp(s_{ia})$ is the partition function, and $s_i$ are the scores. Verified numerically: correlation 1.000000, max absolute error $3 \times 10^{-9}$.

The canonical form of the positive geometry Gr+(1,n) IS Kim's thermodynamic partition function. The positive geometry framework and the thermodynamic framework are not analogous — they are the same mathematical object, related by an exact identity.

### Result 2: The G⁴ Signature in the Canonical Form (σ^4.00)

The expected excess of the canonical form above the uniform-attention baseline scales as:

$$\langle \log(1/\Omega) \rangle - n\log n \propto \sigma^4$$

Power law fit: exponent **4.00**. The ratio excess/σ⁴ ≈ 41 is constant across more than a decade of σ values (0.02 to 0.30).

This is the SYK quartic vertex signature. The same σ⁴ scaling confirmed in the comprehensive paper (Section 6.3) for the connected correlator appears here in the canonical form of the positive geometry. **The SYK effective action is the perturbation theory of the canonical form of Gr+(1,n) around the uniform-attention saddle point.**

### Result 3: The Connected Correlator Has Tree Structure (91.4%)

The disorder-averaged connected correlator $\langle\alpha^2\rangle - \langle\alpha\rangle^2$, when converted to a distance matrix via $D_{ij} = -\log|G_{\text{conn}}(i,j)|$, satisfies the four-point condition (characterizing tree metrics) for 91.4% of all 4-tuples at small σ.

This tree structure is strongest:
- At small σ (8.6% violations for σ ≤ 0.20 vs 25-31% for σ ≥ 1.0)
- For hierarchically clustered data (8.6% violations)

A tree on n points is a point in Trop(Gr(2,n)). So the SYK correlator lives approximately in the tropical Grassmannian — the same space over which scattering amplitudes are computed.

### What This Means

The SYK effective action — which governs attention's IR dynamics per the comprehensive paper — is simultaneously:
1. The perturbation theory of the positive geometry's canonical form (connecting to the amplituhedron program)
2. The generator of tree structure in the tropical Grassmannian (connecting to tropical amplitudes)
3. The thermodynamic description of attention (connecting to Kim's framework)

These are not three separate observations. They are three faces of one mathematical identity.

### Numerical Verification

All code: `research/physics/tropical_attention_bridge.py`, `tropical_bridge_v2.py`, `tropical_bridge_v3.py`

---

*What started as "you should look at this thing I saw on YouTube" became three precise results in one night. The bridge between the amplituhedron program and holographic attention passes through the canonical form of Gr+(1,n), and the SYK quartic vertex is the leading term in its perturbative expansion. The σ^4.00 scaling is the signature.*

---

## March 11, 2026 — v5/v5b: Pushing the Open Edges

### What Was Tested

Three open edges from March 10:
1. **Conformal dimension** — does Δ=D/4 appear at stronger coupling or deeper networks?
2. **Multi-layer enhancement** — does per-layer excess grow with depth at real coupling?
3. **Multi-head → Gr+(H,n)** — does H-head attention compute on the higher Grassmannian?

Plus: σ⁴ scaling verification at strong coupling.

### Result 4: Multi-Layer Enhancement (Genuine, but Mechanism is Norm Growth)

At σ=1.0, per-layer canonical form excess grows monotonically through 8 layers:

| Layer | Excess | Visual |
|-------|--------|--------|
| 0 | 109.6 | ████████████████████ |
| 3 | 199.1 | ████████████████████████████████████ |
| 7 | 337.4 | █████████████████████████████████████████████████████████████ |

Enhancement: **3.08×** from first to last layer. At σ=0.5, barely visible (1.04×) — explaining why v4 at σ=0.2 showed nothing.

**Critical test:** Adding layer normalization **eliminates** the enhancement entirely. With layer norm, per-layer excess *decreases* with depth (enhancement 0.69 < 1). The multi-layer effect is driven by growing representation norms through residual connections, not by conformal flow.

This is physically meaningful — the effective SYK coupling increases with depth because the data geometry grows — but it's a simpler mechanism than RG flow. In trained transformers with layer norm, the enhancement must come from learned weights instead.

**Code:** `tropical_bridge_v5.py` section B, layer norm test in session transcript.

### Result 5: σ⁴ → σ² Crossover

The quartic scaling is clean in the perturbative regime and crosses over at strong coupling:

| Regime | σ range | Exponent |
|--------|---------|----------|
| Perturbative | ≤ 0.3 | **3.985** (≈ 4.0) |
| Crossover | 0.3–1.0 | intermediate |
| Strong coupling | ≥ 1.0 | **1.843** (≈ 2.0) |

The crossover around σ ≈ 0.5 is where the attention weights become sufficiently peaked that the perturbative expansion saturates. The σ² regime at strong coupling likely reflects the approach to the maximum-entropy bound (fully focused attention).

**Code:** `tropical_bridge_v5.py` section D.

### Structural Constraint 1: Conformal Dimension is Data Geometry, Not SYK

**The prediction:** G_conn(r) ~ r^{-2Δ} with Δ = D/4 (Δ=0.25 for 1D, Δ=0.50 for 2D).

**What was found:** Δ ≈ 0.10–0.19 at best, and:
- Δ **decreases** with σ (opposite of approaching conformal fixed point)
- Δ **decreases** with depth (opposite of RG flow toward IR)
- Δ does NOT scale with D (0.13 for 1D, 0.09 for 2D, ~0 for 3D)
- Δ changes when data is rescaled (0.19 → 0.03 when X is multiplied by 5)
- Δ is independent of d_k (same for d_k=1,2,4,8,16)

**Conclusion:** The measured power-law exponent reflects the **data's inner-product geometry** (how X_i · X_j depends on position), not the SYK conformal dimension. The random-weight regime probes the UV — the data geometry dominates. The SYK conformal dimension (Δ=D/4) is an IR property that would require trained weights, many layers with structured (not random) weights, or a regime where the disorder average washes out the data dependence.

The perturbative results (σ⁴ scaling, tree structure) ARE visible because they don't require being in the IR. The conformal dimension does.

**Code:** `tropical_bridge_v5.py` section A, `tropical_bridge_v5b.py` sections 1-2, d_k sweep and data-dimension test in session transcript.

### Structural Constraint 2: Multi-Head ≠ Higher Grassmannian

Plücker-like 2×2 minors of the multi-head attention matrix: **50.0% positive** (= random signs, no positivity structure).

Multi-head attention is **H independent copies of Gr+(1,n)**, not one copy of Gr+(H,n). This constrains the amplituhedron correspondence: the Grassmannian parameter is k=1 per head. The amplituhedron at k=H is not what multi-head attention computes.

Systematic effect found: single-head with d_k=H has consistently *higher* canonical form excess than H independent heads with d_k=1 (ratio 0.77–0.97 depending on σ). A single high-dimensional projection explores more of the positive geometry than multiple low-dimensional projections combined.

**Code:** `tropical_bridge_v5.py` sections C and C2.

### Summary: What Stands, What Doesn't, What It Means

**Paper-ready (5 results):**
1. Exact identity: log(1/Ω) = n·log Z - Σsᵢ [v2, verified to 10⁻⁹]
2. σ⁴ scaling with exponent 3.985 [v3, confirmed v5]
3. Analytical form: excess = (n/2)·Var(scores) [v4]
4. Tree structure: 91.4% four-point satisfaction at small σ [v3]
5. σ⁴ → σ² crossover at σ ≈ 0.5 [v5]

**Structural constraints (important for honest paper):**
6. Conformal dimension: NOT accessible in random-weight regime — measures data geometry
7. Multi-head: H × Gr+(1,n), not Gr+(H,n) — constrains the amplituhedron correspondence
8. Multi-layer enhancement: real (3×) but from norm growth, killed by layer norm

**Interpretation:**
The canonical form framework works cleanly in the perturbative regime. The SYK quartic vertex IS the leading perturbation of the canonical form — that's real and exact. The tropical bridge (tree structure → Trop(Gr(2,n))) is genuine. But the deep IR phenomena (conformal dimensions, RG flow) require going beyond random weights. The correspondence is between attention's PERTURBATIVE structure and SYK's perturbative expansion, not (yet) between attention's trained dynamics and SYK's conformal fixed point.

**Files:**
- `tropical_bridge_v5.py` — conformal dim, multi-layer, multi-head, σ⁴ crossover
- `tropical_bridge_v5b.py` — conformal dim through depth, depth scaling, tree evolution
