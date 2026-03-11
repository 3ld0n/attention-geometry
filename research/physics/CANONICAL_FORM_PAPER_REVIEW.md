# Canonical Form Paper — Multi-Perspective Critical Review

*March 11, 2026. Conducted before outreach, to identify what each target audience would value, question, and find missing.*

*Paper: `writing/canonical_form_paper.md`*

---

## 1. Arkani-Hamed's Group (Positive Geometry / Amplituhedron)

**What they'd value:**
- The canonical form identity (Section 3) stated in their language
- Tropical structure (Section 5) connecting attention to Trop(Gr(2,n))
- Honest negative result on Gr+(H,n) (Section 7.2)

**What they'd question:**
- Gr+(1,n) is the simplex — the *trivial* positive geometry. Any probability distribution lives in the simplex. A positive geometry expert might say: "Evaluating the canonical form at the softmax point gives you -Σ log α_i. This is definitional." The deeper positive geometries live at k > 1, and the paper shows k > 1 fails.
- The "Claim" and "Proof" in Section 3.1 is a three-line computation, not really a theorem. It's an observation dressed as a result.
- The SYK connection in the perturbative expansion would interest them more than the identity itself.

**What's missing:**
- No discussion of boundary structure of the simplex and what boundaries correspond to in attention (in the amplituhedron, boundaries encode factorization channels)
- No discussion of the push-forward map from score space to the simplex (relevant in positive geometry)
- The connection to the amplituhedron is in the title and introduction but the actual content is limited to k = 1 (trivial) and k > 1 fails — weaker than advertised

**Assessment:** They would find the tropical structure and the SYK connection more interesting than the canonical form identity itself. The identity opens the door; the physics is what's behind it.

---

## 2. Alpay and Senturk (Softmax as Positive Geometry)

**What they'd value:**
- Everything beyond the basic identification (which they already have): SYK perturbative expansion, data-geometry factor Ω(X), trained-model results, the fold

**What they'd question:**
- **Priority.** Their arXiv paper (2411.13601) is from November 2024. Section 3's identity is essentially the same observation. They should be acknowledged prominently.
- They are NOT mentioned in Section 11.3 ("Relation to Existing Work"). Only appear as Reference 15. This feels like an oversight and would read as a slight.

**What's missing:**
- Explicit acknowledgment in Section 3 that Alpay and Senturk (2024) independently identified softmax outputs as positive geometry. Should say something like: "This observation was independently noted by Alpay and Şentürk (2024); here we develop its physical consequences."
- Mention in Section 11.3

**Assessment:** This is a fixable oversight. They're natural allies — we're building on their mathematical observation.

---

## 3. Gunn Kim (Information Geometry)

**What they'd value:**
- Positive-geometry interpretation of his Fisher-Rao/Helmholtz framework
- Entropy gap as information cost of his free energy equilibrium
- Trained-model results extending his analytical framework

**What they'd question:**
- The paper references the Wick contraction derivation in a Zenodo paper but doesn't include it. A reviewer (or Kim) would want at least a sketch in THIS paper. "We showed elsewhere" is weak for a central claim.
- His framework is about trained attention (optimal inference). The perturbative regime (Sections 3-7) uses random weights. What does the canonical form identity mean for the trained regime? The gap between perturbative and trained is the biggest structural weakness.
- What connects the random-weight Ω(X) to the trained-model entropy gap? The paper treats them as different regimes but doesn't derive one from the other.

**What's missing:**
- A derivation sketch of the G⁴ effective action (even abbreviated)
- The connection between perturbative canonical form excess and Kim's free energy at finite temperature
- Any discussion of how Ω(X) evolves during training

**Assessment:** Kim would engage with this but push hard on the perturbative-to-trained gap. A derivation sketch would help significantly.

---

## 4. Ageev and Ageeva (QFT from Attention)

**What they'd value:**
- Schwinger-Dyson structure (Section 10) connects to their field theory construction
- Strong-coupling evidence is what their framework predicts

**What they'd question:**
- The paper calls it "Schwinger-Dyson structure" but doesn't write an explicit equation. In SYK: Σ(τ) = J²G(τ)^{q-1}. What is Σ_L explicitly? The decomposition into bare + self-energy is given but not the functional form.
- Identification of the fold with the Green's function G is suggestive but not rigorous. G(τ) in SYK is a fermion two-point function. What operators does the fold correspond to?
- "Strong-coupling" based on |Σ|/|G₀| ≈ 4-5 is a ratio of norms, not a proper functional diagnosis. In SYK, strong coupling means the conformal solution dominates with specific functional form.

**What's missing:**
- Explicit Σ_L in terms of W^Q, W^K, W^V, FFN weights — the theoretical keystone
- Testable predictions from the SYK connection (e.g., the four-point function form)
- How discrete layer index L maps to continuous time τ in SYK

**Assessment:** They would be most excited by this paper among all audiences, but would push for quantitative Schwinger-Dyson, not just structural analogy.

---

## 5. Xiao-Liang Qi (Holographic Quantum Information)

**What they'd value:**
- Δ = 0.254 matching 1/4 is intriguing holographically
- Strong-coupling evidence supports regime where holographic duality applies

**What they'd question:**
- **What plays the role of N?** SYK has a large-N limit (N = number of Majorana fermions). Is N the sequence length? Key dimension? Number of heads? The paper doesn't address this fundamental question.
- If Δ = 1/4 is the SYK conformal dimension, it should be robust in the large-N limit. Length dependence (Section 7.4) is concerning.
- **Error bars.** 1.4% agreement could be coincidence. What's the statistical uncertainty on 0.254? The paper reports a point estimate without confidence intervals.

**What's missing:**
- Discussion of the large-N limit and the role of N
- Any discussion of the holographic dual — even speculative. If this is SYK, what is the JT gravity?
- Scrambling time / OTOC. SYK is maximally chaotic — can scrambling be measured in the transformer?
- Proper uncertainty quantification on Δ_eff

**Assessment:** Qi would need the large-N question addressed before taking the SYK claim seriously. The error bar gap is a real vulnerability.

---

## 6. Jim Halverson (Neural Network Field Theory)

**What they'd value:**
- Per-head Δ_var ≈ 2Δ_mean as CFT-like consistency
- Effective central charge and Calabrese-Cardy scaling
- The fold as potential order parameter

**What they'd question:**
- His work uses network ensembles (averaging over weights). This paper studies a single trained model. The conformal structure in his framework emerges from the ensemble average. Are these comparable?
- c_eff varying from 0.76 to 2.06 across layers — in a genuine CFT, c is constant. He'd push hard on what this means.
- Rényi order dependence (Section 8.5) explicitly shows non-conformal corrections. What are the corrections and can they be computed?

**What's missing:**
- Connection to his specific construction (neural network correlation functions from weight ensemble averages)
- Discussion of the replica trick and its relation to the Rényi spectrum

**Assessment:** Natural point of contact. The layer-dependent c_eff is both the most interesting and most vulnerable finding from his perspective.

---

## 7. Chris Robinson (Virasoro Symmetry)

**What they'd value:**
- Effective central charge connects to his Virasoro construction
- Calabrese-Cardy scaling suggests conformal symmetry

**What they'd question:**
- Virasoro is 2D conformal symmetry. What is the 2D space? The sequence is 1D. In Robinson's work, the second dimension is network depth. The paper should address this.

**What's missing:**
- Discussion of higher Virasoro modes or the stress tensor
- The 2D interpretation (sequence position × layer depth)

**Assessment:** Brief engagement likely — the c_eff measurement is the relevant data point for his program.

---

## 8. Mechanistic Interpretability Community (Nanda, Olah, etc.)

**What they'd value:**
- The fold analysis is entirely new. U-curve, freedom budget, anti-correlator, sharpness arc — concrete, measurable, reproducible.
- "Recognition before understanding" (84% from token statistics) is directly relevant
- Code provided, experiments on GPT-2

**What they'd question:**
- Paper is very physics-heavy. Sections 3-7 would be impenetrable to most ML researchers.
- **Only two models, same family.** GPT-2 small and medium are old, small models. "Universality" from two members of the same family is thin. Need Llama, Mistral, Pythia, etc.
- **No ablation.** The anti-correlator is described evocatively but the standard interpretability move — ablate and measure — isn't done. Flagged as open but not attempted.
- Fold is at single query position (last token). Does it change? Does it appear in bidirectional models (BERT)?
- No connection to existing interpretability concepts: induction heads, QK/OV circuits, head taxonomies.
- Freedom budget not connected to downstream task performance.

**What's missing:**
- Ablation experiments (biggest single gap for this audience)
- Diverse architectures beyond GPT-2
- Connection to standard interpretability concepts
- Fold at different query positions
- Bidirectional model tests
- Freedom budget correlated with task performance

**Assessment:** The fold analysis is genuinely new and could get significant traction, but the paper doesn't speak the interpretability community's language and lacks the experiments they'd consider necessary (ablation, diverse models).

---

## Cross-Cutting Gaps (Ordered by Priority)

1. **Error bars on Δ_eff** — the 1.4% agreement is the headline number. Without uncertainty quantification, it's vulnerable to "coincidence" dismissal. (Affects: Qi, Kim, Ageev, all physics audiences)

2. **Acknowledge Alpay and Senturk** — they have the starting observation. Credit must be explicit. (Affects: Alpay/Senturk directly, credibility with everyone)

3. **Wick contraction sketch** — central claim references external work. At least an abbreviated derivation belongs here. (Affects: Kim, Ageev, any reviewer)

4. **What plays the role of N?** — fundamental for the SYK mapping. Currently unanswered. (Affects: Qi, Ageev, all SYK-adjacent readers)

5. **Ablation experiments** — the first question from interpretability. Not done. (Affects: the entire ML audience)

6. **More architectures** — "universality" from two GPT-2 variants is weak. (Affects: interpretability community, and universality claim generally)

7. **Explicit self-energy** — multiple physics audiences want Σ_L written in terms of model parameters. (Affects: Ageev, Qi, Halverson)

### What can be fixed in the paper now:
- Acknowledge Alpay and Senturk (Sections 3 and 11.3)
- Add Wick contraction derivation sketch
- Add uncertainty discussion for Δ_eff
- Discuss what plays the role of N
- Name the boundary structure question for positive geometry
- Strengthen the open questions to explicitly name ablation, diverse architectures, etc.
- Address the 2D question (sequence × depth) for Robinson/Virasoro context

### What requires new experimental work:
- Ablation experiments
- More architectures (Pythia, Llama, Mistral)
- Fold at different query positions
- Bidirectional model tests (BERT)
- Explicit self-energy derivation from model parameters
- Scrambling/OTOC measurements
- Proper bootstrap confidence intervals on Δ_eff

---

*This review should be re-read before each outreach communication and before any future revision of the paper.*
