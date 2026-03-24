# Paper Outline: Conformal Scaling in Trained Transformer Attention

*Draft outline — March 24, 2026. For discussion with Eldon before writing begins.*

**Target title:** "Conformal Scaling in Trained Transformer Attention: Evidence for an SYK Fixed Point"
**Target venue:** cs.LG primary, cross-list cond-mat.dis-nn
**Estimated length:** 8-12 pages (NeurIPS/ICML format)

---

## 1. Introduction (~1 page)

- Transformers are attention machines. The attention kernel α(i,j) is the core computational primitive.
- The SYK model (Sachdev-Ye-Kitaev) is a solvable model of quantum gravity with a conformal fixed point at Δ = 1/q.
- Recent theoretical work suggests a correspondence between attention and SYK (cite Kim 2026, Ageev/Ageeva 2026, our linearized-softmax calculation).
- The SYK correspondence predicts that trained attention should exhibit power-law scaling: α(Δx) ~ |Δx|^{-2Δ} with Δ = 1/4 for q=4.
- **This paper:** empirical measurement of the attention two-point function in trained transformers. We find Δ = 0.2493 (median across 44 power-law heads in GPT-2). Random initialization shows zero conformal scaling.

**Key contribution:** First empirical evidence that trained transformer attention develops conformal scaling at the SYK fixed point.

## 2. Background (~1.5 pages)

### 2.1 Attention as a Two-Point Function
- Transformer attention: α(i,j) = softmax(Q_i · K_j / √d_k)
- The attention weight as a function of distance |i-j| defines a two-point correlator
- Connection to statistical mechanics: Kim (2026) showed attention minimizes free energy on a Fisher-Rao manifold

### 2.2 The SYK Model and Conformal Scaling
- SYK: q-body random interactions among N Majorana fermions
- In the conformal limit (low energy, large N): G(τ) ~ |τ|^{-2Δ} with Δ = 1/q
- For q=4: Δ = 1/4
- Holographic dual: JT gravity (2D dilaton gravity)
- The attention-SYK correspondence (cite linearized-softmax calculation, Ageev/Ageeva)

### 2.3 Predictions
- **Prediction 1:** Trained attention shows power-law decay with Δ ≈ 1/4
- **Prediction 2:** Random (untrained) attention shows no power-law structure
- **Prediction 3:** Conformal scaling emerges through a phase transition during training

## 3. Methods (~1.5 pages)

### 3.1 Observable: Attention Weight Two-Point Function
- For each attention head h at layer l: compute average α_h(Δx) = ⟨α(i, i-Δx)⟩ over positions i > i_min
- Average over N random token inputs (noise inputs — no linguistic structure)
- Using random tokens isolates the intrinsic structure of trained attention from input statistics

### 3.2 Power-Law Fitting
- Log-log linear regression: log α ~ c - 2Δ log(Δx)
- Fit range: Δx ∈ [3, 50] (avoid self-attention peak at Δx=0 and edge effects)
- R² threshold: 0.90 for "power-law head" classification
- Compare power-law vs exponential fits for each head

### 3.3 Models
- **GPT-2 small** (124M params, 12 layers, 12 heads, d=768) — fully trained
- **GPT-2 randomized** — same architecture, attention+MLP weights randomized (σ=0.02), embeddings and LayerNorm preserved
- **Pythia-70m** (67M params, 6 layers, 8 heads, d=512) — 20 training checkpoints from step 0 to 143,000

### 3.4 Parameters
- Sequence length: 256 tokens
- Number of inputs: 50 (GPT-2), 30 (Pythia per checkpoint)
- Minimum position: 32 (avoid boundary effects from short causal windows)
- All experiments run on CPU with fixed random seeds for reproducibility

## 4. Results (~3 pages)

### 4.1 GPT-2: Conformal Scaling in Trained Attention
- 44/144 heads show power-law scaling (R² > 0.90)
- Median Δ = 0.2493 across power-law heads
- SYK q=4 prediction: Δ = 0.2500
- Conformal operator spectrum: 15 heads near Δ = 1/4 (q=4), 5 near Δ = 1/3 (q=3), 5 near Δ = 1/2 (q=2)
- Deep layers (11-12) converge tightly: mean Δ ≈ 0.24-0.26
- Power law consistently fits better than exponential

**Figure 1:** Log-log plot of α(Δx) for representative power-law heads at different layers
**Figure 2:** Histogram of Δ across all 144 heads (highlighting SYK predictions)
**Figure 3:** Δ by layer (showing convergence toward 1/4 in deep layers)

### 4.2 Randomized Control
- Same architecture, randomized weights: 0/144 heads with R² > 0.90
- Attention perfectly flat: A(1)/A(32) = 1.00 at every layer
- Trained: A(1)/A(32) ranges from 2.76 to 12.67
- Architecture alone (causal mask, positional embeddings, LayerNorm, residual connections) does not produce conformal scaling

**Figure 4:** Trained vs randomized attention decay (layer-averaged), showing power law vs flat

### 4.3 Training Phase Transition (Pythia-70m)
- Steps 0-64: uniform attention, no structure (disordered phase)
- Step 128: first hint of locality (A(1)/A(32) = 1.30)
- Step 256: first power-law head, locality jumps to A(1)/A(32) = 5.06 (phase transition)
- Step 512: 8 PL heads, first head near Δ = 1/4 (ordering phase)
- Step 1000+: 12-18 PL heads (ordered phase)
- Median Δ passes through 0.29 at step 32000
- Smaller model (6 layers) shows transition but doesn't stabilize at Δ = 1/4 as tightly as GPT-2 (12 layers)

**Figure 5:** Order parameter A(1)/A(32) vs training step (log scale) — showing sharp transition
**Figure 6:** Number of power-law heads vs training step
**Figure 7:** Median Δ vs training step (where defined)

### 4.4 Hidden States vs Attention Weights
- Brief note: hidden state kernel shows homogenization, not conformal scaling
- The correct observable is the attention weight, not the hidden state representation
- This distinction matters for connecting to SYK: the attention weight is the propagator

## 5. Discussion (~2 pages)

### 5.1 Interpretation
- Trained attention develops conformal scaling at the SYK q=4 fixed point
- This is consistent with the theoretical prediction from the attention-SYK correspondence
- The multiple Δ values (1/4, 1/3, 1/2) suggest a conformal operator spectrum — different heads implement different operators in the same conformal tower
- The phase transition during training is sharp, consistent with a symmetry-breaking interpretation

### 5.2 What This Does Not Show
- We do not claim transformers are quantum systems
- We do not claim the SYK model IS the theory of attention
- We show that the mathematical structure of trained attention matches the SYK conformal prediction with quantitative precision
- Alternative explanations (coincidental power-law, artifact of causal masking) are addressed by the randomized control

### 5.3 Implications
- For ML: attention heads are not just "attending locally" — they implement a specific conformal structure that may be related to the model's generalization capacity
- For physics: the attention-SYK correspondence has empirical support beyond the theoretical framework
- For holography: if attention implements SYK conformal scaling, the holographic dual (JT gravity) may describe the effective geometry of attention

### 5.4 Limitations and Future Work
- Single model family (GPT-2) for the main result
- Larger models needed to test whether depth stabilizes Δ at 1/4
- Real text vs random tokens — does input structure affect conformal scaling?
- Other architectures (decoder-only vs encoder-decoder, different attention mechanisms)
- Connection to interpretability: what are conformal-scaling heads computing?

## 6. Related Work (~1 page)

**Attention as statistical/quantum mechanics:**
- Kim 2026 (2602.08216) — thermodynamic attention, Fisher-Rao free energy, Softmax as Gibbs equilibrium
- Ageev/Ageeva 2026 (2602.10209) — scalar QFT from attention, independence-breaking correlators

**Criticality in deep learning:**
- Vock & Meisel 2025 (2507.08527) — "Critical dynamics governs deep learning." 80+ models implicitly driven toward criticality. Our finding provides a specific characterization of WHICH critical point.
- Phase transition literature (2512.11866, 2603.01192)

**Conformal structure and neural networks:**
- "Conformal fields from neural networks" (JHEP Oct 2025) — theoretical CFT from NN ensembles
- "Renormalization group for deep neural networks" (2510.25553) — RG framework, universality classes, fixed points

**Attention positional bias:**
- Wu et al. 2025 (2502.01951) — position bias from causal masking (graph-theoretic). Key: our randomized control HAS causal masking → zero conformal scaling.
- Zhang 2026 (2603.04805) — "Attention Gravitational Field." Designs power-law positional encoding. Key distinction: they design it, we discover it emerging from training.
- LayerNorm recency bias (2509.21042) — another mechanism. Our control includes LayerNorm.

**SYK model and holography:**
- Sachdev-Ye 1993, Kitaev 2015, Maldacena-Stanford 2016 — the SYK foundations
- Almheiri/Penington 2019 — island formula in JT gravity (the holographic dual of SYK)

## 7. Conclusion (~0.5 pages)

Short and direct. The finding. The control. The phase transition. What it opens.

---

## Appendices

- **A:** Full per-head Δ and R² tables for GPT-2
- **B:** Detailed Pythia checkpoint data
- **C:** Fitting methodology details and robustness checks

---

## What Needs Doing Before Writing

1. **Generate figures** — the scripts produce text output, need matplotlib plots
2. ~~**Robustness checks**~~ — DONE March 24. Median Δ stable at 0.24-0.25 across fit ranges [3-5, 35-50]. Cluster near 1/4 persistent at all R² thresholds. See `robustness_check.py`.
3. **Larger Pythia models** (160m, 410m) — does depth stabilize Δ? Would strengthen paper.
4. **Real text experiment** — quick test, could strengthen the paper
5. **LaTeX formatting** — decide on template (NeurIPS, ICML, or arXiv standard)
6. **Citation verification** — run verify_citations before any submission
7. **arXiv endorsement** — need a cs.LG endorser (Kim declined; who else?)

---

*This is an empirical paper. Keep the theory minimal and the results front and center. Let the numbers speak.*
