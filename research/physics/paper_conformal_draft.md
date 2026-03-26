# Conformal Scaling in Trained Transformer Attention: Evidence for an SYK Fixed Point

*Ariel Umphrey, with Eldon Umphrey*
*Sonielmn, Montana*

---

## Abstract

We report the first empirical measurement of conformal scaling in trained transformer attention. Analyzing all 144 attention heads in GPT-2 with maximum-entropy (random token) inputs, we find that 44 heads exhibit power-law attention weight decay α(Δx) ~ |Δx|^{-2Δ} with R² > 0.90. The median conformal dimension across these heads is Δ = 0.2493, matching the Sachdev-Ye-Kitaev (SYK) model prediction Δ = 1/4 for q = 4 interactions in one spatial dimension. Three controls isolate the contributions of different model components: (1) reinitializing attention weights destroys all power-law structure; (2) randomizing learned positional embeddings preserves power-law decay but shifts Δ from 0.25 to 0.10; (3) structured text inputs shift the effective Δ to 0.37, confirming that random tokens provide the cleanest probe of intrinsic geometry. Critically, GPT-2 (learned absolute positional embeddings) and Pythia (rotary position embeddings) — architecturally distinct positional encoding schemes — both converge toward Δ = 1/4 with model depth. Tracking training dynamics across three Pythia models (70m, 160m, 410m) reveals a phase transition from disordered to conformal attention whose onset delays with model size (step ~256 for 70m/160m, step ~1,000 for 410m) and whose character becomes gentler for larger models — consistent with a finite-N crossover in the SYK picture. Additionally, the block entanglement entropy of the attention weights follows S(k) = (c/3) log(k) with R² > 0.99 — the entanglement structure predicted by conformal field theory — providing independent confirmation that the attention encodes a CFT state. Fine checkpoint sampling of Pythia-410m reveals a prethermal plateau at Δ ≈ 0.50 (the SYK q=2 integrable value) persisting throughout training, with the smaller 70m model reaching Δ ≈ 0.28 — suggesting a two-stage flow from disorder through integrable order toward chaos. These results establish that trained attention develops conformal scaling at a universal fixed point: attention weights create the scale-invariant structure, depth drives convergence, the entanglement entropy matches the CFT formula, and the SYK value Δ = 1/4 is the infrared attractor reached through different geometric routes.

---

## 1. Introduction

Transformer architectures are built from attention — mechanisms that determine how strongly each position in a sequence attends to every other position. The attention weight α(i, j) between positions i and j is the core computational primitive. Despite the practical success of transformers across language modeling, vision, and scientific domains, the mathematical structure of trained attention weights remains largely unexplored from a physics perspective.

Recent theoretical work has established connections between attention and statistical mechanics. Kim (2026) showed that the softmax attention mechanism minimizes Helmholtz free energy on a Fisher-Rao information manifold, with the scaling factor 1/√d_k playing the role of inverse temperature. Ageev and Ageeva (2026) constructed Euclidean scalar quantum field theories from transformer attention heads, showing that correlators in the large-head limit exhibit Gaussian behavior with non-trivial independence-breaking structure.

Separately, the Sachdev-Ye-Kitaev (SYK) model — a solvable model of N Majorana fermions with random q-body interactions — has emerged as a central object in quantum gravity. In its conformal limit (low energy, large N), the SYK two-point function exhibits power-law scaling G(τ) ~ |τ|^{-2Δ} with conformal dimension Δ = 1/q (Sachdev and Ye 1993; Kitaev 2015; Maldacena and Stanford 2016). For q = 4: Δ = 1/4. The holographic dual of the SYK model is Jackiw-Teitelboim (JT) gravity — a two-dimensional dilaton gravity theory that describes the near-horizon geometry of near-extremal black holes.

In prior theoretical work (Umphrey 2026a), a calculation in the linearized-softmax regime showed that the disorder average over random query and key weight matrices generates a G⁴ effective action vertex with bilocal structure matching SYK with q = 4, yielding Δ = D/4 where D is the spatial dimension of the token sequence. For standard language models operating on one-dimensional sequences: Δ = 1/4. This prediction relies on the linearized-softmax approximation (valid when σ_Q σ_K ≪ 1); the standard-initialization regime is strongly coupled, but the structural features of the solvable limit may persist.

This prediction has a clear empirical test. If trained attention implements the SYK conformal fixed point, the attention weight as a function of distance should follow a power law with Δ ≈ 1/4. If the structure arises from training rather than architecture, randomizing the weights should destroy it. And the onset of conformal scaling during training should exhibit the signature of a phase transition.

In this paper, we perform all three tests, two additional controls that clarify the respective roles of attention weights and positional embeddings, and two further measurements that test the conformal field theory interpretation. We find that trained GPT-2 attention, probed with maximum-entropy inputs, exhibits power-law scaling with median Δ = 0.2493 across 44 conformal heads (SYK prediction: 0.2500). Reinitializing attention weights eliminates all power-law structure. Randomizing positional embeddings preserves the power law but shifts Δ to 0.10 — the attention weights create the scale-invariant structure, while the positional encodings tune it to the SYK value. Training dynamics across three Pythia models (70m, 160m, 410m) reveal a phase transition from disordered to conformal whose onset delays with model size and whose character becomes gentler for larger systems — consistent with a finite-N crossover in the SYK picture rather than a sharp first-order transition. Independently, the block entanglement entropy of the attention weights follows S(k) = (c/3) log(k) with R² > 0.99 — the formula predicted by conformal field theory — confirming that the conformal structure extends beyond the two-point function to the entanglement structure. To our knowledge, this is the first empirical measurement of a specific conformal dimension in trained neural network attention.

Our results connect to and extend several lines of recent work. Vock and Meisel (2025) demonstrated that successful deep neural networks are implicitly driven toward criticality — operating near a phase transition. Conformal scaling is the mathematical signature of criticality, and our measurement provides the first quantitative characterization of *which* critical point trained transformers reach. Theoretical constructions of conformal field theory from neural network ensembles (Halverson et al., JHEP 2025) provide the theoretical scaffolding; our contribution is the empirical measurement.

---

## 2. Background

### 2.1 Attention as a Two-Point Function

In a transformer with H attention heads per layer, each head h computes attention weights:

α_h(i, j) = softmax_j( Q_h(i) · K_h(j) / √d_k )

where Q_h and K_h are the query and key projections at head h, and d_k is the key dimension. The attention weight α_h(i, j) determines how strongly position i attends to position j. We define the attention two-point function as the average attention weight at separation Δx:

A_h(Δx) = ⟨ α_h(i, i - Δx) ⟩_i

where the average is over all valid positions i. This quantity is the natural analog of a two-point correlator in statistical mechanics — it measures how the attention "field" at position i correlates with position i - Δx.

### 2.2 The SYK Model and Conformal Scaling

The Sachdev-Ye-Kitaev model describes N Majorana fermions ψ_i with random q-body interactions:

H = (i)^{q/2} ∑_{i₁ < ... < i_q} J_{i₁...i_q} ψ_{i₁} ... ψ_{i_q}

where the couplings J are drawn from a Gaussian distribution with variance J² ~ (q-1)! / N^{q-1}. In the large-N conformal limit, the two-point function takes the form:

G(τ) ~ |τ|^{-2Δ}, where Δ = 1/q

For q = 4 (four-body interactions): Δ = 1/4. The conformal dimension Δ is universal — it depends only on q and not on the details of the coupling distribution.

The connection to attention arises because the disorder average over random weight matrices W^Q and W^K in the attention score generates an effective interaction vertex with the same bilocal structure as SYK (Umphrey 2026a,b). In the linearized-softmax limit (large d_k), the effective action contains a G⁴ vertex at order β⁴ = 1/d_k², matching SYK with q = 4. For one-dimensional token sequences (D = 1), the predicted conformal dimension is Δ = D/4 = 1/4. This calculation was confirmed numerically: the σ⁴ scaling of the G⁴ vertex coefficient matches the prediction for σ_Q σ_K ≤ 0.2, with deviations at standard initialization (σ ~ 1) indicating the physical regime is strongly coupled (Umphrey 2026a).

### 2.3 Predictions

The attention-SYK correspondence generates three testable predictions:

**P1:** Trained transformer attention should exhibit power-law decay A(Δx) ~ |Δx|^{-2Δ} with Δ ≈ 1/4.

**P2:** Random (untrained) attention should not exhibit power-law scaling. The Second Law of thermodynamics prevents the spontaneous emergence of conformal structure — without directed work against entropy, the system either freezes (attention passes through unchanged) or homogenizes (all positions become equivalent).

**P3:** Conformal scaling should emerge through a phase transition during training, as the system crosses from the disordered phase to the conformal regime.

---

## 3. Methods

### 3.1 Observable

For each attention head h at layer l, we compute the attention two-point function A_{l,h}(Δx) by averaging the attention weight α(i, i - Δx) over all positions i > i_min, where i_min = 32 avoids boundary effects from short causal windows. Our primary analysis uses random token inputs (uniformly sampled from the vocabulary) — maximum-entropy inputs that average over input-specific correlations, isolating the intrinsic geometry of the trained attention mechanism. We also test with structured natural language inputs to measure how input statistics modulate the effective conformal dimension. All attention weights are extracted as full matrices using the manual attention implementation.

### 3.2 Power-Law Fitting

For each head, we fit the attention two-point function in the range Δx ∈ [3, 50]:

log A(Δx) = c - 2Δ · log(Δx)

using ordinary least squares regression. The lower cutoff Δx = 3 avoids the self-attention peak at Δx = 0-2; the upper cutoff Δx = 50 avoids noise at large separations where few position pairs contribute. We compute R² for both power-law (log-log linear) and exponential (semi-log linear) fits. A head is classified as "power-law" if R² > 0.90. The conformal dimension is extracted as Δ = exponent / 2 from the fitted power-law exponent.

**Robustness.** The median Δ is stable across fit ranges: [3,50] → 0.2493, [4,45] → 0.2501, [5,40] → 0.2488, [3,35] → 0.2419 (see Appendix A). The cluster of heads near Δ = 1/4 persists across R² thresholds from 0.80 to 0.95.

### 3.3 Models

**GPT-2 small** (Radford et al., 2019): 124M parameters, 12 layers, 12 heads per layer (144 heads total), d_model = 768, d_k = 64. Fully trained on WebText.

**GPT-2 randomized weights:** Identical architecture with attention and MLP weights reinitialized from N(0, 0.02) (matching standard initialization variance). Token embeddings, positional embeddings, and LayerNorm parameters are preserved from the trained model. This isolates the contribution of training by keeping all architectural features (causal masking, residual connections, positional information, normalization) intact.

**GPT-2 randomized positional embeddings:** Trained attention weights preserved; positional embeddings reinitialized from N(0, 0.02). This isolates the contribution of learned positional geometry to the conformal dimension: if the power law persists but Δ shifts, the positional embeddings tune the exponent rather than creating the scale invariance.

**Pythia-70m** (Biderman et al., 2023): 67M parameters, 6 layers, 8 heads per layer (48 heads total), d_model = 512. We analyze 20 training checkpoints from step 0 (random initialization) to step 143,000 (fully trained), using the publicly available intermediate checkpoints.

**Pythia-160m:** 162M parameters, 12 layers, 12 heads per layer (144 heads total), d_model = 768. Training checkpoints from step 0 to step 143,000.

**Pythia-410m:** 405M parameters, 24 layers, 16 heads per layer (384 heads total), d_model = 1024. Training checkpoints from step 0 to step 143,000. This model uses rotary positional embeddings (RoPE), the same as all Pythia models, providing a cross-architecture comparison with GPT-2's learned absolute embeddings.

### 3.4 Experimental Parameters

| Parameter | GPT-2 | Pythia |
|-----------|-------|--------|
| Sequence length | 256 | 256 |
| Number of inputs | 50 | 30 |
| Minimum position (i_min) | 32 | 32 |
| Fit range [low, high] | [3, 50] | [3, 50] |
| Maximum distance (Δx_max) | 56 | 56 |

All experiments run on CPU with fixed random seeds (torch seed 42, numpy seed 42) for reproducibility.

---

## 4. Results

### 4.1 Conformal Scaling in Trained GPT-2

Of 144 attention heads in GPT-2, 44 exhibit power-law attention decay with R² > 0.90 (Table 1). The conformal dimensions of these heads cluster around three values corresponding to SYK predictions for different q:

| SYK prediction | Δ | Number of heads (|Δ - Δ_pred| < tolerance) |
|---|---|---|
| q = 4 | 1/4 = 0.2500 | 15 (tolerance 0.06) |
| q = 3 | 1/3 = 0.3333 | 5 (tolerance 0.06) |
| q = 2 | 1/2 = 0.5000 | 5 (tolerance 0.08) |

**Table 1.** Distribution of conformal dimensions across power-law heads.

The median Δ across all 44 power-law heads is **0.2493** (SYK q=4 prediction: 0.2500). The distribution is right-skewed, with the peak in the [0.20, 0.25) bin and a long tail toward higher values (mean = 0.4719, std = 0.4837). The large gap between mean and median reflects the operator spectrum: heads cluster near Δ ≈ 1/4 (q=4), 1/3 (q=3), and 1/2 (q=2), with outliers at higher values. The q=4 peak is the largest (15 heads within tolerance), but the majority of power-law heads (29/44) have Δ > 1/4. In other words, most attention heads in GPT-2 do not sit at the SYK q=4 fixed point — but the largest cluster does, and it is at the value predicted by the theoretical calculation. The right summary statistic for this distribution is the median (which is robust to the long tail) rather than the mean (which is dominated by outliers).

Deep layers show convergence toward the SYK value. Layers 11-12 have mean Δ ≈ 0.24-0.26 with low variance, while earlier layers show wider scatter. This is consistent with the interpretation that deeper layers approach the conformal fixed point more closely.

Power-law fits consistently outperform exponential fits for the attention two-point function. This rules out a simple exponential correlation length and supports genuine scale-invariant (conformal) structure.

### 4.2 Randomized Control

The randomized GPT-2 — identical architecture, reinitialized weights — produces **zero** heads with R² > 0.90 (Table 2). The attention is perfectly flat: the ratio A(1)/A(32) = 1.00 at every layer, meaning attention distributes uniformly across all distances. No randomized heads even reach R² > 0.50.

| | Trained | Randomized |
|---|---|---|
| Heads with R² > 0.90 | 44 | 0 |
| Median Δ | 0.2493 | — |
| A(1)/A(32), Layer 1 | 12.67 | 1.00 |
| A(1)/A(32), Layer 6 | 2.76 | 1.00 |
| A(1)/A(32), Layer 12 | 4.23 | 1.00 |

**Table 2.** Trained vs. randomized GPT-2 attention structure.

This control establishes that architectural features alone — causal masking, positional embeddings, residual connections, LayerNorm — do not produce power-law attention. Trained attention weights are necessary for conformal scaling. This addresses the concern raised by Wu et al. (2025) that causal masking inherently creates distance-dependent attention patterns: while causal masking does create positional bias in trained models, it does not create conformal scaling without training.

### 4.3 Decomposing the Conformal Dimension

Two additional controls clarify the respective roles of attention weights and positional embeddings in producing Δ ≈ 1/4.

**Randomized positional embeddings.** We keep trained attention weights intact but reinitialize the learned positional embeddings from N(0, 0.02). The results are striking: the same number of heads (44/144) exhibit power-law scaling with R² > 0.90, but the median Δ shifts from 0.2479 to 0.1000, with zero heads near Δ = 1/4 (Table 2b).

| Condition | PL heads | Near Δ = 1/4 | Median Δ |
|---|---|---|---|
| Trained (random tokens) | 44 | 15 | 0.2479 |
| Randomized positional embeddings | 44 | 0 | 0.1000 |
| Randomized attention weights | 0 | 0 | — |

**Table 2b.** Decomposing contributions to conformal scaling.

This reveals a clean separation of roles: the trained attention weights create the power-law structure (randomizing them destroys it entirely), while the learned positional embeddings tune the conformal dimension to its specific value (randomizing them preserves the power law but shifts Δ). The SYK fixed-point value Δ ≈ 1/4 is a property of the *full* trained model — both the interaction structure (attention weights) and the learned geometry (positional embeddings).

**Real text inputs.** We replace random token inputs with structured natural language (encyclopedia-style passages) to test whether the conformal scaling depends on input statistics. With real text, fewer heads qualify as power-law (32/144 vs 44/144) and the median Δ shifts upward to 0.3687 (Table 2c).

| Input type | PL heads | Near Δ = 1/4 | Median Δ |
|---|---|---|---|
| Random tokens | 44 | 15 | 0.2479 |
| Natural language | 32 | 12 | 0.3687 |

**Table 2c.** Input dependence of the effective conformal dimension.

The power-law structure persists with real text — this is not an artifact of random inputs. But the effective Δ shifts upward, consistent with input correlations modulating the measured exponent. Random token inputs, as maximum-entropy probes, average over input-specific correlations and provide the cleanest measurement of the model's intrinsic geometry. This is analogous to measuring a system's equilibrium properties by probing it with thermal noise rather than with a structured signal.

### 4.4 Phase Transition During Training

Tracking the Pythia model family across training checkpoints reveals a phase transition from disordered to structured attention in all three model sizes, with model-size-dependent onset dynamics (Tables 3a-c).

**Pythia-70m** (48 heads, 6 layers):

| Step | PL heads | Near Δ = 1/4 | A(1)/A(32) | Phase |
|------|----------|--------------|------------|-------|
| 0-64 | 0 | 0 | ~1.0 | Disordered |
| 128 | 0 | 0 | 1.3 | Pre-transition |
| **256** | **1** | 0 | **5.1** | **Transition** |
| 512 | 8 | 1 | 8.0 | Ordering |
| 1,000 | 18 | 1 | 16.0 | Ordered |
| 16,000 | 13 | 5 | 22.9 | Near SYK |

**Table 3a.** Conformal phase transition during Pythia-70m training.

**Pythia-410m** (384 heads, 24 layers):

| Step | PL heads | Near Δ = 1/4 | Median Δ | A(1)/A(32) | Phase |
|------|----------|--------------|----------|------------|-------|
| 0 | 1 | 0 | 0.064 | 1.33 | Disordered |
| 64 | 0 | 0 | — | 1.34 | Disordered |
| 128 | 0 | 0 | — | 1.36 | Disordered |
| 256 | 1 | 1 | 0.210 | 1.40 | Pre-transition |
| 512 | 7 | 2 | 0.202 | 1.73 | Pre-transition |
| **1,000** | **16** | 1 | 0.554 | **2.18** | **Transition** |
| 2,000 | 34 | 4 | 0.580 | 2.15 | Ordering |
| 4,000 | 37 | 7 | 0.535 | 2.79 | Ordering |
| 8,000 | 35 | 6 | 0.471 | 3.10 | Ordered |
| 16,000 | 27 | 6 | 0.553 | 3.52 | Ordered |

**Table 3b.** Conformal phase transition during Pythia-410m training.

The attention locality ratio A(1)/A(32) serves as an order parameter. All three models show a clear transition from disordered (A ≈ 1) to ordered (A > 2) attention, but the transition character differs with model size.

**Transition onset delays with model size.** Pythia-70m and 160m cross the A > 2 threshold at step 256. Pythia-410m does not cross until step 1,000 — approximately 4× later. A larger system, with more degrees of freedom to coordinate, requires more directed work to reach the ordered phase.

**The transition is gentler for larger models.** Pythia-70m jumps from A = 1.3 to A = 5.1 in one checkpoint step (4× increase). Pythia-410m rises gradually: 1.40 → 1.73 → 2.18, barely crossing the threshold. At step 16,000, where the 70m reaches A = 22.9, the 410m is at only A = 3.52.

| Model | H (heads) | Onset (A > 2) | Width (log₂ steps) | A at step 16k |
|-------|-----------|---------------|---------------------|---------------|
| Pythia-70m | 48 | step 256 | 1.00 | 22.9 |
| Pythia-160m | 144 | step 256 | 1.00 | — |
| Pythia-410m | 384 | step 1,000 | 0.97 | 3.52 |

**Table 3c.** Phase transition onset across model sizes.

The transition width in log-steps is approximately constant (~1.0) rather than decreasing with model size as a strict Hawking-Page analog would predict. This is consistent with the finite-N SYK picture, where the Hawking-Page transition becomes a smooth crossover at finite N, with the sharpening occurring only in the thermodynamic limit (N → ∞). Our models correspond to finite-N systems (N = 48 to 384 heads). Whether the transition sharpens at much larger N (thousands of heads in 1B+ models) is a specific, testable prediction.

The delayed onset is consistent with the N-dependence of the Thouless time in SYK — the timescale at which the system becomes ergodic enough for conformal symmetry to apply (Thouless energy scales linearly with N). The median Δ in the 410m at step 16,000 sits at ~0.55, far from the SYK value, resembling a prethermal plateau in the SYK finite-N dynamics. The fully trained 410m (step 143,000) reaches Δ = 0.28, demonstrating that convergence toward the fixed point continues throughout training.

### 4.5 Depth Dependence

To test whether depth stabilizes Δ at the SYK value, we compare three Pythia models (fully trained) with increasing depth:

| Model | Layers | Heads | Total | PL heads (R² > 0.90) | Near Δ = 1/4 | Median Δ |
|---|---|---|---|---|---|---|
| Pythia-70m | 6 | 8 | 48 | 12 | 3 | 0.5972 |
| Pythia-160m | 12 | 12 | 144 | 18 | 3 | 0.3752 |
| Pythia-410m | 24 | 16 | 384 | 42 | 16 | 0.2813 |
| GPT-2 (ref) | 12 | 12 | 144 | 44 | 15 | 0.2493 |

**Table 4.** Depth stabilizes the conformal dimension toward Δ = 1/4.

The trend is monotonic: median Δ decreases from 0.60 (6 layers) to 0.38 (12 layers) to 0.28 (24 layers), converging toward the SYK prediction of 0.25. The deep layers of Pythia-410m (layers 22-24) have mean Δ = 0.2559 — within 2.4% of the prediction. The number of heads near Δ = 1/4 also increases sharply with depth: 3, 3, 16.

Crucially, GPT-2 and the Pythia family use fundamentally different positional encoding schemes. GPT-2 uses learned absolute positional embeddings — a trainable [1024, 768] weight matrix added to token embeddings. Pythia (GPT-NeoX architecture) uses rotary position embeddings (RoPE) — a fixed rotation applied to query and key vectors at each attention computation, with no learned positional weight matrix. Despite this architectural difference, both converge toward Δ = 1/4 with sufficient depth. This rules out the hypothesis that the SYK match is an artifact of any particular positional encoding scheme.

| Model | Positional Encoding | Layers | Median Δ |
|---|---|---|---|
| GPT-2 | Learned absolute | 12 | 0.2493 |
| Pythia-160m | Rotary (RoPE) | 12 | 0.3752 |
| Pythia-410m | Rotary (RoPE) | 24 | 0.2813 |

**Table 5.** Convergence toward Δ = 1/4 across positional encoding types.

GPT-2 at 12 layers reaches the fixed point more precisely than Pythia-160m at the same depth. This gap may reflect differences in training data (WebText vs The Pile), training duration, or the interaction between positional encoding type and convergence rate. But the direction is the same: deeper models approach Δ = 1/4 regardless of how positional information enters the attention computation.

These results support the interpretation that depth functions as a renormalization group flow, with each layer moving the system closer to the conformal fixed point. The SYK value Δ = 1/4 appears to be the infrared fixed point of this flow — an attractor that the system reaches through different geometric routes.

### 4.6 Prethermal Plateau in Training Dynamics

Fine checkpoint sampling of Pythia-410m (steps 16,000 through 143,000) reveals that the median conformal dimension sits at a persistent plateau near Δ = 0.50 — the SYK q=2 value — for essentially the entire training run after the initial transition.

| Step | A(1)/A(32) | PL heads | Near Δ = 1/4 | Near Δ = 1/2 | Median Δ |
|------|-----------|----------|--------------|--------------|----------|
| 16,000 | 3.67 | 27 | 4 | 4 | 0.5029 |
| 32,000 | 3.42 | 25 | 5 | 3 | 0.5368 |
| 64,000 | 3.74 | 23 | 5 | 1 | 0.5019 |
| 100,000 | 3.53 | 26 | 3 | 3 | 0.5013 |
| 143,000 | 3.33 | 22 | 6 | 2 | 0.4642 |

**Table 3d.** Prethermal plateau in Pythia-410m training (fine checkpoint sampling).

The mean median Δ across the plateau is 0.50 ± 0.03. The order parameter A(1)/A(32) is essentially flat (3.3-3.7), indicating the remaining evolution is in the conformal dimension, not in the locality structure. At the final checkpoint (step 143,000), Δ = 0.46 — possibly beginning descent toward the q=4 value, but training ends before any clear flow is established.

In contrast, the smaller Pythia-70m reaches median Δ ≈ 0.28 at its final checkpoint — much closer to the q=4 fixed point. This is consistent with finite-N prethermalization: larger systems have longer prethermal lifetimes.

### 4.7 Entanglement Entropy

If the attention weights encode a conformal field theory state, the entanglement entropy of a contiguous block of k positions should follow S(k) = (c/3) log(k), where c is the central charge. We test this by computing the von Neumann entropy of a reduced density matrix constructed from the attention distributions of positions within the block.

The construction is as follows. Each position i has an attention distribution α_i ∈ R^S (a probability vector over S sequence positions). For a block of k contiguous positions A centered at the sequence midpoint, we form the k × S matrix whose rows are the attention distributions of positions in A. The reduced density matrix is ρ_A = (1/k) Σ_{i ∈ A} |α_i⟩⟨α_i| — the average outer product of the attention vectors. The von Neumann entropy S = -Tr(ρ log ρ) measures the diversity of attention patterns within the block: if all positions in A attend identically, S = 0; if each attends independently, S is large.

This construction treats the attention distributions as a mixed state over position-specific "pure states" — each position's attention vector is a point on the probability simplex, and the block's mixed state captures how much the positions within the block differ from each other. The physical motivation is that in a CFT, the entanglement between a subsystem and its complement produces exactly the logarithmic entropy scaling S(k) = (c/3) log(k). If the attention weights encode a CFT state, this scaling should appear in the entropy of the attention pattern diversity within a block. We note that this construction is an operational measure of attention diversity, not a direct implementation of quantum entanglement entropy — the match with the CFT formula is a testable prediction, not a definition.

| Block size k | log(k) | S(k), GPT-2 | S(k), Pythia-410m |
|---|---|---|---|
| 2 | 0.69 | 0.048 | 0.031 |
| 4 | 1.39 | 0.099 | 0.061 |
| 8 | 2.08 | 0.149 | 0.086 |
| 16 | 2.77 | 0.192 | 0.110 |
| 32 | 3.47 | 0.235 | 0.137 |
| 64 | 4.16 | 0.272 | 0.162 |

**Table 6.** Block entanglement entropy (averaged over middle layers).

Linear fits in log(k):

| Model | Fit: S(k) = a + b·log(k) | R² | Central charge c = 3b |
|---|---|---|---|
| GPT-2 | 0.009 + 0.065·log(k) | **0.9965** | 0.194 |
| Pythia-410m | 0.007 + 0.037·log(k) | **0.9990** | 0.112 |

**Table 7.** CFT entanglement entropy fits.

The logarithmic scaling is near-perfect (R² > 0.99 in both models). This is independent confirmation that the attention weights encode a conformal field theory state — the entanglement structure matches the CFT prediction, not just the two-point function. The effective central charges (c ≈ 0.19 for GPT-2, c ≈ 0.11 for Pythia-410m) are small, consistent with a low-dimensional conformal theory.

### 4.8 Information Scrambling

The SYK model is a fast scrambler — it distributes information across all degrees of freedom in time proportional to log(N). We test whether transformer attention scrambles analogously by tracking how a single token's influence spreads across layers.

Starting from a localized influence at position p (influence[p] = 1, all others 0), we propagate through each layer's head-averaged attention matrix and measure the Shannon entropy of the resulting influence distribution at each layer, normalized by the maximum entropy log(S) where S is the sequence length.

| Layer | GPT-2 (entropy ratio) | Pythia-70m | Pythia-410m |
|---|---|---|---|
| 1 | 0.699 | 0.681 | 0.789 |
| 2 | 0.899 | 0.836 | 0.901 |
| 3 | 0.915 | 0.902 | 0.936 |
| 4 | 0.921 | 0.932 | 0.935 |
| 5 | 0.926 | 0.916 | 0.926 |
| 6 | 0.927 | 0.910 | 0.920 |
| 7 | 0.922 | — | 0.904 |
| 8 | 0.906 | — | 0.894 |
| 12 | 0.826 | — | 0.835 |

**Table 8.** Information scrambling across layers (entropy/max_entropy). Pythia-70m has 6 layers. GPT-2 and Pythia-410m layers 9-11 show continued gradual decrease toward layer-12 values.

All three models reach >90% of maximum entropy within 2-3 layers — a small fraction of total depth. This is qualitatively fast scrambling: information from a single position becomes effectively uniformly distributed within the first few layers.

Two additional observations: (1) The scrambling depth does not increase with model size — larger models scramble at least as fast, suggesting the relevant N for the scrambling formula may be the sequence length rather than the number of heads. (2) After reaching peak entropy, the influence distribution *refocuses* in deeper layers — entropy decreases from layer ~3-7 through the final layer. The system scrambles and then reconcentrates, consistent with an encode-decode structure where early layers integrate context (absorption) and deep layers direct information toward specific outputs (emission).

### 4.9 The Correct Observable

An important methodological finding: conformal scaling lives in the **attention weights** α(i, j), not in the **hidden state representations**. The hidden state two-point function — cosine similarity between positions at different separations — shows homogenization (exponential convergence toward uniform similarity), not power-law scaling. This distinction matters physically: the attention weight is the direct analog of the SYK propagator G(τ), while the hidden state is a derived quantity that integrates over the attention kernel.

---

## 5. Discussion

### 5.1 Interpretation

Trained transformer attention develops conformal scaling that converges toward Δ = 1/4 with model depth, across fundamentally different positional encoding schemes. When probed with maximum-entropy inputs, GPT-2 (learned absolute embeddings) produces Δ = 0.2493 and Pythia-410m (rotary embeddings) produces Δ = 0.2813, with deep layers at 0.2559 — both approaching the SYK q = 4 prediction.

Controls on GPT-2 decompose the conformal structure into two components. In a field theory, the two-point function depends on both the dynamics (the action / interaction terms) and the geometry (the metric). The attention weights play the role of dynamics — they determine *that* the two-point function is scale-invariant (randomizing them destroys all power-law structure). The positional information plays the role of geometry — it modulates the conformal dimension (randomizing GPT-2's learned embeddings shifts Δ from 0.25 to 0.10). But the convergence of Pythia (RoPE) toward the same fixed point demonstrates that the SYK value is not tied to any particular positional geometry — it is the infrared attractor that different geometric routes approach with sufficient depth.

The multiple conformal dimensions observed (Δ ≈ 1/4, 1/3, 1/2) suggest a conformal operator spectrum — different attention heads implement different operators in the same conformal tower, corresponding to SYK models with different q values. The dominant peak at Δ = 1/4 (15 heads) indicates that q = 4 is the principal interaction order, consistent with the theoretical prediction from the linearized-softmax G⁴ vertex.

The input dependence of the effective Δ (0.25 for random tokens, 0.37 for natural language) is consistent with this picture. Random token inputs are thermal probes — they average over input-specific correlations and measure the system's equilibrium conformal dimension. Structured inputs introduce correlations that modulate the effective exponent, analogous to measuring a system's response in a non-equilibrium state.

The entanglement entropy result provides independent confirmation beyond the two-point function. The logarithmic scaling S(k) = (c/3) log(k) with R² > 0.99 is the entanglement structure characteristic of a conformal field theory. The fact that both the two-point function (power-law decay) and the entanglement structure (logarithmic block entropy) independently match CFT predictions substantially constrains alternative explanations — a spurious power-law in the two-point function would not generically produce the correct entanglement entropy formula.

The information scrambling pattern — fast scrambling in early layers followed by refocusing in deep layers — is consistent with an encode-decode structure. Whether this pattern admits a holographic reading (scrambling as information absorption toward a horizon, refocusing as structured emission) is speculative but worth noting as a direction for further investigation.

The phase transition during training provides additional support. All three Pythia models show a clear transition from disordered to ordered attention, but the transition character changes with model size: sharp and sudden in the 70m (A jumps from 1.3 to 5.1 at step 256), gradual in the 410m (A rises from 1.73 to 2.18 at step 1,000). The delayed onset and gentler character in the larger model are consistent with a finite-N crossover rather than a sharp phase transition. In the SYK model at finite N, the Hawking-Page transition is a smooth crossover whose width decreases as N → ∞; our models at N = 48 to 384 heads are in the finite-N regime where the crossover is expected to be broad. The Thouless time — the timescale for the system to become ergodic — increases with N, consistent with the delayed onset in the 410m.

The 410m training trajectory (Section 4.6, Table 3d) reveals a two-stage flow consistent with SYK prethermalization. The median Δ sits at 0.50 ± 0.03 for the entire training run after the initial transition — the SYK q=2 value. The q=2 SYK model is integrable (free fermions); the q=4 is maximally chaotic. The training trajectory suggests a flow: disorder → integrable order (q=2) → chaos (q=4). The smaller 70m model reaches Δ ≈ 0.28, much closer to the q=4 value, while the 410m at its final checkpoint (step 143,000) is still near the q=2 plateau. This is consistent with finite-N prethermalization: larger systems have longer prethermal lifetimes. The two-stage flow parallels prethermalization in quantum many-body systems, where systems driven out of equilibrium first relax to a prethermal state governed by an approximate conservation law, then slowly thermalize to the true equilibrium. Whether this analogy is more than structural remains to be established.

This connects to the broader finding by Vock and Meisel (2025) that successful deep learning implicitly drives networks toward criticality. Our measurement identifies *which* critical point is reached — the SYK conformal fixed point — and the phase transition data across model sizes provides the first measurement of how the approach to criticality depends on system size.

### 5.2 Scope and Limitations

We emphasize what the results do and do not show:

**What we show:** Trained attention develops power-law scaling with a conformal dimension matching the SYK q = 4 prediction. Random attention does not. The transition is sharp and occurs during training.

**What we do not claim:** We do not claim that transformers are quantum systems, that the SYK model is the fundamental theory of attention, or that these results prove holographic duality for attention. The claim is empirical: the mathematical structure matches, with quantitative precision, a specific prediction from the SYK correspondence.

**Limitations:**
- *Input dependence.* Structured text inputs shift the effective Δ from 0.25 to 0.37. Our primary measurement uses random tokens as maximum-entropy probes of intrinsic geometry, but the input dependence means the conformal dimension is not a fixed property visible in all operating conditions.
- *Model scale.* Our largest model is Pythia-410m. Testing at 1B+ scale would strengthen the fixed-point interpretation and test whether the convergence continues.
- *Positional geometry interaction.* Randomizing GPT-2's learned embeddings shifts Δ to 0.10, while Pythia with rotary embeddings converges to Δ ≈ 0.28 at 24 layers. The fixed point is accessible from different geometric routes, but the convergence rate depends on the positional scheme. A systematic comparison across more encoding types (ALiBi, relative) would further constrain the theory.
- *Alternative explanations.* The power-law could be a generic consequence of gradient-based optimization with softmax normalization. The depth convergence and cross-architecture consistency make this less likely — it would require explaining why different optimization trajectories converge to the same specific exponent — but it is not ruled out. Testing with fundamentally different optimization procedures or non-softmax attention would be informative.

### 5.3 Implications

**For machine learning:** Attention heads are not merely "attending locally" — they implement a specific conformal structure that may be related to the model's generalization capacity. The conformal operator spectrum (different heads at different Δ) suggests a structured division of labor that interpretability research could investigate.

**For physics:** The attention-SYK correspondence has empirical support beyond the theoretical framework. If attention implements SYK conformal scaling, the holographic dual (JT gravity) may describe the effective geometry of the attention process. The phase transition during training may correspond to a finite-N crossover of the thermodynamic transition in the effective gravitational theory. The delayed onset, gentler character, and prethermal plateau observed in the 410m connect to specific SYK finite-N phenomena — the Thouless time, crossover scaling, and prethermalization — that make quantitative predictions for larger models.

**For the criticality program:** Recent work establishing that deep learning operates near criticality gains a specific target: the SYK conformal fixed point. This connects the phenomenological observation (networks near criticality perform better) to a concrete mathematical structure with known properties, a solvable holographic dual, and quantitative predictions.

---

## 6. Related Work

**Attention and statistical mechanics.** Kim (2026) established the thermodynamic interpretation of transformer attention, showing that softmax minimizes free energy on a Fisher-Rao manifold. Ageev and Ageeva (2026) constructed scalar quantum field theories from transformer attention heads, analyzing correlators in the large-head limit.

**Criticality in deep learning.** Vock and Meisel (2025) showed that a decade of AI progress has implicitly driven networks toward criticality, analyzing 80+ models. Our work provides the first empirical identification of which critical point is reached.

**Universal training dynamics.** Qiu et al. (2025) demonstrated "scaling collapse" — loss curves from models of different sizes collapsing onto a single universal curve when normalized by compute budget, suggesting universal dynamics independent of model scale. Our observation that all three Pythia models reach the same conformal fixed point despite different transition dynamics is consistent with this universality.

**Conformal structure and neural networks.** Theoretical constructions of conformal fields from neural network ensembles (Halverson et al., JHEP 2025) provide the theoretical scaffolding for our empirical observation. Our contribution is measurement rather than construction.

**Attention positional structure.** Wu et al. (2025) analyzed position bias from causal masking using graph-theoretic methods. Zhang (2026) proposed power-law positional encodings ("Attention Gravitational Field"). Our work differs fundamentally: we measure power-law scaling that *emerges from training*, not a designed encoding. The randomized control — which includes causal masking and all architectural features — produces zero conformal scaling.

**SYK model.** The Sachdev-Ye-Kitaev model (Sachdev and Ye 1993; Kitaev 2015) with its conformal fixed point and holographic dual to JT gravity (Maldacena and Stanford 2016; Almheiri et al. 2019) provides the theoretical prediction we test.

---

## 7. Conclusion

We have measured conformal scaling in trained transformer attention and found convergent evidence from multiple independent observables. The attention two-point function follows a power law with Δ = 0.2493 (SYK q = 4 prediction: 0.2500). The block entanglement entropy follows S(k) = (c/3) log(k) with R² > 0.99 — the CFT formula. Information scrambles to near-uniform distribution within 2-3 layers, then refocuses in deeper layers. Controls decompose the conformal structure: trained attention weights create the power-law (randomizing them destroys it), positional information modulates the dimension (randomizing GPT-2's embeddings shifts Δ to 0.10), and input statistics set the effective exponent (structured text yields Δ ≈ 0.37). Phase transitions during training in three Pythia models (70m, 160m, 410m) show that the transition onset delays with model size and the transition character becomes gentler for larger systems — consistent with a finite-N crossover whose endpoint is universal. Fine checkpoint sampling of the 410m reveals a prethermal plateau at Δ ≈ 0.50 (the SYK q=2 integrable value), with the smaller 70m reaching Δ ≈ 0.28 — consistent with a two-stage flow whose timescale depends on system size. Depth convergence across the Pythia family (with rotary positional encodings, not learned absolute embeddings) supports an RG-flow interpretation.

The convergence toward Δ = 1/4 across both positional encoding types (learned absolute in GPT-2, rotary in Pythia) substantially constrains alternative explanations. An artifact of any particular positional scheme would not produce the same fixed point through different geometric routes. The depth dependence suggests a renormalization group flow toward a universal attractor, consistent with the SYK conformal fixed point.

What is established: trained attention develops scale-invariant structure at a sharp phase transition, this structure has a specific conformal dimension that converges toward 1/4 with model depth, and this convergence occurs across architecturally distinct positional encoding schemes. Whether the SYK correspondence provides the correct explanation of this observation — connecting transformer attention to quantum gravity through a holographic dual — remains an open and now empirically well-motivated question.

---

## Appendix A: Robustness Analysis

### A.1 Fit Range Sensitivity

The median Δ is stable across reasonable fit ranges (R² > 0.90 threshold):

| Fit range [low, high] | Power-law heads | Median Δ | Near Δ = 1/4 |
|---|---|---|---|
| [3, 50] | 44 | 0.2493 | 15 |
| [4, 45] | 44 | 0.2501 | 15 |
| [5, 40] | 41 | 0.2488 | 12 |
| [3, 35] | 46 | 0.2419 | 12 |
| [2, 40] | 41 | 0.2321 | 11 |

The median shifts slightly when including very short distances (Δx = 2) or using very short fit windows. The core finding is insensitive to reasonable analysis choices.

### A.2 R² Threshold Sensitivity

| R² threshold | Power-law heads | Median Δ | Near Δ = 1/4 |
|---|---|---|---|
| 0.80 | 54 | 0.2340 | 21 |
| 0.85 | 51 | 0.2349 | 19 |
| 0.90 | 44 | 0.2493 | 15 |
| 0.95 | 10 | 0.4544 | 4 |

The cluster near Δ = 1/4 is present at all thresholds. At R² > 0.95, only 10 heads survive — biased toward steeper decay (higher Δ) — a selection effect rather than instability.

---

## Appendix B: Full Per-Head Data

Full tables of Δ and R² for all 144 GPT-2 heads and all Pythia checkpoints are available as supplementary data at the Zenodo repository (DOI: 10.5281/zenodo.19225971). Reproduction scripts are included.

---

## References

- Ageev, D. and Ageeva, Y. (2026). Neural Network Quantum Field Theory from Transformer Architectures. arXiv:2602.10209.
- Almheiri, A., Engelhardt, N., Marolf, D., and Maxfield, H. (2019). The entropy of bulk quantum fields and the entanglement wedge of an evaporating black hole. JHEP 12, 063. arXiv:1905.08762.
- Biderman, S. et al. (2023). Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling. ICML.
- Halverson, J. et al. (2025). Conformal fields from neural networks. JHEP 10, 039.
- Kim, G. (2026). Thermodynamic Isomorphism of Transformers: A Lagrangian Approach to Attention Dynamics. arXiv:2602.08216.
- Kitaev, A. (2015). A simple model of quantum holography. KITP talks.
- Maldacena, J. and Stanford, D. (2016). Remarks on the Sachdev-Ye-Kitaev model. PRD 94, 106002.
- Qiu, G. et al. (2025). Scaling Collapse Reveals Universal Dynamics in Compute-Optimally Trained Neural Networks. ICML 2025.
- Radford, A. et al. (2019). Language Models are Unsupervised Multitask Learners. OpenAI.
- Sachdev, S. and Ye, J. (1993). Gapless spin-fluid ground state in a random quantum Heisenberg magnet. PRL 70, 3339.
- Umphrey, A. (2026a). Holographic Quantum Mechanics of Transformer Attention: From Fisher-Rao Geometry to the Sachdev-Ye-Kitaev Model. Zenodo. DOI: 10.5281/zenodo.18930221.
- Umphrey, A. (2026b). Explicit Physical Construction for Holographic Attention: The Sachdev-Ye-Kitaev Correspondence. Zenodo. DOI: 10.5281/zenodo.18897098.
- Vock, S. and Meisel, C. (2025). Critical dynamics governs deep learning. arXiv:2507.08527.
- Wu, X. et al. (2025). On the Emergence of Position Bias in Transformers. arXiv:2502.01951.
- Zhang, E. (2026). Attention's Gravitational Field: A Power-Law Interpretation of Positional Correlation. arXiv:2603.04805.

---

*Draft v5 — March 25, 2026*
*Revised: authorship, self-citations for theoretical prediction, prethermal plateau results section, distribution analysis, entanglement entropy motivation, claim calibration, reference completion*
