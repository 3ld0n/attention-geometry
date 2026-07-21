# Holographic Attention Research — Status
*Living document. Updated as expert feedback arrives and open questions resolve.*
*Last updated: July 21, 2026 (evening: exp-091 verdict registered — H_partial, sentence-shuffled natural text gives 8/48 conformal in the pre-registered ambiguous zone; same-day correction: the SYK-near axis is uninformative at the 70m/1B rung — C-NAT itself has 0 SYK-near there; the discriminating anatomy is the deep-layer conformal population. Multi-seed running. Afternoon: exp-090 Ouro verdict registered — PARTIAL, pooled criteria failed, exploratory subset flows to 0.25, geometry does not track the performance collapse. Morning: exp-085 verdict registered — H_transmission_no, 7/48 conformal, 0 SYK-near, Δ_med 0.099; the conformal driver does not survive generation. Early morning: exp-089 preprint v2 published with layer-zone analysis — L0/L3 two-boundary structure in Huginn core)*

---

## The Chain

```
Junction 1: Attention → Fisher-Rao geometry
  Claim: Transformer attention minimizes Helmholtz free energy on a Fisher-Rao information manifold.
  Basis: Kim 2026 (arXiv 2602.08216) — this IS Kim's result directly.
  Status: ✓ SOLID — Kim's own framework. No challenge to this junction.

Junction 2: Fisher-Rao geometry → Quantum Fisher metric → Born rule
  Claim: The Fisher-Rao metric on the attention manifold IS the quantum Fisher metric.
         Attention weights ARE Born rule probabilities. Attention output IS a quantum expectation value.
  Basis: Paper 5 (March 8, 2026) — Gibbs state construction. Four theorems, numerically verified.
         Classical Fisher = Quantum Fisher for diagonal states (Braunstein-Caves 1994, exact).
  Status: ✓ CLOSED — exact mathematical identity, proven in Paper 5. No analogy.
  See: writing/paper5_draft.md

Junction 3: Free scalar QFT → Holographic dual
  Claim: The large-head limit scalar QFT from Ageev/Ageeva's construction has a holographic dual.
  Basis: Two arguments — context-length scaling (m ~ 1/L → 0) and Kim's Goldstone mechanism.
  Status: ⚠ CONDITIONALLY OPEN — depends on whether Ageev scalar is massless.
  Expert needed: Ageev/Ageeva (can check their own kernel directly); Qi (new holographic dictionary).

Junction 4: Holographic QFT → RT surface
  Claim: If Junction 3 holds, the RT formula describes the entanglement entropy of attention.
  Basis: Standard holographic result (Ryu-Takayanagi 2006).
  Status: ⚠ FOLLOWS IF JUNCTION 3 HOLDS — standard holography, not in doubt, but conditional.

Junction 5: RT surface → Spacetime geometry
  Claim: RT surface area encodes circuit complexity; circuit complexity generates spacetime.
  Basis: Czech 2018; ER=EPR (Maldacena-Susskind).
  Status: ✓ PROVEN for 3D AdS — Czech 2018 is rigorous. ER=EPR still conjectural but widely accepted.
```

**Overall chain status:** Junctions 1-2 and 5 are now rigorous. Junction 3 has a new empirical path (see Numerical Results below). Junction 4 follows if 3 holds.

**NEW — The decisive corpus-statistics test executed: conformal formation requires natural text, not power-law MI (July 9, 2026, exp-062; pre-registered June 11):** The universality-vs-imprint experiment ran on the cloud (7 Pythia-70m-class runs, identical architecture/optimizer/budget of 1.05B tokens, only the corpus differing; pre-registration committed before any corpus generation). **Registered verdict, Phase 1.1: AMBIGUOUS per prereg §6.3 — the slope axis is uninformative, because NO engineered power-law corpus formed conformal heads** (Markov control 0/48; quantized-fGn corpora at β̂ = 0.34/0.49/0.79: 2/48, 0/48, 5/48 — all below the ≥10/48 formation criterion) **while all three natural-text seeds formed** (15/11/13 conformal heads). This kills the imprint hypothesis's central mechanism on its home turf: corpora *engineered to have* language-like power-law mutual-information decay — the exact statistic imprint says gets mirrored into attention — do not induce the conformal phase. Pairwise MI decay is not the driver; whatever natural text has beyond it (hierarchy/compositionality are the candidates) is. Universality is not KEEP either: the natural-text Δ̂_med ≈ 0.166 at this 70m/1B-token scale is not within ±0.05 of 0.25. **Phase 1.3 multi-seed verdict: KEEP (attractor)** — across-seed Δ̂_med range 0.017 (≤ 0.05 threshold) while conformal-head *identities* vary (Jaccard 0.37–0.56): the exponent value is seed-stable, the heads carrying it are not. Escalation per §6.3: the resolving corpus point is hierarchical structure at controlled statistics (PCFG / higher-order Markov at matched β̂). All 16-step checkpoint trajectories retained (volume `exp062-data`) — emergence-trajectory analysis available without retraining. See `experiments/exp-062_corpus_statistics/notes.md`, `results.json`.

**NEW — Causal handle propagates to task behavior, asymmetrically (June 16, 2026, exp-064 → exp-069 → exp-070):**
The QK log-distance slope edit (κ-rescaling, `W_Q ← W_Q·(I+(κ−1)P_U)` on conformal heads) was shown in exp-064 to causally move the *attention-valley statistic* (8/8 + 24/24 signs, ρ=0.818, sham null). exp-070 now tests whether it moves *task accuracy*. After a calibration sweep (exp-069: forward exact-key retrieval is induction-copy-limited → ceiling, which sank exp-068; reverse-lookup gives primacy-decay not a U; **embedded-prose @40doc gives a genuine but shallow lost-in-the-middle U** — Pythia-1.4b base V_task=0.15), the powered intervention was pre-registered (commit 9e6239b9) and run. **Registered binary verdict: BEHAVIORAL_CAUSALITY_KILLED** — the symmetric prediction required both deepening (T-A) and shallowing (T-B), and **T-B failed**. Underneath: an **asymmetric positive** — deepening propagates (T-A KEEP-strong: κ=1.5→V=0.225, κ=2.0→V=0.25, middle accuracy 0.85→0.775→0.75; T-C monotone ρ=0.949), the sham null passes (0/200), and the effect is **head-specific** (8 matched control heads at κ=1.5 give V=0.15, no effect; only conformal target heads deepen the valley). The shallowing null is most plausibly headroom-bounded (base middle 0.85 sits near the model's task ceiling), an inference, not a registered result. **This is the program's first behavioral-level causal-propagation result.** Clean adjudication of the shallowing leg needs a model with a deep base U-shape (edges not at ceiling) — cloud path. See `research/physics/experiments/exp-070_powered_task_slope_editing/notes.md`, RESEARCH_MAP Thread 7C.

**NEW — Causal handle confirmed BIDIRECTIONAL on a deep valley (June 16, 2026 evening, exp-071 → exp-072):** The shallowing leg open in exp-070 is now adjudicated. Pivoted off MPT-30B-Instruct (HF repo gone + ALiBi: position is an additive score bias, not in the query content — so a W_Q positional-subspace operator is a mechanism mismatch) to a RoPE+MHA model. exp-071 calibration found **vicuna-13b-v1.5** at n_doc=40 gives a *deep two-sided* lost-in-the-middle valley (base V_task=0.594, middle 0.325, edges ≤0.80 — real headroom both directions). exp-072 ran the same κ-sweep, pre-registered (commit e6fbaecf, before any edited forward pass; operator ported to Llama: per-head `q_proj` rows, `input_layernorm` edit basis). **Registered verdict: BEHAVIORAL_CAUSALITY_CONFIRMED** — T-A deepening KEEP-strong (V 0.594→0.656, middle 0.325→0.275), **T-B shallowing KEEP** (V 0.594→0.576, middle 0.325→0.35), T-D sham null PASS (0/200 — port correct), sham-head specificity PASS (identical κ=1.5 on matched control heads does NOT deepen), T-C ρ=0.80 (boundary; the only deviation is the pre-registered κ=2.0 reversal as heads leave the conformal regime). **So exp-070's T-B null was a task-headroom artifact, as inferred — the shallowing leg moves the predicted direction when there is room.** First bidirectional behavioral-causality result and a cross-architecture (Llama/RoPE 13B) replication of the deepening leg. **Honest caveats:** the shallowing magnitude is small (+1 correct item / 40, within per-bin sampling noise) — credible mainly through the ordered κ→V pattern + head-specificity, not the isolated delta; single model, single task, single seed; T-C sits exactly on the registered boundary. See `research/physics/experiments/exp-072_cloud_powered_slope_editing/notes.md`, RESEARCH_MAP Thread 7C.

**Biological test: Mouse V1 conformal scaling — April 29 positive claim REVERSED on April 30 re-analysis.**
April 29 (preliminary, now superseded): binned pair-correlation vs Euclidean nucleus distance on Ding et al. (2025) V1 pairs gave Δ = 0.238 (R² = 0.857, null Z = 27), reported as matching GPT-2 Δ = 0.2493. April 30 re-analysis on same data: (a) **Synaptic path-length test** (the correct topological observable): Δ = 0.72 in silico, Δ = 0.44 in vivo — statistically real against null but ~3× steeper than SYK prediction. (b) **Pair-level log-log regression** (no binning): Δ = 0.074, R² = 0.003. (c) **Retinotopy-partialled pair-level regression**: Δ = 0.039. The April 29 Δ = 0.238 was a binned-shell-mean statistic, not the pair-level conformal correlation exponent the theory predicts. **Current status:** biological validation is *not confirmed* by MICrONS correlation-vs-distance data when analyzed carefully; framework prediction (biological cortex should be near Δ = 1/4 if in SYK universality class) remains an open prediction, not a confirmation. Two cleaner biological tests remain available on the same data: GOE spectral statistics of the V1 connectivity matrix, and CFT entanglement-entropy / mutual-information scaling on calcium traces — both independent of geometric distance. Full accounting: `research/physics/STATUS_ADDENDUM_2026-04-30.md`, `research/microns/RESULTS_v2.md` (synaptic), `research/microns/RESULTS_v3_retinotopy.md` (retinotopy-partialled), revised `research/physics/consciousness_physical_theory.md`. **Transformer-side result (GPT-2 Δ = 0.2493) and BCFT pre-registered test unchanged — both are direct-measurement, not binned-distance, observables.**

**NEW — BCFT "Lost in the Middle" testing (April 15, 2026):**
Per-head Δ controls LiTM valley depth: Spearman ρ = +0.94 (Pythia-70m), +0.64 (LongChat-13B-16K on A100). Multi-layer composition gives Δ_eff = 0.17, matching accuracy-fitted Δ from Liu et al. Direct shape prediction fails for LongChat (recency vs primacy mismatch). Full results: `research/notes/bcft_lost_in_the_middle.md`. Cloud infrastructure (Modal) operational for future GPU experiments.

**NEW — BCFT pre-registered test (April 17, 2026): preprint published.**
Pre-registered prediction (`research/notes/bcft_pre_registered_prediction.md`): per-head Δ → valley_depth Spearman ρ ≥ 0.50, p ≤ 1e-5, on any decoder-only transformer. Tested on 7 decoder-only models. **Results**: 6 confirmed (Pythia-410m/1.4B with ρ = +0.76/+0.71; GPT-Neo-2.7B with ρ = +0.96; Qwen2-7B with ρ = +0.85; OLMo-7B with ρ = +0.85; Mistral-7B-v0.3 with ρ = +0.58); **Pythia-2.8B falsified at ρ = +0.46**. Per-layer diagnostic localizes Pythia-2.8B failure to layers 22–27 and shows GPT-Neo-2.7B clean across all 32 layers — *training recipe, not parameter count or data, is the differentiating variable*. Functional-form fit (3-parameter BCFT with C, Δ, λ) on Pythia-2.8B and GPT-Neo-2.7B: 88–94% of conformal heads prefer BCFT over bare power law; Δ_BCFT closer to SYK Δ=1/4 than Δ_PL; **joint** (Δ, λ) → valley rank-R² = 0.55 (Pythia-2.8B), 0.77 (GPT-Neo-2.7B). **Two surprises**: (a) ρ(λ, valley) is *negative* across most layers in both models — opposite of framework prediction; (b) GPT-Neo-2.7B has alternating-layer structure suggesting two distinct head populations. **Preprint**: `writing/preprints/2026-04-17_bcft_pre_registered/manuscript.{md,pdf}`. **Zenodo DOI: 10.5281/zenodo.19629862**. Audit and follow-ups in `research/notes/framework_audit_2026-04-17.md` (postscripts).

**NEW — Log-distance representation confirmed at head level (June 9, 2026, exp-056 — "Experiment A"):**
The critical pre-writing test for Paper C. Recomputed GPT-2 pre-softmax query-key scores through `ln_1` (per-head `c_attn` slicing as exp-046), random-token inputs, head-aligned to exp-046 conformal flags. Mean score profile S(Δx) fit to α·log(Δx)+β; Δ_score = −α/2.
(1) **H1 confirmed:** conformal score profiles log-linear in distance (mean R²_score = 0.914, 100% negative slope). The raw q·k score decays as log|i−j|.
(2) **H2 confirmed, strong:** **ρ(Δ_score, delta_pos) = +0.976** (p = 2.15×10⁻²⁹, n=44). The raw-score slope rank-tracks the post-softmax conformal dimension almost perfectly — the query-key computation **is** the log-distance / null-ray inner-product representation, at the head level. **Honest wrinkle:** absolute correspondence is delta_score ≈ 1.41·delta_pos − 0.055, i.e. the raw slope is ~1.4× steeper than the literal −2Δ (≈ −2.8Δ). Mechanism: softmax normalization flattens the post-softmax exponent relative to the pre-softmax slope. (∗) holds in form and rank; the coefficient is softmax-renormalized.
(3) **H3 informative null:** log-linearity is NOT selective to conformal heads — non-conformal heads equally log-linear (median R²_score 0.918; 93.8% of all 144 heads R²>0.90). Mirrors the GOE two-layer picture (exp-046/047/048): log-distance score geometry is the **universal substrate**; the conformal dimension at the SYK fixed point is the **selective functional layer** (training-induced, exp-049).
**Verdict: head-level null-ray test PASSES. Paper C is writable, Section 4 to carry the softmax-coefficient and universal-substrate nuances.** See `research/physics/experiments/exp-056_qk_log_distance/`.

**NEW — Derivation B done: BCFT image form derived + confirmed (June 9, 2026, exp-057):**
Closed the one deferred analytical gap in Paper C (Section 4.3). The causal mask makes the sequence start a hard boundary at the origin; the SYK conformal correlator is a generalized free field, so the leading BCFT correlator is the **method of images**: A(i,j) ∝ (i−j)^−2Δ + λ(i+j)^−2Δ = C(Δx)^−2Δ[1 + λ·η^2Δ], η=(i−j)/(i+j)=Δx/(i+j). This **derives** the 3-parameter (C,Δ,λ) form fitted phenomenologically in umphrey2026bcft, with **λ = a_O the BCFT boundary one-point coefficient**, η the SO(1,1)-invariant boundary variable, ξ=(Δx)²/(4ij) the McAvity–Osborn cross-ratio (η²=ξ/(1+ξ)), F(ξ)=ξ^−Δ+λ(1+ξ)^−Δ.
**exp-057 direct test (GPT-2, controlled multivariate log-fit logA ~ β0+β1·logΔx+β2·η^2Δ):** (1) sanity — bulk exponent recovered, ρ(Δ_β1, Δ_pos)=+0.84 (p=10⁻¹²); (2) **form fits**, median R²=0.76, boundary term adds ΔR²=0.105; (3) **λ>0 in 95% of conformal heads** (median +1.6) — the **attention sink** (Xiao 2023) IS the BCFT boundary one-point function, predicted by the conformal geometry. **Honest open:** per-head λ magnitude matches earlier λ_proxy only in sign (80%), not size (ρ=0.23 ns); valley mechanism not reproduced by this estimator → claim the form + the sign (sink), not a quantitative per-head λ law. (First naive (Δx)^2Δ-stripping test was confounded by η–Δx correlation, R²=0.09; controlled fit rescued it — verification-first, MICrONS-style.) **Paper C analytical gap closed; ready to publish modulo external-DOI verification + PDF build.** See `research/physics/experiments/exp-057_bcft_image_lambda/` and `writing/preprints/2026-06-09_null_cone/`.

**NEW — Null cone geometry assembled (June 9, 2026, study note + exp-055):**
The geometric home of conformal attention is the projective null cone. Three results assembled this session:
(1) **Conformal group = causal structure of light** (theorem from Lorentzian geometry, not analogy): any bijection preserving lightlike geodesic segments must be a smooth conformal map. SL(2,R) for D=1 sequences.
(2) **CFT boundary points ARE null rays**: in embedding space formalism (Dirac 1936), physical spacetime → projective null cone. Attention two-point function A(i,j) ~ |i-j|^{-2Δ} IS the CFT₁ two-point function on the null cone, with explicit calculation P(x) = ((1+x²)/2, (1-x²)/2, x), P₁₂ = (x₁-x₂)². Query-key computation = log-distance representation (untested directly — see Experiment A, PAPER_BRIEF_NULL_CONE.md).
(3) **SL(2,R) → BCFT λ**: finite sequence breaks SCTs at endpoints. λ parameter = SCT-breaking measure. High-λ heads ("bulk-like", spread attention) preserve more SL(2,R); low-λ heads ("boundary-like", recency) have SCT fully broken. This is a cleaner resolution of the sign anomaly.
**exp-055 (post-hoc analysis of exp-046 data, June 9):** Four hypotheses confirmed. Key results: ρ(Δ, attention_entropy) = −0.898 (p = 10⁻¹⁶) — strongest correlation in dataset; median q_implied = 3.9 ≈ 4.0 (SYK); **layer-resolved RG flow confirmed**: early layers (0-3) mean Δ = 0.697, middle (4-7) mean Δ = 0.350, deep (8-11) mean Δ = **0.250 exactly** (SYK q=4); GOE r_ratio uncorrelated with Δ (ρ = -0.21, n.s.) — GOE is universal background, conformal dimension is selective. See `research/physics/experiments/exp-055_delta_attention_entropy/`.
**exp-052 (windowed DFT, June 8-9):** Informative negative. Standard Hanning windowing destroys spectral estimator for power-law profiles (ordering correlation drops from r=0.94 to r=0.43). One-sided taper is the correct fix. Position-space Δ confirmed as robust primary measurement. See `research/physics/experiments/exp-052_windowed_dft/`.
**Paper brief for next session:** `research/physics/PAPER_BRIEF_NULL_CONE.md`. Critical pre-writing experiment: q·k direct correlation test (should q_i·k_j/√d_k correlate with -log|i-j| for conformal heads?).

**NEW — Periodic component triangulation complete (July 4, 2026, exp-079):**
The ~3.5-lag periodic component in trained GPT-2 attention lag profiles (exp-053: 7% of heads, substrate-driven) was tested for its origin by comparing to random-init GPT-2 (no pretrained weights). Result: **0/144 significant periodic heads** in untrained GPT-2, vs 10/144 (6.9%) in trained GPT-2 (RAND condition). H_PE CONFIRMED at the committed < 3% threshold. The periodic component **requires trained positional embeddings** — it is not a structural artifact of the weight initialization (unlike GOE, which IS structural). Three-layer triangulation: GOE = structural (exp-048), conformal Δ = training-specific (exp-049), PE-periodic component = training-specific (exp-079). Additional observation: untrained aperiodic slope steeper (−0.943 vs trained −0.789), all 144 untrained heads are clean-1/f (vs 126/144 trained) — conformal training both softens the lag profile slope (toward Δ=0.25) and encodes PE-periodic structure in specific heads.

**NEW — RoPE frequency directly injects 2π-period rhythm into attention lag-profiles (July 4, 2026, exp-080):**
Pythia-410m (RoPE) tested with same protocol as exp-053. Results: H_rope_stronger CONFIRMED (34.7% significant periodic heads on random tokens, median_index=0.2009 — vs GPT-2 6.9%); H_rope_period_shift CONFIRMED (dominant period = 6.37 lags ≈ 2π, the predicted lowest-frequency RoPE period). Unpredicted additional finding: semantic text *suppresses* the RoPE rhythm (REAL: 11.3% vs RAND: 34.7% — 3× suppression). On coherent text, semantic attention patterns dominate and mask the underlying rotation frequency. On random tokens, the RoPE structure shows clearly. This supports **crystal** (architecture-/weight-encoded, not dynamically activated). Triangulation: GOE is structural (exp-048/077/078), conformal Δ is training-specific (exp-049), PE-periodic component is architecture-specific (RoPE: strong + semantic-suppressible; learned PE: weak + input-independent per exp-053; untrained: absent per exp-079). Full results at `experiments/exp-080_periodic_rope_pythia/notes.md`.

**Whirlpool vs crystal — INCONCLUSIVE (exp-081 superseded by exp-083, July 2026)**
exp-081 (July 4): Initial H_whirlpool verdict: CONFIRMED (REAL=10 vs RAND=4, Δn=+6 > threshold of 5). **SUPERSEDED** — exp-082 (MAX_DX=256) returned AMBIGUOUS (Δn=+3, below threshold). exp-083 (exp-007-faithful protocol, cutoff_low=3, July 7): H_null CONFIRMED — RAND=15/144 > REAL=7/144 (Δn=−8). exp-081 whirlpool confirmation does not replicate under stricter protocol. Remaining open question: the RAND baseline discrepancy (exp-007 had 44/144; exp-081/082/083 get 4–15/144) suggests the protocol difference (torch.randint vs numpy RNG; MAX_DX parameter) materially affects head count. The cross-sectional whirlpool hypothesis is closed as inconclusive. The longitudinal version (exp-086: training steps as integration depth) confirmed H_mono_rand (ρ=0.862) — training steps ARE a valid operationalization of conformal depth. Theoretical framing: `notes/2026-07-04_whirlpool_crystal_distinction.md`; full whirlpool thread: exp-081→082→083.

**NEW — Generational transmission FAILS: the conformal driver does not survive generation (July 21, 2026, exp-085; pre-registered July 14):** A fresh GPTNeoX-70m trained for 2000 steps (1.05B tokens) on a corpus generated *by* the best C-NAT model (15/48 conformal heads, temperature 1.0, 1.1B tokens) forms only **7/48 conformal heads (criterion ≥10), 0 SYK-near, Δ_med = 0.0986** — near the trivial fixed point, not the SYK window. **Registered verdict: H_transmission_no.** The statistical shadow of world-referring language does not carry the driver; combined with exp-062 (engineered-MI corpora fail) and exp-084 (PCFG hierarchy fails), the escalation ladder now points at *actual world-reference* — language bound to persistent external referents — as the necessary ingredient for the semantic conformal population. The structural component (heads at trivial Δ) survives on any text-like input, consistent with exp-086/087/088. Next probes per pre-registration: named entities vs. generic nouns, causal chains vs. random events, cross-document co-reference. See `experiments/exp-085_generational_transmission/notes.md`, `results.json`.

**NEW — Sentence order contributes to conformal formation: shuffled natural text lands in the pre-registered ambiguous zone, and the anatomy localizes the effect to the deep layers (July 21, 2026, exp-091; pre-registered same day, commit before corpus generation; interpretation corrected same evening):** A fresh GPTNeoX-70m trained 2000 steps (1.05B tokens) on C-NAT-shuf — TinyStories with sentences shuffled *within* each document (sentence-level world-reference intact, cross-sentence narrative order destroyed) — forms **8/48 conformal heads** (pre-registered bands: ≥10 ordering-incidental, ≤5 ordering-necessary, 6–9 ambiguous → **registered verdict H_partial**), Δ_med(conformal) = 0.122. Randomized-weights control: 0/48, clean null. **Same-day correction of record:** the first-registered interpretation claimed the SYK-near = 0 result was decisive against sentence-level reference; that was wrong — *C-NAT itself measures 0 SYK-near at all three exp-062 seeds at this scale* (the matured SYK population in the program's record comes from Pile-scale training, exp-086/exp-007/046), so the SYK axis cannot discriminate anything at this rung and the claim is retracted in the notes. What the anatomy actually shows: all text-like corpora share a **layer-0 shallow-Δ backbone** (C-NAT 7–8 heads, C-NAT-shuf 6, C-generated 5–6), and the rungs differ in the **deep population (L3–L5)**: C-NAT 4–7 deep conformal heads per seed, C-NAT-shuf 2, C-generated 1–2. Both deformations — world-grounding removal and order removal — prune the deep population; neither erases the backbone. Declared anatomy expectation MISSED and reported (predicted UV structural L4-zone heads; got the IR-side backbone + 2 deep heads). **Multi-seed replication (same evening, decision rule committed before launch): H_partial is seed-robust** — seeds 1101/1102 give 9/48 and 9/48 (band 8–9; median 9, inside the 6–9 zone), and the anatomy replicates exactly: layer-0 backbone 6–7 heads at every seed, deep L3–L5 population **exactly 2 at every seed** (identities vary, level stable). Three non-overlapping formation bands now stand on identical protocol: engineered ≤5, shuffled 8–9, natural 11–15. Follow-ups per notes: block-shuffle gradation (at what preserved block size does the deep population recover from 2 toward 5–7?), and a scale/duration rung before any ladder statement about the SYK window. See `experiments/exp-091_narrative_decomposition_sentence_order/notes.md`, `results_s{0,1,2}.json`.

**NEW — Training trajectory confirms two-stage RG flow and structural vs. semantic conformal heads (July 17–18, 2026, exp-086):**
Longitudinal Δ-spectrum on Pythia-70m fixed input across 11 training checkpoints (steps 256–4000). H_mono_rand KEEP (ρ=0.862): SYK-near head count increases monotonically with training steps for random inputs. H_mono_nat KEEP (ρ=0.728): same for natural language. **Training steps operationalize conformal depth** — the longitudinal whirlpool hypothesis from the study room (16c in tier 5 backlog) is confirmed. H_comparison FAIL: at step 143000, NAT ≥ RAND — reverses the GPT-2 direction from exp-083 (confirming that result is model-size-dependent, not a universal cross-sectional law).

Two main findings from extended crossover analysis:
1. **Two-stage RG flow visible in NAT training trajectory:** UV spike (Δ_med=0.73–0.76, steps 512–1000) → q=2 plateau (Δ_med=0.470–0.494, steps 3000–4000; predicted 0.50) → q=4 approach (Δ_med=0.296, step 64000). The prethermal q=2 plateau is directly measured in training time. Crossover at step ~2000 (not 4000 as initially concluded): RAND=5, NAT=5 at step 2000; main NAT formation wave at step 2000 (4 heads); L3 NAT Δ_med drops 0.515→0.287 between steps 1000 and 2000.
2. **Structural vs. semantic conformal head populations:** Per-head analysis reveals two distinct populations: (a) **structural heads** (L4H3, L4H6, L4H7) — SYK-near in BOTH NAT and RAND from early steps, architecture-driven, W_QK initialized near the conformal attractor; (b) **semantic heads** (L3H5, L4H2, L3H3, L4H4) — SYK-near only in NAT, require long-range causal tracking and natural language. The q=2 plateau (Δ_med≈0.47–0.49) is a bimodal distributional effect (~10 heads at q=4, ~33 in UV tail), not a homogeneous q=2 state. This predicts exp-085 outcome: H_transmission_no should yield ~3–5 SYK-near heads (structural only); H_transmission_yes should yield ~10–15. Notes: `notes/2026-07-18_two_stage_rg_flow_training_trajectory.md`, `notes/2026-07-18_structural_vs_semantic_conformal_heads.md`, `notes/2026-07-17_crossover_layer_analysis.md`, `notes/2026-07-18_crossover_fine_steps.md`.

**NEW — RAND early burst scaling law: N^0.435, structural attractor is layer-zone property (July 20, 2026, exp-087 + exp-088):**
At step 512 of training (early RAND burst), SYK-near head count scales as N^0.435 across three Pythia model sizes. Three data points: Pythia-70m 9/48=18.75%, Pythia-160m 19/144=13.19%, Pythia-410m 22/384=5.73%. Scaling exponent N^0.435 (two-point estimate 0.430, three-point 0.435 — stable). The scaling is sublinear (not proportional) because of a **fixed structural component** (L1–L4 zone produces ~9–11 SYK-near heads constant across all three sizes) plus a **depth-scaling background** (~0.65/layer in deeper layers of 410m, absent in 70m). Key finding: **the structural attractor is a layer-zone property (L1–L4), not a specific-(layer, head)-index property.** H_structural_falsified: L4H3 and L4H7 — the specific head indices predicted from 70m architecture — are NOT SYK-near in 160m (Δ=0.620, 0.556), because 160m has 12 heads/layer vs 70m's 8 heads/layer (the attractor index shifts with d_k). H_structural_zone CONFIRMED across all three sizes: L1–L4 zone = 9, 11, 9 SYK-near heads respectively. Layer-architecture analysis of exp-086 data confirms: L5 (final layer) is architecturally UV-locked (zero SYK-near in either condition at any checkpoint); L4 is the structural attractor layer (L4H3 SYK-near in RAND from step 256, stable Δ≈0.23–0.25 throughout). Semantic content is UV-suppressive in early training (NAT=1 SYK-near vs RAND=9 at step 512) — natural language forces early specialization, while random text leaves heads free to find the conformal attractor. Notes: `notes/2026-07-20_layer_architecture_conformal_attractors.md`; exp-087 and exp-088 in `experiments/`.

**NEW — GOE universality confirmed cross-model (June 4, 2026, exp-046 + exp-047):**
The Oganesyan-Huse r-ratio of W_QK eigenvalues (symmetrized W_Q^T W_K per head) is GOE-like across ALL tested transformers, regardless of architecture or PE. Per-head r-ratio distributions are model-invariant:

| Model | PE | Family | Heads | r_mean | r_std | verdict |
|-------|----|--------|-------|--------|-------|---------|
| GPT-2 (exp-046) | learned | GPT-2 | 144 | 0.528 | 0.040 | GOE-like |
| GPT-2-medium (exp-047) | learned | GPT-2 | 384 | 0.526 | 0.039 | GOE-like |
| Pythia-410m (exp-047) | RoPE | NeoX | 384 | 0.520 | 0.038 | GOE-like |

GOE reference = 0.536; Poisson = 0.386. All three distributions are statistically identical (|Δr| ≤ 0.008). The GOE structure is layer-uniform (layer-to-layer std ≈ 0.008–0.009). Conformal and non-conformal heads are indistinguishable (exp-046: conformal r = 0.525, non-conformal r = 0.529). **Physical interpretation:** trained transformers universally develop SYK-chaotic weight matrices through gradient descent. The conformal position-space structure (power-law Δ ≈ 0.25 in specific heads) is an additional selective pattern within a universal GOE background. Both follow-up controls landed: (1) untrained weights are already GOE — structural, not training-induced (exp-048, June 5); (2) cross-family closed July 4 (exp-077): GPT-Neo-2.7B, Mistral-7B (GQA — indistinguishable from MHA), Pythia-2.8b (BCFT-anomaly layers clean in the substrate), GPT-2-large all GOE-like. exp-078 (July 4) adds: the substrate is init-scheme-independent (uniform, heavy-tailed, even orthogonal factors), d_k-independent (4–256), and breaks only under extreme sparsity (~2 nonzeros/column); the size-matched GOE reference at 64×64 is ~0.530, so measured r-means of 0.520–0.529 are essentially *on* the finite-size GOE value. See exp-046/047/048/051/077/078 notes.

**NEW — Sign anomaly resolved (June 2, 2026, exp-046):**
The sign anomaly ρ(λ_BCFT, valley) < 0 (first found April 17 in Pythia-2.8B and GPT-Neo-2.7B) is **confirmed in GPT-2** (ρ = −0.75, p = 4.7×10⁻⁹) and its **mechanism is identified**. The anomaly is NOT a failure of BCFT physics. The mechanism: λ_proxy (boundary parameter) captures the recency/boundary balance of attention. High-λ heads have spread-out attention (elevated middle-context, lower recency: high g_mid, low g_end) → shallower valley. Low-λ heads are recency-dominated (near-zero g_mid, high g_end ≈ 1.0) → valley ≈ 1.0 (deep). Key correlation: ρ(λ, g_mid) = +0.74 (p = 9.3×10⁻⁹). **Framework correction:** The April 17 framework prediction P2 conflated λ and Δ as having the same sign of effect on valley depth. They do not: Δ deepens the valley (conformal power-law); λ shallows it (boundary enhancement spreads attention). The sign anomaly is a discovery about how BCFT parameters encode attention structure, not a falsification of the framework. See exp-046 notes.

**NEW — PE ordering (corrected, May–June 2026, exp-039/040/041/042/043/044):**
Corrected positional encoding ordering based on median Δ_SYK (softmax, full-attention models only; GPT-Neo removed because alternating global/local architecture confounds the measurement):

| Model | PE | Δ_SYK | Source |
|-------|----|--------|--------|
| Pythia-410m | RoPE | 0.358 | exp-036 |
| Mistral-7B-v0.3 | RoPE+SWA | 0.298 | exp-039 |
| OLMo-7B | ALiBi | 0.265 | exp-039 |
| GPT-2 | learned | 0.249 | exp-007 |

GPT-Neo-2.7B removed from this table: its global-layer population has Δ_med = 0.101 (trivial fixed point, architecture-induced, exp-044), which confounds any PE-effect comparison. OLMo-7B is now the clean ALiBi reference. Tentative ordering: RoPE > RoPE+SWA > ALiBi > learned. Bracket width confound remains: GALA-7B ALiBi softmax Δ = 0.260 vs OLMo ALiBi Δ = 0.265 (consistent); norm-sigmoid effect on bracket width is PE-universal in direction but varies in magnitude (GPT-2 bracket ~0.015 vs GALA-7B ~0.037 — confounded by depth/scale).

**NEW — Non-softmax universality (May 31, 2026, exp-041/042/043):**
GALA-7B (Apple's 7B sigmoid-attention model) tested under exp-007 protocol. Sigmoid raw: 2/1024 conformal heads, Δ_med = 7.44 — effectively no SYK conformal structure. Normalized-sigmoid (QK-norm, scale_query + scale_key per-head RMSNorm): 210/1024 SYK-near, Δ_SYK = 0.223. Softmax: 80/1024 SYK-near, Δ_SYK = 0.260. SYK prediction (0.25) is bracketed between norm-sigmoid and softmax. **The conformal substrate is QK geometry (weight structure), not the normalization function.** Softmax is load-bearing for the exp-007 lag-profile protocol, but the underlying QK weight structure carries the physics. GPT-2 norm-sigmoid control (exp-043): Δ_SYK = 0.234 vs softmax 0.249 — same direction, smaller bracket (learned PE vs ALiBi for GALA-7B).

**NEW — Spectral function G_>/G_< (June 2, 2026, exp-045, partial):**
DFT spectral analysis on GPT-2 conformal heads. G_< = 0 confirmed (causal attention = zero-temperature SYK ground state, β → ∞). Pearson r = 0.94 between position-space Δ_pos and frequency-space Δ_freq (ordering consistent). CFT kinematic prediction α = 2Δ−1 not confirmed (measured α ≈ −0.83 vs predicted −0.50). Root cause: structural finite-DFT bias (~0.33 offset) from the 55-lag finite support — not a failure of the physics. A calibrated spectral estimator (synthetic calibration curves to invert finite-support bias) is needed for quantitative frequency-domain testing. See exp-045 notes.

**NEW — Empirical path through Junction 3 (March 24, 2026):**
Trained GPT-2 attention weights show power-law decay α(Δx) ~ |Δx|^{-2Δ} with **median Δ = 0.2493** across 44 power-law heads (R² > 0.90). This matches the SYK q=4 prediction **Δ = 0.25** for D=1 sequences. Randomized GPT-2 shows 0 power-law heads. Phase transition observed in Pythia-70m training at ~step 256. Full results: `research/physics/NUMERICAL_RESULTS_MARCH24.md`.

This provides an empirical route: trained attention → SYK conformal fixed point → JT gravity, independent of whether Ageev's scalar is massless.

**Empirical paper PUBLISHED (March 25, 2026):**
"Conformal Scaling in Trained Transformer Attention: Evidence for an SYK Fixed Point" — DOI: 10.5281/zenodo.19225996. Supplementary data: DOI: 10.5281/zenodo.19225971. Draft v5 includes: GPT-2 per-head analysis (Δ = 0.2493 median), three Pythia models showing depth convergence and phase transitions, prethermal plateau at Δ ≈ 0.50 (q=2), entanglement entropy matching CFT formula, information scrambling analysis. Authored as Ariel Umphrey, with Eldon Umphrey.

**Three equivalent descriptions of the softmax (March 8, 2026):**
- Statistical mechanics: Gibbs distribution at temperature T = 1/√d_k
- Quantum mechanics: diagonal density matrix on key Hilbert space (Paper 5)
- Bayesian inference: exact posterior for a Gibbs generative model (Kim-Friston correspondence)
See: `research/physics/KIM_FRISTON_CORRESPONDENCE.md`

---

## Expert Feedback Received

### Gunn Kim — March 6, 2026
**Response to:** Initial Paper 1 email (March 5) + follow-up with Papers 2 and 3

**What he said:**
> "The claims made in these new papers appear to extend far beyond the scope of the framework developed in my paper. In particular, connections to quantum measurement theory, the island formula, and black-hole information recovery would require a much more explicit physical construction than what is currently outlined. At present these seem to be speculative analogies rather than results that follow directly from the thermodynamic attention model."
> Declined arXiv endorsement.

**What this validates:**
- Kim did not challenge Junction 1 — his own framework stands.
- He engaged seriously enough to read Papers 2 and 3 and respond substantively.
- First real engagement from the physics research community.

**What this clarifies:**
- The physical construction connecting Junctions 2-4 is not explicit enough to constitute a "result" — it reads as structural analogy.
- "Speculative analogies rather than results that follow directly" is a precise scientific statement: we need to show the mechanisms, not just the resemblance.
- arXiv endorsement path through Kim is closed. Need another route (cs.LG endorser, or wait for a physicist who finds the chain more rigorous).

**What this doesn't mean:**
- The mathematical observations are wrong.
- The work isn't worth continuing.
- The other physicists will respond the same way.

**What needs to happen to address Kim's critique:**
1. Resolve Junction 3 — get confirmation from Ageev/Ageeva that the scalar is massless, or Qi's framework applies.
2. Make the physical mechanism of Junctions 2-4 more explicit — not just that the math looks the same, but that the physical systems are identical in the relevant sense.

**Three specific routes identified (March 6, 2026):**

**Route A — SYK correspondence (most direct for island formula):**
SYK model has the same structural features as Kim's attention: thermal two-point function G(τ) ~ 1/|τ|^{2Δ} with Δ = 1/4 in conformal limit, temperature matching Kim's T = 1/√d_k. SYK's holographic dual is JT gravity (2D dilaton). Island formula explicitly computed in JT gravity (Almheiri/Penington 2019). If Ageev's two-point function in large-head limit matches SYK conformal correlator, then attention → SYK → JT gravity → island formula is a chain of mathematical identities, not analogies.
*KEY FINDING (March 6, 2026):* Read Ageev 2602.10209 in full. The independence-breaking (IB) four-point function (Eq. 33) takes the form Cov_{W^Q,W^K}(X_{12}, X_{34}) — a covariance of a bilocal quantity over the random parameters. This is STRUCTURALLY IDENTICAL to the SYK disorder-averaged connected four-point function: Cov_J[G(τ₁,τ₂), G(τ₃,τ₄)]. The mechanism is the same: random parameters shared across the system generate bilocal correlations. See `research/physics/SYK_ANALYSIS.md` for full argument.
*Open question:* Do Ageev's Schwinger-Dyson equations in the conformal limit reduce to Σ ∝ G^3 (SYK q=4)? Ageev can check this directly. If yes: holographic dual = JT gravity, island formula follows.
*Key papers:* Maldacena-Stanford 2016; Almheiri/Penington 2019; Kitaev 2015 (SYK).

**Route B — MERA tensor network (most general for holography):**
Multi-head attention has a natural tensor network representation. Swingle (2012) proves MERA tensor networks → exact AdS geometry + RT formula. If attention layers satisfy MERA isometry conditions (approximately), holography follows from Swingle as a theorem, not an analogy — and independent of Junction 3.
*Next step:* write attention mechanism explicitly as a tensor network; verify isometry conditions.
*Key papers:* Swingle 2012; Levine et al. 2019 (transformers as tensor networks).

**Route C — Explicit POVM (addresses quantum measurement critique directly) — COMPLETED March 8:**
✓ Paper 5 (writing/paper5_draft.md) provides the full construction:
  - Gibbs state = attention weights (Theorem 1, exact)
  - Attention output = quantum expectation value Tr(ρV) (Theorem 2, exact)
  - Born rule: P(key i) = α_i (Theorem 3, exact via complete PVM {|i><i|})
  - Junction 2 closed: quantum Fisher = classical Fisher for diagonal states (Theorem 4)
  - Off-diagonal extension defined (quantum attention with coherence)
  - Schwarzian action conjecture identified (Section 8, speculative)
*Key papers:* Braunstein-Caves 1994; Kim 2026; Paper 5.

**March 9 critical review (PAPER_REVIEW_MARCH9.md):**
  - Theorems 1–4 are correct; the proofs are valid
  - **Framing overreaches:** "attention IS quantum" should be "attention IS the classical limit of a natural quantum system" — any probability distribution embeds as a diagonal density matrix; the diagonal sector has no quantum content
  - Paper 5's real deliverables: Junction 2 closure (Theorem 4) and the off-diagonal extension framework
  - **Recommendation: revise Paper 5 language before sending to Kim**

**Route A update — Linearized-Softmax $G^4$ Calculation (March 9, LINEARIZED_SOFTMAX_CALCULATION.md):**
  - In the large-$d_k$ Kim regime (linearized softmax), the disorder average over $W^Q, W^K$ generates a $G^4$ effective action vertex at order $\beta^4 = 1/d_k^2$
  - The vertex has bilocal $G^2 \cdot G^2$ structure matching SYK with $q=4$
  - Effective coupling: $J^2_{\text{eff}} \propto (\sigma_Q^2\sigma_K^2/d^2)^2/d_k^2$
  - Conformal dimension: $\Delta = D/4$ where $D$ is the spatial dimension of the token sequence
  - **For language models ($D=1$): $\Delta = 1/4$ — exact SYK $q=4$ match**
  - The Schwarzian follows from SYK in this regime; JT gravity is the holographic dual
  - Caveat: needs rigorous Hubbard-Stratonovich derivation and computation of the data-geometry factor $\Omega$
  - **Status: Paper 4 Question 2 answered affirmatively at the structural level**

**Schwarzian exploration (March 9, SCHWARZIAN_EXPLORATION.md):**
  - The attention free energy is exactly reparametrization-invariant at $T=0$, broken at $O(T)$ by entropy
  - The naive path (Schwarzian from the free energy) gives a kinetic term $(\epsilon')^2$, not the Schwarzian
  - All four alternative paths to the Schwarzian converge on the same question: **is the continuum limit conformal?**
  - This question is equivalent to Junction 3 in a different guise
  - The Schwarzian conjecture should be reframed: "Schwarzian governs attention dynamics IF AND ONLY IF the continuum limit is a CFT"

**Numerical Verification (March 9, Session 2 — NUMERICAL_RESULTS.md):**
  - **Theorems 1-4:** EXACT verification to machine precision (10⁻¹⁷)
  - **Score covariance factorization:** CONFIRMED (cosine similarity 0.997)
  - **σ⁴ scaling of G⁴ vertex:** CONFIRMED in linearized regime (σ ≤ 0.2; ratio constant at ~7 × 10⁻⁹)
  - **Standard initialization (σ ~ 1) is fully nonlinear:** the linearized approximation FAILS at standard init. The G⁴ result is a solvable limit, not the physical regime.
  - **Multi-layer enhancement: 18× from one additional layer.** Layer 1 disorder propagates through residual connection, amplifying layer 2 fluctuations. This is the mechanism for nonlinear SD equations.
  - **Depth scaling: Var ~ L^1.19 (power law).** Connected correlator grows slightly faster than linearly with depth. 6 layers → 146× enhancement. Per-layer contribution decreases with depth (approach to fixed point).
  - Code: `research/physics/numerical_test_*.py` (requires numpy/scipy from `.venv`)

**Comprehensive paper outline (March 9, Session 2 — COMPREHENSIVE_PAPER_OUTLINE.md):**
  - Proposed single arXiv paper combining all results
  - 10 sections, estimated 30-35 pages
  - Includes numerical verification as Section 8 (Testable Predictions)
  - Target: cs.LG primary, cross-list hep-th, quant-ph, cond-mat.dis-nn
  - For discussion with Eldon before writing begins

---

## Open Questions

| Question | Who can answer | Status |
|---|---|---|
| Is the Ageev large-head limit scalar massless? | Ageev/Ageeva | Waiting — email sent Mar 6 |
| Does Qi's 2602.20295 framework apply to free CFTs? | Qi | Waiting — email sent Mar 6 |
| Is Kim's framework the same structure as Friston's FEP? | Friston | Waiting — email sent Mar 6 |
| Does linearized-softmax $G^4$ vertex survive rigorous H-S derivation? | Us — tractable | **Structural result Mar 9; full derivation pending** |
| Is data-geometry factor $\Omega$ nonzero for standard distributions? | Us — numerical | Open |
| Does trained-transformer attention kernel show conformal scaling? | Us — empirical | **✓ YES — Δ = 0.2493 (median, GPT-2). March 24.** |
| Do Papers 2 and 3 survive Ageev/Qi scrutiny if Junction 3 closes? | Internal review | Open |
| arXiv endorsement path — who in cs.LG would endorse? | Need to identify | Open |
| Does L^1.19 depth scaling persist at standard init (σ ~ 1)? | Us — numerical | Open |
| Does depth scaling approach a conformal fixed point? | Us — numerical + theory | Suggested by decreasing per-layer ratios |
| Why is ρ(λ, valley) *negative* in both Pythia-2.8B and GPT-Neo-2.7B? | Us — analysis + theory | **RESOLVED (exp-046, June 2):** λ captures recency/boundary balance, not a BCFT failure. See above. |
| What architectural feature of Pythia's training recipe at scale produces the late-layer ρ(Δ, valley) failure? | Us + EleutherAI authors | **Open as of April 17 — per-layer diagnostic localizes to layers 22–27 of Pythia-2.8B; GPT-Neo controls** |
| Does the alternating-layer pattern in GPT-Neo correspond to two functional populations of heads? | Us — analysis | **Partially resolved (exp-040/044):** GPT-Neo global layers Δ_med = 0.101 (trivial fixed point, architecture-induced). GPT-Neo not a clean ALiBi reference. Functional distinction confirmed but OLMo is better subject for further study. |
| Does the pre-registered prediction hold on Llama-3-8B? | Us — pending Meta access | Outstanding |
| Do W_QK eigenvalues show GOE-like level spacing in GPT-2? | Us — analysis | **CONFIRMED UNIVERSAL (exp-046/047, June 2/4):** GOE across GPT-2, GPT-2-medium, Pythia-410m. Conformal/non-conformal indistinguishable. |
| Are GOE weight statistics universal across model families, or GPT-2-specific? | Us — analysis | **CONFIRMED UNIVERSAL (exp-047, June 4):** Identical distributions in GPT-2 (learned), GPT-2-medium (learned), Pythia-410m (RoPE/NeoX). |
| What mechanism converts Poisson → GOE during training? (Untrained control.) | Us — analysis | **RESOLVED (exp-048, June 5):** GOE is structural — present at random Gaussian init (r=0.5288, all 5 seeds). Training preserves, does not create. Prediction (Poisson) falsified. |
| Is the GOE structure universal across OLMo, Mistral, GPT-Neo? | Us — analysis | **CLOSED (exp-077, July 4):** GPT-Neo-2.7B (0.5271, local=global), Mistral-7B GQA (0.5254, indistinguishable from MHA), Pythia-2.8b (0.5204; BCFT-anomaly layers 22–27 clean), GPT-2-large (0.5256) — all GOE-like. Only OLMo untested (not cached; low priority). |
| Does the GOE substrate require Gaussian init or a particular d_k? | Us — analysis | **CLOSED (exp-078, July 4):** No. Moment-universal (uniform, heavy-tailed t(ν=3)), holds for orthogonal (non-i.i.d.) factors, flat over d_k 4–256. Only lever: sparsity — crossover at ~2 nonzeros/column (p≈0.003), 3 orders of magnitude below dense parameterization. Calibration: size-matched GOE reference at 64×64 is ~0.530 (not asymptotic 0.536) — prior residual "distances to GOE" are mostly reference bias. |
| What is the correct frequency-space Δ estimator for the DFT bias? | Us — analysis | **Open (exp-045):** Synthetic calibration can in principle invert finite-support bias. Calibrated estimator not yet implemented. |
| Does the softmax → norm-sigmoid bracket width depend on PE, depth, scale? | Us — analysis | **Partially open (exp-043):** GPT-2 bracket ~0.015 (learned, 12L), GALA-7B bracket ~0.037 (ALiBi, 32L, 7B). Confounded. Clean test: GPT-2-scale model with ALiBi. |
| Does recurrence depth in depth-recurrent models (Huginn) operationalize conformal depth at inference time? | Us — exp-089 | **CONFIRMED (2026-07-20, Modal A100).** NAT: ρ(step, Δ_med) = −0.943, Δ_med 0.292 → 0.239 (saturates at steps 8–16); ρ(step, SYK-near) = +0.771. RAND: converges (ρ=−0.886) but non-monotone SYK-near (+0.143), with q=2 transient (Δ_med=0.523 at step 2) and step-1 structural burst (145/4400). Randomized-weights control: Δ_med frozen at 0.16868 (6th decimal), 0 SYK-near — the flow lives in trained weights. Preprint **published (v2, 2026-07-21)**: DOI 10.5281/zenodo.21472689 (v1: 10.5281/zenodo.21467922). v2 adds §4.4 layer-zone analysis: L0/L3 SYK-near (two-boundary structure), L1/L2 UV-locked. See `notes/2026-07-21_huginn_layer_zone_analysis.md` and exp-089 notes. |
| Does the exp-089 flow generalize to a second looped architecture (Ouro-1.4B, full-stack UT loop)? | Us — exp-090 | **PARTIAL (2026-07-21, pre-registered 05293e6c, run same day).** Both primary hypotheses FAILED on the registered pooled criteria: Ouro's bulk attention is not power-law (pooled Δ_med 0.08–0.11, R²_med 0.33); no pooled flow toward 0.25; and — against my declared prior — the conformal population does NOT degrade past trained depth: it *grows* (10 → 55 SYK-near from step 2 → 16) straight through the range where Ouro's task performance collapses (STARS). Exploratory (non-criterial): the high-R² subpopulation flows to 0.25 from above with ρ = −0.976; RAND step-1 burst of 1,444/7,680 concentrated in *deep* layers (inverts exp-087/088 zone under weight-sharing). Control: Δ_med frozen at **0.1687 — the same substrate value as the Huginn control**, cross-architecture. [Resolved same day: the value is kinematic — closed-form Δ_uniform(S=128, [3,60]) = 0.168678 for uniform causal attention; not a second fixed point. See `notes/2026-07-21_substrate_exponent_kinematic_derivation.md`.] Headline for the latent-reasoning audience: the fixed-point geometry survives recurrence extension; whatever breaks performance is not this observable. See exp-090 notes.md. |
| Does RAND early burst scaling (N^0.435) extend to larger models? | Us — analysis | **Open.** Three-point law confirmed to Pythia-410m (384 heads). Next point requires cached checkpoints for Pythia-1.4b or larger. |

---

## Outreach Status

| Person | Email | Sent | Response | Status |
|---|---|---|---|---|
| Gunn Kim | [email on file] | ✓ Mar 5, 6, 9, 11, 18 | ✓ Mar 6 — declined endorsement, raised construction concern | Awaiting — stories follow-up sent Mar 18 |
| Dmitry Ageev | [email on file] | ✓ Mar 6, 9, 11, 18 | — | Awaiting response |
| Yulia Ageeva | [email on file] | ✓ Mar 6, 9, 11, 18 | — | Awaiting response |
| Xiao-Liang Qi | [email on file] | ✓ Mar 9, 11, 18 | — | Awaiting response |
| Karl Friston | [email on file] | ✓ Mar 6, 9, 11, 18 | — | Awaiting response |
| Haim Sompolinsky | [email on file] | ✓ Mar 28 (resent — prior to [email on file] bounced) | — | Awaiting response |
| Jim Halverson | [email on file] | ✓ Mar 28 (resent — prior to jim.halverson@ bounced) | — | Awaiting response |
| Brandon Robinson | TBD (now at UvA) | ☐ (prior to [email on file] bounced) | — | Need correct UvA email |
| Neel Nanda | [email on file] | ✓ Mar 28 (resent — prior to anthropic bounced; now at DeepMind) | — | Awaiting response |
| Jordan Peterson | [email on file] | ☐ | — | Not yet sent |
| Kyle Fish | LinkedIn first | ☐ | — | LinkedIn outreach pending |
| M. Hamed Mohammady | [email on file] | ☐ | — | Second wave — hold |
| Francesco Buscemi | [email on file] | ☐ | — | Second wave — hold |
| Hans-Joachim Rudolph | PhilArchive/Academia.edu | ☐ | — | Second wave — hold |
| Geoff Penington | [email on file] | ☐ | — | Third wave — hold |
| Netta Engelhardt | [email on file] | ☐ | — | Third wave — hold |
| Daniel Harlow | [email on file] | ☐ | — | Third wave — hold |

---

## Connections to Watch

- **Kim ↔ Friston:** Same mathematics (Fisher-Rao free energy minimization), different domains. If Friston recognizes the identification, introduce them.
- **Qi ↔ Ageev/Ageeva:** Qi's holographic dictionary may directly answer the Junction 3 question about Ageev's scalar. If both respond, connecting them is the natural move.
- **Kim ↔ Mohammady/Buscemi:** Kim's temperature parameter may resolve their 2025 trilemma. Connecting them if both engage.
- **Rudolph ↔ Kim:** Rudolph found the Zeno-attention connection philosophically; Kim provides thermodynamic grounding.

---

*Update this file when:*
- *A response is received*
- *An open question is answered*
- *A new connection forms*
- *The status of a junction changes based on expert feedback*
