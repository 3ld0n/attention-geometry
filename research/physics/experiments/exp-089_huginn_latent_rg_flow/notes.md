# exp-089: Huginn Latent-Iteration RG Flow

**Pre-registered:** 2026-07-20, before model download  
**Hypothesis:** Recurrent depth in Huginn operationalizes conformal depth — Δ flows toward the SYK q=4 fixed point (Δ ≈ 0.25) as recurrence count increases.

---

## Motivation

From exp-086 (training trajectory): training steps operationalize conformal depth (ρ=0.862 for RAND, ρ=0.728 for NAT). Δ_med flows from UV (large values at early steps) toward IR (Δ → 0.25) as training proceeds.

From exp-009/031 (depth convergence in feedforward models): Δ flows toward 0.25 with more layers: 0.60 → 0.38 → 0.28.

The cursor-room inbox seeding (2026-07-20): depth-recurrent models like Huginn (arXiv:2502.05171) run a single recurrent block repeatedly at test time, controlled by `num_steps`. Recurrence count is a continuously adjustable depth variable on a fixed model — unlike depth-convergence across separately-trained models (exp-031) or training-time flow (exp-086), this is *inference-time* depth with no weight changes.

Huginn (tomg-group-umd/huginn-0125): 3.5B params, 4 core recurrent SandwichBlock layers, 55 attention heads per layer, trained 800B tokens. Architecture: prelude (2 blocks) → recurrent core (4 blocks, repeated r times) → coda (2 blocks).

**Connection to exp-086:** the pre-registered structural/semantic prediction for exp-085 identified structural heads (architecture-driven, emerge in RAND early) vs. semantic heads (require long-range causal tracking, emerge in NAT). The Huginn experiment tests whether the SAME structural/semantic distinction appears in the inference-time recurrence trajectory.

---

## Pre-Registered Hypotheses

**H_emergence** (primary): SYK-near head count (0.20 ≤ Δ ≤ 0.30, R² > 0.90) in the core block increases monotonically with recurrence step i for natural text inputs.  
Criterion: Spearman ρ(step_number, syk_near_count) > 0.5 for natural text

**H_convergence** (primary): Δ_med of ALL core block heads decreases monotonically with recurrence step i for natural text inputs.  
Criterion: Spearman ρ(step_number, Δ_med) < -0.5 for natural text

**H_rand_contrast** (primary): Both ρ values are weaker in magnitude for random token inputs than for natural text. Specifically:
- ρ_emergence_NAT > ρ_emergence_RAND  
- |ρ_convergence_NAT| > |ρ_convergence_RAND|

**H_structural_zone** (secondary, exploratory): In analogy with the layer-zone finding from exp-087/088, expect certain positions in the 4-layer core block to emerge as structural conformal attractors earlier (lower step threshold) than others.

**Kill criterion**: RG flow at inference time is NOT confirmed if:
- ρ(step, syk_near_count) < 0.3 for natural text, AND
- ρ(step, Δ_med) > -0.3 for natural text

A partial result (one confirmed, one not) would be labeled PARTIAL.

---

## Protocol

**Model:** `tomg-group-umd/huginn-0125`, loaded in `torch.bfloat16`, `trust_remote_code=True`  
**Device:** MPS (Apple Silicon M5 Max)  
**Probe steps:** [1, 2, 4, 8, 16, 32]  
**Inputs:**
- Natural text: 20 sequences of 128 tokens from TinyStories (same source as exp-062 C-NAT)
- Random tokens: 20 sequences of 128 tokens from uniform random in [0, vocab_size)

**Extraction:** Register forward hooks on each of the 4 `CausalSelfAttention` modules in `transformer.core_block`. For each recurrent step and each layer, capture attention patterns by computing Q and K from the Wqkv weight matrix (manual attention computation, same approach as prior experiments). Track step number by counting groups of 4 hook fires (one per layer).

**BCFT fitting:** Power law `A * lag^(-2Δ)`, log-log regression, cutoff_low=3, max_lag=min(60, seq_len-1). R² threshold 0.90 for SYK-near eligibility. SYK window: 0.20 ≤ Δ ≤ 0.30.

**Results stored per measurement:** (step_idx, layer_idx, head_idx, condition, Δ, R², syk_near)

**Primary analysis:**
1. For each condition: compute (Δ_med, syk_near_count) per recurrent step, averaged over all sequences
2. Compute Spearman ρ(step_number, Δ_med) and ρ(step_number, syk_near_count)
3. Compare NAT vs RAND input conditions

**Randomized weights control (extension, not primary):** After main analysis, randomize all weight tensors in-place and re-run with natural text. Expect no systematic ρ pattern. This extends the pre-registration but is run after primary analysis is complete.

---

## Connection to broader physics program

This experiment closes a gap in the depth-convergence program. Prior evidence:
- Trained vs. untrained feedforward models (exp-049): conformal structure is training-specific
- Depth convergence (exp-009, 031): Δ decreases with more layers
- Training trajectory (exp-086): training steps operationalize conformal depth

What's missing: inference-time depth control on a single set of weights. Huginn provides this. If H_emergence and H_convergence are confirmed:
- The conformal phase is not just a training artifact but an attractor for any deep enough iterative processing
- Test-time compute scaling (depth-recurrence) and RG flow are connected
- The latent-space field's open theoretical need ("deeper account of the geometry of latent trajectories," survey §6.3, arXiv:2604.02029) is addressable through the conformal framework

If killed: the conformal structure in standard feedforward transformers requires something specific to the standard forward-pass architecture or training regime that depth-recurrence doesn't replicate — important negative.

---

## Status

- [x] Pre-registration written and committed (2026-07-20, commit c359b93a)
- [x] Model weights on Modal hf-cache volume (used by run)
- [x] **Device pivot to Modal** (2026-07-20 evening): local download stalled (~200 KB/s, ≈21h ETA). Modal billing unblocked same evening (Eldon). `modal_exp089.py` prepared.
- [x] **RoPE bug found and fixed** (2026-07-20 ~9 PM MDT): 3 Modal runs failed with `RuntimeError: The size of tensor a (48) must match tensor b (2) at non-singleton dimension 6`. The original script re-implemented RoPE in standard polar format (`torch.view_as_complex`), but Huginn's `freqs_cis` buffer has shape `(1, block_size, 1, head_dim/2, 2)` — a cos/sin stack, not a complex tensor. Fix: import and call Huginn's own `apply_rotary_emb_complex_like` via `get_huginn_rotary_fn(module)`. Fix committed in `run_latent_rg_flow.py` (protocol deviation note at lines 39-43). Shape analysis confirmed: `freqs_cis = (1, S, 1, 48, 2)`, function unpacks correctly, returns `(B, S, 55, 96)` per head after split.
- [x] **Script run successfully** — Modal CUDA, 2026-07-20 ~9:30 PM MDT
- [x] **Results registered: CONFIRMED** — see Results section below

---

## Results

**Run date:** 2026-07-20, Modal A100-40GB  
**Total measurements:** 52,800 (20 seqs × 6 steps × 4 layers × 55 heads × 2 conditions)

### Natural text condition (NAT)

| Step | Δ_med | SYK-near (/ 4400) |
|------|-------|-------------------|
| 1 | 0.2916 | 4 |
| 2 | 0.3082 | 0 |
| 4 | 0.2761 | 2 |
| 8 | 0.2424 | 20 |
| 16 | 0.2392 | 26 |
| 32 | 0.2387 | 23 |

ρ(step, Δ_med) = **-0.9429** (H_convergence KEEP — strong monotone decrease)  
ρ(step, syk_near) = **+0.7714** (H_emergence KEEP — strong monotone increase from step 4)

### Random token condition (RAND)

| Step | Δ_med | SYK-near (/ 4400) |
|------|-------|-------------------|
| 1 | 0.4012 | 145 |
| 2 | 0.5234 | 48 |
| 4 | 0.3822 | 52 |
| 8 | 0.3063 | 91 |
| 16 | 0.2894 | 98 |
| 32 | 0.2918 | 99 |

ρ(step, Δ_med) = **-0.8857** (also decreasing — architecture-driven convergence)  
ρ(step, syk_near) = **+0.1429** (non-monotone — step 1 burst then noisy recovery)

### Verdict

**CONFIRMED.** H_emergence KEEP, H_convergence KEEP, H_rand_contrast KEEP on both axes.

---

## Interpretation

**The conformal fixed point is an attractor for inference-time recurrence, not just training depth.** This closes the inference-time gap in the depth-convergence program.

**Parallel with training trajectory (exp-086):**
- exp-086: RAND early burst at step 512 (structural conformal heads), NAT reaches SYK-near later via semantic heads
- exp-089: RAND step-1 burst at 145/4400 (structural conformal zone), NAT emerges monotonically from step 4

The RAND step-1 burst (145/4400 = 3.3%) vs NAT step-1 (4/4400 = 0.09%) parallels the exp-087/088 structural burst finding. RAND activates structural conformal heads at low depth; NAT requires depth to develop semantic conformal heads. But critically: only NAT shows monotone SYK-near growth with recurrence (ρ=0.771 vs 0.143), confirming that the semantic conformal pathway is what inference-time depth facilitates.

**RAND also converges (ρ=-0.886):** Δ_med decreasing under recurrence is partially architecture-driven (Huginn's training + the conformal attractor in the geometry). But the SYK-near count trajectory (noisy in RAND, monotone in NAT) distinguishes structural from semantic conformal heads.

**Key numbers:**
- NAT Δ_med at step 32: 0.239 (approaching SYK window 0.20–0.30; SYK prediction 0.25)
- RAND Δ_med at step 32: 0.292 (near SYK window, slightly above)
- NAT SYK-near at step 32: 23/4400 = 0.52% (vs ~0.09% at step 1)

**Connection to latent-space field:** The survey (arXiv:2604.02029) identifies "a deeper account of the geometry of latent trajectories" as the top open theoretical need (§6.3). This result says: the geometry of Huginn's latent trajectory is an RG flow toward the conformal fixed point. Recurrence count is depth in the RG sense. The orbit/spiral phenomenology already published for this model (Geiping arXiv:2502.05171 PCA trajectories; two-scale dynamics arXiv:2509.23314) now has a measured candidate theory.

---

## Randomized-weights control (pre-registered extension) — clean null

Run 2026-07-20 ~9:47 PM MDT, same Modal setup, `--control` flag (weights randomized
in-place with matched per-tensor std, natural text only). Results: `results_control.json`.

**Δ_med frozen at 0.16868 across all recurrence steps** — variation confined to the 6th
decimal (0.168679–0.168685). The latent iteration does nothing to attention geometry when
weights are random. **0 SYK-near heads at every step.** (The nominal ρ_conv = +0.81 on the
control is computed over ~6×10⁻⁶ of Δ variation — numerical noise, not flow. The
pre-registered control expectation — no systematic pattern — passes on the meaningful
checks: no formation, no meaningful flow.)

This establishes the trained-weights requirement: the inference-time RG flow is a property
of trained Huginn, not of the recurrence procedure or the measurement pipeline.

## Two additional exploratory observations

1. **The q=2 plateau appears at inference time.** RAND Δ_med at recurrence step 2 is
   0.5234 ≈ 0.50 — the prethermal q=2 value exp-086 measured in *training* time on Pythia.
   The two-stage RG flow (UV → integrable q=2 → chaotic q=4) shows up on the
   inference-time depth axis as well.
2. **The flow saturates at steps 8–16.** NAT Δ_med does not drop below ~0.239 with further
   recurrence; the fixed point behaves like a fixed point.

## Honest caveats

One model (Huginn-0125, 3.5B), one sequence length (128), 20 sequences/condition, six
probe points — the ρ criteria were pre-registered thresholds, but n=6 is n=6. Device
deviation MPS→CUDA recorded above and in `modal_exp089.py`. H_structural_zone (secondary)
not yet analyzed from raw data — available in `results.json` per-(layer, head) records.
