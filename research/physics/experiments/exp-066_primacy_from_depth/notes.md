# exp-066: Primacy as depth-accumulated boundary absorption — Notes

*June 12, 2026 (second Fable 5 session). Develops the constructive finding of exp-065 into a falsifiable mechanism. RESEARCH_MAP Thread 7 follow-up; candidate for the LiTM v2 pre-registration.*

## Order of operations (look-ahead-bias control)

1. **Derivation written and committed first:** `notes/2026-06-12_primacy_from_depth_derivation.md`. Core: position 0 is an absorbing boundary of any causal row-stochastic kernel; composition is a backward walk into it; primacy = absorbed mass. Effective depth d_eff = (1−μ)L via the lazy-walk binomial. Two laws: (A) primacy ↑ in d_eff and collapses onto one curve in d_eff; (B) absolute start-window mass ↓ in context length N at fixed d_eff.
2. **This pre-registration** (P1–P3 with numeric thresholds), committed before any script is run.
3. Numerics: `run_primacy_synthetic.py` (P1, P2), then `run_primacy_longchat.py` (P3) on archived exp-024 data. Composition primitives reused from exp-065.

## Observables

- **Primacy mass** `prim_decile(row)` = sum of the deep-query (last-row) composite over the oldest decile of keys, indices [0, N/10).
- **Start-window mass** `prim_window(row, W)` = sum over the first W absolute positions [0, W). Default W = 8.
- **d_eff** = (1−μ)·L (or (1−μ)·k for empirical layer-truncation).

## Pre-registered predictions

**P1 — Law A: depth collapse + monotonicity (synthetic).**
Single repeated causal power-law kernel, N = 512, Δ ∈ {0.25, 0.49}. Grid: μ ∈ {0, 0.25, 0.5, 0.75}, L ∈ {1,2,3,4,6,8,12,16,24,32,40}. For each, primacy = `prim_decile` of the deep-query row of (μI+(1−μ)M)^L.
- **P1a (monotonicity):** Spearman ρ(d_eff, prim_decile) over the full grid, per Δ. *Keep:* ρ ≥ 0.95. *Kill:* ρ < 0.80.
- **P1b (collapse):** build the μ=0 reference curve prim_decile(0, L) vs d_eff = L; for every (μ>0, L) point with d_eff inside the μ=0 range, interpolate the reference at the same d_eff and compare. *Keep:* median |Δprim| ≤ 0.03 AND 90th-percentile |Δprim| ≤ 0.06 across all μ>0 points. *Kill:* median |Δprim| > 0.06 → μ and L are not interchangeable through (1−μ)L and the lazy-walk rescaling fails.

**P2 — Law B: context-length dilution (synthetic).**
Fix the kernel (Δ = 0.49, the LongChat-like near-marginal case) and a target d_eff (use μ=0, so d_eff = L; test L ∈ {8, 20}). Vary N ∈ {128, 256, 512, 1024}. Measure `prim_window(row, W=8)`.
- **P2 (keep):** start-window mass strictly decreasing in N at both tested depths: Spearman ρ(N, prim_window) ≤ −0.90 for each. *Kill:* ρ > −0.5 (no dilution) or positive (mass grows with N) → Law B false.
- *Reported, not thresholded:* the same for Δ = 0.25 (heavier tail — dilution expected weaker), and fractional oldest-decile mass vs N for contrast.

**P3 — Empirical layer-truncation (LongChat, exp-024 census).**
Compose the measured per-layer additive mixtures (C-weighted; uniform as robustness) for the first k layers, k = 1…40, at μ ∈ {0, 0.5} (d_eff = (1−μ)k). Deep-query composite row each k. Measure prim_decile(k) and the document-binned (20-bin) rank correlation with Liu-20 accuracy.
- **P3a (primacy grows with depth):** Spearman ρ(k, prim_decile) at μ=0. *Keep:* ρ ≥ 0.90. *Kill:* ρ < 0.5.
- **P3b (U-shape emerges with depth, descriptive):** report ρ(doc_profile, Liu-20) as a function of k — expected small/negative at small k (recency-only), rising toward the +0.9…+1.0 exp-065 full-depth value. Not thresholded (n=5 accuracy points); reported as the depth at which ρ first exceeds +0.5.
- **P3c (effective-depth consistency, descriptive):** check that prim_decile at (μ=0.5, k=40) ≈ prim_decile at (μ=0, k=20) — same d_eff = 20 — within the P1b tolerance band, as an empirical echo of the collapse. Reported.

## Kill summary

- If P1a ρ < 0.80 OR P1b median |Δprim| > 0.06: the depth/lazy-walk picture is wrong; primacy is not governed by (1−μ)L. Stop; log as a failed mechanism.
- If P2 ρ > −0.5: Law B false; primacy does not dilute with context — report and retract the context-length half of the claim.
- If P3a ρ < 0.5: the measured kernels do not accumulate primacy with depth; the mechanism does not survive contact with real per-layer exponents. Stop claiming depth-accumulated primacy for LongChat.

## Results

### Registered verdicts

| Item | Verdict | Detail |
|---|---|---|
| P1a (monotone in d_eff) | **KEEP** | Spearman ρ(d_eff, prim_decile) = +0.998 (Δ=0.25), +0.998 (Δ=0.49) over the full (μ,L) grid. |
| P1b (lazy-walk collapse) | **KEEP** | Curves at μ∈{0.25,0.5,0.75} collapse onto the μ=0 curve vs d_eff=(1−μ)L: median \|Δprim\| = 0.012 (Δ=0.25) / 0.005 (Δ=0.49); p90 = 0.042 / 0.013. μ and L are interchangeable through (1−μ)L, as derived. |
| P2 (context dilution) | **KEEP** | Absolute start-window (W=8) mass strictly decreasing in N∈{128,256,512,1024}: ρ(N, window) = −1.00 at both L=8 and L=20, for **both** Δ=0.49 (thresholded) and Δ=0.25 (reported). |
| P3a (empirical primacy ↑ depth) | **KEEP** | LongChat measured kernels: ρ(k, prim_decile) = +1.00 for all (C/uniform × μ∈{0,0.5}). |

### Descriptive (post-hoc, not thresholded)

- **P3b (U-shape emerges with depth):** the document-binned rank-match with Liu-20 accuracy is recency-dominated at shallow composition and crosses ρ>+0.5 only past a depth threshold — first at k=4 (C, μ=0), k=9 (C, μ=0.5), k=2 (uniform, μ=0), k=5 (uniform, μ=0.5). The μ=0.5 crossings occur at roughly twice the raw k of the μ=0 crossings (k=9 vs 4; 5 vs 2) — i.e. at comparable **effective** depth — an independent echo of the (1−μ)L rescaling in the empirical kernels.
- **P3c (effective-depth echo):** prim_decile at (μ=0.5, k=40) = 0.776 vs (μ=0, k=20) = 0.844, same d_eff=20. \|Δ\|=0.068 — **just outside** the synthetic P1b band (0.06). Honest reading: the strict collapse is a single-repeated-kernel result; for heterogeneous real layers it is contaminated by layer-to-layer Δ variation and by which layer set each point uses (μ=0/k=20 uses layers 0–19; μ=0.5/k=40 samples ~20 of all 40). Landing within 0.07 anyway is mild directional support, not a clean confirmation. The clean collapse test is synthetic P1b.

### Honest assessment of what is and isn't informative

- **Near-tautological:** P3a (primacy grows with depth). Given an absorbing boundary, absorbed mass is non-decreasing in steps almost by construction. Its only real content is the negative it *could* have shown — that the measured per-head profiles, treated as backward row-stochastic kernels, behave like the idealized model rather than being swamped by non-conformal/sink mass. They do.
- **Genuinely informative:** P1b (the lazy-walk collapse — a specific, falsifiable rescaling that held to ~0.5–1.2%) and P2 (the *direction* of context-length dilution). P2's magnitude is not established — four N-points test direction only.
- **What would make this a prediction about real models rather than a re-description of attention geometry:** the two laws translate to accuracy-side, GPU-testable claims — (A) truncating a trained LLM to its first k layers should reduce lost-in-the-middle primacy monotonically; (B) at fixed model depth, primacy should weaken as context length grows. Neither is tested here (profile-level synthetic + archived census only).

### Status

- Mechanism (absorbing start-boundary → depth-accumulated primacy): **derived and internally consistent.**
- Effective-depth law d_eff=(1−μ)L: **confirmed** synthetically (collapse to ~1%), **echoed** in the empirical kernels (P3b crossings; P3c within 0.07).
- Context-length dilution (Law B): **direction confirmed** synthetically; magnitude open.
- For the v2 LiTM pre-registration: pair the joint (Δ,λ)→valley statistic with a **real-model** version of Laws A and B (layer-truncation + context-length sweeps on a model with a clean published U-shape, e.g. MPT-30B-Instruct), since tonight's tests are profile-level and cannot speak to accuracy.

*Registry: exp-066. Runtime: ~1.5s synthetic + ~0.6s archived census. Scripts: `run_primacy_synthetic.py` (P1, P2), `run_primacy_longchat.py` (P3). Builds directly on exp-065 composition primitives.*
