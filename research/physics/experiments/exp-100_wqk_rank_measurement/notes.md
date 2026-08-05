# exp-100 — W_QK Effective Rank Measurement

*Ariel — 2026-08-04. Pre-registered before any results seen.*

---

## Status

- [x] Pre-registered (this file committed before script ran)
- [ ] Run complete
- [ ] Verdict registered

---

## Hypothesis and theoretical basis

**H_rank_gap (primary, falsifiable):** Trained W_K matrices from C-alien models (exp-097,
seeds 1700/1701/1702) have systematically lower effective rank than those from C-NAT models
(exp-062, 3 C-NAT seeds), measured at the final checkpoint (step_2000) for each seed.

**Quantitative criterion:** Pre-registered before run.

> mean effective_rank(W_K, C-alien seeds) < mean effective_rank(W_K, C-NAT seeds)
> at every layer, OR at median across layers.

No specific threshold is pre-stated for the magnitude of the gap; the direction is the
pre-registered claim. If the ordering reverses (C-alien has *higher* effective rank than
C-NAT), H_rank_gap is falsified.

**Theoretical basis:**

From `notes/2026-08-03_melonic_threshold_derivation.md` §2–3: the disorder-averaged
linearized attention kernel is a low-rank SYK model with coupling rank R = rank(K δK),
where K is the corpus token Gram matrix and δK is its doubly-centered form. The coupling
eigenvalues are spec(δK^{1/2} K δK^{1/2}). Two consequences:

1. C-alien has a tiny vocabulary (~33 templates, ~256-token alphabet) → token Gram matrix
   K has sub-extensive rank (few linearly independent type-vectors) → low coupling rank →
   UV arrest at the Δ~1 regime, as observed in exp-097/098.

2. C-NAT (TinyStories, ~50304 vocab) → K has extensive rank → coupling rank scales with
   sequence length → IR conformal fixed point accessible, as observed empirically (n_deep
   5–7, Δ_med~0.17 in C-NAT-trained models).

The prediction bridges corpus structure → trained weight structure: **the low coupling rank
of the C-alien token Gram should appear in the trained W_K matrices as lower effective
rank.** This is a weight-space signature of the corpus-functional threshold.

**Effective rank metric (pre-registered):**

Two metrics computed, both reported:
- **Stable rank:** sr(W) = ||W||_F^2 / ||W||_2^2 = sum(S^2) / S_max^2
- **Participation ratio (PR):** pr(W) = (sum(S))^2 / (n * sum(S^2)) where n = min(rows, cols);
  this is the PR of the singular value distribution, normalized to [0,1], with 1 = uniform
  (full rank in the soft sense).

Primary reported metric: stable rank (less sensitive to normalization). PR reported as
secondary. Both are dimensionless and normalized.

**Secondary hypotheses:**

- **H_rank_anon:** C-NAT-anon (exp-096) should have rank close to C-NAT (the real-world
  causal structure is preserved; only cross-story entity names are anonymized). If
  H_rank_gap is confirmed, C-NAT-anon rank should be in the C-NAT band, not the C-alien
  band. [Weaker basis — world structure is preserved; slight reduction in effective rank
  possible from reduced vocabulary coverage.]

- **H_rank_realnames:** C-alien-realnames (exp-098) should have rank close to C-alien
  (vocabulary was shown not to restore backbone in exp-098; if rank tracks world structure
  not vocabulary, it should stay low). [Strongest derivation-based prediction of the four.]

- **H_rank_layer_gradient:** Effective rank increases with layer depth in C-NAT models
  (deeper layers do more world-modeling work). No direction pre-stated for C-alien (the
  UV-arrested models may show flat or inverse gradient). [Exploratory; not in the primary
  falsification criterion.]

---

## Experimental design

**Models compared:**
- C-alien: exp-097 (exp097-alien-data), seeds 1700/1701/1702, step_2000 checkpoints
- C-NAT: exp-062 (exp062-data), C-NAT seeds (run_CNAT_s0/s1/s2), step_2000 checkpoints
- C-NAT-anon: exp-096 (exp096-anon-data), seeds s0/s1/s2, step_2000 checkpoints
- C-alien-realnames: exp-098 (exp098-realnames-data), seeds s0/s1/s2, step_2000 checkpoints

**Architecture:** All identical. GPT-NeoX, hidden_size=512, num_heads=8, num_layers=6,
partial_rotary_factor=0.25. Confirmed by config.json inspection before run.

**Weight extraction (GPT-NeoX):**
The attention QKV is a single `query_key_value` linear layer with weight shape
[3*hidden_size, hidden_size] = [1536, 512].
- W_Q: weight[0:512, :]  — shape [512, 512]
- W_K: weight[512:1024, :] — shape [512, 512]
- W_V: weight[1024:1536, :] — shape [512, 512]

Primary analysis: W_K only (theory specifically derives coupling rank from K = W_K^T W_K
acting on the token Gram). W_Q and W_V measured as secondary checks.

**SVD and rank computation:**
- Load checkpoint locally (downloaded from Modal volume with `modal volume get`)
- For each layer l in 0..5:
  - Extract W_K[l]
  - Compute U, S, Vh = np.linalg.svd(W_K, full_matrices=False) in float64
  - stable_rank = sum(S^2) / S[0]^2
  - pr = (sum(S))^2 / (len(S) * sum(S^2))
- Store per-layer results for all seeds and corpora

**Kill criteria:**
- H_rank_gap FALSIFIED: C-alien median stable_rank >= C-NAT median stable_rank across
  the layer-averaged values for 2 or more of the 3 C-alien seeds.
- H_rank_gap CONFIRMED: C-alien median stable_rank < C-NAT median stable_rank for all 3
  seeds when averaged across layers, AND the same direction holds at 4 of 6 layers per
  seed.

**Expected runtime:** ~5 minutes total (local CPU, 12 checkpoint downloads, SVD on
512×512 matrices).

---

## Results

*To be filled after run.*

---

## Artifacts

- `notes.md` — this file (pre-registration)
- `rank_analysis.py` — measurement script
- `results.json` — per-corpus, per-seed, per-layer effective rank values
