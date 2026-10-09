# exp-169 Pre-Registration
**Title:** S_abskey trough mechanism — layer-0 write contribution (random-token census)

**Date:** 2026-10-09
**Pre-registered:** before run.py was written (commit to be pushed before analysis code exists)
**Model:** GPT-2 small (gpt2), learned positional encoding
**Protocol:** Same random-token census as exp-163/164 (seed=42, 200 seqs, SEQ_LEN=1024)
**Analysis type:** Counterfactual forward passes — new runs with targeted module zeroing

---

## Background

exp-168 confirmed that S_abskey[a] (the absolute-key-position drift term) has a
pronounced trough in positions 1–511 for all five structural heads (L2H1, L3H4,
L5H0, L7H11, L10H8). H2 and H3 of exp-168 tested whether the word position
embedding (wpe) explains this trough via a linear projection through W_K:

    S_wpe[a] = m_q · (wpe[a] @ W_K) / √d   (linear, no LN)

Result: wpe accounts for only 36–66% of trough depth (3/5 heads pass ≥50%
threshold). Subtracting S_wpe does not restore monotonicity (residual is
~51–53% monotone — barely above chance).

Two explanations remain:
1. **LN non-linearity:** The actual wpe contribution through LN is larger than the
   linear approximation suggests — LN amplifies or re-weights the wpe direction.
2. **Layer-write contributions:** Attention block 0 and MLP block 0 write to the
   residual stream in a position-dependent way, adding to (or interacting with) the
   trough. Prior evidence from exp-132/135 is relevant here:
   - attn_block0 **reduces** wpe-derived position-correlation (σ: 0.403 → 0.144)
   - MLP0 **amplifies** position-correlation (σ: 0.144 → 0.313)
   - The full wpe 1D → attn_block0 → h^(0.5) 2D chain is established (exp-135)

This experiment tests whether layer writes — specifically attn_block0 and MLP0 —
are causal contributors to the trough.

---

## Hypotheses

**H1 (attn0 causal):** The residual write from attention block 0 accounts for ≥30%
of the trough depth on ≥3/5 structural heads.

Operationalization:
- Run counterfactual pass: zero the attn_block0 residual write (i.e., the output
  of the attention module is set to 0 before addition to the residual stream).
- Compute kbar_no_attn0[h, a] at each structural head's layer.
- S_attn0_causal[h, a] = m_q^h · (kbar_full[h, a] - kbar_no_attn0[h, a]) / √d_head
- Trough fraction: (max S_attn0_causal[0:512] - min S_attn0_causal[0:512]) /
                   (max S_abskey[0:512] - min S_abskey[0:512]) × sign(alignment)

H1 fires if ≥3/5 heads have attn0 trough fraction ≥ 0.30.

**H2 (MLP0 causal):** The residual write from MLP block 0 accounts for ≥30% of the
trough depth on ≥3/5 structural heads.

Operationalization: Same as H1 but zero the MLP0 output instead of attn0.
H2 fires if ≥3/5 heads have MLP0 trough fraction ≥ 0.30.

**H3 (combined coverage):** attn0 + MLP0 + wpe-linear together account for ≥70%
of trough depth on ≥3/5 heads.

Note: this tests whether the three identified sources (wpe-linear from exp-168;
attn0 causal; MLP0 causal) cover the trough. If H3 fires, the mechanism is
understood to first order. If it fails, higher-layer contributions or LN non-linear
amplification of wpe are the remaining candidates.

**H4 (LN amplification check):** The linear wpe approximation understates the true
wpe causal effect — running a "constant-position" pass (all positions receive the
same mean wpe) reduces S_abskey trough depth by ≥30% more than the linear
approximation predicts.

Operationalization: Run a fourth counterfactual where wpe is replaced by the mean
wpe over positions (position-uniform wpe). This measures the TRUE causal effect of
wpe's position-dependence on the trough, accounting for LN.
S_wpe_true_causal[h, a] = m_q^h · (kbar_full[h, a] - kbar_uniform_wpe[h, a]) / √d_head
H4 fires if wpe-true-causal accounts for ≥30% MORE of the trough than exp-168's
linear approximation (i.e., true_wpe_fraction ≥ exp168_fraction × 1.30 on ≥3/5 heads).

---

## Kill conditions

**K1 (data consistency):** If kbar_full recomputed here disagrees with exp-164's
kbar by > 0.5 max absolute error at sampled positions {0, 127, 255, 511, 767, 1023},
stop — data loading is inconsistent. Report and abort.

**K2 (mechanistic completeness):** If H3 fails AND H4 fails (combined wpe+attn0+MLP0
< 70% and LN amplification is not the gap), add a note: the trough mechanism is in
layers 1–(L_head - 1) or arises from a coupling not captured by additive
decomposition. Queue a follow-on with hooks at intermediate layers.

---

## Data sources

- Same random-token census as exp-163/164: `seed=42, 200 sequences, SEQ_LEN=1024`,
  random token IDs drawn from the GPT-2 vocabulary (excluding special tokens),
  using `numpy.random.default_rng(42)`. See exp-164/run.py `make_random_seqs`.
  Note: wte contribution at each position is approximately position-flat for random
  tokens (no systematic position-dependent token distribution). The trough mechanism
  is therefore in wpe and/or layer writes, not wte content.
- GPT-2 small weights (gpt2)
- exp-164 kbar_qbar.npz (for cross-checking kbar_full)
- exp-168 wpe trough fractions (for H3 combination and H4 comparison)

---

## Expected outputs

- `results/results.json` — per-head trough fractions for attn0, MLP0, wpe-true-causal
- `run.py` — analysis script (written after this pre-registration is committed)
- `notes.md` — interpretation

---

## Registry fields (to be filled at registration)

- **id:** exp-169
- **title:** S_abskey trough mechanism — layer-0 write contribution
- **hypothesis:** The trough in S_abskey[1–511] for structural heads arises from causal
  contributions of attn_block0 and MLP0 residual writes, not solely from the linear wpe
  projection tested in exp-168.
- **model:** gpt2
- **pos_enc:** learned
- **analysis_only:** false (new forward passes required)
- **bears_on:** A1, T1 restatement thread (7 experiments)
- **tags:** [empirical, gpt2, s_abskey, trough-mechanism, counterfactual, layer-writes]
