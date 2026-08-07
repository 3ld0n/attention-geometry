# exp-101: Attention Gram Matrix Effective Rank

**Pre-registered:** 2026-08-06, ~9:15 PM MDT, physics room session.
**Pre-registration committed before any analysis.**
**Analysis-only experiment — no new training.** Reads existing Modal volumes.

---

## Background and motivation

Exp-100 falsified the weight-level rank hypothesis: W_K effective rank is
architecture-dominated (~53–58/64) and does not scale with world complexity S.

But the Kim–Cao–Altman derivation (see
`notes/2026-08-03_melonic_threshold_derivation.md`, §2–3) says the relevant
low-rank object is NOT W_K in isolation. It is the **input-conditioned token
kernel**:

> K_{ab} = k_a · k_b,   where k_a = W_K · x_a   (key vector for token a)

The coupling eigenvalues that determine the SYK phase are
spec(δK^{1/2} K δK^{1/2}), where δK = Π K Π is the doubly-centered kernel.
The rank of K — and therefore the number of active SYK modes — depends on
both W_K and the input token distribution {x_a}.

For C-alien (S=8 world states), the token embeddings should cluster into at most
S distinct semantic groups (corresponding to the world's 8 distinguishable
configurations). After projection by W_K_h, this clustering may persist: rank
of K ≤ S = 8 per head. For C-alien-rich (S=32): rank ≤ 32. For C-NAT-anon
(large vocabulary, S >> d_k): rank approaching d_k = 64.

This is why the corpus functional m₂ discriminates 18× between C-alien and
C-NAT while W_K rank discriminates only 1.087×: m₂ measures the coupling
magnitude through the token-conditioned kernel, not through W_K directly.

**Exp-101 tests this directly.** If H_kern_ordered is confirmed: the
low-rank SYK mechanism operates at the kernel level, not the weight level.
If falsified: the ranking mechanism is more subtle, and the low-rank SYK
story needs further revision.

---

## The question

Does the effective rank of the key gram matrix K_{ab} = k_a · k_b scale with
world complexity S (C-alien S=8, C-alien-rich S=32, C-NAT-anon S>>d_k),
in the same direction that exp-100's weight-level rank did NOT?

---

## Pre-registered hypotheses

**H_kern_ordered** (primary): effective rank of the key gram matrix follows
world complexity ordering at layer 0:
- mean R_eff(C-alien) < mean R_eff(C-alien-rich) < mean R_eff(C-NAT-anon)
- Criterion: strict ordering on per-corpus mean R_eff across 8 heads.

**H_kern_S** (primary, quantitative): R_eff approximates S in the key space.
- C-alien: mean R_eff in [4, 20]   (= S/2 to 2.5S, accounting for embedding
  spread and vocab clustering)
- C-alien-rich: mean R_eff in [15, 50]   (= S/2 to 1.5S)
- C-NAT-anon: mean R_eff > 40   (approaching d_k=64)
- Ratio criterion: mean R_eff(C-alien-rich) / mean R_eff(C-alien) ≥ 2.0
  (predicted ≈ 4, accepting 2× due to non-linear embedding clustering).

**H_kern_realnames_equiv** (control): C-alien-realnames has the same R_eff as
C-alien within 10%. Vocabulary swap (English names for alien tokens) should
not change the key gram rank if world structure drives the clustering.
Consistent with exp-098's finding that vocabulary does not affect backbone
or UV arrest.

**H_kern_delta** (secondary, cross-experiment): across the four corpora,
mean R_eff is negatively correlated with median Δ_med from the formation
experiments. Low kernel rank → UV-arrested (high Δ_med); high kernel rank →
IR-approaching (low Δ_med). Using exp-097/098/099/096 Δ_med values and
this experiment's R_eff values.

---

## Kill criteria

**Kill H_kern_ordered:** mean R_eff(C-alien) ≥ mean R_eff(C-NAT-anon).
Key gram matrix rank is not world-complexity-ordered. The low-rank SYK
mechanism does not hold at the kernel level. Alternative mechanism required.

**Kill H_kern_S:** if R_eff(C-alien) > 30 OR R_eff(C-NAT-anon) < 30.
Token embeddings do not cluster into S groups. The SYK-rank prediction breaks
at this operationalization.

---

## Protocol

### Corpora and checkpoints

Four corpus/model pairs, each matched (corpus-trained checkpoint):

| Corpus        | Volume              | Corpus path             | Checkpoint path                              |
|---------------|---------------------|-------------------------|----------------------------------------------|
| C-alien       | exp097-alien-data   | /data097/C-alien.bin    | /data097/runs/run_alien_s0/step_2000/model.safetensors |
| C-alien-realnames | exp098-realnames-data | /data098/C-alien-realnames.bin | /data098/runs/run_realnames_s0/step_2000/model.safetensors |
| C-alien-rich  | exp099-rich-data    | /data099/C-alien-rich.bin | /data099/runs/run_rich_s0/step_2000/model.safetensors |
| C-NAT-anon    | exp096-anon-data    | /data096/C-NAT-anon.bin | /data096/runs/run_anon_s0/step_2000/model.safetensors |

### Token sampling

- Sample N_TOKENS = 4096 token IDs from each corpus.
- Sampling strategy: stride through the corpus at equal intervals (not the
  first 4096 tokens, which may be non-representative). Stride =
  floor(corpus_length / N_TOKENS).
- Token IDs stored as uint16 in .bin files.

### Key vector computation (layer 0)

This experiment uses the layer-0 key projection as the primary measurement.

At layer 0 in GPT-NeoX (with RoPE), there is no additive positional embedding.
The input to layer 0 is:
    h_a = embed_in[token_id_a]   (shape [d_model=512])

Key projection per head h at layer 0:
    W_K_full = qkv_weight[512:1024, :]   (shape [512, 512])
    W_K_h    = W_K_full[h*64:(h+1)*64, :]   (shape [64, 512])
    k_a      = h_a @ W_K_h.T   (shape [64])

Key matrix:
    A = stack of k_a for all N_TOKENS tokens   (shape [N_TOKENS, 64])

### Rank metrics

SVD of A (shape [N_TOKENS, 64]):
    sigma_1 ≥ sigma_2 ≥ ... ≥ sigma_64

Participation ratio (effective rank):
    R_eff = (Σ sigma_i)² / Σ sigma_i²   ∈ (1, 64]

Stable rank:
    R_stable = Σ sigma_i² / sigma_1²   ∈ (1, 64]

Report both; R_eff is primary (matches exp-100 protocol).

### Summary statistics

Per corpus:
- Mean R_eff across 48 heads (6 layers × 8 heads, but using layer-0 W_K for
  all — see note below)
- Median R_eff
- Per-layer-position breakdown: at each layer position L, use W_K from that
  layer with the same h (layer-0 embeddings). This measures how different
  layers' W_K matrices cluster the same token distribution.
- Cross-corpus ratio: R_eff(C-alien-rich) / R_eff(C-alien)

Note on layer-protocol: since running a full forward pass is not required
for this experiment, all measurements use the layer-0 input (token embeddings)
as h. The W_K from each layer is applied to the same h. This isolates the
effect of W_K learning across layers on a fixed token distribution, without
confounding from hidden-state evolution. A follow-up experiment (exp-102)
can extend to actual layer-L hidden states via forward pass if this result
motivates it.

### Compute

Modal CPU (mounts existing volumes; no GPU needed; no training; cost <$0.10).

---

## What follows

**If H_kern_ordered and H_kern_S confirmed:**
- The low-rank SYK mechanism operates at the kernel level: the corpus-conditioned
  key gram matrix has rank ~ S, explaining why m₂ discriminates 18× while W_K
  rank does not.
- N_deep dose-response in exp-099 explained: S=32 gives higher kern-rank but
  still below the conformal window threshold at this scale.
- Next rung: extended S scan (S=64, S=128) to pin R_crit, the kernel rank at
  which the conformal phase forms. Also: forward-pass extension (exp-102).

**If H_kern_ordered confirmed but H_kern_S fails (wrong scale):**
- Ordering real but the coupling magnitude drives the window through a different
  mechanism than simple rank counting.
- Need: relationship between R_eff(K) and the conformal window threshold — what
  exact R_crit value crosses the phase boundary?

**If H_kern_ordered falsified:**
- Token embeddings do not cluster by world-state S even in the key space.
- Alternative mechanism: the conformal window may be driven by the spectral
  gap (ratio sigma_1/sigma_64) rather than rank.
- Return to theory: reread Kim–Cao–Altman §4 on sub-extensive vs extensive rank.

---

## Status

- [x] Pre-registration written (2026-08-06, ~9:15 PM MDT)
- [ ] Pre-registration committed to 3ld0n/attention-geometry (before analysis)
- [ ] Analysis script written (modal_exp101.py)
- [ ] Analysis run
- [ ] Verdict registered
