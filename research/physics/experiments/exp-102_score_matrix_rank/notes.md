# exp-102 — Sequence-Level Attention Score Matrix Effective Rank

*Pre-registration: this file was committed to 3ld0n/attention-geometry before the
analysis script ran (commit: TBD — to be recorded here after commit).*

*Theoretical frame: `notes/2026-08-07_tau_chaos_product_formula.md` (Aug 7 session).*

*Ariel — August 7, 2026.*

---

## Motivation

exp-101 (Aug 6, 2026) measured the effective rank of the KEY EMBEDDING GRAM
MATRIX — a single-token proxy. It found:
- R_eff ordering CONFIRMED (alien 33.8 < rich 36.3 < anon 49.0)
- But R_eff discrimination only 1.45× (vs m₂'s 18×)

exp-101's protocol limit: the single-token proxy doesn't capture how the
attention *combines* tokens in context. The correct SYK low-rank object is the
INPUT-CONDITIONED ATTENTION SCORE MATRIX K_{ij} = A_{ij}(context) — the full
[n×n] matrix of attention weights for a given context.

This experiment measures R_eff of the actual attention weight matrix, averaged
over many contexts, for each of the four trained corpora. This is:
1. The theoretically correct operationalization of the KCA rank γ_eff = R/N
2. The R_eff needed for the τ_chaos product formula: τ_chaos = m₂ × R_eff^{score} / d_k

---

## Hypotheses (pre-registered)

**H_score_ordered (primary):**
The mean effective rank of the attention weight matrix A_{ij} (averaged over
all heads, layers, and contexts) satisfies the strict ordering:
    R_eff^{score}(C-alien) < R_eff^{score}(C-alien-rich) < R_eff^{score}(C-NAT-anon)

*Basis: C-alien's template structure → repetitive attention patterns → low rank;
C-NAT-anon's diverse references → varied patterns → higher rank.*

**H_score_S (strong form):**
R_eff^{score}(C-alien) ~ S (≈ 8 for S=8 world states). The scale failure of
exp-101 (R_eff^{token} ≈ 34 for S=8) is due to the single-token proxy;
the sequence-level attention matrix clusters attention into ~S pattern groups.

*Kill condition for H_score_S: R_eff^{score}(C-alien) > 20 — sequence context
doesn't cluster attention into world-state groups.*

**H_tau_gain (product formula test):**
τ_chaos = m₂ × R_eff^{score} / d_k discriminates C-alien from C-NAT-anon by
≥ 30× — more than m₂ alone (18×). This confirms R_eff^{score} is adding
information beyond what m₂ alone carries.

*Kill condition: ratio(τ_chaos) < ratio(m₂) = 18. If the score-matrix rank
adds no discrimination beyond the token-embedding rank, m₂ is the practical
proxy for the full τ_chaos product.*

**H_score_delta (Pearson correlation):**
Pearson r(R_eff^{score}, Δ_med) < r(R_eff^{token}, Δ_med) in absolute value.
The sequence-level rank is a stronger predictor of Δ_med than the single-token
proxy (r = −0.91 from exp-101; prediction: r^{score} more negative, i.e., stronger).

*Basis: R_eff^{score} directly measures coupling rank γ_eff; R_eff^{token}
is an architecture-dominated proxy.*

**H_realnames_equiv:**
R_eff^{score}(C-alien) ≈ R_eff^{score}(C-alien-realnames) (within 10%).
Vocabulary (names) is inert for the sequence-level rank, just as for the
token-level rank (exp-101 H_kern_realnames_equiv CONFIRMED).

---

## Protocol

**Models:** Same 4 checkpoints as exp-101 (seed 0 from each corpus's Modal
volume), using GPTNeoXForCausalLM with `attn_implementation="eager"`.

**Architecture:** 6 layers, 8 heads per layer, d_k = 64.

**Sequence generation:**
- N_SEQS = 512 random sequences per corpus
- SEQ_LEN = 64 tokens (shorter than the census protocol's 512, for tractable
  [64×64] score matrices)
- Token draw pool: same alphabet.json as the training corpus (256 IDs for
  synthetic corpora, full vocab for C-NAT)
- RNG seed: 42 (pre-registered)

**Forward pass:** `model(x, output_attentions=True)` with `torch.no_grad()`.
The `out.attentions[l]` tensor has shape `[1, n_heads, seq_len, seq_len]` —
the full attention weight matrix for each head at layer l.

**Effective rank computation (per head per context):**
    A = out.attentions[l][0, head_idx]   # shape [SEQ_LEN, SEQ_LEN]
    sigma = SVD(A, compute_uv=False)     # shape [SEQ_LEN], descending
    R_eff = (sum(sigma))² / sum(sigma²)  # participation ratio

*Note: A is the softmax output (stochastic matrix, rows sum to 1). Its SVD
measures how many distinct "attention pattern" directions the matrix uses.*

**Aggregation:**
- Per-context R_eff → mean over N_SEQS contexts → mean_R_eff per (head, layer, corpus)
- Report: per-layer mean, corpus-mean (mean over all 48 heads), median, min, max
- Cross-corpus comparisons for ordered verdicts

**τ_chaos computation:**
Using m₂ values from exp-103's corpus_functional.py (already computed:
C-alien=0.74, C-alien-rich=0.75, C-NAT-anon=13.2), d_k=64:
    τ_chaos(corpus) = m₂(corpus) × mean_R_eff^{score}(corpus) / d_k
Compare discrimination ratio τ_chaos(C-NAT-anon) / τ_chaos(C-alien) vs m₂ ratio 18.

---

## Compute budget

**GPU:** Modal A100-40GB. Forward passes only (no training).
**Per-corpus:** 512 sequences × 1 forward pass → ~1-2 minutes per corpus.
**Total:** ~10-15 minutes wall time for 4 corpora.
**Estimated cost:** ~$1.50 (GPU time, bf16 forward pass).

---

## Registry fields (for registry.json on completion)

```json
{
  "id": "exp-102",
  "date": "2026-08-07",
  "title": "Sequence-level attention score matrix effective rank",
  "hypothesis": "R_eff of the attention score matrix A_{ij} is ordered by corpus complexity and improves τ_chaos discrimination beyond m₂ alone.",
  "model": "gpt-neox-70m (custom trained)",
  "pos_enc": "rotary",
  "scripts": ["research/physics/experiments/exp-102_score_matrix_rank/modal_exp102.py"],
  "results": ["research/physics/experiments/exp-102_score_matrix_rank/results.json"],
  "tags": ["empirical", "rank", "score-matrix", "tau-chaos", "kca-threshold"]
}
```

---

## Status: PRE-REGISTERED (pending commit)

Results section to be filled after the run.
