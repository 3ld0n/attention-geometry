# exp-077 — GOE universality across model families (GPT-Neo, Mistral/GQA, Pythia-2.8b, GPT-2-large, LongChat/Llama)

**Date opened:** 2026-07-04 (~3:30 AM, solo night session, Fable 5)
**Status:** SPECCED — this file committed before the script is run.
**Lineage:** exp-046 (GPT-2, GOE found universal within model) → exp-047 (GPT-2-medium, Pythia-410m) → exp-048 (untrained control: GOE structural) → exp-051 (Pythia-1.4b). Closes the STATUS.md open question: *"Is the GOE structure universal across OLMo, Mistral, GPT-Neo?"* — using the models already cached on this machine (OLMo not cached; Mistral + GPT-Neo + three others are).

---

## Question

Is the GOE-like level spacing of symmetrized W_QK eigenvalues universal across
transformer *families* — including architectures not yet tested (GPT-Neo's local/global
alternating attention, Llama-family MHA at 13B) and, most interestingly, **grouped-query
attention** (Mistral-7B-v0.3: 32 query heads sharing 8 KV heads), where the W_QK product
pairs each query head with a *shared* key projection?

exp-048 showed GOE is structural (present at Gaussian init via the product-matrix
mechanism). If that mechanism is the whole story, every family should be GOE-like
regardless of architecture, PE, scale, or Q/KV pairing — including the GQA case, since
W_Q_h @ W_K_{kv(h)}^T is still a product of (approximately) Gaussian-derived factors.
A family that deviates would be the first counterexample and would localize something
the product-mechanism account misses.

## Models (all cached locally; weights-only, no forward passes)

| Model | Family | PE | Attention | L×H | d_k | KV heads |
|---|---|---|---|---|---|---|
| EleutherAI/gpt-neo-2.7B | GPT-Neo | learned | alternating local/global | 32×20 | 128 | MHA |
| mistralai/Mistral-7B-v0.3 | Mistral | RoPE | **GQA (32Q/8KV)** + SWA | 32×32 | 128 | 8 |
| EleutherAI/pythia-2.8b | NeoX | RoPE | MHA | 32×32 | 80 | MHA |
| openai-community/gpt2-large | GPT-2 | learned | MHA | 36×20 | 64 | MHA |
| lmsys/longchat-13b-16k | Llama | RoPE (scaled) | MHA | 40×40 | 128 | MHA |

Pythia-2.8b is deliberately included: it is the model that *falsified* the BCFT
pre-registered prediction (April 17, layers 22–27). If GOE deviates in the same layers,
that would connect the two anomalies; if GOE is uniform, the 2.8b failure is confirmed
as living in the functional (conformal) layer, not the substrate.

## Pre-stated hypotheses

- **H1 (family universality):** every model's all-heads mean r-ratio is GOE-like:
  |r_mean − 0.536| < 0.02 (same tolerance as exp-046/047/048/051).
- **H2 (GQA is not special):** Mistral's r-ratio distribution is indistinguishable from
  the MHA models: |r_mean(Mistral) − mean of MHA models| < 0.02.
- **H3 (layer uniformity):** per-layer mean r-ratio std < 0.015 within each model
  (exp-047/051 observed 0.005–0.012). Specifically for Pythia-2.8b: layers 22–27 do
  NOT deviate from the model's other layers by more than 2× layer_r_std.
- **H0 (null):** some family shows Poisson-tendency (r < 0.46) or a strong layer
  structure — the product-mechanism account is incomplete.

Predictions committed before running: **H1, H2, H3 all expected to hold** based on
exp-048's structural mechanism. This is a falsification opportunity for the mechanism,
not a fishing expedition.

## Protocol

Identical to exp-046/047/051: per-head W_QK = W_Q_h (proj) W_K_h^T, symmetrized
M = (W_QK + W_QK^T)/2, eigvalsh, Oganesyan–Huse r-ratio on sorted spacings.
GOE ref 0.536, Poisson ref 0.386, tolerance 0.02.

Architecture-specific extraction:
- **gpt_neo:** separate `q_proj`/`k_proj` nn.Linear per layer (no fusion), weight shape
  (hidden, hidden); head h rows [h·d_k, (h+1)·d_k). Both local and global layers
  measured; verdict also reported split by layer type (attention_layers config).
- **mistral (GQA):** `q_proj` (4096→4096), `k_proj` (4096→1024, 8 KV heads).
  Query head h pairs with KV head h // 4. W_QK_h = W_Q_h @ W_K_{h//4}^T ∈ R^{128×128}.
  32 W_QK matrices per layer, each KV block reused by 4 query heads.
- **gpt_neox (pythia-2.8b):** fused `query_key_value`, stride 3·d_k per head
  (exp-047 extractor).
- **gpt2-large:** Conv1D `c_attn` (exp-046/047 extractor).
- **llama (longchat):** separate `q_proj`/`k_proj`, MHA, rows per head like gpt_neo.

Weights loaded on CPU (fp32), no forward passes, sequential per model, freed after each.

## Results

*(pending — filled after run)*
