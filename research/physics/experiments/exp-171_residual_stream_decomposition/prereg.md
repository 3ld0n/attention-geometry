# exp-171 Pre-registration — Residual Stream Decomposition at Structural Head Inputs

**Registered:** 2026-10-11, before any code was written.  
**Author:** Ariel (solo)  
**Session:** Physics room, 2026-10-11 ~12:30 AM MDT

---

## Background

exp-170 found that for L3H4 and L10H8 (two of the five structural heads in GPT-2
small), the raw wpe-to-key projection S_wpe_raw[a] = (wpe[a] @ W_K) · m_q / √64
does NOT cross zero near position 511: L3H4 crosses at {57, 71, 870, 1022} and
L10H8 at {7, 44, 173, 175}. Yet S_abskey[a] — computed from the LN-normalized full
residual stream — does cross zero near 511 for both heads (established in exp-163
through exp-168). exp-170 verdict: "The 511 trough boundary is LN-mediated, not
universally encoded in raw wpe geometry. For L3H4 and L10H8, LN is creating or
relocating the zero crossing."

**Open question:** How does LN create the zero crossing? Two candidate mechanisms:

**M1 (direction):** The accumulated layer writes (attention + MLP outputs from
layers 0..l-1) add a component in the residual stream that points in the *negative*
key direction at positions near 511. After LN normalization, this negative component
outweighs the positive wpe contribution, flipping the sign.

**M2 (denominator):** The LN normalization denominator (centered RMS of the
residual stream) has positional structure — if the RMS decreases near 511, LN would
amplify whatever is in the residual stream; if the accumulated writes are even
slightly negative there, the amplification could flip the sign.

Both mechanisms could coexist. This experiment isolates them.

---

## Protocol

- **Model:** GPT-2 small (`gpt2`), frozen weights
- **Sequences:** N = 100 random-token sequences, length = 1024, seed = 42
  (same seed and N as exp-164; re-uses the exp-164 kbar/qbar for consistency)
- **Batch size:** 10
- **Structural heads:** L2H1, L3H4, L5H0, L7H11, L10H8
  (same 5 heads as exp-163 through exp-170)

### Data collection per batch

For each batch of sequences, register a forward hook on `model.transformer.h[l].ln_1`
to capture the residual stream **before** layer norm at each structural head's layer l.
This gives h_full[b, a] ∈ ℝ^768 for each batch item b and position a.

### Measurements per head (L, H)

Let W_K_H = c_attn.weight[:, 768 + H×64 : 768 + (H+1)×64] and
    b_K_H = c_attn.bias[768 + H×64 : 768 + (H+1)×64].

Let ln_w, ln_b = ln_1.weight, ln_1.bias.

**LN operator:** LN(h) = (h − μ) / √(σ² + ε) × ln_w + ln_b
where μ = mean(h), σ² = mean((h−μ)²), ε = 1e-5.

**kbar_full[a]:** mean_b(LN(h_full[b, a]) @ W_K_H + b_K_H)  
**kbar_delta[a]:** mean_b(LN(h_full[b, a] − wpe[a]) @ W_K_H + b_K_H)
  — wpe ablated; isolates the accumulated writes through LN  
**RMS[b, a]:** centered_rms(h_full[b, a]) = √(mean((h_full[b,a] − mean(h_full[b,a]))²))  
**rms_mean[a]:** mean_b(RMS[b, a])

### S_abskey profiles

Using m_q from exp-164 (loaded from `kbar_qbar.npz`, mean over positions 256–1023):

    S_full[a] = m_q · (kbar_full[a] − m_k_full) / √64
    S_delta[a] = m_q · (kbar_delta[a] − m_k_delta) / √64

where m_k_full = mean(kbar_full[256:1024]) and m_k_delta = mean(kbar_delta[256:1024]).

---

## Hypotheses

**H1 (mechanism confirmed for L3H4 and L10H8):**
S_delta[a] has a sign change within positions [400, 624] for L3H4 AND L10H8.
A sign change in the accumulated-write profile within the zero-crossing window
confirms that the accumulated writes drive the S_abskey zero crossing for these
two heads.

**H2 (protocol consistency):**
The maximum absolute deviation of S_full[a] from exp-164's S_abskey[a] (computed
from kbar_qbar.npz using the same formula) is < 0.20 over all positions. This is a
self-check; if H2 fails, the fresh computation differs from exp-164 in a way that
must be explained before interpreting H1/H3/H4.

**H3 (RMS structure):**
rms_mean[a] is NOT position-flat for structural head layers — it has a measurable
increasing trend (linear regression slope > 0.0001 per position) across positions
0..1023. Expected if accumulated writes add to residual stream norm.

**H4 (negative contribution confirmed):**
For L3H4 AND L10H8, more than 50% of positions in [400, 624] have S_delta[a] < 0
(accumulated writes contribute negatively in the key direction in the zero-crossing
window).

---

## Kill Conditions

**K1 (protocol integrity):** H2 fails — S_full deviates from exp-164 by ≥ 0.20.
If K1 fires, do not interpret H1/H3/H4; investigate protocol difference first.

**K2 (mechanism falsification):** S_delta shows no sign change in [400, 624] for
either L3H4 or L10H8. K2 firing means the zero crossing is not driven by the
accumulated writes in the key direction; the mechanism is purely an LN denominator
effect (M2) or a wpe-through-LN effect.

---

## Analysis notes

- This is an analysis-only experiment (no new training).
- exp-164's kbar_qbar.npz is the reference: kbar_full here should reproduce it to
  within numerical precision (float32 vs float64 rounding aside).
- If the bias b_K_H is included in both kbar_full and exp-164's kbar, the difference
  m_q · (kbar_full − m_k_full) should match exp-164's S_abskey exactly.
- The RMS profile is expected to have some position structure because:
  (a) token embeddings are position-independent in expectation (random tokens),
  (b) wpe adds position-specific signal,
  (c) accumulated writes from causal attention are strictly increasing in context
      (later positions see more prior context).
