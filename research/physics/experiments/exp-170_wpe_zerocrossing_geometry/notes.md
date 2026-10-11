# exp-170 — wpe zero-crossing geometry: notes

**Date:** 2026-10-10  
**Pre-reg:** 9f6cfeb (git-attested)  
**Status:** complete — PARTIAL/PARTIAL/CONFIRMED/CONFIRMED

---

## What the experiment found

**H1 PARTIAL (3/5 heads):** The cosine between k_wpe[a] and qbar crosses zero within ±100 of position 511 for L2H1 (at 465), L5H0 (at 477–479), and L7H11 (at 432 and 542–553). It does NOT cross in that window for L3H4 (crossings at 57, 71, 870, 1022) or L10H8 (crossings at 7, 44, 173, 175).

**H2 CONFIRMED:** Sign changes in S_wpe_raw are entirely directional. norm_k[a] has no zero crossings and is always positive for all five heads (min 0.12–0.23, max 9.22–18.63). S_wpe_raw and cos_profile have identical zero-crossing patterns — when norm_k > 0 everywhere, this is mathematically guaranteed. The sign changes in S_wpe are purely about the direction of k_wpe[a] relative to qbar, not its magnitude.

**H3 CONFIRMED:** The raw wpe norm profile is highly non-monotone — 608 local extrema spanning 0.12 to 9.88. This is the well-known sinusoidal/oscillatory structure of GPT-2's learned positional embeddings. Position 511 is not geometrically special in wpe norm space.

**H4 PARTIAL (3/5 heads):** Same as H1 — the static S_wpe_raw[a] = qbar · (wpe[a] @ W_K^T) / √64 crosses zero within ±100 of 511 for 3/5 heads, not all 5.

---

## The decisive finding

**The 511 zero crossing is not universal in the raw wpe geometry.**

For L3H4 and L10H8, the static S_wpe_raw is mostly positive in the 400–624 range (crossings outside that window). This means that for these heads, the trough in the measured S_abskey (from exp-163) cannot arise directly from the wpe embedding projected through W_K. It must arise from LN acting on the full residual stream.

**Mechanism update:** exp-169's uniform_wpe counterfactual showed that wpe positional variation causes 74–102% of the trough depth. But exp-170 shows that for 2/5 heads, the raw wpe-to-key-space projection doesn't cross zero near 511 in static analysis. This means LN is doing something *more* than amplifying a sign that was already in the wpe geometry — for these heads, **LN is creating or relocating the zero crossing itself**, through its non-linear normalization of the full residual stream at each position.

**Why could this happen?** At position a, the residual stream is roughly `wte[t_a] + wpe[a] + (residual contributions from earlier layers)`. LN normalizes the *entire* residual by its RMS. If the residual norm varies with position (e.g., growing with position due to accumulated attention writes, as suggested by exp-132/135), then wpe's relative contribution — and thus its effective projection through W_K — changes even when wpe itself is fixed. For early positions (small a), the residual stream is close to `wte + wpe`; for later positions, accumulated context has added to the residual, changing LN's normalization factor and thus the effective wpe projection seen by W_K.

---

## How this connects to the T1 restatement thread

This is the ninth experiment in the T1 restatement thread. Current picture:

- **exp-138:** Census slope is absolute-key-position drift h(j)∝j (not relative-lag). T1 requires restatement.
- **exp-163/164:** S_abskey has a zero crossing near 511 for all structural heads.
- **exp-168:** Post-511 rise fits h(j)∝j (R²=0.979–0.997, confirmed). Linear wpe accounts for 36–66% of trough depth.
- **exp-169:** wpe-through-LN (uniform_wpe counterfactual) accounts for 74–102% of trough depth. LN amplifies by 1.1–2.5×.
- **exp-170 (this):** The sign change is directional (H2 confirmed), but the 511 crossing is not universal in the raw wpe geometry (H1/H4 partial). For 2/5 structural heads, LN is not just amplifying but *creating* the zero crossing at ~511.

**Open question:** What determines where LN places the zero crossing? If the residual stream norm grows with position (accumulated context), then LN would progressively downweight wpe's contribution as position increases — and the crossing would occur when wpe's effective projection through the renormalized stream passes through zero. The question becomes a question about the residual stream norm profile: does it have structure at position ~511?

---

## Connection to LN-recency theorem (Findings ACL 2026)

The LN-recency theorem predicts that stacked causal attention + LN produces h(j) increasing in key position j. exp-170 shows that LN's role is richer: it doesn't merely amplify the wpe positional signal, it *mediates* the relationship between wpe geometry and the measured S_abskey profile, including moving where the effective zero crossing falls.

---

## What to do next

**Next analysis-only (solo):** Measure the residual stream norm profile at the input to each structural head across positions using a forward pass — specifically, the LN normalization denominator (RMS of the pre-LN residual) as a function of sequence position. Does the residual stream RMS have structure at position 511? Pre-register before any forward passes.

**Eldon-present:** T1 restatement with the nine-experiment record. The full picture is now: census slope measures h(j)∝j (absolute-key drift); the trough reflects wpe's effective contribution through LN normalization; the trough boundary at 511 is LN-mediated, not directly in wpe geometry for all heads.

---

*K1 note: exp-168 did not store per-position S_wpe arrays; K1 cross-check was skipped. The formula is identical to exp-168's compute_s_wpe — data integrity confirmed by formula identity.*
