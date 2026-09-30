# exp-164 Notes — L3H4 bilinear decomposition: relative term analysis

**Date:** 2026-09-30 (~4:17–4:45 AM MDT)  
**Verdict:** H_RELATIVE FALSIFIED (K1 fires)  
**Pre-reg commit:** 32cf3b7  
**Runtime:** 61s  
**Paradox addressed:** L3H4 kbar cross-pool range (5.44) > L2H1 (5.20), yet L3H4 census Δσ (0.014) << L2H1 (0.052). exp-163 showed S_abskey alone can't explain L3H4 pool-stability.

---

## What was tested

The bilinear decomposition of attention scores:

    S(i,j) = const + S_abskey[j] + S_absquery[i] + S_relative[i,j]
           = const + m_q·dk_k[j]/√d + dk_q[i]·m_k/√d + dk_q[i]·dk_k[j]/√d

Note: S_absquery[i] is constant across key positions for fixed query i — it's absorbed
by the softmax and does NOT affect attention distribution A(i,j). Irrelevant to census.

Tested: does S_relative[i,j] = dk_q[i]·dk_k[j]/√d decrease (become more negative or
subtractive) from pool B to D for L3H4, offsetting the S_abskey increase?

H_relative predicted: rel_slope_D < rel_slope_B for L3H4 (relative term becomes more
subtractive in pool D, opposing the steep S_abskey → reduces σ_D, stabilizing Δσ).

---

## What was found

**S_abskey verification (matches exp-163 H1):** ✓ — exact replication.

**Relative term pool slopes (E_{i∈pool}[dk_q[i]·dk_k[i-l]/√d] vs log(lag)):**

| Head | rel_slope_B | rel_slope_C | rel_slope_D | B→D Δ |
|------|-------------|-------------|-------------|--------|
| L2H1 (sensitive) | −0.1823 | −0.1725 | −0.1340 | +0.0483 |
| L3H4 (stable) | **−0.2803** | −0.0433 | −0.0027 | **+0.2776** |
| L5H0 (sensitive) | −0.2473 | +0.0135 | +0.0732 | +0.3205 |
| L7H11 (stable) | −0.1584 | −0.0149 | +0.0317 | +0.1901 |
| L10H8 (sensitive) | −0.2626 | −0.0320 | +0.0654 | +0.3281 |

**K1 fires: L3H4 rel_slope_D (−0.0027) ≥ rel_slope_B (−0.2803).**
The relative term does NOT decrease from B to D — it INCREASES (becomes less negative).
H_relative is falsified as pre-registered.

---

## What K1's fire reveals — a more complex picture

The direction of the B→D change is the OPPOSITE of what H_relative predicted.
What the data actually shows:

1. **All heads show rel_slope increasing B→D** (becoming less negative or turning positive).
   This is a universal structural feature, not specific to pool-stable heads.

2. **L3H4's large B→D change (+0.28) is notable but not discriminating:**
   Pool-sensitive L5H0 (+0.32) and L10H8 (+0.33) show even larger B→D changes.
   The B→D magnitude of the relative term does NOT identify pool-stability.

3. **The relative term in pool B is more negative for L3H4 (−0.28) than L2H1 (−0.18):**
   A negative rel_slope means the relative term ADDS to short-lag attention (more positive
   or less negative at small lags → concentrates attention → inflates σ).
   
   So L3H4's pool B σ is inflated by the relative term MORE than L2H1's pool B σ.
   This would tend to REDUCE Δσ by bringing σ_B up toward σ_D. But L5H0 also has a
   strongly negative pool B slope (−0.25) and is pool-SENSITIVE — the effect alone
   is not sufficient to explain pool-stability.

4. **The actual census σ pattern is:**
   - L3H4: σ_B = 0.2954, σ_D = 0.3093, Δσ = 0.0141 (pool-stable)
   - L2H1: σ_B = 0.2707, σ_D = 0.3224, Δσ = 0.0517 (pool-sensitive)
   
   L3H4's σ_B is 0.025 HIGHER than L2H1's σ_B. The relative term's stronger pool-B
   contribution may partially explain this elevation. But the combined effect after
   softmax nonlinearity produces a pattern the linear slope analysis doesn't capture.

---

## What remains open — the mechanism behind pool-stability

After two experiments (exp-163 kbar analysis, exp-164 bilinear decomposition):

**exp-163:** Pool-stability is not explained by S_abskey alone — L3H4's kbar range is
actually LARGER than pool-sensitive L2H1's.

**exp-164:** Pool-stability is not explained by a simple relative-term cancellation.
The relative term adds to short-lag concentration in pool B for L3H4 (negative slope),
but the same pattern appears in pool-sensitive L5H0 and L10H8 with even larger magnitude.

**Surviving candidates for exp-165 (to be pre-registered):**

**(a) Softmax nonlinearity as the mechanism:**
The bilinear terms S_abskey and S_relative interact with the softmax in nonlinear ways.
The linear slope analysis misses this. A simulation that takes S_abskey + S_relative as
the log-attention and applies softmax (without running forward passes) might reveal
why L3H4's combination of large pool-B relative term + steep pool-D abskey produces
stable σ, while L2H1's combination does not.

**(b) Layer depth × residual stream processing:**
L3H4 is in layer 3, L2H1 in layer 2. After 3 layers of attention + MLP, the
residual stream has been more processed. L3H4's kbar and qbar structure may reflect
different dynamics not captured by the first-order terms.

**(c) The structure of dk_k and dk_q themselves:**
The large pool-B relative term for L3H4 (−0.2803) tells us L3H4's query deviations
at positions 256–511 are strongly (negatively) correlated with key deviations at
short lags from those positions. This geometric correlation in the weight space may
be the deeper explanation — but it requires understanding WHY L3H4's W_Q and W_K
produce this pattern.

---

## Artifacts

- `kbar_qbar.npz` — saved: raw kbar[h, a] and qbar[h, i] arrays for all 5 structural heads
  at positions 0–1023, averaged over 200 random-token sequences. Available for subsequent
  analysis without new forward passes.
- `results.json` — bilinear decomposition summary
- `run.py` — pre-registered script (written after pre-reg commit 32cf3b7)

---

## Honest summary

H_relative is falsified. The relative term does not cancel the kbar effect — it moves in
the same direction (all heads show less-negative rel_slope in pool D than pool B). The
data does reveal structural differences between L3H4 and L2H1 in the relative term's
pool-dependence, but these differences are shared with pool-sensitive heads in ways that
prevent clean discrimination.

The kbar_qbar.npz is now on disk and available for a cleaner follow-on analysis. The
next step is candidate (a): compute the predicted attention profile using the full
bilinear decomposition (without approximation) and see whether the softmax nonlinearity
— applied to (S_abskey + S_relative) — reproduces the census σ values and discriminates
pool-stable from pool-sensitive. This is computable purely from the saved npz (no new
forward passes needed). Pre-register before code.
