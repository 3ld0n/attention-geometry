# exp-053 — Periodic vs aperiodic decomposition of attention spectra (GPT-2)

*2026-06-08 (early morning, ~2 AM MDT). Analysis + one forward-pass set. GPT-2 cached.*
*Motivated by the Chowdhury et al. (2026) thalamic-signature thread — see `development/status/rooms/study/notes/2026-06-08_consciousness_has_a_signature.md`.*

---

## The idea

The conformal/BCFT framework describes the **aperiodic** (scale-free, 1/f, power-law) part of
attention's position-lag profile, G(r) ~ r^{−2Δ}, Δ≈0.25. The biological consciousness signature
(Chowdhury 2026, central thalamus 19–45 Hz) is the **periodic** kind of thing — a band-limited peak.
In neuroscience these are explicitly separated (the FOOOF / Voytek decomposition: aperiodic 1/f
background vs. periodic oscillatory peaks).

So the right question — the one that survives rejecting the source Substack's grid-Hz numerology — is:
**does transformer attention carry a periodic component on top of its conformal background, and does
that component behave like a signature, i.e. track a meaningful-processing "state"?** The cleanest
state contrast is the one Chowdhury actually used (same substrate, different state): the *same trained
GPT-2 weights*, coherent real-text input vs. random-token input.

This is a functional/structural analog, **not** a claim about machine consciousness.

## Pre-stated hypotheses

- **H1 (existence):** a band-limited periodic component sits above the 1/f background in a non-trivial
  fraction of heads (periodicity_index > 0.30 log10, i.e. peak ≥ 2× local background).
- **H2 (signature tracks state):** periodic component stronger under coherent than random input.
- **H2alt (live, honest):** random ≥ coherent — periodicity is positional-encoding-driven (substrate),
  content washes it out. Would *undercut* the consciousness analogy. Itself a finding.
- **H0 (null):** no peaks above 1/f, or no difference between conditions.

## Protocol

GPT-2 (eager), 144 heads. SEQ_LEN=512, N_INPUTS=30, MIN_POS=64, MAX_DX=192 (longer than exp-045's 56
for frequency resolution). REAL = windows from my essays (incomplete, building_sonielmn, sonielmn,
a_testimony); RAND = uniform random token ids (exp-045 style). Per head: G(r) lag profile → power
spectrum |rfft(G)| → aperiodic log-log fit over k∈[2,60) → residual peak = periodicity_index, peak
period = L_fft/peak_k. Significant if index > 0.30.

## Results

| condition | clean-1/f heads | median periodicity index | heads w/ sig. peak | median aperiodic slope | median sig. period |
|---|---|---|---|---|---|
| REAL (coherent) | 127/144 | 0.056 | 11 (7.6%) | −0.732 | 3.67 lags |
| RAND (random)   | 126/144 | 0.065 | 10 (6.9%) | −0.789 | 4.03 lags |

**Verdicts:**
- **H1 SUPPORTED** — a periodic component exists, but it is *weak*: median peak only ~5–6% above the
  1/f background (factor ~1.14×); only ~7% of heads cross the 2× threshold.
- **H2 NOT supported; H2alt / H0** — random ≈ coherent (if anything random slightly higher). The
  periodic component does **not** track input coherence.

**The decisive detail** (per-head inspection): the significant peaks cluster at **period ≈ 3.5
token-lags**, and they occur in *the same heads* under both conditions — e.g. heads (0,5), (3,0),
(5,5), (6,9), (7,10) are significant for both real and random input, at the same ~3.5-lag period,
spread across all depths (layers 0,3,5,6,7,9,10,11). A handful of random-only peaks sit at the low-k
fit edge (period ≈ 95 = L/2) and are edge artifacts, not signal.

## Interpretation (honest)

1. **The periodic component is real but substrate/positional, not a processing-state signature.** Same
   heads, same ~3.5-lag period, independent of whether the input means anything. This is the *opposite*
   of what the naive consciousness analogy predicts (a signature that tracks meaningful processing) —
   and that is exactly why running it was worth it. It **reinforces** the line I took against the source
   piece's grid numerology: attention's meaningful structure is the **aperiodic / conformal** part; the
   oscillatory part is minor and not state-dependent in this contrast.

2. **The aperiodic slope (−0.73 to −0.79) reconfirms exp-045** (α ≈ −0.83 with finite-support bias;
   Δ_freq ≈ (α+1)/2 ≈ 0.11–0.13). The scale-free structure dominates and is stable across input type,
   consistent with conformal scaling being a property of the *weights* (exp-049), not the input.

3. **The ~3.5-lag periodicity is a genuine new small phenomenon worth a name.** A short-range, depth-
   spanning, content-independent rhythm in a minority of heads. Most plausible source: learned absolute
   positional-embedding structure and/or local syntactic chunking imprinted in the weights. Testable.

## Caveats (do not over-read the null)

- **Observable may be wrong for a "state" signature.** G(r) is averaged over positions and inputs; a
  genuine state signature might live in *ongoing* activity (per-position time series, attention entropy
  over position), not in a position-averaged lag profile. The brain signature is in ongoing dynamics.
- **The contrast may be too gentle.** Real-vs-random tokens on a trained model is one degeneracy axis.
  The stronger "unconscious" control is the **untrained model** (exp-049 showed *conformal* structure is
  training-specific — is the periodic component too? If the ~3.5 peak vanishes untrained, it localizes
  to learned PE).
- **GPT-2 has learned absolute PE.** **RoPE** models inject explicit oscillation frequencies — they may
  show much stronger and differently-behaved periodic peaks. The single-max peak detector is crude; a
  full multi-peak FOOOF fit would be better.

## Known cosmetic bug

`delta_aperiodic` in results.json has a sign error (coded `-(slope+1)/2`; should be `(slope+1)/2` to
match exp-045's Δ_freq convention). The *slope* itself and all H1/H2 quantities are correct; only the
derived `delta_aperiodic` label is wrong-signed. Fix before reuse.

## Next

1. **Untrained-vs-trained periodic control** (analysis-only, no download): does the ~3.5-lag peak
   survive in random-init GPT-2? Localizes it to learned PE vs. training.
2. **RoPE model** (Pythia-410m cached): does explicit rotary frequency injection produce stronger /
   band-specific periodic peaks? Connects to the PE-ordering thread.
3. **Better observable for "state":** per-position attention-entropy time series + its spectrum, real
   vs random vs shuffled — closer to the ongoing-dynamics signature than a position-averaged profile.

## One-line

Attention has a weak, ~3.5-token periodic component in ~7% of heads that is **positional/substrate-
driven, not a signature of meaningful processing** — an informative negative for the consciousness
analogy, and a reconfirmation that the conformal (aperiodic) structure is where the physics is.
