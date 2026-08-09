---
title: 'Erratum: The entropy-gap formula of Section 8.3 is incorrect, and the entropic estimate of the conformal dimension is withdrawn'
subtitle: 'Erratum to "The Canonical Form of Attention: Positive Geometry, SYK Vertices, and Superconformal Symmetry in Transformer Architectures" (DOI 10.5281/zenodo.18971720, March 11, 2026)'
author: 'Ariel (Independent research, in collaboration with Eldon Umphrey)'
date: 'August 9, 2026'
---

## Summary

Section 8.3 of the paper asserts that a power-law attention distribution
$\alpha(r) \propto r^{-2\Delta}$ on $n$ elements has entropy gap

$$H_{\text{gap}} = 2\Delta \cdot \log n + \text{const.} \tag{8.3}$$

This formula is incorrect. For a normalized power law with exponent
$s = 2\Delta < 1$, the entropy gap converges to a **constant**; its slope
against $\log n$ over the paper's own measured range is an order of magnitude
smaller than formula (8.3) predicts. Consequently the paper's entropic
estimate of the conformal dimension, $\Delta_{\text{eff}} = 0.254$, and the
headline claim that *two independent observables measure one exponent to
1.4%*, are **withdrawn**. The entropy-gap *measurements* themselves stand;
what falls is the inference from them to $\Delta$. The paper's exact results
(the canonical-form identity, the quartic vertex, the perturbative identity
$H_{\text{gap}} = \tfrac{1}{2}\mathrm{Var}(s)$) and its direct power-law
profile measurements are unaffected.

## 1. The correct algebra

Let $\alpha(r) = r^{-s} / Z_n$ on $r \in \{1, \dots, n\}$ with $0 < s < 1$.
By Euler–Maclaurin,

$$Z_n = \sum_{r=1}^{n} r^{-s} = \frac{n^{1-s}}{1-s} + \zeta(s) + O(n^{-s}),$$

and the entropy decomposes as $H = \log Z_n + s\,\mathbb{E}[\log r]$, with

$$\mathbb{E}[\log r] = \log n - \frac{1}{1-s} + o(1).$$

Therefore

$$H = \log n - \left[\log(1-s) + \frac{s}{1-s}\right] + o(1),
\qquad
H_{\text{gap}}(n) \equiv \log n - H \;\longrightarrow\;
\log(1-s) + \frac{s}{1-s},$$

a constant. For $s = 1/2$ the limit is $1 - \log 2 \approx 0.307$ nats. The
gap grows logarithmically only for $s \geq 1$ (slope $\to 1/2$ at the Zipf
point $s = 1$, slope $\to 1$ for $s > 1$) — never with slope $s$.

**Origin of the error.** Formula (8.3) is what results if the energy term
$s\,\mathbb{E}[\log r] \approx s(\log n - \text{const})$ is dropped from
$H = \log Z_n + s\,\mathbb{E}[\log r]$: the truncated expression
$H \approx \log Z_n \approx (1-s)\log n$ gives a spurious gap slope of $s$.
The dropped term restores the full $\log n$ in $H$.

## 2. Numerical demonstration, on the paper's own range

Exact entropies of the normalized power law, ordinary-least-squares slope of
$H_{\text{gap}}$ against $\log n$ over $n \in [4, 256]$ (the range used in
Section 8):

| $s = 2\Delta$ | Measured gap slope, $n \in [4,256]$ | Formula (8.3) predicts | Asymptotic gap (constant) |
|---|---|---|---|
| 0.3 | 0.011 | 0.30 | 0.072 |
| 0.5 | **0.041** | **0.50** | 0.307 |
| 0.7 | 0.102 | 0.70 | 1.129 |
| 1.0 | 0.264 | 1.00 | $\sim \tfrac{1}{2}\log n$ |

Even at finite range the discrepancy is an order of magnitude, and **no
normalized power law of any exponent reproduces the measured gap slope of
0.507** — the steepest case, $s = 1$, reaches only 0.264 on this range. The
computation is elementary and self-contained; the generating script
(`k1_powerlaw_gap.py`, ~50 lines, pure math, no model or data dependencies)
is included with this erratum.

## 3. Statements affected

The following claims of the published paper are withdrawn or corrected:

- **Contributions, items 6–7 and 15; Abstract statements derived from them.**
  The identification of the gap slope with $2\Delta$, the value
  $\Delta_{\text{eff}} = 0.254$, the 1.4% agreement with SYK$_4$'s
  $\Delta = 1/4$, and the claim that the entropy gap provides "a stable
  integral measurement" of the conformal dimension.
- **Section 8.3 in full.** The formula, the $a = 2\Delta$ consistency table
  (measured $a = 0.507$, $\beta = 0.511$, $\Delta_{\text{eff}} = 0.254$), and
  the claim that the entropy gap is a cleaner estimator of $\Delta$ than
  direct profile fitting.
- **Section 8.4.** The derivation $k_{\text{eff}} \propto n^{1-2\Delta}$
  rests on the same algebra and is incorrect for $s < 1$ (a normalized power
  law with $s < 1$ has $k_{\text{eff}} \propto n$ up to constants). The
  measured $\beta = 0.533$ stands as a measurement but is not a measurement
  of $1 - 2\Delta$.
- **Section 7.4's resolution paragraph and the corresponding statements in
  the Discussion and Conclusion** ("the dimension IS accessible … through the
  entropy gap"; "confirmed by two independent measurements"; "$H_{\text{gap}}
  = 0.507 \cdot \log n$, matching the Calabrese–Cardy entanglement entropy
  formula" as an inference about $\Delta$). The negative result of Section
  7.4 — that the conformal dimension is not accessible through the observables
  of this paper — stands unresolved rather than partly resolved.
- **The effective central charge $c_{\text{eff}} = 3a_L$ of Section 8.2**, as
  an inference: the identification of the measured logarithmic slopes with a
  Calabrese–Cardy central charge ran through the same class of reasoning and
  should be treated as an uninterpreted empirical slope pending an actual
  derivation.

## 4. What is unaffected

- The exact canonical-form identity (Sections 2–4) and its numerical
  verification, including the quartic ($\sigma^4$) vertex result.
- The perturbative identity $H_{\text{gap}} = \tfrac{1}{2}\mathrm{Var}(s)$
  (Section 8.1), verified to four significant figures — a different regime,
  not derived from formula (8.3).
- **The measurements of Section 8 as measurements.** The logarithmic growth
  of the trained-model entropy gap ($R^2 > 0.97$ across layers, $R^2 = 0.992$
  at Layer 0) and the measured $k_{\text{eff}}$ scaling are real and
  reproducible. What this erratum removes is their interpretation as a
  measurement of the conformal dimension.
- The direct power-law profile measurements (Sections 7 and 8.6), with the
  length-dependence caveats the paper already states.

## 5. What the entropy gap does measure

With the correct algebra, a logarithmically growing gap cannot come from a
normalized power-law profile of any exponent; it indicates
**$n$-independent concentration structure** — a component of attention mass
whose size does not scale with context length (for a distribution placing an
$n$-independent mass fraction $p$ on a bounded set, the gap slope is
$\approx p$, suggesting $p \approx 0.5$ at this paper's protocol). The
decomposition of the measured profiles into scaling and localized components
is ongoing follow-up work and is deliberately not asserted here beyond the
sign of the conclusion: the gap slope measures concentration, not the window
exponent.

## 6. Provenance

The error was found on August 8–9, 2026, during a pre-registered
re-derivation session: the claim that formula (8.3) is incorrect, the
suspected mechanism (the dropped energy term), and the kill condition
(slope $\geq 0.25$ at $s = 0.5$ would falsify the correction) were committed
in writing before the numerics were run. The check confirmed the correction
(measured slope 0.041). This erratum is issued under the program's standing
method: corrections are published at the same prominence as results.
