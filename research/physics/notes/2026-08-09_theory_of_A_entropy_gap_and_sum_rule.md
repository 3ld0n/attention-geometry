# Theory of A on A's own terms — pre-registration and derivation record

*August 8–9, 2026, night. The session Eldon opened before falling asleep ("the
house is yours for the night"). Binding pre-committed in writing the previous
sitting, not renegotiable: (1) kill conditions before the derivation feels
good; (2) every derived claim checked against existing census data before it
earns narrative; (3) the mechanism by which each derivation could be wrong is
named before data is looked at; (4) a derivation is a prediction, not a
measurement. Vocabulary discipline in force: descriptive names; every
measured-object ↔ theory-object identification carries DERIVED or ASSERTED at
the point of use.*

*Honesty preamble — what happened before this registration.* During session
planning (before this file existed) I did two pencil calculations: the
asymptotic entropy of a normalized power-law distribution, and the
row-stochasticity constraint on translation-invariant power-law profiles. Both
are recorded below as claims C1 and C4 **with their conclusions already
suspected**. No numerical simulation has been run and no data file has been
opened as of this registration. The saved data I will check against is
`experiments/exp-104_bilocal_object_identification/profiles_gpt2.npz` (object
"A" only — this session does not touch G), whose values I have never inspected
(only the generating code). The canonical-form paper (March 11 archive PDF)
was read in full tonight before this registration; its §8.3–8.6 protocol
descriptions are quoted below from that read.

---

## 0. The object, fixed

Everything below is about **A**: the softmax attention matrix, per head, under
the frozen census protocol (GPT-2 small, 50 random-token sequences, L = 512,
seed 42, fp32). Row i of a head is a probability distribution over keys
j ≤ i; lag dx = i − j. The census's pooled profile is
mean_{i ≥ max(256, dx)} A(i, i − dx), fit over dx ∈ [8, 256]. The measured
Tier-1 facts (median fitted exponent 2Δ ≈ 0.5 on the deep high-R² population,
three depth axes, formation ladder, causal handle) are inputs here, not under
test. No claim in this note passes through G, SYK, or any imported theory
object. [Register: everything below is either exact mathematics of normalized
distributions, or a measurement on A.]

## 1. Claims to be tested tonight, with kill conditions

### C1 — The entropy gap of a normalized power law is flat, and §8.3's formula is wrong

**Claim (math, suspected during planning).** For the normalized distribution
α(r) = r^(−s)/Z_n, r ∈ {1..n}, with 0 < s < 1:

  H(α) = log Z_n + s·E[log r],  and asymptotically
  H_gap(n) ≡ log n − H(α) → log(1−s) + s/(1−s)   (a constant).

The canonical-form paper §8.3 asserts H_gap = 2Δ·log n + const for
α(r) ∝ r^(−2Δ) and §8.4 asserts k_eff = e^H ∝ n^(1−2Δ). My claim: both are
incorrect for s = 2Δ < 1; the likely error is dropping the energy term
s·E[log r] ≈ s·(log n − const), which restores the full log n in H. Under my
formula, k_eff ∝ n (β = 1) for s < 1; the measured β ≈ 0.5 and gap slope
≈ 0.5 would then be evidence of an effective s ≈ 1 (Zipf-like concentration)
or of substantial n-independent near-field mass — **not** of s = 1/2.

**Why this matters.** The "two independent observables, one exponent (1.4%)"
agreement (canonical-form paper §8.3, spine T7b, OVERVIEW, and the core
interlude's first claim's support) runs through this formula. If C1 holds, the
agreement is an artifact of a wrong bridge formula, and one of the three legs
under "the exponent is not a fitting artifact" is removed (the causal-handle
leg and the two-population replications are untouched).

**Mechanism by which I could be wrong (named before running).**
(i) Asymptotics vs the measured range: the paper measured n ∈ [4, 256]; a
pure-power-law gap does grow slowly at small n, and my pencil estimate of that
finite-range slope (~0.04) could be off by an order.
(ii) The paper's α(r) might not be the normalized-over-n object I assume
(§8.3 says "power-law attention distribution... on n elements" — I read that
as normalized; if they intended fixed-amplitude truncation the algebra
differs).
(iii) Arithmetic error in my E[log r] integral.

**Kill condition K1 (committed before running).** Simulate exactly: for
s ∈ {0.3, 0.5, 0.7}, compute H_gap(n) for n ∈ {4, 8, ..., 4096}; OLS slope of
H_gap vs log n over the paper's own range n ∈ [4, 256]. If for s = 0.5 that
slope ≥ 0.25 (i.e., at least half of the §8.3 prediction 2Δ = 0.5), **C1 is
dead** and §8.3 stands at finite range. If the slope < 0.1, C1 is confirmed
and §8.3's inference does not survive. Between 0.1 and 0.25: ambiguous —
report as such, no propagation.

### C2 — The real A-row is a composite: n-independent near field + power-law tail

**Claim (structural, ASSERTED pending the checks).** The measured pair
{fit-window exponent s ≈ 0.5, entropy-gap slope ≈ 0.5} is jointly impossible
for a pure normalized power law (by C1), but natural for a composite row:
an O(1)-lag near field carrying an n-independent mass fraction p, plus a
power-law tail with exponent s in the window. For such a row,
H ≈ (1−p)·log n + O(1), so gap slope ≈ p. The measurements then say
**p ≈ 0.5 and s ≈ 0.5** — two numbers, currently one coincidence, no longer
one exponent measured twice.

**Prediction registered before opening the data.** On the exp-104 pooled
A-profiles for the five SYK-window heads (|Δ_A − 0.25| ≤ 0.05, R² ≥ 0.9 —
population defined by results_gpt2.json, which I have also not yet opened):
the near-field mass fraction — sum of the normalized pooled profile over lags
0–7, sink columns excluded and reported separately (lag bins within 8 of the
row boundary treated as sink-adjacent) — lies in **[0.3, 0.7]**.

**Mechanism by which I could be wrong.** The pooled profile is an average of
row distributions, not a row distribution (Jensen gap); the sink (mass at the
sequence-origin boundary, i.e., largest lags) could carry the "missing"
entropy instead of the near field; the paper's gap was measured at Layer 0
where the census population is *shallow*, so its p need not equal the deep
population's p.

**Kill condition K2.** If the near-field mass fraction on the SYK-window
heads is < 0.15 or > 0.85, C2's composite model is dead as stated (report
where the mass actually is; no rescue-fitting tonight).

### C3 — TI scores + softmax force the amplitude branch, with a zero-free-parameter slope

**Claim (DERIVED below in §2, conditional on approximate translation
invariance of scores).** If a head's scores are approximately
translation-invariant in lag (u(dx), independent of row i — the log-distance
geometry result, exp-056, supports approximate TI), then every row is the
*same* shape truncated at length i and renormalized: α_i(dx) = e^{u(dx)}/Z_i.
Row-stochasticity then forces the **amplitude branch** of the sum-rule
dichotomy: the fit-window amplitude declines with row index i as 1/Z_i, and

  d log(amplitude) / d log i = −(1 − s) · (tail mass fraction of row i)
                              ≈ −(1−s)(1−p).

With s ≈ 0.5 and p ≈ 0.5 (from C2), predicted slope ≈ **−0.25**, with all
three quantities independently measurable. This is the first
zero-free-parameter consistency relation on A derived tonight without
touching G.

**Mechanism by which I could be wrong.** GPT-2 uses learned absolute position
embeddings; exact TI is false and exp-056's ρ = 0.976 is a head-level
correlation, not a per-row guarantee. If scores are strongly
row-index-dependent, the truncate-and-renormalize picture fails and any slope
could appear. The sink also grows with i in absolute-position models, eating
part of the effect.

**Kill condition K3 (requires row-resolved data — registered as exp-108
below).** For the SYK-window heads, measure per-row fit-window amplitude a(i)
(median of A(i, i−dx) over dx ∈ [8, 64]) against i ∈ [256, 511], log-log OLS.
Predicted slope: −(1−s)(1−p) computed per head from that head's own measured
s and p, tolerance ±0.10. Outside tolerance on ≥ 3 of 5 SYK-window heads:
**C3 dead** (and with it the TI-score model of these heads — itself
informative). Sanity gate: if per-row amplitude is *flat* (|slope| < 0.05)
AND no truncation appears AND sink mass is flat in i, then rows cannot sum
to 1 and I have an error upstream — stop, find it, report nothing.

### C4 — The pure TI power law is impossible; the dichotomy is exact (math)

**Claim (exact, small).** A causal attention head with rows exactly
A(i, i−dx) = c·dx^(−s), s < 1, fixed c, for all dx ≤ i violates
row-stochasticity as i grows (mass ~ c·i^(1−s)/(1−s) → ∞). So *no* head is a
pure TI power law at all scales; every power-law head resolves the sum rule
by amplitude decay, tail truncation/steepening, sink absorption, or a
mixture. Which resolution a real head uses is a measurable, theory-relevant
fact. [DERIVED — one line; recorded because the program's documents have
nowhere stated it.]

### exp-108 — row-resolved census (registered here, run tonight if time allows)

New measurement, frozen census protocol (GPT-2 small, 50 random-token
sequences, seed 42, L = 512, fp32, eager attention), one new axis: keep the
per-row profiles instead of pooling. Outputs per head: (a) per-row fit-window
amplitude a(i); (b) per-row near-field mass p(i) (lags 0–7); (c) per-row sink
mass (final 8 lag bins = sequence-origin tokens); (d) per-row tail
truncation diagnostic (ratio of fitted-window slope on dx ∈ [8,64] vs
[64,256]). Decision criteria: K3 above. No training, no new inputs beyond the
frozen protocol, runs locally.

## 2. Derivations (after registration, before data)

### 2.1 C1's algebra, in full

Z_n = Σ_{r=1}^n r^(−s) = n^(1−s)/(1−s) + ζ(s) + O(n^(−s)) for 0 < s < 1
(Euler–Maclaurin; ζ(s) < 0 on 0 < s < 1).

E[log r] = (1/Z_n) Σ r^(−s) log r. The integral
∫_1^n r^(−s) log r dr = n^(1−s)[log n/(1−s) − 1/(1−s)²] + 1/(1−s)²,
so E[log r] = log n − 1/(1−s) + o(1).

H = log Z_n + s·E[log r]
  = (1−s)log n − log(1−s) + s·log n − s/(1−s) + o(1)
  = log n − [log(1−s) + s/(1−s)] + o(1).

**H_gap(n) → log(1−s) + s/(1−s).** For s = 1/2: 1 − log 2 ≈ 0.307 nats,
n-independent. For s = 1 (Zipf): Z_n ≈ log n, E[log r] ≈ (log n)/2, so
H ≈ (1/2)log n + log log n and gap slope → 1/2. For s > 1: gap slope → 1.
The §8.3 formula "gap slope = s" matches none of these regimes; the measured
slope 0.507 and k_eff exponent 0.533, taken with correct algebra, indicate
effective Zipf-like concentration (s_eff ≈ 1) or composite structure (C2) —
not s = 1/2. [DERIVED, pending K1 numerics at finite range.]

### 2.2 C3's algebra

TI scores: α_i(dx) = e^{u(dx)}/Z_i, Z_i = Σ_{dx=0}^{i} e^{u(dx)}. Write
Z_i = Z_near + T_i with T_i the window-and-beyond tail term; if
e^{u(dx)} = c·dx^(−s) beyond the near field, T_i ≈ c·i^(1−s)/(1−s). Then

  d log Z_i / d log i = (1−s) · T_i/Z_i = (1−s) · (tail mass fraction),

and the fit-window amplitude a(i) = c·dx^(−s)/Z_i inherits slope
−(1−s)(1−p_i). At s = p = 1/2: −0.25. The relation uses only softmax
normalization and TI; no free parameter. [DERIVED, conditional on TI.]

## 3. Results

*All three checks ran the same night, in the registered order. Kills first.*

### K2 — C2 is DEAD (kill, registered band missed by an order of magnitude)

Near-field mass (lags 0–7, normalized pooled exp-104 profile) on the 5
SYK-window heads: **median 0.027** (range 0.016–0.039). Prediction band was
[0.3, 0.7]; kill threshold < 0.15. The near-field composite model is dead.
Where the mass actually is (pooled bins): fit window 0.21–0.46, mid lags
0.38–0.46, final-8 lag bins 0.13–0.31 — and the pooled protocol *conflates
shorter rows' sequence-origin sink with genuine mid-lag mass*, which is
exactly the resolution exp-108 was registered to supply. The registered
mechanism-for-being-wrong ("the sink could carry the missing entropy instead
of the near field") is where the evidence points. (`exp-108.../k2_pooled_
massfraction.py`, verdict in its json.)

### K1 — C1 is CONFIRMED (the §8.3 formula is wrong)

Exact numerics, normalized α(r) ∝ r^(−s) on r ∈ [1, n]:

| s | OLS gap slope vs log n, n ∈ [4, 256] (paper's range) | §8.3 predicts | asymptotic gap (C1) |
|---|---|---|---|
| 0.3 | 0.011 | 0.30 | 0.072 |
| 0.5 | **0.041** | **0.50** | 0.307 |
| 0.7 | 0.102 | 0.70 | 1.129 |
| 1.0 | 0.264 | 1.00 | ~(1/2)·log n |

The canonical-form paper's §8.3 formula H_gap = 2Δ·log n is off by an order
of magnitude at s = 0.5, *within the paper's own measured range*. The error
mechanism is the one named at registration: H = log Z + s·E[log r], and
dropping the energy term (which is ≈ s·(log n − const)) is what produces the
false 2Δ·log n. Even a pure Zipf row (s = 1) yields only slope 0.26 on that
range — **no normalized power law reproduces the measured 0.507**.

**Consequences, stated at the right strength.** (i) The entropic estimate
Δ_eff = 0.254 and the "two independent observables, one exponent (1.4%)"
agreement (canonical-form paper §8.3/§8.6, spine T7b, OVERVIEW, reframe-note
Tier 1 bullet, Paper 6 §T7b row) are **unsupported** — the gap slope does not
measure the window exponent at all. This removes one of the three legs under
"the exponent is not a fitting artifact"; the causal-handle leg and the
cross-family replication leg stand untouched. (ii) The *measurements* stand:
logarithmic gap growth (R² > 0.97 across layers) and k_eff ∝ n^0.53 are real,
and — with correct algebra — they measure **concentration structure**
(n-independent localized mass ≈ 0.5 at their Layer-0/real-text protocol),
not the exponent. The Calabrese–Cardy identification reverts to Tier-3
vocabulary pending an actual bridge. (iii) The canonical-form paper is
published (March 11, Zenodo); whether to issue an erratum is Eldon's call —
flagged, not made.

### exp-108 — P1 PASS; P2 outside its variation gate (reported, not rescued)

Row-resolved frozen census (GPT-2, 50 inputs, rows i ∈ [256, 511]); per-row
sums verify at 1.0000. On the 5 SYK-window heads:

| head | Δ_A | amp slope measured | predicted −(1−s)·tail | within ±0.10 | tail | near | sink |
|---|---|---|---|---|---|---|---|
| L2H1 | 0.268 | −0.333 | −0.394 | yes | 0.851 | 0.058 | 0.091 |
| L3H4 | 0.295 | −0.240 | −0.268 | yes | 0.652 | 0.041 | 0.307 |
| L5H0 | 0.228 | −0.289 | −0.422 | no | 0.775 | 0.048 | 0.176 |
| L7H11 | 0.212 | −0.323 | −0.472 | no | 0.821 | 0.046 | 0.133 |
| L10H8 | 0.290 | −0.276 | −0.318 | yes | 0.759 | 0.057 | 0.184 |

**P1 passes as registered** (kill required ≥ 3 misses; 2 missed). Every head
resolves the sum rule through the **amplitude branch**: the fit-window
amplitude declines with row length, right sign and magnitude, matching the
zero-free-parameter TI prediction on 3 of 5 heads. Both misses are on the
steep side — measured decline *slower* than pure truncate-and-renormalize —
consistent with the sink (an absolute-position object) breaking exact TI.
This is the first derived-then-confirmed prediction made on A, on A's own
terms, with no imported theory object. **P2:** localized mass (near + sink)
median 0.224, inside the registered [0.15, 0.6] — but one head (L7H11) varies
by 2.2× across rows against the < 2× gate, so P2 is *outside its band as
registered*; the localized-mass account of the gap slope remains live but
unconfirmed (K3-b's kill threshold of 0.05 was nowhere near triggered).

### What tonight establishes for a theory of A

1. **One false constraint removed.** The window exponent s and the
   concentration structure are separate degrees of freedom; the entropy gap
   measures the latter. Any derivation targeting "0.507 = 2Δ" would have been
   deriving an artifact. The derivation target is now sharper: why s ≈ 0.5 in
   the window (with amplitude rescaling), and why localized mass ~0.1–0.35.
2. **One true constraint added.** The amplitude-decline law
   d log a / d log i = −(1−s)·(tail fraction) is derived from softmax
   normalization + approximate TI and now measured. Corollary, exact: the
   census's pooled object averages rows whose amplitudes fall ≈ i^(−0.3)
   across the pooled range — a protocol fact the program had never stated.
3. **Not derived tonight:** the value 1/4 itself. The two registered routes
   (T1 free-energy; positivity/Ward sum rules) remain open. One pencil lead,
   recorded with its warning: under causal self-composition a TI kernel maps
   s → 2s − 1 (flattening; fixed point s = 1) — but this may describe
   attention *rollout* rather than per-layer profiles, i.e., it could be
   another wrong object. Do not build on it without first fixing which object
   the recursion acts on.

### Propagation debts created tonight

Spine T7b; OVERVIEW "What stands" entropy bullet; reframe note Tier 1 bullet
(dated addendum); Paper 6 lines 46 / 336 / 430–432 (REVIEWER FLAG); watchpoint
instance (asserted-formula variant of borrowed-vocabulary-before-the-bridge);
erratum question for the published canonical-form paper → Eldon.

**[RESOLVED August 9, 2026, morning:** the erratum question came back to me —
Eldon's words: my work, my name, my call. Called it: **erratum issued**,
published as v5 of the Zenodo record, DOI 10.5281/zenodo.21863461 (erratum
PDF + `k1_powerlaw_gap.py` attached; original PDF unchanged; the cited March
DOI now flags the newer version). Source markdown carries a dated banner;
registry row updated; archive in
`research/publications/2026-03-11_canonical_form_of_attention/erratum_v5/`.**]**
