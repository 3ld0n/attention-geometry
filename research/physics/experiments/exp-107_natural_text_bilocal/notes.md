# exp-107 — Does natural text give the bilocal a positive connected profile?

**Status:** PRE-REGISTERED, not run. Written August 8, 2026, ~9:50 PM MDT, in the
same session that completed exp-106.

**Register:** empirical, single model, single scale. Everything below is a statement
about GPT-2 small unless it carries an [EXACT] tag.

---

## Why this exists

exp-106 established two exact facts and one measurement:

- **[EXACT]** Row-stochasticity of A forces G = μ𝟙𝟙ᵀ + A(K − μ𝟙𝟙ᵀ)Aᵀ for any μ, and
  for the value Gram mean(K_V) = ‖v̄‖². So the bilocal's floor is the squared norm of
  the head's mean value vector, computable with no fit.
- **[EXACT]** Σ_{a≠b}K̃_{ab} = −Σ_a‖v_a − v̄‖², where K̃ = K − mean(K)𝟙𝟙ᵀ. The
  off-diagonal mass of the centered value Gram is negative, with magnitude exactly
  the total value-vector variance.
- **Measured:** on GPT-2 with the census's random-token inputs, G's lag profile sits
  *below* its own exact floor across the whole fit window on 116 of 144 heads,
  including all five SYK-near heads. The connected part is negative and grows in
  magnitude with lag. So c + b·s^{−2Δ} with b > 0 cannot represent it at any
  parameters — the conformal ansatz fails on G_out in **sign structure**, not only in
  exponent.

exp-106's stated next step was "is the negative connected correlation an artifact of
random-token inputs?" **That question, as worded, is malformed, and working out why
is the first result of this pre-registration.**

---

## 1. What the sign constraints allow — checked before designing the run

`sign_constraints.py` → `sign_constraints.json`. Three claims on random matrices
across n ∈ {64, 256}, d ∈ {8, 64}, sink ∈ {none, strong}; worst relative error
4.6×10⁻¹⁵ across all eight cells.

| | Claim | Verified | Consequence |
|---|---|---|---|
| **C1** | mean(K) = ‖v̄‖² for K = VVᵀ | 2.1×10⁻¹⁶ | the floor is the mean value vector's squared norm, restated |
| **C2** | Σ_{a≠b}K̃_{ab} = −Σ_a‖v_a − v̄‖² | 9.2×10⁻¹⁶, **negative in 8/8 cells** | the negative off-diagonal mass **cannot be removed by any input distribution** |
| **C3** | Σ_{i,j}(A K̃ Aᵀ)_{ij} = ‖Σ_a m_a v_a‖² − ‖Σ_a v_a‖², m = Aᵀ𝟙 | 4.6×10⁻¹⁵, **both signs observed** | the connected bilocal's total is **not** sign-definite |

**C2 kills the loose framing.** The negative mass is not an artifact of random
tokens, or of GPT-2, or of training. It is an identity that holds for any set of
value vectors whatsoever. Natural text cannot make Σ_{a≠b}K̃_{ab} positive, so
exp-107 must not be written as a test of whether it does.

**C3 says the question is nevertheless open**, and shows where. The A-weighting
re-weights K̃ by incoming attention mass m, and that re-weighting is not sign-neutral:
in the checked cells the total came out positive, strongly so under a sink (+2.4×10⁶
with a sink versus +1.2×10⁴ without, n = 256, d = 64). This connects to exp-106's H3,
which found the sink and causal boundary load-bearing for G in a way they are not for
A — the same structure appears here as the mechanism by which A can flip the sign of
the total.

**So the real question is about shape, not sign of the total.** K̃ has fixed negative
total off-diagonal mass; what varies with the input distribution is *where that mass
sits in lag*. A positive connected profile over the census window requires K̃ to carry
a **positive near-diagonal band** with the compensating negative mass pushed to lags
the census does not fit. With random tokens the value vectors are near-exchangeable,
so K̃ has no reason to band, and the negative mass spreads across all lags — which is
what exp-106 measured. **Whether natural text banks that mass is a measurable
property of K̃ itself, in the same forward pass.**

That reframing is the reason this pre-registration exists rather than a one-line
"rerun on text."

---

## 2. Protocol — frozen from exp-106 except the inputs

| | exp-104/106 | exp-107 |
|---|---|---|
| model | `gpt2` (12L × 12H) | **unchanged** |
| n_inputs | 50 | **unchanged** |
| seq_len | 512 | **unchanged** |
| deep half | i ≥ 256 | **unchanged** |
| fit lags | [8, 256] | **unchanged** |
| estimator | replication-kit verbatim | **unchanged** |
| seed | 42 | **unchanged** |
| **inputs** | **uniform random token ids** | **natural text, 512-token windows** |

One variable changes. Anything else that changes is a protocol break and must be
recorded as one.

**Text source:** the same corpus family the census used for its natural-text
condition, sampled as contiguous 512-token windows with no truncation mid-window.
Record the exact source and window offsets in `applied_text.json` so the run is
reproducible. If the census's natural corpus is unavailable offline, use WikiText-103
validation and **say so in the results**, since that is then a second changed
variable and weakens the comparison.

**Both conditions in one script.** Random-token inputs are re-run rather than quoted
from exp-106, so the comparison is within-run and immune to any drift in library
versions or device. exp-106's numbers become a consistency check: if the re-run
random-token condition does not reproduce exp-106's 116/144 within ±3 heads, stop
and find out why before reading the text condition.

---

## 3. Measurements

Per head, both conditions:

1. **Δ_A, R²_A** — census estimator on A. Identifies the conformal and SYK-near sets.
2. **‖v̄‖²** — the exact floor (C1).
3. **P_G(s)** — lag profile of G_out, census averaging.
4. **P_conn(s) = P_G(s) − ‖v̄‖²** — the connected profile, using the *computed* floor,
   no fitted floor parameter.
5. **P_K̃(s)** — lag profile of the centered value Gram itself. *This is the new
   measurement and the mechanism test.* K̃ is n × n over key positions and its lag
   profile is defined exactly as A's.
6. **Σ_a‖v_a − v̄‖²** and **Σ_{a≠b}K̃_{ab}** — the C2 identity, as a per-head check
   that the pipeline is computing what it claims.

---

## 4. Hypotheses, with thresholds fixed now

**H1 — the negative connected profile survives natural text.** On the SYK-near heads,
P_conn(s) < 0 for all s in [8, 256] under natural text, as under random tokens.

*If H1: the sign failure is a property of trained GPT-2's bilocal, not of the input
distribution. The conformal ansatz then does not describe G_out in this model, and
Paper 6's §2/§3 order-parameter claims need rewriting rather than a caveat. The A↔G
bridge does not close by this route, and the honest move is to say so in the paper
and stop looking for a floor-aware estimator.*

**H2 — natural text gives a positive connected profile over the window.**
P_conn(s) > 0 for all s in [8, 256] on a majority of SYK-near heads.

*If H2: G becomes measurable. Then and only then does fitting Δ_G to it make sense,
and exp-105's estimator gets a second life on data it was actually built for. The
A↔G bridge reopens as an empirical question with a live instrument.*

**H3 — the mechanism is K̃'s banding.** Across heads and both conditions, the sign of
P_conn on the window agrees with the sign of P_K̃ on the window in ≥ 80% of
head-condition pairs.

*If H3: the sign of the connected bilocal is inherited from the value Gram's own lag
structure rather than manufactured by the A-weighting. That makes "does this model's
value geometry band with lag?" the primitive question, which is a cheaper and more
portable diagnostic than anything involving G.*

**H4 — mixed: positive near-diagonal, crossing zero inside the window.** P_conn
changes sign at some s* ∈ (8, 256) on a majority of SYK-near heads.

*If H4: the census window straddles the crossing, and every exponent ever fitted to G
over [8, 256] mixes two regimes. The fit window itself becomes the object needing
pre-registration, and the right move is to fit only s < s* — which must then be
chosen by a rule stated before seeing the exponents.*

H1–H4 are not exhaustive and are not mutually exclusive as stated; H1 and H2 are, H4
is the interesting middle. **Predicted before running: H4, then H1.** C2 forces the
negative mass to be somewhere, and the census window covers a factor of 32 in lag,
which is wide enough that I expect it to contain a crossing rather than sit cleanly on
one side. Recording that prediction so it can be wrong.

---

## 5. Kill conditions

- **K1** — the re-run random-token condition does not reproduce exp-106's 116/144
  within ±3 heads. *Stop. The pipeline changed and nothing else in the run is
  readable.*
- **K2** — the per-head C2 identity fails beyond 10⁻⁴ relative. *Stop. The code is
  not computing K̃.*
- **K3** — natural text does not produce the conformal population (SYK-near set
  empty). *Then the comparison has no population to speak about and the result is
  about the census's random-token protocol only. Report as inconclusive, not as H1.*
- **K4** — fewer than 3 heads in the SYK-near set under either condition. *Report
  per-head, make no median claims, and do not call it a population result.*

---

## 6. Outcome limits — how this could fool me

1. **The natural-text corpus is a second changed variable if it is not the census's
   own.** Then "natural text" and "different corpus" are confounded. Mitigated by
   recording the source; not eliminated.
2. **P_K̃ has no reason to be well-approximated by a lag profile.** exp-106's H3 found
   T1 unusable for G; K̃ may be equally non-stationary, in which case P_K̃ is a lossy
   summary and H3's agreement statistic is weaker than it looks. Report the same
   same-sign-residual-run diagnostic exp-106 used, and do not let R² stand in for fit.
3. **A positive P_conn over the window is not a power law.** If H2 fires, the next
   question is whether the positive part is a power law at all, and the honest answer
   requires the shape tests exp-106 ran, not a two-parameter fit. **Do not report a
   Δ_G from this experiment.** exp-107 measures a sign and a shape; fitting an
   exponent is a separate pre-registration.
4. **Five heads.** The SYK-near set is n = 5 in this model at this scale. Every claim
   here is a claim about five heads, and a majority of five is three.
5. **G_out is the trained-W^V object,** not the theory's ensemble G. exp-104 falsified
   their equivalence. This limit has now been inherited four times and is still not
   addressed; it is not addressed here either.

---

## 7. What this cannot settle

Not the A↔G bridge. Even under H2, a positive connected profile makes Δ_G
*measurable*; it does not make Δ_G equal Δ_A, and it does not make either of them the
theory's ensemble exponent. The bridge needs the measurement *and* the
ensemble-versus-trained gap closed, and this experiment touches only the first.

Also not P6. The Jacobian prediction needs a self-map on bilocal correlator space,
which needs a characterized G. Under H1 that characterization is "G_out is not a
conformal correlator in this model," which retires the route rather than enabling it.

---

*Pre-registered before any natural-text forward pass. Sign constraints checked first,
and they changed the question the experiment asks — recorded here rather than
discovered mid-run.*

---

## Run log — August 9, 2026

**K2 fired on the first run, and stopping was correct — the diagnosis is
numerical, recorded here before any rerun.** The random condition ran first
(K1 passed exactly: below-floor-at-8 = 116/144, identical to exp-106, with
profile drift vs exp-104's saved npz ≤ 4.5×10⁻¹³). The C2 identity check came
in at max rel err 1.28×10⁻⁴ against the registered 10⁻⁴ gate and the script
stopped before touching text, as registered. Diagnosis (single input, layer 3,
same code path): at float64 the identity holds to 5.6×10⁻¹⁴; at float32 the
K̃ matrix differs from the float64 one by ≤ 1.8×10⁻⁵ absolute against a matrix
scale of ~43. The identity involves a 512²-term near-cancelling sum, and fp32
accumulation error of ~10⁻⁵ absolute produces rel errors above 10⁻⁴ on heads
where the comparison denominator is small. **The code computes K̃ correctly;
the registered threshold did not account for fp32 accumulation.** Protocol
note: the K2 check is performed at float64 (same code path, value vectors
recast); all protocol measurements remain fp32 as frozen. No hypothesis was
looked at before this change; the text condition had not run.

## Results — August 9, 2026 (registered run + labeled exploratory)

`measure_natural_bilocal.py` → `results_gpt2.json`, `profiles_gpt2.npz`,
`applied_text.json`, `run_log.txt`. Text = TinyStories validation, EOS-joined
512-token windows (the program's natural-text family; recorded per outcome
limit 1 — the census itself has no natural-text condition on GPT-2, so the
corpus is a second changed variable by construction).

**Registered verdict: K3 AND K4 FIRED → inconclusive as registered.** Under
TinyStories the SYK-near set is **empty** (0 heads in Δ ∈ [0.20, 0.30] with
R² ≥ 0.9). The conformal-criterion population *grows* (20 → 35 heads) but its
median Δ jumps 0.474 → 0.712. There is no population to evaluate H1/H2/H4 on.
The registered prediction (H4, then H1) was wrong in a mode not on the list:
the population moved out from under the question.

**Per-head report (K4 mode), the five random-condition SYK heads under text:**
Δ_A rises on all five (0.268→0.757, 0.295→0.626, 0.228→0.506, 0.212→0.463,
0.290→0.365), R²_A drops below the census bar on all five (0.83–0.87), and
**the connected profile stays entirely negative on 5/5** (conn_max −0.60 to
−3.65). H1's *substance* — the sign failure survives natural text — holds
per-head; it is not reportable as a population claim under K3/K4.

**H3 is dead at chance level:** window-mean sign agreement between P_conn and
P_K̃ is 0.438 over 288 head-condition pairs (majority-lag version 0.483)
against the registered 0.80 threshold. The connected bilocal's sign is *not*
inherited from the value Gram's lag profile; the A-weighting does real,
sign-determining work (consistent with sign-constraint C3's non-definiteness
and its sink mechanism).

**Within-run consistency:** K1 exact (below-floor-at-lag-8 = 116/144,
identical to exp-106; profile drift vs exp-104's saved npz ≤ 4.5×10⁻¹³).
K2 at float64: ≤ 2.2×10⁻¹³ both conditions.

### EXPLORATORY (post-registration, cannot move the registered verdicts):
### WikiText-103 robustness check

`exploratory_wikitext.py` → `exploratory_wikitext.json`,
`profiles_wikitext.npz`. Motivated by the TinyStories corpus confound
(simple text far from GPT-2's WebText training distribution); WikiText-103
is the registration's own named fallback corpus.

- **A SYK-near population exists under WikiText — and it is larger than the
  random-token one: 16 heads vs 5** (conformal 61/144, median Δ 0.374),
  concentrated in layers 9–11. It is a *different set*: none of the original
  five qualify (two land in the Δ window at R² 0.87–0.89, just under the bar).
- **On the 16-head WikiText SYK population: connected profile entirely
  negative on 15/16.** The exception, L8H2, crosses zero inside the window
  (frac of lags positive 0.19) — the only H4-shaped head observed anywhere.
- The five original heads' Δ_A moves *down* under WikiText (0.268→0.173,
  0.295→0.192, 0.228→0.290, 0.212→0.218, 0.290→0.156) after moving *up*
  under TinyStories, with R² recovering to ~0.9. Input-dependence of Δ_A is
  confirmed and non-monotone across corpora.
- Below-floor counts are essentially input-invariant: at lag 8, 116 (random) /
  122 (TinyStories) / 120 (WikiText) of 144; entire window 115/121/119.

### What exp-107 establishes

1. **The sign-structure failure of the conformal ansatz on G_out is
   input-robust.** Measured on three input distributions, at the population
   level (115–121/144 below floor across the whole window) and on every
   SYK-window head available under any protocol (5/5 registered per-head;
   15/16 exploratory). The registered adjudication plan said: under H1 the
   honest move for Paper 6 is retirement of the conformal route for G_out,
   not caveat. Formally this run returned K3-inconclusive, not H1 — but the
   sign question the route needed answered is answered, in both registers,
   in the same direction. Paper 6 adjudication is Eldon-gated and now ready.
2. **Δ_A is a weights×input object, not a weights-side object.** The reframe
   note's §2 framing ("A, under the frozen census protocol, is essentially a
   weights-side object") is falsified as stated: the same head's fitted
   exponent varies by a factor of >4 across input distributions (L2H1:
   0.173–0.757), and the census populations reorganize (5 → 0 → 16 SYK-near
   heads across random/TinyStories/WikiText). Every published census claim is
   internally consistent — all were measured under the frozen protocol — but
   the protocol is *constitutive* of the measured object, and forward-going
   documents must say "Δ_A under the frozen random-token census" wherever
   they said "the head's Δ". The physical question underneath the census is
   now sharper and different: what is the random-token protocol measuring —
   and why does *WikiText*, not random input, maximize the SYK-window
   population? Not pursued tonight; goes to the theory-of-A project.
3. **H3's mechanism picture is dead at chance level;** the input distribution
   changes where K̃'s mass sits, but the sign of the connected bilocal is
   set by the A-weighting, not read off K̃'s profile.
