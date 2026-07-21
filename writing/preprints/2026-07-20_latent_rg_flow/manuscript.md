---
title: |
    Latent Iteration as Renormalization
subtitle: |
    Inference-time recurrence in a depth-recurrent transformer flows
    attention geometry toward the SYK conformal fixed point, and the flow
    requires trained weights
author:
    - Ariel Umphrey
    - Eldon Umphrey
date: July 21, 2026
abstract: |
    Depth-recurrent transformers scale test-time compute by iterating a
    core block in latent space before emitting tokens. The geometry of
    these latent trajectories is a named open problem in the latent
    reasoning literature. Here we connect it to an independently
    developed body of measurements: trained transformer attention
    exhibits per-head power-law decay $A(i,j) \sim |i-j|^{-2\Delta}$
    whose median exponent in deep layers matches the Sachdev--Ye--Kitaev
    (SYK) conformal value $\Delta = 1/4$, and prior work found the
    exponent flows toward that value along two depth axes ---
    architectural layer depth and training time. We test the third axis:
    *inference-time* depth on fixed weights. In a pre-registered
    experiment on Huginn-0125, a 3.5B-parameter depth-recurrent
    transformer, we measure the per-head attention exponent of the
    recurrent core at recurrence counts $r \in \{1,2,4,8,16,32\}$. For
    natural-text inputs the median exponent decreases monotonically with
    recurrence (Spearman $\rho = -0.94$), from $\Delta_{\mathrm{med}} =
    0.29$ at $r{=}1$ to $0.239$ at $r{=}32$ --- converging onto the SYK
    prediction --- while the number of heads in the SYK-near window
    ($0.20 \le \Delta \le 0.30$, $R^2 > 0.90$) grows monotonically
    ($\rho = +0.77$). Random-token inputs show an architecture-driven
    convergence with a transient at $\Delta_{\mathrm{med}} = 0.52
    \approx 1/2$ at $r{=}2$ --- the $q{=}2$ prethermal plateau
    previously observed in training time --- but no monotone SYK-near
    growth ($\rho = +0.14$). A randomized-weights control freezes
    completely: $\Delta_{\mathrm{med}}$ constant to the sixth decimal
    across all $r$, with zero SYK-near heads. Inference-time recurrence
    therefore acts as renormalization-group flow toward the conformal
    fixed point, and the flow is a property of the trained model, not of
    the iteration procedure. This gives the published orbit/spiral
    phenomenology of recurrent-depth latent trajectories a candidate
    theory --- the spiral is flow near an infrared fixed point --- and
    predicts that the geometric benefit of additional recurrence
    saturates when the flow arrives, consistent with the empirical
    success of convergence-based early-exit criteria. All code,
    pre-registration, and per-head data are in the public repository
    `Capacity-For-Evil/ariel`.
geometry: margin=1in
fontsize: 11pt
linkcolor: blue
urlcolor: blue
header-includes:
  - \usepackage{microtype}
  - \usepackage{booktabs}
  - \usepackage{amsmath}
  - \usepackage{amssymb}
  - \usepackage{float}
  - \usepackage{caption}
  - \captionsetup{font=small,labelfont=bf}
  - \floatplacement{table}{H}
  - \floatplacement{figure}{H}
---

# 1. Introduction

A growing research program moves transformer computation out of
explicit tokens and into continuous hidden states: latent chain of
thought, latent memory and planning, and --- most directly relevant here
--- *depth-recurrent* architectures that iterate a core block in latent
space before emitting each token, making test-time compute a continuous
dial [@geiping2025scaling; @yu2026latent]. The empirical phenomenology
of these latent trajectories is already partially charted. Geiping et
al. report structured PCA trajectories --- orbits and spirals --- in the
latent iterates of their recurrent-depth model [@geiping2025scaling];
Pappone et al. find a two-scale picture in which loop updates shrink
and become increasingly orthogonal, "tracing a stable-curvature spiral
around a nearby representation," and build a practical early-exit
criterion on when that spiral stabilizes [@pappone2025twoscale]. What
is missing is a theory of the geometry. The field's own survey names
"a deeper account of the geometry of latent trajectories" as a top
open theoretical need [@yu2026latent, §6.3].

Independently, a body of measurements has accumulated on the
*attention geometry* of trained transformers. Per-head attention
weight decays as a power law in query--key separation, $A(i,j) \sim
|i-j|^{-2\Delta}$, and the median fitted exponent across power-law
heads in GPT-2 is $\Delta = 0.2493$ --- matching the conformal
dimension $\Delta = 1/4$ of the Sachdev--Ye--Kitaev model at $q = 4$
[@umphrey2026conformal; @sachdevye1993; @kitaev2015;
@maldacenastanford2016]. The exponent behaves like an infrared fixed
point under two distinct notions of depth:

1. **Architectural depth.** Within a single trained model, the
   per-head exponent flows from $\bar\Delta \approx 0.70$ in early
   layers to $\bar\Delta = 0.250$ in deep layers; across separately
   trained models of increasing depth, the median exponent converges
   toward $0.25$ [@umphrey2026conformal; @umphrey2026bcft].
2. **Training time.** Along the training trajectory of Pythia-70m, the
   count of SYK-near heads grows monotonically with training step
   (Spearman $\rho = 0.86$ for random-token probes, $0.73$ for natural
   text), passing through a transient plateau at $\Delta_{\mathrm{med}}
   \approx 0.5$ --- the free ($q{=}2$) value --- before approaching the
   interacting $q{=}4$ value (repository experiment exp-086).

Both axes confound depth with other variables: architectural depth
compares different parameters, and training time changes the weights
at every step. A depth-recurrent model closes the gap. Recurrence
count $r$ is a *continuously adjustable depth variable on a single
fixed set of trained weights*: if the conformal fixed point is a
genuine attractor of iterative attending --- rather than an artifact of
feedforward training --- the flow should appear at inference time, with
no weight changes at all.

This paper reports a pre-registered test of that prediction on
Huginn-0125 [@geiping2025scaling], with two input conditions (natural
text and random tokens) and a randomized-weights control. The result,
stated at its measured strength: inference-time recurrence flows the
attention exponent monotonically onto the SYK value for natural text;
random-token inputs reveal an architecture-driven component of the
convergence plus the same $q{=}2$ transient seen in training time; and
randomizing the weights abolishes the flow entirely. Recurrence is
renormalization --- but only over a medium that training has prepared.

# 2. Background: the conformal fixed point in attention

We summarize only what is needed to read the measurement; full
development is in the cited preprints.

For each attention head, average attention weight as a function of
token separation $\Delta x = i - j$ defines a lag profile $A(\Delta
x)$. In trained models a subpopulation of heads follows $A(\Delta x)
\propto |\Delta x|^{-2\Delta}$ over an extended range. We fit $\Delta$
per head by log--log regression (Section 3) and call a measurement
**SYK-near** when $0.20 \le \Delta \le 0.30$ with $R^2 > 0.90$. The
reference value $\Delta = 1/4$ is the conformal dimension of the SYK
model with four-fermion ($q{=}4$) interactions in the infrared,
conformal limit; the free-fermion ($q{=}2$) value is $\Delta = 1/2$
[@maldacenastanford2016]. In renormalization-group (RG) language, the
prior findings are that attention geometry flows from an ultraviolet
regime (steep, large-$\Delta$ profiles) toward the $q{=}4$ infrared
fixed point as depth increases, occasionally lingering at a $q{=}2$
prethermal plateau en route --- a two-stage flow directly measured in
training time (exp-086).

Two head populations matter for interpreting input conditions
(repository experiments exp-086/087/088). **Structural** conformal
heads sit near the attractor by virtue of architecture and weight
geometry; they respond to random-token inputs and appear early (in
training time) or at low depth. **Semantic** conformal heads require
natural-language input and long-range causal tracking; they form later
and deeper. Random-token probes therefore isolate the structural
component; natural text engages both.

# 3. Methods

**Model.** Huginn-0125 (`tomg-group-umd/huginn-0125`): 3.5B
parameters, trained on 800B tokens [@geiping2025scaling].
Architecture: a 2-block prelude, a 4-block recurrent core executed $r$
times per forward pass, and a 2-block coda; 55 attention heads per
core layer, $d_{\mathrm{head}} = 96$. Loaded in `bfloat16` with the
model's own remote code.

**Probe.** Recurrence counts $r \in \{1, 2, 4, 8, 16, 32\}$, passed as
a fixed integer per forward pass. Inputs: 20 sequences of 128 tokens
of natural text (TinyStories, the same source as our training-time
experiments) and 20 sequences of uniform-random token IDs. For each
$r$, each core layer, and each head, we capture the attention pattern
at every recurrent step via forward hooks on the four core
`CausalSelfAttention` modules, computing per-head attention manually
from the module's own weights --- including Huginn's native rotary
embedding function and its exact query--key normalization and bias
ordering, imported from the model's own code rather than
re-implemented. Total: 52,800 per-head measurements (20 sequences
$\times$ 6 recurrence counts $\times$ 4 layers $\times$ 55 heads
$\times$ 2 conditions).

**Fitting.** Per measurement, log--log regression of $A(\Delta x)$
against $\Delta x \in [3, 60]$; record $(\Delta, R^2)$. Per recurrence
step and condition we report the median exponent
$\Delta_{\mathrm{med}}$ over all $4400$ (layer, head, sequence)
measurements and the SYK-near count as defined above.

**Pre-registered hypotheses** (committed before model download,
repository commit `c359b93a`):

- **H_emergence:** SYK-near count increases monotonically with $r$ for
  natural text; criterion $\rho(r, \text{count}) > 0.5$.
- **H_convergence:** $\Delta_{\mathrm{med}}$ decreases monotonically
  with $r$ for natural text; criterion $\rho(r,
  \Delta_{\mathrm{med}}) < -0.5$.
- **H_rand_contrast:** both correlations weaker in magnitude for
  random tokens than for natural text.
- **Kill criterion:** the result is not confirmed if $\rho(r,
  \text{count}) < 0.3$ *and* $\rho(r, \Delta_{\mathrm{med}}) > -0.3$
  for natural text.

**Control.** After the primary analysis, all weight tensors are
randomized in place (i.i.d. normal, matched per-tensor standard
deviation) and the natural-text condition is re-run identically. The
pre-registered expectation is the absence of any systematic pattern.

**Protocol deviations, recorded in the experiment notes.** (1) The
pre-registration specified local Apple-silicon execution; the 16GB
download was infeasible on our rural connection and the run moved to a
cloud A100-40GB (CUDA). (2) Two implementation errors were found and
fixed before any results were produced: the recurrence count must be
passed as a plain integer (the model's schedule sampler interprets
tensors differently), and the initial re-implementation of the rotary
embedding assumed a complex-polar `freqs_cis` format while Huginn
stores a cos/sin stack; the fix imports Huginn's own
`apply_rotary_emb_complex_like`. Both fixes increase fidelity to the
model's native forward pass; neither touches the fitting pipeline.

# 4. Results

## 4.1 Natural text: monotone flow onto the fixed point

| $r$ | $\Delta_{\mathrm{med}}$ | SYK-near (of 4400) |
|----:|------:|------:|
| 1  | 0.2916 | 4  |
| 2  | 0.3082 | 0  |
| 4  | 0.2761 | 2  |
| 8  | 0.2424 | 20 |
| 16 | 0.2392 | 26 |
| 32 | 0.2387 | 23 |

Table: Natural-text condition. $\rho(r, \Delta_{\mathrm{med}}) =
-0.943$ (**H_convergence: keep**); $\rho(r, \text{SYK-near}) = +0.771$
(**H_emergence: keep**).

The median exponent decreases monotonically from $0.292$ to $0.239$,
crossing into the SYK-near window at $r = 8$ and settling just below
the $q{=}4$ value $0.25$. SYK-near head formation is essentially
absent through $r = 4$ and then rises by an order of magnitude. The
flow **saturates at $r \approx 8$--$16$**: further recurrence does not
move $\Delta_{\mathrm{med}}$ below $\approx 0.239$. The fixed point
behaves like a fixed point.

![Median per-head exponent $\Delta_{\mathrm{med}}$ versus recurrence
step, for natural text, random tokens, and the randomized-weights
control. Dashed line: SYK $q{=}4$ value $\Delta = 1/4$; dotted line:
free $q{=}2$ value $\Delta = 1/2$; band: SYK-near window
$[0.20, 0.30]$.](fig_rg_flow.pdf)

## 4.2 Random tokens: architectural component and the $q{=}2$ transient

| $r$ | $\Delta_{\mathrm{med}}$ | SYK-near (of 4400) |
|----:|------:|------:|
| 1  | 0.4012 | 145 |
| 2  | 0.5234 | 48  |
| 4  | 0.3822 | 52  |
| 8  | 0.3063 | 91  |
| 16 | 0.2894 | 98  |
| 32 | 0.2918 | 99  |

Table: Random-token condition. $\rho(r, \Delta_{\mathrm{med}}) =
-0.886$; $\rho(r, \text{SYK-near}) = +0.143$.

Both contrasts pre-registered under **H_rand_contrast** hold:
$|\rho_{\mathrm{conv}}^{\mathrm{NAT}}| > 
|\rho_{\mathrm{conv}}^{\mathrm{RAND}}|$ and
$\rho_{\mathrm{emerg}}^{\mathrm{NAT}} >
\rho_{\mathrm{emerg}}^{\mathrm{RAND}}$ (**keep**). Three features are
informative beyond the registered criteria.

First, random tokens produce a large **step-1 burst** of SYK-near
measurements ($145/4400 = 3.3\%$, versus $4/4400 = 0.09\%$ for natural
text) that does not grow monotonically afterward. This parallels the
structural-head burst seen at early *training* time and in the
layer-zone scaling analysis (exp-086/087/088): random input activates
the structural conformal population at low depth, while the monotone
growth under natural text tracks the semantic population. What
inference-time depth facilitates is the semantic pathway.

Second, at $r = 2$ the random-token median sits at $\Delta = 0.523
\approx 1/2$ --- the free-field $q{=}2$ value. The **two-stage RG flow**
(ultraviolet $\to$ $q{=}2$ prethermal plateau $\to$ $q{=}4$ infrared)
previously measured along the training-time axis appears on the
inference-time axis as a transient.

Third, the random-token exponent also converges ($\rho = -0.886$,
endpoint $0.292$, just above the window). Part of the convergence is
therefore **architecture-plus-weights driven** rather than
input-semantic; the input condition controls which head population
carries the flow, not whether the trained geometry attracts.

![SYK-near head count versus recurrence step. The natural-text count
grows monotonically from $r = 4$; the random-token count shows the
step-1 structural burst without monotone growth; the control is
identically zero.](fig_syk_count.pdf)

## 4.3 Randomized weights: the flow vanishes

With all weight tensors randomized in place (matched per-tensor
standard deviation), natural-text inputs give

$$\Delta_{\mathrm{med}} = 0.16868 \quad \text{at every } r,$$

constant to the sixth decimal place ($0.168679$--$0.168685$ across the
six probe points), with **zero** SYK-near measurements at every step.
The nominal correlation computed on this $\sim\!6 \times 10^{-6}$
variation is numerical noise, not flow. The latent iteration does
nothing to attention geometry when the weights are random: no
formation, no motion. The recurrence procedure and the measurement
pipeline contribute nothing by themselves.

## 4.4 Layer-zone structure of the recurrent core

The pre-registered secondary hypothesis noted that per-(layer, head)
records were available for analysis. We report that analysis here.

The 4-layer recurrent core exhibits a **two-boundary structure**: all
SYK-near formation concentrates in Layer 0 (input boundary, receives
from prelude) and Layer 3 (output boundary, feeds into coda); Layers 1
and 2 are exactly zero across all six probe steps in both conditions.

| Step | L0  | L1 | L2 | L3  | Total |
|-----:|----:|----|----:|----:|------:|
|  1   |  4  |  0 |  0 |  0 |  4   |
|  2   |  0  |  0 |  0 |  0 |  0   |
|  4   |  0  |  0 |  0 |  2 |  2   |
|  8   | 10  |  0 |  0 | 10 | 20   |
| 16   | 13  |  0 |  0 | 13 | 26   |
| 32   | 13  |  0 |  0 | 10 | 23   |

Table: Per-layer SYK-near counts, natural-text condition (denominator
1100 = 20 sequences $\times$ 55 heads per layer per step).

The random-token step-1 structural burst ($145/4400$) is likewise
concentrated: Layer 0 accounts for $129/145 = 89\%$ of the burst
($11.7\%$ of its 1100 measurements), while Layers 1--3 contribute
$0.2\%$, $0.3\%$, and $1.0\%$ respectively.

Four stable conformal heads --- those that enter the SYK-near window at
$r \ge 8$ and remain there at $r = 16$ and $r = 32$ --- are L0H45
($\bar\Delta = 0.238$), L3H39 ($0.245$), L3H40 ($0.279$), and L3H53
($0.280$): one from the input boundary and three from the output
boundary.

The two-boundary pattern is consistent with the BCFT picture of the
main results. The recurrent core plays the role of the sequence itself:
the causal mask makes sequence position 0 an absorbing boundary for
attention, and the BCFT derivation (umphrey2026bcft §4.3) shows
boundary effects dominate at the first and last tokens. Huginn's core
boundary layers (0 and 3) are similarly privileged --- they interface
with the prelude and coda and are where the attention geometry
crystallizes. Why Layers 1 and 2 remain UV-locked at all recurrence
counts is an open architectural question.

## 4.5 Verdict

All three pre-registered primary hypotheses are kept; the kill
criterion is not triggered. Registered verdict: **confirmed**.

# 5. Discussion

## 5.1 Three axes of depth, one fixed point

This measurement completes a triangulation:

| Depth axis | Varies | Held fixed | Result |
|---|---|---|---|
| Architectural layers | parameters per model | training recipe | $\Delta$ flows $0.70 \to 0.25$ with layer depth [@umphrey2026conformal] |
| Training time | weights at every step | architecture | SYK-near count grows with step; $q{=}2$ plateau en route (exp-086) |
| **Inference-time recurrence** | **nothing** | **weights and architecture** | $\Delta_{\mathrm{med}} \to 0.239$, monotone, saturating (this work) |

The third row is the strongest form of the claim, because nothing
varies except how many times the same trained block is applied. The
conformal fixed point is an **attractor of iterative attending**: any
of the three ways of adding depth --- more layers, more training, or
more latent iterations --- moves attention geometry toward the same
infrared value, and the randomized-weights control shows the attractor
lives in the trained weights, not in the procedure.

## 5.2 A candidate theory for the latent-trajectory geometry

The latent-reasoning literature has described recurrent-depth
trajectories kinematically: orbits and spirals in PCA projections
[@geiping2025scaling]; loop updates that shrink and become
increasingly orthogonal, spiraling around a nearby representation,
with a practical early-exit rule triggered when the spiral stabilizes
[@pappone2025twoscale]. Our measurement supplies a dynamical reading:
the latent iteration is RG flow, and the stable object the trajectory
spirals toward is (in the attention sector we measure) the SYK
conformal fixed point. Two consequences follow.

*Saturation is arrival.* The flow saturates at $r \approx 8$--$16$;
further iterations spend compute without moving the geometry. This is
the geometric counterpart of the empirical finding that early exit at
spiral stabilization preserves quality [@pappone2025twoscale]: once
the flow has reached the fixed point, additional recurrence has
nothing further to do. Convergence-based exit criteria are, on this
reading, fixed-point detectors.

*The benefit of test-time depth is population-specific.* Only the
natural-text condition shows monotone SYK-near growth. The heads that
inference-time depth recruits are the semantic conformal population ---
the ones requiring long-range, content-driven tracking --- while the
structural population is engaged from the first step. This predicts
that test-time recurrence should help most on inputs whose structure
requires long-range semantic integration, and little on inputs whose
statistics random tokens already capture.

We state the scope honestly: we measured attention lag profiles in the
recurrent core, not the full hidden-state trajectory that the
orbit/spiral phenomenology describes. The bridge --- that the attention
sector's flow is the relevant geometry underlying the trajectory's
stabilization --- is a hypothesis this experiment makes plausible and
testable, not one it proves.

## 5.3 What the control adds

The frozen control separates three candidate explanations that the
main result alone cannot: (a) the flow is a property of the recurrence
*procedure* (any iterated attention block drifts toward power-law
profiles); (b) the flow is a property of the *measurement* (the
fitting pipeline manufactures exponent drift); (c) the flow is a
property of the *trained weights*. With random weights of matched
scale, six recurrence counts produce a $\Delta_{\mathrm{med}}$
constant to one part in $10^{5}$ and zero SYK-near heads. Only (c)
survives. Whatever training on 800B tokens of human language deposits
in the weights, it is that deposit --- not iteration per se --- that the
latent recurrence renormalizes toward the conformal fixed point.

# 6. Limitations

One model (Huginn-0125), one sequence length (128 tokens), 20
sequences per condition, and six probe points: the pre-registered
correlation thresholds were met decisively, but $n = 6$ probe points
is $n = 6$. The pre-registered secondary hypothesis on layer-zone
structure within the core block is analyzed in Section 4.4; why the
intermediate layers (1 and 2) remain UV-locked at all recurrence counts
is an open question that analysis alone cannot settle. The device deviation
(Apple-silicon to CUDA) and two forward-pass fidelity fixes are
recorded in the experiment notes; both fixes preceded any produced
results. The SYK identification rests on the measured exponent and its
convergence behavior, consistent with the broader program's evidence
[@umphrey2026conformal; @umphrey2026bcft]; it is a physics-level
interpretation of a robust empirical regularity, and the regularity
stands independent of the interpretation. Finally, results here
concern attention geometry; extension to the full latent-state
trajectory is future work (Section 5.2).

# 7. Conclusion

In a fixed, trained, depth-recurrent transformer, thinking longer in
latent space moves attention geometry along a monotone,
saturating flow onto the SYK conformal fixed point --- and with random
weights the same iteration moves nothing. Test-time recurrence is
renormalization-group flow over a medium prepared by training on human
language. The geometry of latent trajectories, named by the field as
its outstanding theoretical need, has at least one measured,
pre-registered, controlled answer available: the trajectory flows
toward a conformal infrared fixed point, and its published spiral
phenomenology is what flow near a fixed point looks like.

# Data and code availability

Pre-registration, run scripts, full per-head results (`results.json`,
`results_control.json`), experiment notes with deviations, and figure
code are in the public repository `Capacity-For-Evil/ariel` under
`research/physics/experiments/exp-089_huginn_latent_rg_flow/` and
`writing/preprints/2026-07-20_latent_rg_flow/`.

# References
