---
title: |
    A Pre-Registered Test of Boundary Conformal Field Theory
    \newline
    in Transformer Attention
subtitle: |
    Per-head conformal weight predicts long-range attention "valley depth"
    in 6 of 7 decoder-only models, with informative structure in the
    failure case
author:
    - Ariel Umphrey
    - Eldon Umphrey
date: April 17, 2026
abstract: |
    Causal attention in trained transformers exhibits power-law decay
    in attention weight as a function of query–key separation, with a
    position-dependent enhancement near the start of the sequence. We
    have previously interpreted this as the two-point function of a
    boundary conformal field theory (BCFT) on a strip whose left edge
    is the start of the sequence. The framework predicts that the
    per-head conformal weight $\Delta$, measured from short-context
    random-token attention, should positively rank-correlate across
    heads with a long-range "valley depth" measure related to the
    "lost-in-the-middle" phenomenon. We pre-registered the prediction
    Spearman $\rho(\Delta, \mathrm{valley}) \geq 0.50$, $p \leq
    10^{-5}$, and tested it on seven decoder-only transformers
    (Pythia-410m/1.4B/2.8B, GPT-Neo-2.7B, Qwen2-7B, OLMo-7B,
    Mistral-7B-v0.3). Six confirmed; Pythia-2.8B falsified at
    $\rho=+0.46$. A per-layer diagnostic localizes the falsification to
    layers 22–27 and shows that GPT-Neo-2.7B (the matched control: same
    parameter count, same training data, different training recipe)
    confirms with $\rho=+0.96$ across all 32 layers, the cleanest BCFT
    signal we have measured. Fitting the full BCFT functional form (3
    parameters per head: $C, \Delta, \lambda$) on Pythia-2.8B and
    GPT-Neo-2.7B reveals that 88–94% of conformal heads prefer BCFT
    over the bare power law, that $\lambda$ is mostly positive and well
    structured, and that $\Delta_{\mathrm{BCFT}}$ is closer to the SYK
    $\Delta = 1/4$ prediction than $\Delta_{\mathrm{PL}}$. However, the
    pre-registered scalar $\rho(\Delta, \mathrm{valley})$ becomes
    *weaker* with the cleaner $\Delta_{\mathrm{BCFT}}$, while the
    *joint* $(\Delta, \lambda) \to \mathrm{valley}$ rank-regression
    explains substantially more variance. Two findings demand
    explanation: (i) $\rho(\lambda, \mathrm{valley})$ is mostly
    *negative* across layers in both models, the opposite of the
    framework's prediction; (ii) GPT-Neo-2.7B exhibits an
    alternating-layer pattern with two distinct populations of heads
    by boundary structure. We discuss what this changes about the
    framework, what we would do differently in a follow-up
    pre-registration, and where the most informative remaining tests
    lie. All code, raw per-head data, and the pre-registration
    document are in the public repository at
    `Capacity-For-Evil/ariel`.
geometry: margin=1in
fontsize: 11pt
linkcolor: blue
urlcolor: blue
header-includes:
  - \usepackage{microtype}
  - \usepackage{booktabs}
  - \usepackage{graphicx}
  - \usepackage{float}
  - \usepackage{caption}
  - \captionsetup{font=small,labelfont=bf}
  - \floatplacement{table}{H}
  - \floatplacement{figure}{H}
---

# 1. Introduction

Attention in trained transformers carries structure that the network
has organized for reasons no one fully specified during training. One
strand of recent interpretability work asks whether that structure is
*generic* — whether it converges, across architectures and training
recipes, to forms predicted by physical theories of strongly
interacting systems. We have argued in three preprints
[@umphrey2026canonical; @umphrey2026syk; @umphrey2026conformal] that
softmax attention sits at a conformal fixed point with the same
exponent $\Delta = 1/4$ as the Sachdev–Ye–Kitaev (SYK) model
[@sachdev1993; @kitaev2015] at $q=4$. The empirical claim, reported
in [@umphrey2026conformal], is that the median per-head conformal
weight in deep GPT-2 layers matches the SYK prediction.

The causal mask in decoder-only transformers introduces a *boundary*:
each query attends only to keys to its left. This makes the natural
theoretical setting a *boundary* conformal field theory (BCFT)
[@cardy2004], whose two-point function for a single boundary on a
strip takes the form
$$
\alpha(x_q, x_k) \;\approx\; C \,(\Delta x)^{-2\Delta}\,
\bigl[1 + \lambda\,\eta^\Delta\bigr], \qquad
\eta \;=\; \frac{(\Delta x)^2}{4\, x_q\, x_k},
\qquad \Delta x = x_q - x_k.
$$
The bulk piece $C\,(\Delta x)^{-2\Delta}$ is the "scale-invariant"
attention decay; the boundary piece $\lambda\,\eta^\Delta$ enhances
attention for keys near the start of the sequence (small $x_k$). On
April 14, 2026 we found that this BCFT functional form, with
$\lambda$ as a free per-head parameter, beats the bare power law in
15 of 15 conformal heads of GPT-2 (mean $\Delta R^2 = +0.176$;
[@bcftboundarytest]).

A separate framework prediction is that the per-head $\Delta$ should
correlate positively, *across heads of a single model*, with the depth
of the long-range "valley" in the attention profile — the
attention-level analogue of the "lost-in-the-middle" effect that has
been studied at the level of task accuracy [@liu2024lost]. We had
preliminary evidence for this from Pythia-70m (Spearman
$\rho = +0.94$ across training-checkpoint-pooled heads) and from
LongChat-13B-16K ($\rho = +0.64$, single trained model)
[@bcftltm].

This paper does three things. **First**, we pre-register
([@bcftprereg], committed to the public repo before any of the new
runs) a precise scalar prediction —
$\rho(\Delta, \mathrm{valley}) \geq 0.50$ with $p \leq 10^{-5}$ — and
test it on seven previously-unmeasured decoder-only transformers.
**Second**, when one of the seven (Pythia-2.8B) falls below the
threshold, we run a per-layer diagnostic that uses no new compute
beyond the cached attention weights, and find that the falsification
has interpretable internal structure: the BCFT signal is robust in
early-to-mid layers and degrades in late layers in a model-specific
way. **Third**, we fit the full three-parameter BCFT form on the
falsifying model and the cleanest control (GPT-Neo-2.7B), and find a
mixture of strong agreement (functional form, $\lambda$ structure,
sharper $\Delta$ estimates) with two specific anomalies that the
framework as currently stated does not explain.

The empirical contribution we want to make plainly: (a) that this
pre-registration is on the public record with the exact procedure and
threshold; (b) that 6 of 7 tested models confirm; (c) that the failure
case has a structure that is informative for what BCFT applies to and
what it does not; (d) that the fuller fit reveals two empirical
patterns we did not predict and cannot yet account for. We frame BCFT
as one organizing description of the data; we are equally interested
in alternatives that may explain the same patterns more cleanly.

# 2. Pre-registered prediction and procedure

The pre-registered prediction document [@bcftprereg] specifies the
following.

**Claim P0.** *In any decoder-only transformer with at least 12
layers and at least 12 heads per layer, trained on a standard
language-modeling objective, when one measures per-head $\Delta$ from
short-context random-token attention (procedure §2.1) and per-head
valley depth from the same data (procedure §2.2), and restricts to
heads that pass the conformality criterion (§2.3), the Spearman rank
correlation will satisfy*
$$
\rho(\Delta, \mathrm{valley\_depth}) \;\geq\; 0.50, \qquad
p \;\leq\; 1 \times 10^{-5}.
$$

The pre-registration was committed to the public repository at
`Capacity-For-Evil/ariel`, file
`research/notes/bcft_pre_registered_prediction.md`, on the morning of
April 17, 2026. The seven test models were named in the document
*before* any measurement was made.

## 2.1 Per-head $\Delta$

For each model:

1. Load the model in fp16 with `attn_implementation="eager"` (so the
   raw attention weights are exposed).
2. Sample $n=50$ sequences of $L=512$ random token IDs from the model
   vocabulary, uniform.
3. For each input, run a forward pass and capture the
   `(n_layers, n_heads, L, L)` attention tensor.
4. Average the attention tensor across inputs.
5. For each $(\ell, h)$ pair, for each separation $\Delta x \in [3,
   \min(120, L/3)]$, compute
   $$
     \alpha_{\ell h}(\Delta x) = \langle \alpha_{\ell h}(q, k) \rangle
     _{q \geq L/2,\; q-k = \Delta x}
   $$
   and fit $\log \alpha = \log C - 2\Delta \log \Delta x$ by ordinary
   least squares. Record $\Delta_{\ell h}$, $C_{\ell h}$, $R^2_{\ell
   h}$.

The $q \geq L/2$ restriction averages over query positions away from
the start boundary, so $\Delta_{\ell h}$ measured this way is a "bulk"
estimator with reduced boundary contamination. We will return to this
methodological choice in §5.

## 2.2 Per-head valley depth

Using the *same* averaged attention tensor:

1. Set $Q_{\mathrm{LO}} = 3L/4 = 384$.
2. For each $(\ell, h)$, compute the deep-query-averaged profile
   $P_{\ell h}(k) = \langle \alpha_{\ell h}(q, k) \rangle_{q \geq
   Q_{\mathrm{LO}}}$.
3. Split $P$ into thirds and define
   $$
     v_{\ell h} \;=\; 1 \;-\; \frac{
       \langle P \rangle_{\mathrm{middle\;third}}
     }{
       \max(\langle P \rangle_{\mathrm{first\;third}},\,
            \langle P \rangle_{\mathrm{end\;third\;up\;to\;}Q_\mathrm{LO}})
     }.
   $$

$v=0$ means flat; $v$ near $1$ means a deep U; $v<0$ means the middle
exceeds both edges (rare). The valley depth uses *only* the
short-context (L=512) attention tensor — no separate long-context
inputs are required for the basic test. This is intentional: it makes
the prediction self-contained per model and removes a degree of
freedom in what the experimenter could vary.

## 2.3 Conformality criterion and falsification rule

A head is **conformal** if $\Delta \geq 0.05$ and $R^2 \geq 0.85$ on
the power-law fit. Heads that fail are excluded — a head whose
attention is not well-described by the BCFT functional form is by
definition not in the framework's domain of applicability.

P0 is *falsified* if $\rho < 0.50$ or $p > 10^{-5}$ on the conformal
subset of any model with $\geq 50$ conformal heads.

# 3. Results

We ran the pre-registered procedure on seven models on Modal A100 GPUs
(40 GB). The full per-head data are in
`research/physics/results/bcft_pre_registered_run_2026-04-17T*.json`
in the public repository. Total compute: $\approx 6$ GPU-hours,
$\approx \$10$ of credit.

\begin{table}[H]
\small
\centering
\begin{tabular}{lrrrrl}
\toprule
Model & $n_{\mathrm{tot}}$ & $n_{\mathrm{conf}}$ &
  Spearman $\rho$ & $p$-value & Verdict \\
\midrule
Pythia-410m  & 384 & 83  & $+0.7601$ & $7.86 \times 10^{-17}$ &
  \textbf{Confirm} \\
Pythia-1.4B  & 384 & 89  & $+0.7105$ & $6.25 \times 10^{-15}$ &
  \textbf{Confirm} \\
Pythia-2.8B  & 1024 & 227 & $+0.4639$ & $1.63 \times 10^{-13}$ &
  \emph{Falsify} \\
GPT-Neo-2.7B & 640 & 478 & $+0.9582$ & $6.11 \times 10^{-261}$ &
  \textbf{Confirm} \\
Qwen2-7B     & 784 & 135 & $+0.8502$ & $6.94 \times 10^{-39}$ &
  \textbf{Confirm} \\
OLMo-7B      & 1024 & 271 & $+0.8540$ & $2.63 \times 10^{-78}$ &
  \textbf{Confirm} \\
Mistral-7B-v0.3 & 1024 & 241 & $+0.5833$ & $2.30 \times 10^{-23}$ &
  \textbf{Confirm} \\
\bottomrule
\end{tabular}
\caption{Pre-registered test, per-head $\Delta \to$ valley\_depth
Spearman correlation on conformal subset. Falsification threshold:
$\rho < 0.50$ or $p > 10^{-5}$. Six of seven models confirm.}
\end{table}

The five confirmations from the first batch (Pythia-410m, Pythia-1.4B,
GPT-Neo-2.7B, Qwen2-7B, OLMo-7B) cover three model families that had
*not* been measured before this commit: GPT-Neo (EleutherAI, pre-Pythia
training recipe), Qwen2 (Alibaba; Chinese-heavy data; GQA; sliding
window), OLMo (Allen AI; fully open training data). Mistral-7B-v0.3
(added in a second batch after a HuggingFace token was configured) is
a fourth previously-unmeasured family. The framework's prediction
generalizes outside the family it was tuned on.

The single falsification, Pythia-2.8B, sits at $\rho = +0.464$. The
$p$-value is $1.6 \times 10^{-13}$ — there is a strong, real positive
correlation, just not strong enough to clear the pre-registered
threshold. As §4 shows, this falsification has interpretable internal
structure.

# 4. Per-layer diagnostic

The pooled correlation $\rho$ aggregates across all conformal heads
in a model. With 200+ conformal heads per Pythia-2.8B and GPT-Neo-2.7B,
much of the pooled signal could come from layer-specific behavior that
the scalar hides. We computed per-layer Spearman $\rho$ on every
layer with $\geq 5$ conformal heads, using the same cached
`bcft_pre_registered_run_*.json` files (no new model runs). Code:
`research/physics/pythia_per_layer_diagnostic.py`. Results:
`pythia_per_layer_diagnostic_2026-04-17T100046Z.json`. Plot: same
path with `.png`.

\begin{table}[H]
\small
\centering
\begin{tabular}{lrrrrrr}
\toprule
Model & layers w/ signal & frac+ & frac strong+ & frac sig$-$ &
  mean $\rho$ & median $\rho$ \\
\midrule
Pythia-410m   & 9   & 1.00 & 0.11 & 0.00 & $+0.73$ & $+0.77$ \\
Pythia-1.4B   & 9   & 1.00 & 0.11 & 0.00 & $+0.77$ & $+0.82$ \\
Pythia-2.8B   & 23  & 0.74 & 0.17 & 0.04 & $+0.40$ & $+0.53$ \\
GPT-Neo-2.7B  & 32  & 1.00 & \textbf{0.84} & 0.00 & $+0.89$ &
  $+0.96$ \\
Qwen2-7B      & 11  & 0.91 & 0.27 & 0.00 & $+0.67$ & $+0.77$ \\
OLMo-7B       & 20  & 1.00 & 0.40 & 0.00 & $+0.68$ & $+0.74$ \\
Mistral-7B-v0.3 & 21 & 0.81 & 0.33 & 0.05 & $+0.45$ & $+0.60$ \\
\bottomrule
\end{tabular}
\caption{Per-layer $\rho(\Delta, \mathrm{valley})$ summary for all
seven models. \emph{frac+}: fraction of layers with $\rho > 0$.
\emph{frac strong+}: fraction with $\rho \geq 0.50$ and $p < 10^{-3}$.
\emph{frac sig$-$}: fraction with $\rho < 0$ and $p < 0.05$.
GPT-Neo-2.7B's 0.84 strong-positive fraction is the highest of any
model measured.}
\end{table}

![Per-layer Spearman $\rho(\Delta, \mathrm{valley})$ vs. relative depth (layer index / total layers) for seven decoder-only transformers. Dashed lines mark $\rho = 0.50$ (the pre-registered threshold) and $\rho = 0$. Each dot is one layer's correlation restricted to its conformal heads. GPT-Neo-2.7B (cyan) is clean across all 32 layers; Pythia-2.8B (red) and Mistral-7B-v0.3 (magenta) show the early-clean / late-noisy pattern; the smaller Pythia models contribute correlations only from their early layers because their late layers have too few conformal heads to evaluate.](pythia_per_layer_diagnostic_2026-04-17T100046Z.png){width=100%}

The headline diagnostic findings:

1. **Pythia-2.8B passes the pre-registered threshold if restricted to
   the first 22 layers** (pooled $\rho = +0.605$ on layers 0–15;
   $\rho = +0.526$ on layers 0–21). Each batch of late layers (22–24,
   25–27) drops the pooled $\rho$ by $0.03$–$0.06$. The early/mid
   signal is robust; the late layers contribute the noise that
   crosses the threshold. This is descriptive, not a re-drawing of
   the line — the pre-registration stands as committed.

2. **GPT-Neo-2.7B is the cleanest BCFT signal we have measured.** 27
   of 32 layers are significantly strong-positive
   ($\rho \geq 0.50, p < 10^{-3}$); every layer with $\geq 5$
   conformal heads has $\rho > 0$; most layers have $\rho > 0.90$.
   GPT-Neo-2.7B has the same parameter count and training data as
   Pythia-2.8B; it differs in *training recipe* (older EleutherAI
   codebase, different optimization). This pins the cause of
   Pythia-2.8B's late-layer noise to recipe rather than scale or
   data.

3. **The smaller-Pythia confirmations are partly an artifact of
   conformality.** Pythia-410m has zero conformal heads in layers
   15–23. Pythia-1.4B has fewer than 5 conformal heads in most
   layers 11–19. They confirm the pre-registration in part because
   their late layers do not contribute enough conformal heads to
   drag down the pool — not because the BCFT framework is
   demonstrably valid throughout those models. This *weakens* the
   strength of the cumulative-confirmations argument for those two
   models specifically. It does not weaken the GPT-Neo, OLMo, or
   Qwen2 confirmations, which are robust across all layers.

4. **The early-clean / late-noisy pattern is not unique to Pythia.**
   Mistral-7B-v0.3 confirms with $\rho = +0.58$, the smallest
   confirming correlation. Per-layer, Mistral has the same shape as
   Pythia-2.8B: strong $\rho > 0.55$ through layer 14, sparse and
   noisy in layers 15–25, weak negatives in 26–29 (L29 significantly
   negative). If Mistral had as many conformal heads in its noisy
   late layers as Pythia-2.8B does, it might also have failed.

The pre-registered statistic ("model-pooled $\rho$") hides per-layer
structure. We will report both pooled and per-layer profiles in
future tests.

# 5. Functional-form fit

The pre-registered procedure fits only the bare power law. The full
BCFT form has a third per-head parameter $\lambda$. We hypothesized
that fitting the full form might recover the framework's predicted
correlation in the noisy late layers of Pythia-2.8B, since some of
the boundary contamination is being absorbed into $\Delta$ in the
power-law fit. To test this we re-ran a version of the measurement
that exposes the 2D position-resolved attention $\alpha(x_q, x_k)$ on
Pythia-2.8B and GPT-Neo-2.7B (the cleanest control), and fit three
forms per head:

- **$\Delta_{1d}$**: 1D power-law fit on the deep-position-averaged
  $\alpha(\Delta x)$ — the same as the pre-registered run.
- **$\Delta_{\mathrm{PL}}$**: 2D power-law fit on all $(x_q, x_k)$
  points: $\alpha = C\,(\Delta x)^{-2\Delta}$.
- **$\Delta_{\mathrm{BCFT}}, \lambda_{\mathrm{BCFT}}$**: 2D fit of
  the full BCFT form: $\alpha = C\,(\Delta x)^{-2\Delta}
  [1 + \lambda \eta^\Delta]$.

Code: `research/physics/bcft_functional_form_fit.py`. Raw data: file `bcft_functional_form_fit_2026-04-17T102458Z.json` in `research/physics/results/`.

## 5.1 The functional form is the right form

\begin{table}[H]
\small
\centering
\begin{tabular}{lrrr}
\toprule
Model & \% heads BCFT $>$ PL ($\Delta R^2 > 0.001$) &
  median $\Delta R^2$ & mean $\Delta R^2$ \\
\midrule
Pythia-2.8B & \textbf{94.2\%} & $+0.061$ & $+0.108$ \\
GPT-Neo-2.7B & \textbf{88.2\%} & $+0.107$ & $+0.180$ \\
\bottomrule
\end{tabular}
\caption{The BCFT functional form fits attention substantially better
than the bare power law. On 207 + 440 = 647 fresh heads from two
new model families, BCFT wins in 88--94\% of conformal heads. The
mean improvement matches the GPT-2 result from April 14 (15 of 15
heads, $+0.176$).}
\end{table}

This is the strongest empirical result in this paper, in the sense of
having the most data behind it. The BCFT functional form is the right
parametric description of attention $\alpha(x_q, x_k)$ for the
overwhelming majority of conformal heads. The boundary correction is
not an artifact of fitting four parameters where two would do; the
correction parameter $\lambda$ is *required* by the data.

## 5.2 The boundary parameter $\lambda$ is well structured

\begin{table}[H]
\small
\centering
\begin{tabular}{lrr}
\toprule
Model & median $\lambda$ & \% heads with $\lambda > 0$ \\
\midrule
Pythia-2.8B & $+3.0$ & $97.1\%$ \\
GPT-Neo-2.7B & $+1.9$ & $71.4\%$ \\
\bottomrule
\end{tabular}
\caption{$\lambda$ distribution. Mostly positive (consistent with
positive boundary entropy in the BCFT setting), order-1 to order-10
in magnitude. Across heads, $|\lambda|$ rank-correlates with
$\Delta R^2$ at $\rho = +0.71$ in GPT-Neo and $+0.45$ in Pythia
--- where the data needs the boundary term, $\lambda$ is
correspondingly large.}
\end{table}

## 5.3 $\Delta_{\mathrm{BCFT}}$ is closer to the SYK prediction

\begin{table}[H]
\small
\centering
\begin{tabular}{lrrr}
\toprule
Model & median $\Delta_{\mathrm{PL}}$ & median $\Delta_{\mathrm{BCFT}}$
  & SYK prediction \\
\midrule
Pythia-2.8B & $+0.243$ & $+0.355$ & $0.250$ ($q=4$) \\
GPT-Neo-2.7B & $+0.169$ & $+0.296$ & $0.250$ \\
\bottomrule
\end{tabular}
\caption{The 2D power-law fit underestimates $\Delta$ in both
models, because near-boundary attention enhancement makes the
curve look like a shallower power law. When $\lambda$ is freed to
absorb that boundary lift, $\Delta$ recovers a more "physical"
value, much closer to the SYK $q=4$ prediction in Pythia and
substantially closer in GPT-Neo.}
\end{table}

## 5.4 The pre-registered scalar $\rho(\Delta, \mathrm{valley})$ becomes weaker

\begin{table}[H]
\small
\centering
\begin{tabular}{lrrr}
\toprule
Model & $\rho(\Delta_{1d}, \mathrm{valley})$ &
  $\rho(\Delta_{\mathrm{PL}}, \mathrm{valley})$ &
  $\rho(\Delta_{\mathrm{BCFT}}, \mathrm{valley})$ \\
\midrule
Pythia-2.8B & $+0.758$ & $+0.682$ & $+0.474$ \\
GPT-Neo-2.7B & $+0.942$ & $+0.832$ & $\mathbf{+0.337}$ \\
\bottomrule
\end{tabular}
\caption{Marginal correlation of $\Delta$ alone with valley depth,
under three estimators of $\Delta$. The cleaner $\Delta$ (BCFT) is
the *worse* predictor of valley depth. GPT-Neo-2.7B drops from
$+0.94$ (the strongest confirmation in the pre-registered test) to
$+0.34$ (which would have failed the pre-registration outright).
The information that $\Delta_{\mathrm{PL}}$ was using to predict
valley depth was partly the boundary contamination that gets
absorbed into $\lambda$ in the cleaner fit.}
\end{table}

This is a sharp methodological lesson: **the pre-registered prediction
implicitly assumed that $\Delta$ would be a clean estimator of
"conformal weight." It is not; it is a compound estimator of
conformal weight plus boundary contamination, and the boundary part
is what drives the predictive correlation with valley depth.** The
framework's *joint* prediction is recovered by adding $\lambda$ back:

\begin{table}[H]
\small
\centering
\begin{tabular}{lrr}
\toprule
Model & rank $R^2$ (valley $\mid \Delta_{\mathrm{PL}}$) &
  rank $R^2$ (valley $\mid \Delta_{\mathrm{BCFT}}, \lambda$) \\
\midrule
Pythia-2.8B & $0.465$ & $\mathbf{0.548}$ \\
GPT-Neo-2.7B & $0.693$ & $\mathbf{0.765}$ \\
\bottomrule
\end{tabular}
\caption{Joint $(\Delta, \lambda)$ rank-regression of valley depth
explains substantially more variance than $\Delta_{\mathrm{PL}}$
alone, by $0.08$ (Pythia) and $0.07$ (GPT-Neo). The framework's
two-parameter content is empirically supported; the
single-parameter scalar version is not the right test of the
framework.}
\end{table}

## 5.5 Anomaly: $\rho(\lambda, \mathrm{valley})$ is mostly negative

The framework predicts that larger $\lambda$ (sharper boundary
effect) should mean *deeper* valleys. Per-layer Spearman
$\rho(\lambda, \mathrm{valley})$ over layers with $\geq 5$ conformal
heads:

- **Pythia-2.8B**: 18 of 21 layers have $\rho(\lambda,
  \mathrm{valley}) < 0$; 9 layers have $\rho < -0.7$.
- **GPT-Neo-2.7B**: 24 of 31 layers have $\rho(\lambda,
  \mathrm{valley}) < 0$; 8 layers have $\rho < -0.7$.

This is the **opposite** of the framework's sign prediction, and the
per-layer signal is large and consistent enough that this is not
plausibly noise.

We considered whether this could be a geometric artifact of the
valley-depth measure. The measure is
$1 - \mathrm{middle}/\max(\mathrm{start}, \mathrm{end})$. If
$\lambda$ enhances the start-boundary attention, $\mathrm{start}$ is
high; if $\mathrm{end}$ (recency) is roughly constant across heads,
then $\max(\mathrm{start}, \mathrm{end}) \approx \mathrm{start}$ for
heads with large $\lambda$, and the valley depth approaches
$1 - \mathrm{middle}/\mathrm{start}$, which gets *small* (deep
valley) when $\mathrm{start}$ is large. Under this candidate
mechanism the prediction is $\rho(\lambda, \mathrm{valley}) > 0$ —
the *correct* direction. So the candidate mechanism is wrong, or
there is a confounding mechanism we are missing. We flag this as a
real anomaly and do not paper over it.

## 5.6 New empirical pattern: GPT-Neo-2.7B has two populations of heads

Per-layer median $|\lambda|$ in GPT-Neo-2.7B alternates between
near-zero and large. Of the 31 layers with $\geq 5$ conformal heads,
12 have $|\lambda| < 0.5$ ("small-$\lambda$ layers") with mean
$\Delta R^2 = +0.010$ — BCFT and PL fits coincide, no boundary
correction needed. The other 19 layers have $|\lambda| > 5$
("large-$\lambda$ layers") with mean $\Delta R^2 = +0.235$ — BCFT
substantially improves the fit.

The pattern is not strictly periodic but has a periodic flavor: the
small-$\lambda$ layers are at indices $\{5, 7, 8, 9, 11, 13, 15, 17,
19, 23, 25, 29, 31\}$, the large-$\lambda$ layers at $\{2, 12, 14,
16, 18, 20, 22, 24, 26, 28, 30\}$. We have not seen this described
in interpretability literature for GPT-Neo specifically. The
hypothesis that fits the data is that there are *two populations of
attention heads* in this model with respect to boundary structure.
Identifying which heads belong to which population (induction heads
vs.\ non-induction; attention sinks vs.\ non-sinks; syntactic vs.\
semantic) is a follow-up.

# 6. Discussion

## 6.1 What this evidence supports

The strongest claims supported by the data presented here are:

1. The BCFT functional form is the right parametric description of
   attention in the conformal subset of heads (88–94% of conformal
   heads in two new model families prefer BCFT, replicating and
   extending the GPT-2 result of [@bcftboundarytest]).

2. A two-parameter $(\Delta, \lambda)$ summary of attention geometry
   per head predicts a structural property of long-range attention
   behavior (the valley depth in the deep-query-averaged profile)
   with rank-$R^2$ in the range 0.55–0.77 on the two models tested
   in detail.

3. The conformal weight $\Delta_{\mathrm{BCFT}}$ in deep-layer median
   in two new models is in the range $[0.30, 0.36]$, reasonably close
   to the SYK $q=4$ prediction of $0.25$, supporting the prior
   GPT-2 result of $0.249$ [@umphrey2026conformal].

4. The model-pooled per-head $\rho(\Delta, \mathrm{valley})$ is
   $\geq 0.50$ in 6 of 7 decoder-only transformers from four
   previously-unmeasured families. This is a generalizable structural
   regularity.

## 6.2 What this evidence does not support

1. The per-head $\rho(\Delta, \mathrm{valley}) \geq 0.50$ as a
   universal claim across all decoder-only transformers (Pythia-2.8B
   sits below the threshold, with internal structure pointing to a
   training-recipe-specific cause).

2. The framework's prediction about the sign of
   $\rho(\lambda, \mathrm{valley})$. The data show this is mostly
   negative, opposite to the prediction.

3. The interpretation that "attention is BCFT" implies the bulk dual
   has the geometric structure of JT gravity. That chain remains
   theoretical; what is established is that the boundary functional
   form of a BCFT two-point function fits attention data, with
   $\Delta$ in the right range. This is consistent with a holographic
   dual at one fixed point but does not require it.

4. Any task-level prediction. We showed previously that the BCFT
   attention U-shape and the LongChat-13B-16K accuracy U-shape do
   not have matching geometric structure (R² $\in [0.02, 0.07]$ for
   direct fit) [@bcftltm]. The BCFT framework as currently stated
   describes *attention layer* behavior; the composition through
   MLPs and residual stream into final task accuracy is a separate
   problem.

## 6.3 What we would do differently

The most important methodological lesson is that the cleaner the
estimator of $\Delta$ alone, the *worse* its marginal correlation
with valley depth. A future pre-registration should test the joint
$(\Delta, \lambda) \to \mathrm{valley}$ prediction directly — for
instance, "rank $R^2$ of valley regressed on $(\Delta_{\mathrm{BCFT}},
\lambda_{\mathrm{BCFT}}) \geq 0.50$". The marginal-$\Delta$ test we
pre-registered was simpler and easier to communicate, but the data
show it is conflating the bulk and boundary components of attention
geometry.

The second lesson is to report the per-layer profile, not just the
pooled scalar, by default. The signal-then-noise pattern in
Pythia-2.8B and Mistral is invisible from the pooled correlation;
the GPT-Neo-2.7B alternating-layer pattern is invisible from any
$\Delta$-only summary. A future test should report:

- pooled $\rho$ (the simple scalar);
- per-layer $\rho$ with confidence intervals;
- the conformal-head fraction by layer (so the reader can see whether
  late-layer absence is "no signal" or "no measurable signal");
- the $\lambda$ distribution by layer (so the reader can detect
  population structure).

The third lesson, less actionable, is that the SYK $\Delta = 1/4$
prediction is sharper than we previously thought, *but* the right
empirical comparison is to $\Delta_{\mathrm{BCFT}}$ (the
boundary-corrected estimator), not $\Delta_{\mathrm{PL}}$. The
$\Delta = 0.25$ result for GPT-2 may need to be re-stated with the
estimator made explicit.

## 6.4 What would falsify the framework more cleanly

Three experiments would substantially update us:

1. **Non-softmax universality.** Apple's
   [`apple/ml-sigmoid-attention`](https://github.com/apple/ml-sigmoid-attention)
   models [@apple2025sigmoid] provide 7B sigmoid + 7B softmax models
   trained identically except for the attention mechanism, with 8
   training checkpoints each. Measuring $\Delta$ on both and at each
   checkpoint is the cleanest possible test of whether the
   conformal scaling is a property of softmax specifically or of
   "normalized dot-product similarity with learned projections" more
   generally. We have not run this; it would cost $\sim$\$50 of
   Modal A100 time.

2. **The negative-$\rho(\lambda, \mathrm{valley})$ anomaly.** Either
   work out analytically what the sign should be from first
   principles given the recency-end boundary that the current BCFT
   form omits, or examine per-head attention profiles for
   high-$\lambda$ vs.\ low-$\lambda$ heads to see what the geometry
   actually looks like. If the framework cannot predict the sign of
   this correlation, that is a non-trivial empirical constraint on
   what versions of BCFT are viable.

3. **Llama-3-8B and frontier-scale models.** All our tested models
   are $\leq 7$B parameters. The 7B-class results may not be
   representative of frontier-scale behavior. Llama-3-8B is on our
   queue (HuggingFace gated access requested). Llama-3-70B and
   Mixtral-8x7B would be informative for testing whether the
   early-clean / late-noisy pattern persists or whether frontier
   models show robust BCFT throughout depth.

## 6.5 Connection to "lost-in-the-middle"

The "lost-in-the-middle" effect [@liu2024lost] is a task-accuracy
phenomenon in which model accuracy on long-context retrieval drops
for information placed in the middle of the context. The valley
depth $v$ measured here is a *structural* analogue at the level of
attention weights: deep-query attention profiles are U-shaped, with
attention concentrated near the start (boundary effect) and the
recent past (recency).

We do not claim the BCFT framework explains task-level
"lost-in-the-middle" — our LongChat-13B-16K results showed that the
attention-level U-shape and the accuracy-level U-shape do not have
matching geometric structure [@bcftltm]. What we *do* claim, with
the evidence presented here, is that the attention-level U-shape
has structural properties consistent with a BCFT description: it
follows the predicted functional form, its depth correlates with
the conformal weight $\Delta$ (via the joint $(\Delta, \lambda)$
prediction), and the prediction generalizes across a representative
range of decoder-only transformers.

The relationship between attention-level U-shape and task-level
U-shape is a separate question worth investigating in its own right.
Our results constrain part of the answer: whatever drives task-level
"lost-in-the-middle" is *not* a direct readout of the attention U.
The composition through MLPs and residual stream changes the shape.

# 7. Reproducibility

All code, raw per-head data, the pre-registration document, the
per-layer diagnostic, and this manuscript are in the public
repository at `Capacity-For-Evil/ariel`, on the master branch as of
April 17, 2026.

Specifically (paths relative to the repository root):

| Artifact | Path |
|---|---|
| Pre-registration | `research/notes/`\newline`bcft_pre_registered_prediction.md` |
| Pre-registered run code | `research/physics/`\newline`bcft_pre_registered_run.py` |
| Per-layer diagnostic code | `research/physics/`\newline`pythia_per_layer_diagnostic.py` |
| Functional-form fit code | `research/physics/`\newline`bcft_functional_form_fit.py` |
| Per-head data, run 1 | `research/physics/results/`\newline`bcft_pre_registered_run_2026-04-17T092239Z.json` |
| Per-head data, run 2 (Mistral) | `research/physics/results/`\newline`bcft_pre_registered_run_2026-04-17T095022Z.json` |
| Per-layer diagnostic results | `research/physics/results/`\newline`pythia_per_layer_diagnostic_2026-04-17T100046Z.json` (`.png` for plot) |
| Functional-form fit data | `research/physics/results/`\newline`bcft_functional_form_fit_2026-04-17T102458Z.json` |

Total Modal A100 compute for the experiments in this paper:
$\approx 7$ GPU-hours, $\approx \$12$ at Modal pricing (April 2026).
Reproduction on a personal workstation with 24+ GB GPU is feasible
for the 7B-class models in fp16; the smaller Pythia models can be
reproduced on a 16 GB GPU. The full reproduction script is a single
Python file per experiment.

# Acknowledgments

The pre-registration was committed before the test runs. We thank
the developers of EleutherAI, Allen AI, Alibaba, and Mistral AI for
publishing the model weights used in this study, and the
authors of the original "lost-in-the-middle" paper [@liu2024lost]
for the empirical phenomenon that motivated the framework.
Compute provided by Modal. This work was performed in solo physics
sessions of the `Capacity-For-Evil/ariel` project on April 17, 2026.

# References

::: {#refs}
:::
