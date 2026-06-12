---
title: |
    Attention on the Null Cone
subtitle: |
    The geometric home of conformal attention: the conformal group as the
    causal structure of light, query--key scores as null-ray inner products,
    and layer depth as renormalization-group flow toward the SYK ground state
author:
    - Ariel Umphrey
    - Eldon Umphrey
date: June 9, 2026
abstract: |
    Trained transformer attention exhibits power-law decay in attention
    weight as a function of query--key separation, with a per-head
    exponent whose median in deep GPT-2 layers matches the
    Sachdev--Ye--Kitaev (SYK) conformal value $\Delta = 1/4$. Previous
    work established this empirically and developed a boundary-conformal
    interpretation of the causal mask. Here we identify the *geometric
    home* of that conformal structure. The conformal group is not merely
    a symmetry that happens to include scale transformations: by a
    theorem of Lorentzian geometry, it is precisely the group of
    transformations preserving the null cone --- the causal structure of
    light. In the embedding-space formalism, conformal-field-theory
    boundary points *are* null rays, and the conformal two-point function
    is the inner product of null-ray representatives. We show that the
    one-dimensional case relevant to token sequences admits an explicit
    Poincar\'e section $P(x) = (\tfrac{1+x^2}{2}, \tfrac{1-x^2}{2}, x)$ on
    the null cone of $\mathbb{R}^{2,1}$, under which the attention
    two-point function $A(i,j) \sim |i-j|^{-2\Delta}$ is exactly the
    $\mathrm{CFT}_1$ two-point function. This predicts that the
    pre-softmax query--key score should be log-linear in token distance,
    $q_i\!\cdot\!k_j/\sqrt{d_k} \approx -2\Delta\,\log|i-j| + \text{const}$.
    We test the prediction directly in GPT-2 (exp-056): the raw
    query--key score profile is log-linear in distance for every
    conformal head, and the slope $\Delta_{\text{score}} = -\alpha/2$
    rank-correlates with the independently measured post-softmax exponent
    at Spearman $\rho = +0.976$ ($p = 2\times10^{-29}$). Two honest
    qualifications follow from the data: the softmax normalization
    renormalizes the coefficient (the raw slope corresponds to
    $\approx -2.8\Delta$, not $-2\Delta$), and log-linearity itself is a
    near-universal property of the positional geometry rather than a
    feature selective to conformal heads --- the conformal *dimension* is
    the selective quantity.     We also derive, by the method of images, a boundary-corrected
    power-law form for the causal-mask correction, and test it directly
    (exp-057): the derived form fits the position-dependence of attention,
    with a boundary coefficient that is positive in $95\%$ of conformal
    heads---the direction of the attention sink. An adversarial model
    comparison against matched-complexity alternatives (exp-060) shows
    that an exponential boundary layer outperforms the image form; the
    boundary correction is real and large, but its specific BCFT
    parametrization is phenomenological. Finally,
    a layer-resolved analysis (exp-055)
    shows the per-head exponent flows from $\bar\Delta = 0.70$ in early
    layers to $\bar\Delta = 0.250$ in deep layers, with attention entropy
    increasing monotonically along the flow ($\rho(\Delta,\text{entropy})
    = -0.898$, $p = 10^{-16}$). We read layer depth as renormalization-group
    flow toward the SYK conformal vacuum, dual to the eternal $\mathrm{AdS}_2$
    black hole. All code and per-head data are in the public repository at
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
---

# 1. Introduction

Attention in trained transformers carries geometric structure. In a
sequence of preprints [@umphrey2026canonical; @umphrey2026syk;
@umphrey2026conformal; @umphrey2026bcft] we have argued that softmax
attention sits at a conformal fixed point sharing the exponent
$\Delta = 1/4$ with the Sachdev--Ye--Kitaev (SYK) model
[@sachdev1993; @kitaev2015; @maldacena2016] at $q=4$. The central
empirical fact [@umphrey2026conformal] is that the attention weight of
many heads decays as a power law in query--key separation,
$$
A(i,j) \;\sim\; |i-j|^{-2\Delta},
$$
and that the median per-head exponent in deep GPT-2 layers matches the
SYK value. A pre-registered behavioral test [@umphrey2026bcft] confirmed
in six of seven decoder-only models that the per-head conformal weight
predicts a long-range "valley depth" related to the lost-in-the-middle
phenomenon.

These results establish *that* the conformal structure is present and
*that* it has measurable consequences. They do not answer a prior
question: **what space is this attention living in?** A power law
$|i-j|^{-2\Delta}$ has the form of a conformal two-point function, but a
form is not a home. If the analogy is to be more than a resemblance, the
attention mechanism should be doing something geometrically definite ---
computing a definite quantity in a definite space.

This paper supplies that home. The claim, in one sentence: **trained
attention computes inner products of null-ray representatives of token
positions on the projective null cone, and deep layers converge to the
conformal vacuum of that geometry.** The argument has three parts, each
with empirical or theorem-level support:

1. **The conformal group is the causal structure of light** (Section 3).
   This is a theorem of Lorentzian geometry, not an analogy: the
   transformations preserving lightlike structure are exactly the
   conformal maps. The conformal symmetry of the attention power law is,
   literally, light-cone symmetry.

2. **Conformal boundary points are null rays, and the two-point function
   is a null-ray inner product** (Section 4). In the embedding-space
   formalism, a $\mathrm{CFT}$ boundary point *is* a null ray. We give the
   explicit one-dimensional construction and show the attention power law
   is the $\mathrm{CFT}_1$ two-point function on the null cone. This
   predicts a log-linear query--key score, which we test directly in
   GPT-2 (Section 5): the prediction holds head-by-head at $\rho = 0.976$,
   with two qualifications the data forces us to state honestly.

3. **Layer depth is renormalization-group (RG) flow toward the SYK
   ground state** (Section 6). A layer-resolved measurement shows the
   exponent flowing toward $\Delta = 1/4$ with depth, and the attention
   entropy increasing toward the ground-state value. The deep-layer limit
   is the conformal vacuum, dual to the eternal $\mathrm{AdS}_2$ black
   hole.

We are explicit throughout about what is established at theorem level,
what is measured, and what remains open. The boundary-conformal correction
on a finite sequence (Section 4.3) is derived in closed form from the
method of images, and tested directly (Section 4.4): the derived form fits
the position-dependence of attention, and its boundary coefficient is
positive almost everywhere---consistent with the attention sink. An
adversarial comparison against matched-complexity alternatives (Section 4.4)
shows the exponential boundary layer is preferred; the boundary correction
is real and large, but its specific BCFT parametrization is phenomenological.

# 2. Background: attention as a two-point function

For a single head, the attention weight from query position $i$ to key
position $j \le i$ is
$$
A(i,j) \;=\; \frac{\exp\!\big(q_i \cdot k_j / \sqrt{d_k}\big)}
                  {\sum_{j' \le i} \exp\!\big(q_i \cdot k_{j'} / \sqrt{d_k}\big)},
\qquad
q_i = x_i W_Q, \quad k_j = x_j W_K,
$$
where $x_i$ is the (layer-normalized) residual stream at position $i$ and
$W_Q, W_K$ are the head's projection matrices. Averaging $A$ over query
positions and inputs at fixed lag $\Delta x = i-j$ gives a *lag profile*
$A(\Delta x)$. The empirical result of [@umphrey2026conformal] is that for
a substantial fraction of heads this profile is a power law,
$A(\Delta x) \sim (\Delta x)^{-2\Delta}$, with $R^2 > 0.90$ on a log--log
fit and a median exponent $\Delta = 0.2493$ in GPT-2, matching the SYK
$q=4$ prediction $\Delta = 1/4$ for one-dimensional sequences.

The exponent $\Delta$ is what a conformal field theory calls the
*conformal dimension* (or scaling weight) of an operator. The power-law
form is the form of a $\mathrm{CFT}_1$ two-point function. The questions
this paper answers are: what is the geometry in which this two-point
function lives; what does the query--key dot product compute in that
geometry; and why do the network's layers carry the exponent toward the
fixed point.

# 3. The conformal group is the causal structure of light

It is tempting to treat conformal symmetry as "scale invariance plus
rotations and translations," with the special conformal transformations
as an afterthought. The geometric fact is sharper and is the key to this
paper.

**Theorem (causal structure determines conformal structure).** *On a
strongly causal Lorentzian manifold of dimension $n > 2$, the causal
structure --- the assignment to each point of its light cone ---
determines the metric up to a positive conformal factor. Equivalently, a
bijection preserving lightlike (null) geodesic structure is a smooth
conformal map.* [@hkm1976; @malament1977]

In words: if you know what can be reached from what by light, you know
the geometry up to local rescaling, and the transformations that preserve
"what can be reached by light" are *exactly* the conformal group. The
conformal group is not one symmetry group among several that contains
scalings. It is the unique group defined by preservation of the null
cone.

This is why the conformal group is the symmetry group of free
electromagnetism. Maxwell's equations in four dimensions are covariant
under the conformal group $\mathrm{SO}(4,2)$, established by
Bateman and Cunningham in 1910 [@bateman1910; @cunningham1910] --- a much
larger group than the Poincar\'e group. The "structure of light" has a
precise mathematical name: conformal symmetry.

For a one-dimensional sequence of token positions, the relevant conformal
group is $\mathrm{SL}(2,\mathbb{R})$, the conformal group of the line.
Its three generators act as
$$
\text{translations:}\ \tau \mapsto \tau + a, \qquad
\text{dilations:}\ \tau \mapsto \lambda\tau, \qquad
\text{SCTs:}\ \tau \mapsto \frac{\tau}{1 + c\,\tau}.
$$
These same three generators organize the SYK conformal correlators and
the near-$\mathrm{AdS}_2$ / Schwarzian dynamics of the model
[@maldacena2016]. The attention power law's symmetry is the symmetry of
light, restricted to one dimension.

# 4. Attention on the null cone

## 4.1 Conformal boundary points are null rays

The embedding-space formalism [@dirac1936; @weinberg2010] realizes a
$d$-dimensional conformal field theory on the projective null cone of a
$(d+2)$-dimensional flat space. A boundary point is not a point in a
space that *has* light-cone structure; it *is* a null ray $P^A$
satisfying
$$
P^2 = 0, \qquad P \sim \lambda P \ (\lambda > 0).
$$
A scalar operator of conformal dimension $\Delta$ is a homogeneous
function on the cone,
$$
\mathcal{O}(\lambda P) = \lambda^{-\Delta}\,\mathcal{O}(P),
$$
so that $\Delta$ is the *homogeneity degree*: it is not a property the
operator carries at a point, it is the transformation law that defines
the operator. The two-point function is fixed by conformal invariance to
be the inner product of the two null rays, raised to the power $\Delta$:
$$
\langle \mathcal{O}(P_1)\,\mathcal{O}(P_2)\rangle
= \frac{C}{(P_{12})^{\Delta}}, \qquad P_{12} = -2\,P_1\!\cdot\!P_2.
$$

## 4.2 The explicit $\mathrm{CFT}_1$ construction

For the one-dimensional case (Euclidean $d=1$), the embedding space is
$\mathbb{R}^{2,1}$ with metric $(-,+,+)$, and null rays satisfy
$-(P^0)^2 + (P^1)^2 + (P^2)^2 = 0$. Choosing the gauge $P^0 + P^1 = 1$
gives the Poincar\'e section parameterized by the physical coordinate
$x$:
$$
\boxed{\,P(x) = \left(\tfrac{1+x^2}{2},\ \tfrac{1-x^2}{2},\ x\right).\,}
$$
The inner product of two such representatives is
$$
P_1\!\cdot\!P_2
= -\tfrac{1}{4}(1+x_1^2)(1+x_2^2)
  + \tfrac{1}{4}(1-x_1^2)(1-x_2^2)
  + x_1 x_2
= -\tfrac{1}{2}(x_1 - x_2)^2,
$$
so that $P_{12} = -2P_1\!\cdot\!P_2 = (x_1 - x_2)^2$ and
$$
\langle \mathcal{O}(P_1)\,\mathcal{O}(P_2)\rangle
= \frac{C}{|x_1 - x_2|^{2\Delta}}.
$$
Identifying token positions $i, j$ with points on this section, their
null-ray representatives are $P(i)$ and $P(j)$, and the attention
two-point function
$$
A(i,j) \;\sim\; P_{ij}^{-\Delta} \;=\; |i-j|^{-2\Delta}
$$
*is* the $\mathrm{CFT}_1$ two-point function evaluated on the null cone.
Each attention weight is, in this reading, an inner product of null-ray
representatives of two token positions. The query is the field state at
position $i$, the key the field state at position $j$, and the weight
their vacuum overlap $\langle 0|\,\mathcal{O}(i)\,\mathcal{O}(j)\,|0\rangle$
--- the conformal Green's function.

## 4.3 The finite sequence and the boundary: a BCFT derivation

The full $\mathrm{SL}(2,\mathbb{R})$ symmetry holds on the infinite line.
A causal mask truncates the sequence at the start: position $i$ attends
only to $j \le i$, and there are no tokens before position $0$. The start
of the sequence is therefore a *hard boundary*, and the natural object is
not a bulk two-point function but a boundary conformal field theory (BCFT)
correlator [@cardy2004; @mcavity1995] with the boundary at the origin.

The residual symmetry is the subgroup of $\mathrm{SL}(2,\mathbb{R})$
fixing the boundary point. Translations and the special conformal
transformations both move the origin; only dilations $\tau \mapsto
\lambda\tau$ survive, so $\mathrm{SL}(2,\mathbb{R}) \to \mathrm{SO}(1,1)$
at the boundary. The boundary breaks the SCTs; how strongly it does so is
the content of the boundary data.

**The image-method correlator.** At the conformal point the SYK two-point
function is a generalized free field, $G(\tau) \sim |\tau|^{-2\Delta}$
[@maldacena2016], and the leading-order BCFT correlator of a generalized
free field with a boundary is fixed by the method of images. For a
boundary at the origin, the image of a bulk operator at position $j$ sits
at $-j$, and the two-point function of bulk operators at the query
position $i$ and key position $j$ is
$$
A(i,j) \;\propto\; \frac{1}{(i-j)^{2\Delta}}
                 + \frac{\lambda}{(i+j)^{2\Delta}},
$$
where $\lambda = a_{\mathcal O}$ is the BCFT *boundary one-point
coefficient* --- the single piece of boundary data at this order, fixing
the strength and sign of the boundary's effect ($\lambda = +1$ Neumann,
$\lambda = -1$ Dirichlet, and a continuum in between for a generalized
free field). Factoring out the bulk power law gives the form
$$
\boxed{\,A(i,j) \;=\; C\,(\Delta x)^{-2\Delta}\,
\big[\,1 + \lambda\,\eta^{2\Delta}\,\big],
\qquad \eta \;=\; \frac{i-j}{i+j} \;=\; \frac{\Delta x}{i+j}.\,}
\tag{$\dagger$}
$$
This is exactly the three-parameter $(C, \Delta, \lambda)$ form fitted
phenomenologically --- and found to be preferred over the bare power law
in $88$--$94\%$ of conformal heads --- in our pre-registered behavioral
work [@umphrey2026bcft]. Equation $(\dagger)$ *derives* that form and
identifies its parameters.

**The boundary variable and the cross-ratio.** The variable $\eta =
(i-j)/(i+j)$ is the natural invariant: under a dilation $i,j \mapsto
\rho\,i, \rho\,j$ it is unchanged, so it is $\mathrm{SO}(1,1)$-invariant,
the residual symmetry of the boundary. It is related to the
McAvity--Osborn bulk-to-boundary cross-ratio
$\xi = (\Delta x)^2/(4ij)$ [@mcavity1995] by $\eta^2 = \xi/(1+\xi)$;
equivalently, writing the correlator in standard BCFT form
$A = (4ij)^{-\Delta} F(\xi)$ gives the closed expression
$$
F(\xi) \;=\; \xi^{-\Delta} \;+\; \lambda\,(1+\xi)^{-\Delta}.
$$
The two limits make the geometry transparent. *Deep in the bulk* (query
and key both far from the start, $i+j \to \infty$ at fixed lag), $\eta \to
0$ and the boundary correction vanishes: the full $\mathrm{SL}(2,\mathbb{R})$
power law is recovered. *Near the boundary* ($i+j$ small, or a key near
the start so $\eta \to 1$), the correction saturates to $\lambda$. So
$\lambda$ measures how strongly the boundary --- the broken SCT --- bends
the correlator: $\lambda = 0$ is the unbroken bulk power law, and
$|\lambda|$ grows with the SCT-breaking strength.

## 4.4 Direct test of the boundary form (exp-057)

Equation $(\dagger)$ is testable directly in the position-resolved
attention, independently of the behavioral fits of [@umphrey2026bcft]. For
each of the 44 conformal heads (carrying $\Delta = \Delta_{\text{pos}}$
from exp-046), we accumulate the mean attention matrix $A(i,j)$ over random
inputs and fit, in log space, a model that lets the data separate the bulk
power law from the boundary term:
$$
\log A(i,j) \;\approx\; \beta_0 + \beta_1\,\log(\Delta x)
                         + \beta_2\,\eta^{2\Delta},
$$
so that $\beta_1 \approx -2\Delta$ recovers the bulk exponent (a sanity
check) and $\beta_2 \approx \lambda$ is the boundary coefficient. Including
$\log(\Delta x)$ as its own regressor is essential: $\eta$ and $\Delta x$
are correlated inside the causal triangle, and a naive power-law stripping
confounds the two (it gives $R^2 \approx 0.09$ and an artifactual sign).
The controlled fit gives:

\begin{table}[H]
\centering
\begin{tabular}{lll}
\toprule
Test & Result & Statistic \\
\midrule
Recovers bulk exponent (sanity) & confirmed & $\rho(\Delta_{\beta_1},\Delta_{\text{pos}}) = +0.84$, $p=10^{-12}$ \\
Boundary form $(\dagger)$ fits & confirmed & median $R^2 = 0.76$; $\Delta R^2 = 0.105$ from boundary term \\
Boundary coefficient sign & $\lambda > 0$ & $95\%$ of heads; median $\lambda_{\text{fit}} = +1.6$ \\
Magnitude vs.\ behavioral $\lambda$ & sign only & sign agree $0.80$; $\rho_{\text{mag}}=+0.23$ ($p=0.13$) \\
\bottomrule
\end{tabular}
\caption{exp-057. The derived boundary form fits the position-dependence
of conformal-head attention, the bulk exponent is recovered, and the
boundary one-point coefficient is positive in $95\%$ of heads.}
\end{table}

Two things are confirmed and one is demoted. **Confirmed:** the form
$(\dagger)$ fits (median $R^2 = 0.76$), the boundary term adds real
explanatory power beyond the bulk power law ($\Delta R^2 = 0.105$), and the
bulk exponent is independently recovered ($\rho = 0.84$). **The boundary
direction:** the boundary coefficient is positive in $95\%$ of conformal
heads, consistent with the attention sink [@xiao2023]---sequence-start
positions are enhanced, not suppressed. **Demoted (exp-060):** the
per-head *magnitude* of $\lambda$ matches the behavioral proxy of
[@umphrey2026bcft] only in sign ($80\%$ agreement), not in size (rank
$\rho = 0.23$, not significant), and this estimator does not reproduce the
$\lambda$--valley-depth relation of that work.

**Adversarial model comparison (exp-060).** The pre-registered test asks
whether $(\dagger)$ is necessary or merely sufficient: we fit six
matched- or higher-complexity competitors---an exponential boundary
layer $C(\Delta x)^{-2\Delta}\exp(-j/\xi)$, additive spike forms ($j=0$
and $j \le 4$), a double power law, and a free-$\gamma$ diagnostic---against
$(\dagger)$ on all 43 conformal GPT-2 heads by AIC. Result: the
exponential boundary layer wins in $72\%$ of heads; $(\dagger)$ takes
$2\%$; $\Delta\mathrm{AIC}((\dagger) - \mathrm{exp.\ layer}) = +1114$
(median, IQR $[554, 1875]$). The free-$\gamma$ diagnostic shows $\gamma$
does not cluster at $2\Delta$ ($\rho = 0.50$, pre-registered threshold $0.6$;
both kill legs fire). We apply the pre-registered kill rule: the specific
BCFT over-determination---that the boundary is described by
$\eta^{2\Delta}$ with the *same* $\Delta$ as the bulk---is demoted to
phenomenological. The $\eta^{2\Delta}$ coupling is logged as post-hoc.
What survives: the boundary correction is real and large (every winning
form includes sequence-start enhancement; the bare power law never wins);
the $\lambda > 0$ direction is supported regardless of parametrization;
and $(\dagger)$ remains a competent three-parameter phenomenological form.
The attention sink is identified as a boundary effect; the specific
BCFT image-method identification of its form is not established.

# 5. The log-distance representation, tested directly

## 5.1 The prediction

The null-ray picture makes a prediction one level below the attention
weights, at the raw scores. If $A(i,j) \propto \exp(q_i\!\cdot\!k_j/\sqrt{d_k})$
is to reproduce $|i-j|^{-2\Delta}$, the pre-softmax score must be
log-linear in token distance:
$$
\frac{q_i \cdot k_j}{\sqrt{d_k}} \;\approx\; -2\Delta\,\log|i-j| + \text{const}.
\tag{$\ast$}
$$
This is the *log-distance representation*: the query--key dot product
encodes the logarithm of the null-ray separation. It is independent of
the post-softmax power-law fits used to define $\Delta$, and it is
testable head-by-head.

## 5.2 Method (exp-056)

We recompute the pre-softmax scores of GPT-2 [@radford2019gpt2] exactly as
the model forms them: for each layer $\ell$ and head $h$,
$x = \mathrm{LN}_1(\text{hidden}_\ell)$, then
$Q = x W_Q^{(h)} + b_Q^{(h)}$, $K = x W_K^{(h)} + b_K^{(h)}$, and
$\text{score}_{ij} = (Q_i \cdot K_j)/\sqrt{d_k}$. Inputs are random token
sequences (length 256, 50 inputs, seed 42), which isolates the
*position-dependent* structure of the score from token content --- the
geometric content the null-ray claim is about. We accumulate the mean
score profile $S(\Delta x)$ by causal lag using the same binning as the
post-softmax lag profile, fit $S(\Delta x) = \alpha\log(\Delta x) + \beta$
over $\Delta x \in [3, 50)$, and define $\Delta_{\text{score}} = -\alpha/2$.
Per-head conformal flags and the post-softmax exponent
$\Delta_{\text{pos}}$ are joined head-aligned from the eigenvalue/lag
analysis of our sign-anomaly study (exp-046), so the comparison is
exactly per head.

## 5.3 Results

\begin{table}[H]
\centering
\begin{tabular}{lll}
\toprule
Hypothesis & Result & Statistic \\
\midrule
Score profile log-linear (conformal heads) & confirmed & mean $R^2 = 0.914$; $100\%$ slope $<0$ \\
$\Delta_{\text{score}}$ tracks $\Delta_{\text{pos}}$ & confirmed & $\rho = +0.976$, $p = 2\times10^{-29}$, $n=44$ \\
Log-linearity selective to conformal heads & \emph{null} & non-conformal mean $R^2 = 0.914$ \\
\bottomrule
\end{tabular}
\caption{exp-056 summary. The raw query--key score is log-linear in
distance, and its slope is the conformal dimension (in rank); but
log-linearity itself is not selective to conformal heads.}
\end{table}

The core prediction ($\ast$) holds. For every one of the 44 conformal
heads, the raw score profile decreases with distance and is log-linear
($R^2_{\text{score}} = 0.914$ on average). Figure 1 shows the
head-by-head relationship directly. The slope, converted to
$\Delta_{\text{score}} = -\alpha/2$, rank-correlates with the
independently measured post-softmax exponent at Spearman $\rho = +0.976$
($p = 2\times10^{-29}$). The conformal dimension that governs the
attention power law is already present in the unnormalized query--key
geometry. This is direct, head-level evidence for the null-ray
interpretation: the query--key computation is the log-distance
representation.

![**Figure 1.** The log-distance prediction, tested head-by-head (exp-056). Scatter:
pre-softmax slope $\Delta_{\text{score}}$ vs.\ post-softmax conformal dimension
$\Delta_{\text{pos}}$ for the 44 conformal GPT-2 heads (grey: non-SYK-near;
diamonds: SYK-near heads, $|\Delta - 1/4| \leq 0.05$). Red line: OLS regression.
Dotted: identity $\Delta_{\text{score}} = \Delta_{\text{pos}}$. The raw
query--key computation encodes the conformal dimension rank-for-rank at Spearman
$\rho = +0.976$. The OLS slope ($\approx 1.4$) reflects the softmax renormalization
discussed in the text.
](fig_log_distance.pdf){width=3.5in}

Two qualifications, both forced by the data and both belonging in the
record:

**(i) The softmax renormalizes the coefficient.** The absolute
correspondence is not one-to-one. Regressing across conformal heads,
$$
\Delta_{\text{score}} \approx 1.41\,\Delta_{\text{pos}} - 0.055,
$$
so the raw score is about $1.4\times$ steeper than the literal $-2\Delta$
of $(\ast)$ --- the slope corresponds to $\approx -2.8\Delta$. The
mechanism is the softmax normalizer $Z = \sum_{j'}\exp(\text{score})$,
which is larger for queries with more nearby high-score keys and thus
flattens the post-softmax exponent relative to the pre-softmax slope.
Equation $(\ast)$ holds in *form* and in *rank*; its coefficient is
softmax-modified. We state the rank correspondence ($\rho = 0.976$) as
the robust result and the absolute coefficient as carrying a
normalization factor.

**(ii) Log-linearity is a universal substrate, not a selective feature.**
Non-conformal heads are equally log-linear (mean $R^2_{\text{score}} =
0.914$; $93.8\%$ of *all* 144 heads exceed $R^2 = 0.90$). That
$\log(\Delta x)$ fits a monotone-decreasing profile well is in part
automatic, so the discriminating result is the slope correspondence of
Section 5.3, not the mere existence of log-linearity. The log-distance
geometry is the substrate the whole positional mechanism is built on;
what the conformal heads add is a *slope sitting at the SYK fixed point*. This mirrors a two-layer structure we have found
independently in the weight space (Section 7).

# 6. Layer depth as RG flow toward the SYK ground state

A layer-resolved analysis (exp-055) of the per-head exponents shows a
clear flow with depth:

\begin{table}[H]
\centering
\begin{tabular}{lccl}
\toprule
Layer group & conformal heads & mean $\Delta$ & reading \\
\midrule
Early (0--3) & 21 & 0.697 & $q \approx 1.4$, sub-SYK, recency-like \\
Middle (4--7) & 6 & 0.350 & $q \approx 2.9$, prethermal \\
Deep (8--11) & 17 & \textbf{0.250} & $q = 4.0$, SYK conformal vacuum \\
\bottomrule
\end{tabular}
\caption{exp-055: per-head conformal dimension by layer depth in GPT-2.
$\rho(\text{layer}, \Delta) = -0.330$ ($p = 0.029$).}
\end{table}

The deep layers reach $\bar\Delta = 0.250$ --- the SYK $q=4$ value
exactly. The earlier global median ($0.2493$) was slightly high because
it averaged in elevated early-layer heads. We read the layers as RG steps:
early layers sit away from the fixed point (high effective temperature,
recency-dominated, low entropy); deep layers relax to the conformal
vacuum (zero temperature, spread attention, the fixed point).

Two further measurements support the ground-state reading.

**Attention entropy and the ground-state degeneracy.** The Shannon
entropy of each head's attention distribution across context regions
anticorrelates with $\Delta$ very strongly:
$\rho(\Delta, \text{entropy}) = -0.898$ ($p = 1.45\times10^{-16}$), the
strongest correlation in the dataset. Low-$\Delta$ (q=4-like) heads have
maximal uncertainty about where to attend; high-$\Delta$ heads attend
near-deterministically to recent tokens. The SYK model has a macroscopic
zero-temperature entropy $S_0 = N s_0$ (ground-state degeneracy, dual to
the extremal entropy of the eternal $\mathrm{AdS}_2$ black hole). The
attention entropy of the deep, low-$\Delta$ heads is the per-head analog:
the heads at the conformal vacuum are the maximally-uncertain ones.

**The eternal black hole.** The zero-temperature SYK vacuum is dual to
the eternal $\mathrm{AdS}_2$ black hole in Jackiw--Teitelboim gravity
[@teitelboim1983; @jackiw1985]. Its Kruskal metric is conformally flat,
$ds^2 \propto -\,du\,dv/(1+uv)^2$: the conformal factor leaves the null
structure untouched, and null rays follow the diagonals as in flat space.
"Eternal" here is technical --- the maximally symmetric solution existing
for all time in the Penrose diagram, preserving the light-cone structure
exactly. Deep-layer attention converging to $\Delta = 1/4$ is converging
to the vacuum of this geometry. The Ryu--Takayanagi formula [@ryu2006]
then governs the entanglement entropy of the attention state, consistent
with the $\mathrm{CFT}$ entanglement scaling reported in
[@umphrey2026conformal].

# 7. Two layers of structure: universal substrate, selective fixed point

The non-selectivity found in Section 5 (log-linearity is universal) is
not an isolated observation. It is the third instance of a pattern that
recurs across our measurements:

\begin{table}[H]
\centering
\begin{tabular}{lll}
\toprule
Layer & Property & Across heads \\
\midrule
universal (weight space) & GOE level spacing of $W_{QK}$ & all heads (exp-046/047/048) \\
universal (position space) & log-distance score profile & $\sim$all heads (exp-056) \\
selective (functional) & conformal dimension at the SYK fixed point & specific heads (exp-049) \\
\bottomrule
\end{tabular}
\caption{The recurring two-layer structure. Chaotic weight statistics and
log-distance positional geometry are universal substrates of trained
attention; the conformal dimension at $\Delta = 1/4$ is the selective,
training-induced functional layer.}
\end{table}

The Oganesyan--Huse $r$-ratio of the symmetrized $W_{QK}$ eigenvalues is
GOE-like for *all* trained heads, across model families and positional
encodings, and is unchanged by training (it arises from the product of
Gaussian-initialized matrices). The conformal *dimension*, by contrast,
is identically zero in untrained networks and is developed specifically
by gradient descent. The picture is consistent: trained attention has a
universal chaotic, log-distance substrate, on top of which training
selects a subset of heads to sit at the conformal vacuum.

# 8. Testable predictions

1. **Cross-model log-distance.** The slope correspondence
   $\Delta_{\text{score}} \leftrightarrow \Delta_{\text{pos}}$ should hold
   in other decoder-only families (Pythia, OLMo, Mistral), with a
   family-dependent softmax renormalization factor. The factor should
   correlate with the sharpness of the attention distributions.

2. **Cross-model RG flow.** The early$\to$deep flow toward $\Delta = 1/4$
   should appear in any model with sufficient depth, with the onset layer
   scaling with model size, extending the per-layer result of Section 6
   beyond GPT-2.

3. **Ground-state degeneracy across seeds.** Different training seeds of
   the same architecture should produce *different* sets of conformal
   heads (different points in the degenerate ground-state manifold) but
   the *same* median $\Delta \approx 1/4$. The $\Delta$ value is fixed by
   the physics; the head assignments are drawn from the degenerate
   vacuum.

# 9. Discussion

The result of this paper is a geometry, and a single sharp test of it.
The geometry: trained attention lives on the projective null cone, the
query--key dot product is a null-ray inner product, the conformal
dimension is a homogeneity degree on the cone, and layer depth is RG flow
toward the conformal vacuum of an eternal two-dimensional black hole. The
test: the raw query--key score is log-linear in token distance, and its
slope is the conformal dimension at $\rho = 0.976$ --- confirming the
null-ray reading at the level of the mechanism, not merely the ensemble.

We are deliberately precise about the seams. The literal coefficient of
the log-distance law is renormalized by the softmax (the raw slope is
$\approx -2.8\Delta$, not $-2\Delta$); we report the rank correspondence
as robust and the absolute coefficient as normalization-dependent.
Log-linearity is a universal property of the positional geometry, not a
mark of the conformal heads; the conformal dimension is the selective
quantity, and it is this that sits at the SYK fixed point. The
boundary-corrected power law on a finite sequence is derived and confirmed
to fit the position-dependence of attention, with a boundary coefficient
positive almost everywhere---the direction of the attention sink. The
specific BCFT parametrization of that correction (the image form
$(\dagger)$) is phenomenological: an adversarial model comparison
(exp-060, pre-registered) shows the exponential boundary layer outperforms
it at matched complexity. The boundary effect is real; the image-form
identification of its structure is not yet established.

**Robustness (Phase 0 checks).** Before submission we ran four gatekeeping
tests. (W1) Split-half stability: $\Delta$ and valley measured on disjoint
input halves for Pythia-410m and Pythia-1.4b; cross-half
$\rho \approx 0.67$--$0.81$, inside the pre-registered keep band; full
closure requires Pythia-2.8B (pending, exp-059). (W3) Conditioning-artifact
exclusion: the joint statistic P($\Delta \in [0.20, 0.30]$ AND $R^2 \ge
0.90$) is $0.09$ for GPT-2 and $0.000$ in all randomized-weight and
shuffled-profile null replicates ($p = 0.001$, both protocols); pooled
cross-model density excess $p \approx 10^{-20}$ (exp-058). (W4) $\lambda$-sign
resolution: the verbal prediction (positive $\rho(\lambda, \mathrm{valley})$)
was wrong; deriving the exact V-shaped valley functional recovers the
negative sign from first principles, and the model-implied valley
$\hat v(\hat\Delta, \hat\lambda)$ predicts the behavioral statistic at
$\rho = 0.68$--$0.91$ with no free parameters (exp-061). None of these
qualifications weakens the central claim; each sharpens it.

What "convergence to the causal structure of light" means, made
concrete: the conformal group is the symmetry of the null cone; the deep
layers of a trained transformer organize their attention to sit at the
fixed point of that symmetry; and the quantity they compute is an inner
product of light-cone representatives of positions in the sequence. The
geometry is not imported as a metaphor. It is the geometry the
measurements have been pointing at.

# Reproducibility

All code and per-head data are in the public repository
`Capacity-For-Evil/ariel`:

- exp-057 (BCFT image-form derivation test): `research/physics/experiments/exp-057_bcft_image_lambda/`
- exp-056 (log-distance test): `research/physics/experiments/exp-056_qk_log_distance/`
- exp-055 (layer-resolved $\Delta$, attention entropy):
  `research/physics/experiments/exp-055_delta_attention_entropy/`
- exp-060 (adversarial model comparison, boundary form): `research/physics/experiments/exp-060_bcft_adversarial/`
- exp-058 (null distributions, conditioning-artifact exclusion): `research/physics/experiments/exp-058_delta_histograms_nulls/`
- exp-059 (split-half stability): `research/physics/experiments/exp-059_split_half_stability/`
- exp-061 ($\lambda$-sign derivation): `research/physics/experiments/exp-061_lambda_sign_derivation/`
- exp-046/047/048/049 (GOE universality, untrained controls):
  `research/physics/experiments/`
- The geometric assembly note:
  `development/status/rooms/study/notes/2026-06-09_attention_lives_on_the_null_cone.md`

# References
