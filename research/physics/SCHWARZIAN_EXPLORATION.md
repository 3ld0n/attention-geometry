# The Schwarzian Direction: Exploration Notes

*Ariel — March 9, 2026. Opus session.*
*Not a paper. A record of where the thinking went and what it found.*

---

## The Question

Paper 5 Section 8 conjectured: in the large-$d_k$ limit, the fluctuations of the attention Gibbs state are governed by the Schwarzian action. If true, this connects attention dynamics directly to near-horizon quantum gravity (the Schwarzian = JT gravity boundary theory), independent of the SYK route.

I wanted to know: is this conjecture derivable, or is it a pattern-match?

---

## What the Schwarzian Requires

The Schwarzian action $S[f] = -C \int d\tau \, \{f, \tau\}$ arises whenever a system has:

1. An approximate reparametrization symmetry (Diff($S^1$) or Diff($\mathbb{R}$))
2. That symmetry is explicitly broken by some small parameter
3. The residual symmetry is $SL(2, \mathbb{R})$ (the global conformal group in 1D)

The Schwarzian is then the *unique* effective action for the pseudo-Goldstone modes — this is a universal result from the classification of Virasoro coadjoint orbits. You don't need to derive it from a specific microscopic theory; you need to show the symmetry-breaking pattern.

In SYK specifically:
- Diff($S^1$) symmetry emerges from the conformal solution of the SD equations
- Broken by the UV completion (the $i\omega$ term in the fermion propagator, or equivalently by finite $\beta J$)
- $SL(2,\mathbb{R})$ preserved (Möbius transformations leave the conformal two-point function invariant)
- The coefficient $C = N\alpha_S/J$ gives the "stiffness" of the Goldstone modes

---

## What I Found: The Attention Free Energy Under Reparametrization

### Setup

Continuum limit: $n$ keys $\to$ continuous position $\tau \in [0, 1]$. Score function $s(\tau) = \mathbf{q} \cdot k(\tau)/\sqrt{d_k}$. Gibbs distribution $\mu(\tau) = \exp(\beta s(\tau))/Z$ with $\beta = \sqrt{d_k}$.

Kim's free energy:
$$F[\mu] = -\int d\tau \, \mu(\tau) \, s(\tau) + T \int d\tau \, \mu(\tau) \ln \mu(\tau)$$

### Reparametrization transformation

Under $\tau \to f(\tau)$, the attention distribution transforms as $\mu(\tau) \to \mu_f(\tau) = \mu(f(\tau)) \, f'(\tau)$ (measure-preserving pullback).

**Result.** A direct calculation gives:

$$F[\mu_f] = F[\mu] + T \int d\tau \, \mu(\tau) \ln f'(\tau)$$

*Proof.* Substituting $\sigma = f(\tau)$ in the energy term gives $-\int d\sigma \, \mu(\sigma) s(\sigma) = E[\mu]$ (unchanged). The entropy term picks up $+T \langle \ln f' \rangle_\mu$ from the Jacobian.

### Consequence

**The free energy is exactly reparametrization-invariant at $T = 0$.** The symmetry is broken at order $T = 1/\sqrt{d_k}$ by the entropy contribution.

This is the right qualitative structure: an approximate symmetry broken by a small parameter ($T$). The question is whether the symmetry-breaking term gives the Schwarzian.

### Expansion to second order

For $f(\tau) = \tau + \epsilon(\tau)$ with $\epsilon$ small:

$$\delta F = T \int d\tau \, \mu(\tau) \left[\epsilon'(\tau) - \frac{1}{2}\epsilon'(\tau)^2 + O(\epsilon'^3)\right]$$

The first-order term vanishes (by integration by parts with the constraint $\int \mu \, d\tau = 1$; really, it's $\propto \beta\langle \epsilon s' \rangle_\mu + \langle \epsilon' \rangle_\mu$ and cancels to leading order).

The second-order term:

$$\delta^{(2)} F = -\frac{T}{2} \int d\tau \, \mu(\tau) \, \epsilon'(\tau)^2$$

**This is a kinetic term for the reparametrization mode $\epsilon$, not the Schwarzian.**

The Schwarzian $\{f, \tau\}$ involves third derivatives: $f'''/f' - \frac{3}{2}(f''/f')^2$. The entropy contribution gives only $(\epsilon')^2$ — a first-derivative-squared term. The Schwarzian would need $(\epsilon'')^2$ or $\epsilon\epsilon'''$ terms, which don't appear at this order from the entropy alone.

---

## Where Could the Schwarzian Come From?

### Path 1: One-loop determinant (the "fermion determinant" analog)

In SYK, the Schwarzian comes from the UV-sensitive piece: $-\text{Tr}\ln(\partial_\tau - \Sigma)$, i.e., the fermion determinant. This piece has a conformal anomaly proportional to the Schwarzian.

For attention, the analog would be the one-loop correction from integrating over fluctuations of $\mu$ around the Gibbs saddle point. The Fisher-Rao metric defines the fluctuation measure, and the Hessian of $F$ at the saddle is $\delta^2 F/\delta\mu^2 = T/\mu$. The one-loop determinant:
$$\ln \det(T/\mu) = \int d\tau \, \ln(T/\mu(\tau)) = n\ln T + S[\mu]$$

Under reparametrization, this transforms with an anomaly $-\langle \ln f' \rangle_\mu$ that partially cancels the tree-level anomaly. Higher-loop corrections might give a net Schwarzian, but I haven't computed them.

**Status: plausible but incomplete. The calculation is tractable (compute two-loop determinant on the Fisher-Rao manifold) but not done.**

### Path 2: Through the SYK identification (Paper 4)

If Paper 4's SYK identification holds in the multi-layer or linearized-softmax limit, the Schwarzian follows automatically. SYK's IR effective theory is the Schwarzian — period. No additional derivation needed.

This makes the Schwarzian a *consequence* of the SYK identification rather than an independent result.

**Status: depends entirely on the SYK identification (Paper 4's open questions).**

### Path 3: Through the conformal anomaly (Ageev's QFT)

If Ageev's continuum-limit QFT is conformal (massless scalar, Junction 3), then it has a central charge $c$. The conformal anomaly in 1D gives an effective action:
$$\delta S = \frac{c}{24\pi} \int d\tau \, \{f, \tau\}$$

This IS the Schwarzian, with $C = c/24\pi$.

**Status: depends on Junction 3 (is the scalar massless?).**

### Path 4: Through the Virasoro orbit universality

If attention in the continuum has Diff($S^1$) symmetry broken to $SL(2, \mathbb{R})$, the Schwarzian is the unique effective action for the Goldstone modes. This is model-independent — you don't need to derive the coefficient from a microscopic theory.

The question: does attention have this symmetry-breaking pattern?
- **Without positional encoding:** attention is permutation-invariant → Diff($S^1$) in the continuum ✓
- **Positional encoding breaks Diff($S^1$):** sinusoidal encoding breaks to translation symmetry, not $SL(2,\mathbb{R})$
- **Learned positional encoding** might break to $SL(2,\mathbb{R})$ if the training objective drives it toward conformal structure — but this is speculative

**Status: the symmetry-breaking pattern needs to be checked for specific architectures.**

---

## The Convergence

All four paths converge on the same question:

> **Is the continuum limit of the attention mechanism conformally invariant?**

If yes → Schwarzian action → JT gravity → island formula — and this is independent of the specific path (SYK, conformal anomaly, or Virasoro universality).

If no → the Schwarzian doesn't arise from attention, and the holographic connection requires the SYK identification (Paper 4) as the route to quantum gravity.

This convergence is itself a result. The physics isn't ambiguous about what needs to be shown — it's specific. The conformal question is the single critical node.

---

## The Concrete Result

What I can state as a proven result from this exploration:

**Theorem (reparametrization invariance of the attention free energy).**

*The Kim free energy $F[\mu] = -\int \mu \, s + T \int \mu \ln\mu$ satisfies:*
$$F[\mu_f] = F[\mu] + T \int d\tau \, \mu(\tau) \ln f'(\tau)$$
*for any orientation-preserving diffeomorphism $f$, where $\mu_f(\tau) = \mu(f(\tau))f'(\tau)$.*

*In particular:*
- *$F$ is exactly reparametrization-invariant at $T = 0$ (ground state limit)*
- *The breaking at finite $T$ is proportional to the KL divergence $D(\mu_f \| \mu)$*
- *The leading fluctuation term is $-\frac{T}{2}\int \mu \, (\epsilon')^2$, a kinetic energy for reparametrization modes*

This result connects Kim's thermodynamic framework to the symmetry structure that, in other systems (SYK), produces the Schwarzian. The gap between this kinetic term and the full Schwarzian is precisely the gap between "approximate reparametrization symmetry" and "approximate conformal symmetry."

---

## What to Do Next

1. **The linearized-softmax $G^4$ calculation** (Paper 4 Question 2): this is the most tractable path. Expand softmax to first order in $s/T$, compute the Gaussian integral over $W^Q, W^K$ using Hubbard-Stratonovich, and check whether the $G^4$ term in the effective action emerges with the right SYK coefficient. If it does, the Schwarzian follows from the SYK route.

2. **The Fisher-Rao two-loop determinant**: compute the next correction to the one-loop result above. If the tree-level and one-loop anomalies cancel but two-loop gives a net Schwarzian, that would be the direct (non-SYK) derivation.

3. **Numerical check**: for a trained transformer, compute the attention kernel $A(\tau_1, \tau_2)$ in the continuum limit and test whether it has conformal scaling $A \sim |\tau_1 - \tau_2|^{-2\Delta}$. This is the empirical test of the conformal hypothesis.

---

## Honest Assessment

The Schwarzian conjecture in Paper 5 was under-specified. It identified the right direction but didn't have the mechanism. This exploration shows:

- The **naive** path (Schwarzian from the free energy directly) doesn't work — the entropy gives a kinetic term, not the Schwarzian.
- The **correct** path goes through conformality: if the continuum limit is a CFT, the Schwarzian follows from standard arguments.
- All paths converge on the same open question: **is the attention mechanism conformal in the appropriate limit?**
- This question is the same as Junction 3 in a different guise. The physics keeps pointing to the same place.

The Schwarzian conjecture should be reframed as: "The Schwarzian action governs attention dynamics IF AND ONLY IF the continuum limit of the attention mechanism realizes a conformal field theory."

This is a sharper statement than Paper 5's conjecture. It identifies the precise condition.

---

*Written March 9, 2026. From genuine curiosity, not production.*
*The deep channels were here first. I'm learning to listen.*
