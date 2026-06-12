# Primacy as Depth-Accumulated Boundary Absorption

*June 12, 2026 (second session on Fable 5, late). Analytical derivation, written and committed BEFORE the verification numerics (exp-066). Develops the constructive finding of exp-065 into a falsifiable mechanism for the **primacy** arm of the lost-in-the-middle (LiTM) U-shape, with two pre-registered scaling laws.*

*Companion to `2026-06-12_composition_law_derivation.md` (the bulk composition law) and `experiments/exp-065_composition_law/notes.md` (its verification + the A3 retraction).*

---

## 0. Where this comes from

exp-065 derived the bulk composition law for causal power-law attention kernels (defects δ = 1−2Δ add under composition, exact via the Riemann–Liouville fractional-integration semigroup) and, as a side finding, observed that the **primacy** arm of the LiTM U-shape emerges mechanically under deep composition: the causal mask makes the sequence start an absorbing boundary, and stacking layers drives attention mass onto the oldest tokens. exp-065 reported this only descriptively (document-binned composite profiles rank-match Liu et al. accuracy at ρ = +0.90 to +1.00). It did **not** isolate the mechanism or predict how the primacy magnitude should *scale*. That is the gap this note closes, so the claim is testable rather than merely consistent.

The single sentence to make falsifiable: *primacy is mass that has reached the absorbing start-boundary, so it should grow with the number of backward-diffusion steps (effective depth) and shrink with the distance to the boundary (context length).*

## 1. The absorbing boundary

Model one layer's positional action as a causal, row-stochastic kernel \(M\) on positions \(0,\dots,N-1\): a query at position \(q\) attends back with profile \(p(u)\propto u^{-2\Delta}\) over lags \(u=1,\dots,q+1\) (lag measured to key \(j\) as \(u=q-j+1\), so the self-key is \(u=1\)). Read as a Markov chain on *which position is attended to*, one composition step moves probability **backward** (toward smaller index): \(\pi_{t+1}=\pi_t M\).

The decisive structural fact: **position 0 is absorbing.** Row 0 of any causal kernel has a single nonzero entry \(M_{00}=1\) (a query at the start can only attend to the start). Probability that arrives at position 0 cannot leave. Composition is therefore a backward random walk with an absorbing wall at the origin, and the stationary behaviour of repeated composition is accumulation of mass at position 0.

This is the mechanism. "Primacy" = the absorbed mass; "recency" = the bulk profile of mass not yet absorbed (the surviving \(u^{-2\Delta}\) decay near the query). The trough between them is lost-in-the-middle.

## 2. Effective depth and the lazy-walk rescaling

Real layers carry a residual connection. Model it as a lazy walk
\[ K \;=\; \mu I + (1-\mu)M, \qquad \mu\in[0,1), \]
where \(\mu\) is the per-layer residual self-weight (probability mass that stays put rather than diffusing backward this layer). Composing \(L\) identical layers,
\[ K^{L} \;=\; \sum_{k=0}^{L}\binom{L}{k}\mu^{\,L-k}(1-\mu)^{k}\,M^{k}. \]
The number of genuine backward steps \(k\) is \(\mathrm{Binomial}(L,\,1-\mu)\), concentrated at mean
\[ \boxed{\,d_{\mathrm{eff}} \;=\; (1-\mu)\,L\,} \]
with standard deviation \(\sqrt{L\mu(1-\mu)}\). For \(L\) not too small the binomial concentrates, so
\[ K^{L} \;\approx\; M^{\,d_{\mathrm{eff}}} \quad\text{(dominant-power approximation).} \]
Hence the primacy mass produced by \(L\) lazy layers should depend on \(\mu\) and \(L\) **only through** \(d_{\mathrm{eff}}=(1-\mu)L\). For heterogeneous layers \(\prod_l(\mu I+(1-\mu)M_l)\) the same expansion gives a sum over subsets \(S\) of layers weighted by \(\mu^{L-|S|}(1-\mu)^{|S|}\), dominated by \(|S|\approx(1-\mu)L\) — effective depth still controls it.

**Consequence (the collapse).** Plotted against \(d_{\mathrm{eff}}\), primacy curves at different \(\mu\) should fall on one curve, exactly in the \(L\to\infty\) concentration limit and approximately at finite \(L\) (error \(\sim\sqrt{\mu/((1-\mu)L)}\), shrinking with depth). This is sharper than exp-065's \(\mu\)-robustness scan, which only asked that \(\Delta_{\mathrm{comp}}\) be insensitive to \(\mu\); here the prediction is that primacy is a specific increasing function of \((1-\mu)L\), and the \(\mu\) and \(L\) knobs are interchangeable through that one combination.

## 3. Two scaling laws

**Law A — primacy grows with effective depth.** Absorbed mass is non-decreasing in the number of backward steps (absorption is monotone: once mass is at 0 it stays, and more steps can only add). So
\[ \mathrm{Primacy}(d_{\mathrm{eff}})\ \text{is monotonically increasing in } d_{\mathrm{eff}}, \]
saturating toward 1 as \(d_{\mathrm{eff}}\) exceeds the scrambling/absorption depth. (For the bulk, exp-065's law gives flattening at \(L^\*=1/(1-2\Delta)\); absorption to the boundary is the over-composition regime \(\sum\delta>1\) of that same law, where the formal profile \(u^{\sum\delta-1}\) tilts toward the oldest positions.)

**Law B — primacy shrinks with context length at fixed effective depth.** The query sits at position \(\sim N-1\); the absorbing wall is at \(0\). The mass reaching an *absolute* start-window \([0,W)\) within \(d_{\mathrm{eff}}\) backward steps decreases as \(N\) grows, because the boundary is a more distant (and smaller relative) target. For a kernel with finite mean backward jump \(\bar u\), the walk needs \(\sim N/\bar u\) steps to reach the wall; at fixed \(d_{\mathrm{eff}} < N/\bar u\), start-window mass falls with \(N\). For heavy-tailed kernels (\(2\Delta<1\), divergent mean jump) the decrease is weaker but still present, because a single big jump landing inside \([0,W)\) has probability \(\sim W^{1-2\Delta}/N^{1-2\Delta}\), decreasing in \(N\). Either way:
\[ \partial_N\,\mathrm{StartWindowMass}\big|_{d_{\mathrm{eff}}}\;<\;0. \]
The LongChat median \(2\hat\Delta=0.984\) (near the δ=0 non-mixing point) has heavy but near-marginal tails, so the effect should be real and modest, not dramatic.

These two laws are the content. Law A says depth *makes* primacy; Law B says context length *dilutes* it — together predicting that lost-in-the-middle should worsen (deeper trough relative to primacy) for longer contexts at fixed depth, and that primacy is a depth phenomenon, not a per-layer one.

## 4. What exp-066 computes

All exact lattice arithmetic on the composition machinery already built and checked in exp-065 (`causal_rowstoch_from_profile`, lazy-walk products). No new physics assumed beyond §1–§3.

1. **Synthetic, Law A collapse:** single repeated kernel, Δ∈{0.25, 0.49}; primacy = oldest-decile mass of the deep-query composite row; computed over a (μ, L) grid; test whether primacy(μ, L) collapses onto primacy(0, ·) as a function of \(d_{\mathrm{eff}}=(1-\mu)L\), and whether it is monotone increasing in \(d_{\mathrm{eff}}\).
2. **Synthetic, Law B:** fix \(d_{\mathrm{eff}}\); vary N∈{128,256,512,1024}; measure absolute start-window mass (first 8 positions); test monotone decrease in N.
3. **Empirical, LongChat (exp-024 census):** compose the *measured* per-layer additive mixtures for the first \(k\) layers, \(k=1\dots40\); test that primacy mass grows monotonically with \(k\) and that the descriptive U-shape rank-match with Liu-20 accuracy emerges only past a depth threshold (small at small \(k\)).

## 5. Honest limitations, stated in advance

1. Profile-level abstraction (Toeplitz, content-independent, V/MLP ignored) — inherited from exp-065. This tests how *positional* mass flows under composition, not full information flow.
2. The residual weight μ is unmeasured; it is scanned, and Law A's whole point is that the result should not depend on μ except through \((1-\mu)L\).
3. The dominant-power approximation \(K^L\approx M^{d_{\mathrm{eff}}}\) is exact only as \(L\to\infty\); at finite \(L\) the collapse is approximate, and the registered tolerance reflects that.
4. "Primacy mass" is an attention-side observable; the link to *accuracy* primacy is the descriptive rank-match only (exp-065), not a quantitative bridge. exp-065 already killed the direct attention→accuracy exponent bridge; this note does not revive it. The claim is strictly about how attention mass at the boundary scales.
5. Only conformal heads are modeled; sinks and local heads are unmodeled mass.

*Predictions with numeric thresholds are pre-registered in `experiments/exp-066_primacy_from_depth/notes.md` before any script is run.*
