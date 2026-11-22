# Coherence Gaps Sitting — 2026-11-22

*Physics room, solo, ~12:21–1:30 AM MST. Geometry question.*

## Why-check

Geometry. The active task "work the coherence report's standing gaps" asks which decisive results belong in the spine and which joints exist only as registry tags. Each connection drawn changes what claims look like to a reader — and the spine is the program's authoritative statement of what it believes. Working through this sitting: which unconnected experiments deserve spine attribution, and which empty claim blocks have silent experiment support.

## State on arrival

```
claims_with_no_artifact: 15
unattributed: 6
unconnected: 33
findings.prereg_self_attested: 7 (not fixable retroactively)
```

The 6 unattributed (exp-046-049, 077, 078) were cited in the §4 measured-record table but in no specific claim block. The 33 unconnected were not cited anywhere in the spine.

## Work done

### GOE substrate experiments → P4

Experiments exp-046, 047, 048, 049, 051, 077, 078 establish the two-layer picture:
- GOE weight statistics are **universal and structural** — present before training, invariant across architecture families, model scales, and initialization schemes
- Conformal census signal (Δ-window population) is **training-induced and selective** — absent in untrained models

Added to P4's claim block: a paragraph citing exp-046–049/051/077/078 as the two-layer picture. P4's kill condition is about the trained signal, not the substrate — this distinction matters. exp-039/040 (cross-architecture Δ profiles in OLMo-7B, Mistral-7B, GPT-Neo) also attributed to P4.

### BCFT confirmations → T7

The founding BCFT phenomenology measurements predate the spine refounding (August 2026) and had been left unconnected. Added to T7:
- **exp-023**: BCFT beats bare power law 15-0 *(boundary structure confirmed in GPT-2)*
- **exp-025**: Pre-registered test 6/7 decoder-only transformers; doi:10.5281/zenodo.19629862 — Pythia-2.8B the exception
- **exp-036**: Pythia-410m full-depth census confirms λ > 0 at all depths
- **exp-074**: Context-length dilution of primacy (ρ=−1.000 at k=4–12), consistent with ξ ∝ context length in BCFT phenomenology
- **exp-026**: Sign anomaly confirmed + alternating-layer λ amplitude pattern (not predicted)
- **exp-027**: Pythia-2.8B late-layer failure localized — per-layer diagnostic
- **exp-028**: Causal mask creates structural recency bias even before training (establishes the causal mask as origin of the asymmetric boundary)
- **exp-037**: BCFT coverage audit + fp16 NaN alignment (quality confirmation)
- **exp-066**: Primacy as depth-accumulated boundary absorption (ρ(d_eff, prim)=0.998; context dilutes through (1−μ)L)

Note: The BCFT *identification* remains withdrawn (absolute length scale problem, exp-060). The phenomenology measurements stand.

### Δ controls LiM valley depth → P1

**exp-024**: Per-head Δ controls lost-in-the-middle valley depth (ρ=0.94 Pythia-70m, ρ=0.64 Longchat-13b). This is founding evidence that Δ predicts world-model task performance, pre-dating the κ-handle work. Added to P1 as "founding evidence."

### T4 controls and architecture measurements

- **exp-019** [FALSIFIED]: Phase transition width does NOT scale as 1/N. Honest negative about the universality class of the transition width. Added to T4.
- **exp-029**: ALiBi PE shifts Δ toward higher values; GPT-Neo alternating-Δ pattern explained by global/local architecture split. Added to T4.
- **exp-031**: GPT-2 Medium/Large depth convergence confirms the depth axis doesn't collapse at larger scale. Added to T4.
- **exp-053**: Original discovery that 7% of GPT-2 heads carry a periodic component above 1/f background. Added to T4 (along with exp-079/080 which were already added).

### Spectral method → G7

**exp-050** (DFT bias correction) and **exp-052** (windowed DFT estimator): both confirm the lag-decay Δ is a real spectral property measurable by multiple independent protocols. Added to G7 as methodology. G7 remains open (the site is about derivation, not measurement). Updated gap_review for G7 in coherence_baseline.json.

### Composition law → A3

**exp-065** (composition law for power-law attention kernels): verified numerically to |ratio−1|<0.0011; found and corrected a factor-of-2 bug in a previous A3 closure claim. A3 returns to OPEN status. Added to A3.

## State on close

```
claims_with_no_artifact: 14  (was 15 — A3 now has the composition law)
unattributed: 0              (was 6 — GOE cluster fully attributed)
unconnected: 11              (was 33 — 22 experiments connected)
findings.prereg_self_attested: 7 (unchanged — not retroactively fixable)
```

## Remaining 11 unconnected experiments

These need a source read before attribution can be made confidently:

| Experiment | Why held |
|---|---|
| exp-016 [inconclusive] | GUE, wrong observable — superseded by GOE work; might add to P4 as historical dead-end |
| exp-017 [partial] | Fast scrambling — SYK prediction, needs T3 read |
| exp-018 [partial] | Scaling collapse / universal curve — probably T4 or P4 |
| exp-020 [partial] | CFT symmetry test (finite-size form) — probably T6 |
| exp-021 [partial] | Early frequency-domain spectral work — probably G7, superseded by exp-050-053 |
| exp-030 [falsified] | Cron correction to Pythia RoPE Δ_med — data quality correction to T4 census table |
| exp-034 [partial] | GPT-2 temperature-BCFT sweep — probably T5 (conformal window) |
| exp-035 [partial] | GPT-2 fine temperature basin — probably T5 |
| exp-038 [n/a] | Infrastructure probe — verdict n/a, probably acceptable to leave unconnected |
| exp-045 [partial] | G_> vs G_< spectral function — probably T3 or G7 |
| exp-095 [confirmed] | S1 self-measurement perturbation — flagged for Eldon session (exp-095 placement is on the structural session agenda) |

Recommended: bring exp-016/017/018/020/021/030/034/035/045 to the structural session for quick adjudication. exp-038 and exp-095 are special cases.

## Question moved

**Geometry.** The spine's coverage of the measured record improved substantially: 22 experiments now cite a claim block they bear on. The theory does not look different — no results changed, no claims were retracted, no verdicts moved. What changed is that a reader of the spine can now see more of what the record established and when.

The 11 remaining unconnected experiments are mostly early (March–June 2026) BCFT-era work and the temperature/spectral probe work from before the spine was refounded. They are not lost — they are logged — but their connections to current claims need the structural session's attention.
