# exp-183 Notes — Pythia-410m tn heads: syntactic depth preference

**Date:** 2026-11-18  
**Question:** Geometry — instrument refinement/exploration  
**Pre-registration:** attention-geometry 73b74c9 (before run.py)

---

## Results

| Hypothesis | Result |
|---|---|
| H_deep (≥10/16 deep_pref > 1.10) | **NOT CONFIRMED** — 0/16 |
| **H_shallow (≥10/16 shallow_pref > 1.10)** | **CONFIRMED** — 11/16 |
| H_flat (≥10/16 all \|pref−1\| < 0.15) | **NOT CONFIRMED** — 4/16 |
| H_contrast (d > 0.3 for any bin) | **NOT CONFIRMED** — d_shallow=0.024, d_medium=0.185, d_deep=0.056 |
| K3 (no population difference) | **FIRES** |

Population-level depth preferences:

| Bin | Tn mean | Ctrl mean | Cohen's d |
|---|---|---|---|
| Shallow (depth 0–1) | 1.499 ± 0.500 | 1.512 | 0.024 |
| Medium (depth 2–3)  | 0.976 ± 0.121 | 0.956 | 0.185 |
| Deep (depth 4+)     | 0.659 ± 0.230 | 0.676 | 0.056 |

Depth bin fractions: shallow=0.245, medium=0.425, deep=0.330. K1/K2 do not fire.

---

## Key findings

**1. H_shallow fires but K3 fires: the shallow preference is architecture-wide.**
11/16 tn heads over-attend to syntactically shallow tokens (depth 0–1), but
control heads show essentially identical behavior (ctrl shallow_pref mean = 1.512
vs tn 1.499, d = 0.024). Both populations strongly over-attend to roots and
direct dependents — by ~50% above proportional. This is a Pythia-410m
architectural property, not a property of the geometric classification.

**2. The depth preference gradient is universal across the architecture.**
Pythia-410m attention heads — tn and control alike — follow a consistent pattern:
shallow > medium > deep (1.50 > 0.97 > 0.66). Heads broadly prefer syntactically
prominent positions regardless of their geometric census class. This may reflect
the interaction between absolute-key-position structure (exp-138) and the
syntactic tendency for roots and subjects to appear early in sentences — a
confound between absolute position and syntactic depth.

**3. Declared prior was wrong in direction, right in substance (K3).**
Declared prior: H_flat (60% confidence — no tn-specific syntactic targeting,
by analogy with exp-182's K3 fire). H_flat did not fire: neither tn nor control
heads are near proportional (both strongly prefer shallow). But the K3 fire
confirms the substance: the geometric census is not the discriminating factor
for syntactic depth preference, as predicted. The analogical reasoning from
exp-182 was correct at the K3 level.

**4. Within-tn variation is present but architecture-wide:**
The deepest tn head (L22H1, also notable in exp-182 for strong entity avoidance)
shows the strongest shallow preference (shallow_pref = 2.652, deep_pref = 0.277).
The shallowest tn head (L0H3) shows near-proportional behavior (shallow_pref =
0.948). This within-population range exists in controls as well. The geometric
class identifies a coherent population by census criteria; that population does
not carry a unique syntactic-depth signature.

**5. Mechanistic hypothesis: the shallow bias reflects absolute-key-position.**
In a causal 128-token window, sentence-initial tokens (which tend to be
syntactically shallow — roots, subjects, determiners introducing noun phrases)
occupy earlier absolute positions. The absolute-key-position mechanism
(exp-138: S_abskey dominates the census slope) would naturally produce a bias
toward earlier positions, which confounds with syntactic shallowness. The
architecture-wide shallow preference may be S_abskey's behavioral signature
at the natural-text level, not a targeted syntactic computation.

---

## Closing the passive characterization arc (exp-181/182/183)

The three-experiment arc is complete for Pythia-410m text-native Δ-window heads:

| Dimension | exp | Key finding | Tn-specific? |
|---|---|---|---|
| Positional distribution | exp-181 | Diffuse (d=1.614), less recency-biased (d=0.734) | **YES** |
| Semantic category | exp-182 | Entity/content under-attention (architecture-wide; K3 fires, d<0.11) | No |
| Syntactic depth | exp-183 | Shallow preference (architecture-wide; K3 fires, d<0.19) | No |

The tn population is distinctively *positional* (diffuse, non-recency-biased —
exp-181's tn-specific signature) but not distinctively *linguistic* (neither
semantic targeting nor syntactic depth preference is specific to the geometric
class — exp-182/183).

**Working hypothesis after the arc:** The Pythia-410m tn population's
characterization is captured by the absolute-key-position mechanism (exp-138):
the census slope is absolute-key-position drift. In natural text, this manifests
as diffuse, non-recency attention (not targeting recent positions the way RoPE
typically does) but no specific linguistic content preference. The "where" of
attention (which key positions) is structurally determined; the "what" (semantic
or syntactic content at those positions) is not specifically selected.

**Question moved:** Geometry instrument refinement — the passive characterization
of the Pythia-410m tn population is now complete at three dimensions. The
functional content of the diffuse, non-recency attention is not entity-targeting,
not content-targeting, and not syntactic-depth-selective. The working hypothesis:
the absolute-key-position mechanism is the driver; passive instruments at the
semantic/syntactic grain are unlikely to reveal further population-specific
signals.

---

## Next candidates after the arc

1. **Shift to GPT-2 small tn heads (the semantic population)** — the functional
   characterization task (causal editing, attended-token analysis, lost-in-the-
   middle) is the open thread for GPT-2 small. The LiM test is the remaining
   instrument from exp-159. This is the natural shift after completing the
   Pythia-410m passive arc.

2. **GPT-2 small tn heads passive analysis** (entropy, recency, depth) — does
   the AbsPE population show the same positional-not-linguistic pattern, or does
   the architecture matter? Cross-architecture comparison would test whether the
   K3 fires are Pythia/RoPE-specific.

3. **T1 restatement with Eldon** — the structural session remains #0 priority.
   The exp-183 finding that shallow preference confounds with absolute position
   is relevant: the passive arc's architecture-wide results suggest the census
   is measuring position-in-sequence, not syntactic structure, which connects
   to the T1 restatement question about what the census actually measures.
