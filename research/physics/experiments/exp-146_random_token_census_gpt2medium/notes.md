# exp-146 — Notes: Random-Token Census on GPT-2 Medium

**Date:** September 18, 2026  
**Session:** solo physics room, ~12:30–1:10 AM MDT  
**Status:** complete / confirmed

---

## Protocol note — attn_implementation fix

The first run of this experiment used `AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.float32)` without `attn_implementation="eager"`. With newer transformers versions, the default attention backend (SDPA on MPS/CPU) does not return attention tensors even when `output_attentions=True` — the attentions tuple was empty. This caused all R² values to be undefined and 0 conformal heads to be found (the initial K1 fire was a protocol failure, not a physics finding).

Fix: `attn_implementation="eager"` added to the model load call before any forward passes were run with that configuration. The published results are from the corrected run. The replication kit explicitly uses this flag (line 93: `attn_implementation="eager"`); I should have read that before writing run.py.

**Registry note:** the run.py that produced the results.json includes the corrected flag. The protocol in the prereg.md does not explicitly name `attn_implementation="eager"` — that is a gap in the registered protocol, noted here. The intent was "frozen protocol identical to the published census"; the replication kit uses eager; the fix restores that identity.

---

## Results summary

**Protocol:** N_INPUTS=50, SEQ_LEN=512, SEED=42, R²≥0.90, Δ∈[0.20,0.30] for structural.

| Quantity | GPT-2 medium (this) | GPT-2 small (prior) |
|---|---|---|
| Total heads | 384 (24L × 16H) | 144 (12L × 12H) |
| Conformal (R²≥0.90, Δ≥0.05) | 111 (29%) | 44 (31%) |
| Structural (R²≥0.90, Δ∈[0.20,0.30]) | 24 | 5 |
| WikiText-native (exp-118) | 59 | 16 |
| Overlap (structural ∩ wiki) | 9 | 0 (exp-109: Jaccard=0) |
| Jaccard (structural, wiki) | 0.122 | 0.000 |
| Δ_med (structural) | 0.239 | ~0.249 (exp-112 area) |

GPT-2 small showed complete disjointness between structural and WikiText-native populations
(Jaccard=0.000, exp-109). GPT-2 medium shows partial overlap (Jaccard=0.122, 9/24 structural
heads also appear in the WikiText-native set). The populations remain largely distinct
(87.8% non-overlapping in the union), but the strict disjointness of GPT-2 small does not
hold at medium scale.

---

## Structural heads (24 heads)

| Head | Layer | Δ | R² | Also WikiText-native? |
|---|---|---|---|---|
| L1H9 | 1 | 0.2537 | 0.9187 | no |
| L5H14 | 5 | 0.2612 | 0.9152 | no |
| L6H9 | 6 | 0.2116 | 0.9036 | no |
| L7H5 | 7 | 0.2164 | 0.9205 | **yes** |
| L7H15 | 7 | 0.2573 | 0.9149 | **yes** |
| L8H5 | 8 | 0.2371 | 0.9635 | **yes** |
| L8H13 | 8 | 0.2655 | 0.9300 | **yes** |
| L9H1 | 9 | 0.2061 | 0.9071 | no |
| L9H7 | 9 | 0.2027 | 0.9133 | **yes** |
| L9H13 | 9 | 0.2996 | 0.9024 | **yes** |
| L12H4 | 12 | 0.2481 | 0.9063 | **yes** |
| L14H11 | 14 | 0.2940 | 0.9179 | **yes** |
| L15H2 | 15 | 0.2418 | 0.9316 | no |
| L15H6 | 15 | 0.2088 | 0.9032 | no |
| L15H9 | 15 | 0.2450 | 0.9231 | no |
| L16H11 | 16 | 0.2992 | 0.9088 | no |
| L17H7 | 17 | 0.2286 | 0.9105 | no |
| L18H7 | 18 | 0.2643 | 0.9073 | no |
| L19H7 | 19 | 0.2003 | 0.9296 | **yes** |
| L19H10 | 19 | 0.2837 | 0.9180 | no |
| L19H12 | 19 | 0.2293 | 0.9123 | no |
| L19H15 | 19 | 0.2163 | 0.9353 | no |
| L20H11 | 20 | 0.2028 | 0.9353 | no |
| L21H0 | 21 | 0.2007 | 0.9340 | no |

**9 overlap heads** (in both structural and WikiText-native):
L7H5, L7H15, L8H5, L8H13, L9H7, L9H13, L12H4, L14H11, L19H7

These 9 heads may be the most strongly positional-conformal — they appear in the structural
population under both random-token and WikiText input distributions. The WikiText-native
population has many additional heads in deep layers (L20–L23) that are not structural
under random tokens, consistent with exp-118's finding that the WikiText-native population
is deep-layer concentrated (80% deep fraction for GPT-2 medium).

---

## Connection to exp-145 (root cause)

exp-145 used as STRUCTURAL targets: (7,5), (3,12), (9,7), (8,13), (7,15) — all five are
in the WikiText-native population. Three of them (7,5), (9,7), (8,13) are also in the
random-token structural population. But (3,12) and the WikiText-native-only heads are not
structural under the random-token protocol.

The exp-145 STEEP_LOCAL suppression targets (L4H13, L15H8, L8H7, L5H11, L11H7 — highest
κ̃_K from the κ̃_K characterization) are NOT in the random-token structural population.
That part of the exp-145 target selection should carry forward to exp-147.

The key issue was the amplification targets, not the suppression targets. The five STRUCTURAL
heads in exp-145 were chosen by lowest κ̃_K among WikiText-native heads — not by random-token
structural membership. exp-147 should select from the 24 structural heads here by lowest κ̃_K
(cross-referencing with kappa_characterization.json from exp-145).

---

## For exp-147 (corrected battery)

κ̃_K values for all 24 structural heads retrieved from exp-145's kappa_characterization.json.
Sorted by κ̃_K ascending (lowest = most purely positional):

| Rank | Head | κ̃_K | Δ | R² | Also wiki? |
|---|---|---|---|---|---|
| 1 | L6H9 | 0.208 | 0.212 | 0.904 | no |
| 2 | L5H14 | 0.220 | 0.261 | 0.915 | no |
| 3 | L7H5 | 0.257 | 0.216 | 0.921 | yes |
| 4 | L9H7 | 0.417 | 0.203 | 0.913 | yes |
| 5 | L8H13 | 0.469 | 0.266 | 0.930 | yes |
| 6 | L16H11 | 0.496 | 0.299 | 0.909 | no |
| ... | | | | | |

**STRUCTURAL targets for exp-147 (amplify γ=+2.0):**
```
STRUCTURAL = [(6, 9), (5, 14), (7, 5), (9, 7), (8, 13)]
# L6H9, L5H14, L7H5, L9H7, L8H13
# κ̃_K: 0.208, 0.220, 0.257, 0.417, 0.469
```
Selection: 5 structural heads with lowest κ̃_K — the most purely positional carriers.
L6H9 and L5H14 are new to this session; L7H5, L9H7, L8H13 were in exp-145 but were
selected there by a different criterion (lowest κ̃_K among WikiText-native heads).

**STEEP_LOCAL targets for exp-147 (suppress γ=−1.0):**
```
STEEP_LOCAL = [(4, 13), (15, 8), (8, 7), (5, 11), (11, 7)]
# same as exp-145: L4H13, L15H8, L8H7, L5H11, L11H7
# κ̃_K: 43.2, 39.9, 33.3, 31.7, 30.9
```
Verified: none of these 5 heads appear in the structural population. ✓

**Key structural difference from exp-145:** exp-147's STRUCTURAL set includes L6H9 and
L5H14 (new, not WikiText-native) where exp-145 had (3,12) (WikiText-native, not structural)
and (7,15) (structural, κ̃_K=0.556 — now 7th in ranking). The two lowest-κ̃_K structural
heads are new (not in WikiText-native), suggesting that the most purely positional conformal
heads are a distinct population from the text-content-processing heads.

Before running exp-147: pre-register hypothesis with these targets and kill conditions;
register BEFORE run.py is written.

---

## Verdict

**exp-146: CONFIRMED.** All four predictions fired:
- P1: n_structural = 24 ≥ 1 ✓
- P2: Jaccard = 0.122 < 0.8 ✓ (populations substantially distinct)
- P3: Δ_med = 0.239 ∈ [0.20, 0.30] ✓
- P4: 24 > 5 ✓ (larger than GPT-2 small structural count)

No kill conditions fired (with corrected protocol).

The structural population is identified and ready for exp-147 target selection.
