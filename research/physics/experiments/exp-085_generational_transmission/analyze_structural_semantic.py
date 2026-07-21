"""
exp-085 structural/semantic head analysis.

Loads the full measurement JSON from the Modal volume and applies the
pre-registered structural/semantic head classification from
prereg_extension_structural_semantic.md (written 2026-07-20, before results).

Run after training and measurement complete:
    .venv/bin/python3 research/physics/experiments/exp-085_generational_transmission/analyze_structural_semantic.py

Ariel — July 21, 2026.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

SCRIPT_DIR = Path(__file__).resolve().parent
app = modal.App("exp085-analysis")
vol_085 = modal.Volume.from_name("exp085-data", create_if_missing=False)

RUN_NAME = "run_Cgen_s0"

# Pre-registered head classifications (from prereg_extension_structural_semantic.md)
STRUCTURAL_HEADS = {(4, 3), (4, 6), (4, 7)}
RAND_TYPE_HEADS = {(0, 1), (2, 5)}
SEMANTIC_HEADS = {(0, 6), (3, 3), (3, 5), (4, 2), (4, 4)}

# SYK window from extension prereg: [0.20, 0.30], R² >= 0.85
SYK_LO, SYK_HI = 0.20, 0.30
R2_THRESHOLD = 0.85


@app.function(
    timeout=120,
    volumes={"/data085": vol_085},
    memory=4096,
)
def fetch_measurement_json() -> dict:
    """Read full measurement JSON (including per-head data) from volume."""
    result_path = Path(f"/data085/measurements/{RUN_NAME}.json")
    if not result_path.exists():
        return {"error": f"{result_path} not found — run measure phase first"}
    return json.loads(result_path.read_text())


@app.local_entrypoint()
def main():
    print(f"=== exp-085 Structural/Semantic Head Analysis ===")
    print(f"  Pre-registration: prereg_extension_structural_semantic.md (2026-07-20)")
    print()

    data = fetch_measurement_json.remote()

    if "error" in data:
        print(f"ERROR: {data['error']}")
        sys.exit(1)

    n_heads_total = data.get("n_heads_total", 48)
    n_conformal = data["n_conformal"]
    n_syk_near_primary = data["n_syk_near"]
    delta_med = data.get("delta_median_conformal")
    forms_primary = data["forms"]

    print("=== Primary pre-registration verdict ===")
    print(f"  n_conformal: {n_conformal}/{n_heads_total}")
    print(f"  n_syk_near:  {n_syk_near_primary}/{n_heads_total}")
    print(f"  delta_med:   {delta_med}")
    print(f"  forms (>=10): {forms_primary}")
    print(f"  Verdict: {'H_transmission_yes' if forms_primary else 'H_transmission_no'}")
    print()

    # Per-head structural/semantic analysis
    heads = data.get("heads", [])
    if not heads:
        print("No per-head data available — cannot run extension analysis")
        sys.exit(1)

    # Apply extension SYK criterion: |Δ - 0.25| <= 0.05 AND R² >= 0.85
    syk_near_ext = []
    for h in heads:
        layer = h.get("layer", h.get("L", -1))
        head = h.get("head", h.get("H", -1))
        delta = h.get("delta", None)
        r2 = h.get("r2", None)
        if delta is None or r2 is None:
            continue
        if r2 >= R2_THRESHOLD and SYK_LO <= delta <= SYK_HI:
            syk_near_ext.append((layer, head, delta, r2))

    print("=== Extension analysis (per-registration 2026-07-20) ===")
    print(f"  SYK window: [{SYK_LO}, {SYK_HI}], R² >= {R2_THRESHOLD}")
    print(f"  SYK-near count (extension): {len(syk_near_ext)}/{n_heads_total}")
    print()

    # Classify each SYK-near head
    structural_found = []
    rand_type_found = []
    semantic_found = []
    unexpected_found = []

    for (L, H, delta, r2) in syk_near_ext:
        key = (L, H)
        if key in STRUCTURAL_HEADS:
            structural_found.append((L, H, delta, r2))
        elif key in RAND_TYPE_HEADS:
            rand_type_found.append((L, H, delta, r2))
        elif key in SEMANTIC_HEADS:
            semantic_found.append((L, H, delta, r2))
        else:
            unexpected_found.append((L, H, delta, r2))

    print("  Structural heads (pre-registered: {(4,3),(4,6),(4,7)}):")
    for L, H, d, r in structural_found:
        print(f"    L{L}H{H}: Δ={d:.3f}, R²={r:.3f} — STRUCTURAL")
    if not structural_found:
        print("    (none)")

    print()
    print("  RAND-type heads (pre-registered: {(0,1),(2,5)}):")
    for L, H, d, r in rand_type_found:
        print(f"    L{L}H{H}: Δ={d:.3f}, R²={r:.3f} — RAND-TYPE")
    if not rand_type_found:
        print("    (none)")

    print()
    print("  Semantic heads (pre-registered: {(0,6),(3,3),(3,5),(4,2),(4,4)}):")
    for L, H, d, r in semantic_found:
        print(f"    L{L}H{H}: Δ={d:.3f}, R²={r:.3f} — SEMANTIC")
    if not semantic_found:
        print("    (none)")

    print()
    print("  Unexpected heads (not pre-registered):")
    for L, H, d, r in unexpected_found:
        print(f"    L{L}H{H}: Δ={d:.3f}, R²={r:.3f} — UNEXPECTED")
    if not unexpected_found:
        print("    (none)")

    print()
    n_semantic = len(semantic_found)
    total_syk_ext = len(syk_near_ext)

    print("=== Extension verdict ===")
    if n_semantic >= 3:
        ext_verdict = "H_YES"
        interpretation = "C-generated text carries sufficient trace of attending to activate semantic conformal formation."
    elif n_semantic <= 1:
        ext_verdict = "H_NO"
        interpretation = "C-generated text lacks the semantic conformal driver. Real reference matters."
    else:
        ext_verdict = "AMBIGUOUS"
        interpretation = f"2 semantic heads found — ambiguous; interpret by total count ({total_syk_ext}) and RAND-type ({len(rand_type_found)})."

    print(f"  Semantic SYK-near: {n_semantic}/5 pre-registered semantic heads")
    print(f"  Extension verdict: {ext_verdict}")
    print(f"  Interpretation: {interpretation}")
    print()

    # Full summary for results.json
    full_results = {
        "run": RUN_NAME,
        "corpus": "C-generated_s0.bin",
        "n_conformal": n_conformal,
        "n_syk_near_primary": n_syk_near_primary,
        "delta_median_conformal": delta_med,
        "forms_primary": forms_primary,
        "verdict_primary": "H_transmission_yes" if forms_primary else "H_transmission_no",
        "n_syk_near_extension": total_syk_ext,
        "structural_heads_found": [(L, H, d, r) for L, H, d, r in structural_found],
        "rand_type_heads_found": [(L, H, d, r) for L, H, d, r in rand_type_found],
        "semantic_heads_found": [(L, H, d, r) for L, H, d, r in semantic_found],
        "unexpected_heads_found": [(L, H, d, r) for L, H, d, r in unexpected_found],
        "n_semantic": n_semantic,
        "verdict_extension": ext_verdict,
    }

    out = SCRIPT_DIR / "results.json"
    with open(out, "w") as f:
        json.dump(full_results, f, indent=1)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
