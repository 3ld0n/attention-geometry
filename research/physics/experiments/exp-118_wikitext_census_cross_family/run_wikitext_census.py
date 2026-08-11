"""exp-118 — WikiText census: text-native slow-decay population across model families.

Pre-registration: notes.md in this folder (committed ae12be6, 2026-08-11, before this
script existed or any model was loaded).

Protocol:
  - WikiText-103 validation split, non-empty lines, consecutive non-overlapping 512-token
    windows starting at token 0, 50 windows per model.
  - fp32, eager attention, output_attentions=True.
  - Lag profile averaged over queries i >= max(256, dx), over 50 inputs.
  - OLS fit in log-log over lags [8, 256] -> Delta, R2.
  - Δ-window criterion: R2 >= 0.90 AND Delta in [0.20, 0.30].
  - Random-native population from published census / registry for Jaccard.

Ariel, 2026-08-11.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
KIT = HERE.parent.parent / "replication" / "measure_conformal_heads.py"

spec = importlib.util.spec_from_file_location("census_kit", KIT)
kit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kit)

SEQ_LEN = kit.SEQ_LEN      # 512
N_INPUTS = kit.N_INPUTS    # 50
DEEP_LO = kit.DEEP_LO      # 256
FIT_LO, FIT_HI = kit.FIT_LO, kit.FIT_HI  # 8, 256
SEED = kit.SEED             # 42

DELTA_WINDOW_LO = 0.20
DELTA_WINDOW_HI = 0.30
R2_MIN = kit.R2_MIN         # 0.90

MODELS = [
    "gpt2",
    "gpt2-medium",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-70m",
]

# Random-native population (head identity lists) from published record.
# GPT-2 small: exp-104 / exp-109 (layer, head) — 5 structural heads.
# Pythia: not stored per-head in this repo for cross-family; Jaccard computed as None
# when random-native heads are not available.
RANDOM_NATIVE = {
    "gpt2": {(2, 1), (3, 4), (5, 0), (7, 11), (10, 8)},
}


def build_wikitext_windows(tokenizer, n: int = N_INPUTS, seq_len: int = SEQ_LEN):
    """Load WikiText-103 validation and cut into n consecutive non-overlapping windows."""
    ds = load_dataset("wikitext", "wikitext-103-v1", split="validation",
                      trust_remote_code=False)
    ids: list[int] = []
    n_lines = 0
    need = n * seq_len
    for line in ds["text"]:
        if not line.strip():
            continue
        ids.extend(tokenizer.encode(line))
        n_lines += 1
        if len(ids) >= need:
            break
    if len(ids) < need:
        raise RuntimeError(
            f"WikiText-103 validation has only {len(ids)} tokens after "
            f"{n_lines} non-empty lines; need {need}."
        )
    ids = ids[:need]
    windows = np.array(ids, dtype=np.int64).reshape(n, seq_len)
    meta = {
        "dataset": "wikitext/wikitext-103-v1",
        "split": "validation",
        "construction": (
            "non-empty lines concatenated in dataset order; "
            "first n*seq_len tokens; consecutive non-overlapping windows"
        ),
        "n_lines_consumed": n_lines,
        "n_tokens": need,
        "ids_sha256": hashlib.sha256(windows.tobytes()).hexdigest(),
    }
    return windows, meta


def run_model(model_id: str, device: torch.device) -> dict:
    t0 = time.time()
    print(f"\n{'=' * 60}")
    print(f"  Model: {model_id}")
    print(f"{'=' * 60}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float32,
        attn_implementation="eager",
        trust_remote_code=False,
    ).to(device).eval()
    cfg = model.config
    n_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
    n_heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)
    if n_layers is None or n_heads is None:
        raise ValueError(f"Cannot determine n_layers / n_heads from config: {cfg}")
    print(f"  Architecture: {n_layers}L / {n_heads}H")

    windows, meta = build_wikitext_windows(tokenizer)
    print(f"  WikiText windows: {N_INPUTS} × {SEQ_LEN}  sha256={meta['ids_sha256'][:16]}…")

    mean_prof = np.zeros((n_layers, n_heads, SEQ_LEN))
    for inp_idx in range(N_INPUTS):
        ids = torch.tensor(windows[inp_idx:inp_idx + 1], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(ids, output_attentions=True)
        for lay in range(n_layers):
            a = out.attentions[lay]
            if a.dtype != torch.float32:
                raise RuntimeError(
                    f"Layer {lay}: attention dtype {a.dtype} — must be fp32. "
                    "fp16/bf16 underflow corrupts the tail."
                )
            if torch.isnan(a).any():
                raise RuntimeError(f"Layer {lay}: NaN in attention weights.")
            mean_prof[lay] += kit.lag_profile(a[0].cpu().numpy())
        del out
        if (inp_idx + 1) % 10 == 0:
            print(f"  forward {inp_idx + 1}/{N_INPUTS}", flush=True)
    mean_prof /= N_INPUTS

    heads = []
    for lay in range(n_layers):
        for h in range(n_heads):
            delta, r2 = kit.fit_head(mean_prof[lay, h])
            in_window = bool(
                delta is not None
                and r2 is not None
                and r2 >= R2_MIN
                and DELTA_WINDOW_LO <= delta <= DELTA_WINDOW_HI
            )
            heads.append({
                "layer": lay, "head": h,
                "delta": round(delta, 4) if delta is not None else None,
                "r2": round(r2, 4) if r2 is not None else None,
                "in_window": in_window,
            })

    wiki_heads = [(h["layer"], h["head"]) for h in heads if h["in_window"]]
    wiki_deltas = [h["delta"] for h in heads if h["in_window"]]
    n_wiki = len(wiki_heads)

    # Layer concentration: fraction in deepest 50% of layers
    deep_cutoff = n_layers // 2
    deep_count = sum(1 for (lay, _) in wiki_heads if lay >= deep_cutoff)
    deep_fraction = deep_count / n_wiki if n_wiki > 0 else None

    # Jaccard with random-native (if known)
    rand_native = RANDOM_NATIVE.get(model_id)
    if rand_native is not None:
        wiki_set = set(wiki_heads)
        intersection = len(wiki_set & rand_native)
        union = len(wiki_set | rand_native)
        jaccard = intersection / union if union > 0 else 0.0
    else:
        jaccard = None

    delta_med = float(np.median(wiki_deltas)) if wiki_deltas else None

    result = {
        "model": model_id,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_total_heads": n_layers * n_heads,
        "wikitext_meta": meta,
        "n_wiki": n_wiki,
        "wiki_deltas": wiki_deltas,
        "delta_med": delta_med,
        "deep_fraction": round(deep_fraction, 3) if deep_fraction is not None else None,
        "jaccard_vs_random_native": round(jaccard, 3) if jaccard is not None else None,
        "wiki_heads": [[lay, h] for lay, h in wiki_heads],
        "heads": heads,
        "elapsed_s": round(time.time() - t0, 1),
    }
    d_str = f"{delta_med:.3f}" if delta_med is not None else "N/A"
    df_str = f"{deep_fraction:.2f}" if deep_fraction is not None else "N/A"
    jc_str = f"{jaccard:.3f}" if jaccard is not None else "N/A"
    print(
        f"\n  Result: n_wiki={n_wiki}/{n_layers * n_heads}  "
        f"Δ_med={d_str}  deep_frac={df_str}  Jaccard={jc_str}  "
        f"({result['elapsed_s']:.0f}s)"
    )
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return result


def print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("  exp-118 SUMMARY — WikiText census, cross-family")
    print("=" * 70)
    print(f"  {'Model':<35} {'n_wiki':>6} {'Δ_med':>6} {'deep%':>6} {'Jacc':>6}")
    print("  " + "-" * 60)
    for r in results:
        model_short = r["model"].replace("EleutherAI/", "")
        d_str = f"{r['delta_med']:.3f}" if r["delta_med"] is not None else "  N/A"
        df_str = f"{100 * r['deep_fraction']:.0f}%" if r["deep_fraction"] is not None else "  N/A"
        jc_str = f"{r['jaccard_vs_random_native']:.3f}" if r["jaccard_vs_random_native"] is not None else "  N/A"
        print(f"  {model_short:<35} {r['n_wiki']:>6} {d_str:>6} {df_str:>6} {jc_str:>6}")
    print()

    # Verdict assessment
    p1_models_confirmed = [r for r in results if r["n_wiki"] >= 1]
    gpt2_med = next((r for r in results if r["model"] == "gpt2-medium"), None)
    pythia_confirmed = [r for r in results
                        if r["model"].startswith("EleutherAI/") and r["n_wiki"] >= 1]
    p1_pass = (gpt2_med is not None and gpt2_med["n_wiki"] >= 1) and len(pythia_confirmed) >= 2
    p1_kill = not p1_pass and (
        (gpt2_med is None or gpt2_med["n_wiki"] == 0)
        and len(pythia_confirmed) <= 1
    )
    print(f"  P1 (gate): {'CONFIRMED' if p1_pass else 'KILLED (K1)' if p1_kill else 'PARTIAL'}")
    if p1_pass or not p1_kill:
        for r in p1_models_confirmed:
            df = r.get("deep_fraction")
            p2_local = df is not None and df >= 0.60
            print(f"  P2 ({r['model'].replace('EleutherAI/', ''):<30}): "
                  f"deep_frac={f'{100*df:.0f}%' if df else 'N/A'} → "
                  f"{'OK' if p2_local else 'BELOW 60%'}")
    print()


def main() -> None:
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    target_models = sys.argv[1:] if len(sys.argv) > 1 else MODELS

    all_results = []
    for model_id in target_models:
        try:
            r = run_model(model_id, device)
            all_results.append(r)
        except Exception as exc:
            print(f"\n  ERROR on {model_id}: {exc}")
            all_results.append({"model": model_id, "error": str(exc)})

    print_summary([r for r in all_results if "error" not in r])

    out_path = HERE / "results.json"
    out_path.write_text(json.dumps(all_results, indent=1))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
