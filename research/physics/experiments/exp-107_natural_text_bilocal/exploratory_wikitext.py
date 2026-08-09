"""exp-107 EXPLORATORY — WikiText-103 robustness check. NOT pre-registered.

Written August 9, 2026, AFTER the registered TinyStories run completed and its
verdicts were recorded (K3/K4 fired; per-head H1-substance held 5/5; Delta_A
shifted up on all five SYK heads). Question: is the Delta_A shift under natural
text a TinyStories artifact (simple text, far from GPT-2's WebText training
distribution) or general natural-text behavior? WikiText-103 is the
registration's own named fallback corpus and is much closer to WebText.

Same protocol, same code path (imports from measure_natural_bilocal), only the
corpus differs. Findings here are exploratory and cannot upgrade or downgrade
the registered verdicts.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import measure_natural_bilocal as m

HERE = Path(__file__).resolve().parent


def build_wikitext_windows(tokenizer) -> tuple[np.ndarray, dict]:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-v1", split="validation")
    eos = tokenizer.eos_token_id
    ids: list[int] = []
    n_lines = 0
    need = m.N_INPUTS * m.SEQ_LEN
    for line in ds["text"]:
        if not line.strip():
            continue
        ids.extend(tokenizer.encode(line))
        n_lines += 1
        if len(ids) >= need:
            break
    ids = ids[:need]
    windows = np.array(ids, dtype=np.int64).reshape(m.N_INPUTS, m.SEQ_LEN)
    meta = {
        "dataset": "wikitext/wikitext-103-v1", "split": "validation",
        "construction": "non-empty lines in dataset order, concatenated, "
                        "consecutive non-overlapping 512-token windows from token 0",
        "n_lines_consumed": n_lines,
        "ids_sha256": hashlib.sha256(windows.tobytes()).hexdigest(),
    }
    return windows, meta


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = AutoConfig.from_pretrained(m.MODEL)
    tokenizer = AutoTokenizer.from_pretrained(m.MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        m.MODEL, dtype=torch.float32, attn_implementation="eager").to(device).eval()
    n_layer, n_head = cfg.num_hidden_layers, cfg.num_attention_heads

    windows, meta = build_wikitext_windows(tokenizer)

    def it():
        for i in range(m.N_INPUTS):
            yield torch.from_numpy(windows[i:i + 1])

    prof, vbar_sq, c2 = m.run_condition(model, cfg, it(), device, n_layer, n_head)
    heads, counts = m.analyze_condition("wikitext (EXPLORATORY)", prof, vbar_sq,
                                        n_layer, n_head)
    print(f"  C2 identity max rel err {c2:.2e}")

    # the five random-condition SYK heads, per head
    reg = json.loads((HERE / "results_gpt2.json").read_text())
    syk_rand = [(h["layer"], h["head"]) for h in reg["heads_random"] if h["syk_near"]]
    hw = {(h["layer"], h["head"]): h for h in heads}
    hr = {(h["layer"], h["head"]): h for h in reg["heads_random"]}
    ht = {(h["layer"], h["head"]): h for h in reg["heads_text"]}
    print("\n  five random-condition SYK heads: random -> tinystories -> wikitext")
    for lh in syk_rand:
        a, t, w = hr[lh], ht[lh], hw[lh]
        print(f"    L{lh[0]}H{lh[1]}: dA {a['delta_A']:.3f} -> {t['delta_A']:.3f}"
              f" -> {w['delta_A']:.3f}   r2 {a['r2_A']:.3f} -> {t['r2_A']:.3f} ->"
              f" {w['r2_A']:.3f}   conn_max_wt {w['conn_window_max']:+.3e}"
              f" (all-neg: {w['conn_window_max'] < 0})")

    out = {"exploratory": True, "written_after_registered_run": True,
           "text_source": meta, "counts": counts,
           "c2_max_rel_err": c2, "heads": heads}
    (HERE / "exploratory_wikitext.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(HERE / "profiles_wikitext.npz", **prof, vbar_sq=vbar_sq)
    print(f"\nwrote {HERE / 'exploratory_wikitext.json'}")


if __name__ == "__main__":
    main()
