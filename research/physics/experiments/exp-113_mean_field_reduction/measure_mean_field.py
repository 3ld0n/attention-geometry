"""exp-113 — mean-field reduction of the positional-mean score drift.

Pre-registration: notes.md in this folder, written before this file existed.
Native conditions only (random, WikiText), inputs bit-identical to exp-107
(sha256-gated). Captures the ensemble-mean residual stream entering each
registered block, forms q_mf = LN(h-bar) W_Q + b_q (and k_mf likewise), and
compares the pooled mean-field score drift against exp-112's measured
positional-mean drift.

Gates: K1 (input sha256), K2 (sigma_pos recomputed from exp-112's saved
q-bar/k-bar matches exp-112's published raw value to 1e-6).

Ariel — August 9, 2026.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
EXP107 = HERE.parent / "exp-107_natural_text_bilocal"
EXP112 = HERE.parent / "exp-112_score_drift_decomposition"
KIT = HERE.parent.parent / "replication" / "measure_conformal_heads.py"

spec = importlib.util.spec_from_file_location("census_kit", KIT)
kit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kit)
SEQ_LEN, N_INPUTS, DEEP_LO = kit.SEQ_LEN, kit.N_INPUTS, kit.DEEP_LO
FIT_LO, FIT_HI, SEED = kit.FIT_LO, kit.FIT_HI, kit.SEED
assert (SEQ_LEN, N_INPUTS, DEEP_LO, FIT_LO, FIT_HI, SEED) == (512, 50, 256, 8, 256, 42)

spec112 = importlib.util.spec_from_file_location(
    "exp112", EXP112 / "measure_scores.py")
exp112 = importlib.util.module_from_spec(spec112)
sys.path.insert(0, str(EXP107))
spec112.loader.exec_module(exp112)          # imports m107/w107 transitively
pooled_window_profile = exp112.pooled_window_profile
ols_slope = exp112.ols_slope
WINDOW, NW = exp112.WINDOW, exp112.NW

import measure_natural_bilocal as m107      # noqa: E402
import exploratory_wikitext as w107         # noqa: E402

STRUCTURAL = exp112.STRUCTURAL
SEMANTIC = exp112.SEMANTIC
NATIVE = {"random": STRUCTURAL, "wikitext": SEMANTIC}
EXPLORATORY_PAIRS = {"random": SEMANTIC, "wikitext": STRUCTURAL}
K2_TOL = 1e-6
BAND = 0.10
DEAD_BAND = 0.25


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = AutoConfig.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2", dtype=torch.float32, attn_implementation="eager").to(device).eval()
    n_layer, n_head = cfg.num_hidden_layers, cfg.num_attention_heads
    d_head = cfg.hidden_size // n_head
    scaling = d_head ** -0.5

    # ---- inputs, bit-identical to exp-107 (K1) --------------------------------
    wt_windows, wt_meta = w107.build_wikitext_windows(tokenizer)
    rec_wt = json.loads((EXP107 / "exploratory_wikitext.json").read_text())["text_source"]
    assert wt_meta["ids_sha256"] == rec_wt["ids_sha256"], "WikiText windows drifted"

    def random_iter():
        rng = np.random.default_rng(SEED)
        torch.manual_seed(SEED)
        for _ in range(N_INPUTS):
            yield torch.from_numpy(
                rng.integers(0, cfg.vocab_size, size=(1, SEQ_LEN)).astype(np.int64))

    def wt_iter():
        for i in range(N_INPUTS):
            yield torch.from_numpy(wt_windows[i:i + 1])

    iters = {"random": random_iter, "wikitext": wt_iter}

    # ---- capture pre-ln_1 residual stream per block ---------------------------
    captured: dict[int, torch.Tensor] = {}

    def make_prehook(ell):
        def hook(_mod, args, kwargs):
            captured[ell] = args[0] if args else kwargs["hidden_states"]
        return hook

    for ell in range(n_layer):
        model.transformer.h[ell].register_forward_pre_hook(
            make_prehook(ell), with_kwargs=True)

    r112 = json.loads((EXP112 / "results_gpt2.json").read_text())
    npz112 = np.load(EXP112 / "scores_gpt2.npz")

    results = {"protocol": {"band": BAND, "dead_band": DEAD_BAND,
                            "k2_tol": K2_TOL, "n_inputs": N_INPUTS},
               "gates": {}, "conditions": {}}
    npz_out = {}

    for cond in ("random", "wikitext"):
        print(f"condition {cond.upper()}", flush=True)
        all_pairs = NATIVE[cond] + EXPLORATORY_PAIRS[cond]
        layers = sorted({l for l, _ in all_pairs})

        hbar = {ell: np.zeros((SEQ_LEN, cfg.hidden_size)) for ell in layers}
        for m_idx, ids in enumerate(iters[cond]()):
            with torch.no_grad():
                _ = model(ids.to(device))
                for ell in layers:
                    hbar[ell] += captured[ell][0].float().cpu().numpy()
            captured.clear()
            if (m_idx + 1) % 25 == 0:
                print(f"    {m_idx + 1}/{N_INPUTS} inputs", flush=True)
        for ell in layers:
            hbar[ell] /= N_INPUTS

        # ---- K2: sigma_pos recomputed from exp-112's saved qbar/kbar ---------
        # (arrays exist only for exp-112's registered layers per condition;
        #  gate runs over the pairs that have them — all NATIVE pairs do)
        k2_max = 0.0
        for (rl, rh) in all_pairs:
            if f"qbar_{cond}_L{rl}" not in npz112:
                continue
            qb = npz112[f"qbar_{cond}_L{rl}"][rh].astype(np.float64)
            kb = npz112[f"kbar_{cond}_L{rl}"][rh].astype(np.float64)
            sig_pos_raw = -ols_slope(
                pooled_window_profile((qb @ kb.T) * scaling), WINDOW)
            ref = r112["conditions"][cond]["heads_exploratory"][
                f"L{rl}H{rh}"]["sigma_pos_raw"]
            k2_max = max(k2_max, abs(sig_pos_raw - ref))
        k2_ok = k2_max <= K2_TOL
        print(f"  K2: max |sigma_pos recomputed - exp-112 published| = "
              f"{k2_max:.2e} -> {'OK' if k2_ok else 'FAIL'}", flush=True)
        results["gates"][cond] = {"k2_max_abs": k2_max}
        if not k2_ok:
            results["verdict"] = f"STOPPED: K2 failed on {cond}"
            (HERE / "results_gpt2.json").write_text(json.dumps(results, indent=1))
            raise SystemExit("K2 gate failed (fp32 saved arrays vs float64 "
                             "pipeline?); nothing is readable")

        # ---- mean-field objects ----------------------------------------------
        cond_res = {"registered": {}, "exploratory": {}}
        for (rl, rh) in all_pairs:
            block = model.transformer.h[rl]
            with torch.no_grad():
                h = torch.tensor(hbar[rl], dtype=torch.float32, device=device)
                x = block.ln_1(h)
                qkv = x @ block.attn.c_attn.weight + block.attn.c_attn.bias
                q, k, _ = qkv.split(cfg.hidden_size, dim=-1)
                q = q.view(SEQ_LEN, n_head, d_head).permute(1, 0, 2)[rh]
                k = k.view(SEQ_LEN, n_head, d_head).permute(1, 0, 2)[rh]
                q_mf = q.cpu().numpy().astype(np.float64)
                k_mf = k.cpu().numpy().astype(np.float64)
            S_mf = pooled_window_profile((q_mf @ k_mf.T) * scaling)
            sig_mf = -ols_slope(S_mf, WINDOW)

            name = f"L{rl}H{rh}"
            is_reg = (rl, rh) in NATIVE[cond]
            bucket = "registered" if is_reg else "exploratory"
            ref_rec = (r112["conditions"][cond]["registered"].get(name)
                       or r112["conditions"][cond]["heads_exploratory"][name])
            sig_pos = ref_rec.get("sigma_pos_corrected",
                                  ref_rec.get("sigma_pos_raw"))

            # diagnostic D: object-level agreement with exp-112's qbar/kbar
            rec = {"sigma_mf": sig_mf, "sigma_pos_exp112": sig_pos,
                   "abs_diff": abs(sig_mf - sig_pos)}
            if f"qbar_{cond}_L{rl}" in npz112:
                qb = npz112[f"qbar_{cond}_L{rl}"][rh].astype(np.float64)
                kb = npz112[f"kbar_{cond}_L{rl}"][rh].astype(np.float64)
                pool = slice(DEEP_LO, SEQ_LEN)
                dq = np.linalg.norm(q_mf[pool] - qb[pool], axis=-1) / \
                    np.maximum(np.linalg.norm(qb[pool], axis=-1), 1e-30)
                dk = np.linalg.norm(k_mf - kb, axis=-1) / \
                    np.maximum(np.linalg.norm(kb, axis=-1), 1e-30)
                rec["D_q_relerr_median"] = float(np.median(dq))
                rec["D_k_relerr_median"] = float(np.median(dk))
            cond_res[bucket][name] = rec
            if is_reg:
                print(f"    {name:8s} sigma_mf={sig_mf:+.3f} "
                      f"sigma_pos={sig_pos:+.3f} |diff|={rec['abs_diff']:.3f} "
                      f"D_q={rec.get('D_q_relerr_median', float('nan')):.3f} "
                      f"D_k={rec.get('D_k_relerr_median', float('nan')):.3f}",
                      flush=True)
            npz_out[f"S_mf_{cond}_{name}"] = S_mf
            npz_out[f"q_mf_{cond}_{name}"] = q_mf.astype(np.float32)
            npz_out[f"k_mf_{cond}_{name}"] = k_mf.astype(np.float32)
        for ell in layers:
            npz_out[f"hbar_{cond}_L{ell}"] = hbar[ell].astype(np.float32)
        results["conditions"][cond] = cond_res

    # ---- registered verdicts ---------------------------------------------------
    print("\n=== registered verdicts ===", flush=True)
    verdicts = {}
    for label, cond, pairs, conf_n, dead_rule in (
            ("P1", "random", STRUCTURAL, 4, lambda deny: deny >= 3),
            ("P2", "wikitext", SEMANTIC, 12, lambda within: within <= 4)):
        rows = results["conditions"][cond]["registered"]
        within = sum(rows[f"L{l}H{h}"]["abs_diff"] <= BAND for l, h in pairs)
        far = sum(rows[f"L{l}H{h}"]["abs_diff"] > DEAD_BAND for l, h in pairs)
        if label == "P1":
            v = ("CONFIRMED" if within >= conf_n else
                 "DEAD" if far >= 3 else "AMBIGUOUS")
        else:
            v = ("CONFIRMED" if within >= conf_n else
                 "DEAD" if within <= 4 else "AMBIGUOUS")
        verdicts[label] = {"within_band": within, "beyond_dead_band": far,
                           "n": len(pairs), "verdict": v}
        print(f"  {label}: {within}/{len(pairs)} within {BAND} -> {v} "
              f"(prediction on record: CONFIRMED)", flush=True)

    results["registered_verdicts"] = verdicts
    (HERE / "results_gpt2.json").write_text(json.dumps(results, indent=1))
    np.savez_compressed(HERE / "meanfield_gpt2.npz", **npz_out)
    print(f"\nwrote {HERE / 'results_gpt2.json'}", flush=True)


if __name__ == "__main__":
    main()
