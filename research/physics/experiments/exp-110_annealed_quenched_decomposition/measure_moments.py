"""exp-110 — annealed vs quenched decomposition of the census exponent.

Pre-registration: notes.md in this folder, written before this file existed.
Inputs are bit-identical to exp-107's three conditions (sha256-gated). Per
(layer, head, dx) this run accumulates the first four raw moments of log A
over the census pool, alongside the standard P_A profile computed through the
replication kit's own lag_profile (K1 gate: must match exp-107's saved
profiles to <= 1e-10).

Ariel — August 9, 2026.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
EXP107 = HERE.parent / "exp-107_natural_text_bilocal"
KIT = HERE.parent.parent / "replication" / "measure_conformal_heads.py"

spec = importlib.util.spec_from_file_location("census_kit", KIT)
kit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kit)
SEQ_LEN, N_INPUTS, DEEP_LO = kit.SEQ_LEN, kit.N_INPUTS, kit.DEEP_LO
FIT_LO, FIT_HI, SEED = kit.FIT_LO, kit.FIT_HI, kit.SEED
assert (SEQ_LEN, N_INPUTS, DEEP_LO, FIT_LO, FIT_HI, SEED) == (512, 50, 256, 8, 256, 42)

sys.path.insert(0, str(EXP107))
import measure_natural_bilocal as m107          # noqa: E402
import exploratory_wikitext as w107             # noqa: E402

REGISTERED_HEADS = [(2, 1), (3, 4), (5, 0), (7, 11), (10, 8)]
CONDITIONS = ("random", "tinystories", "wikitext")
DIAG_LAGS = (8, 32, 128, 256)
LOG_FLOOR = 1e-45  # clamp; exact zeros counted separately (K2)


def log_moments_accumulate(A: np.ndarray, acc: dict, ell: int) -> None:
    """A: (n_head, L, L) fp32. Accumulate pooled moments of log A per (head, dx)."""
    logA_zeros = (A <= 0)
    if logA_zeros.any():
        # count zeros inside the pooled region only
        for dx in range(SEQ_LEN):
            diag = np.diagonal(logA_zeros, offset=-dx, axis1=-2, axis2=-1)
            k_lo = max(DEEP_LO, dx) - dx
            if k_lo < diag.shape[-1]:
                acc["n_zero"][ell, :, dx] += diag[:, k_lo:].sum(axis=-1)
    lg = np.log(np.clip(A, LOG_FLOOR, None)).astype(np.float64)
    for dx in range(SEQ_LEN):
        diag = np.diagonal(lg, offset=-dx, axis1=-2, axis2=-1)  # (n_head, L-dx)
        k_lo = max(DEEP_LO, dx) - dx
        if k_lo >= diag.shape[-1]:
            continue
        d = diag[:, k_lo:]
        acc["n"][ell, :, dx] += d.shape[-1]
        acc["s1"][ell, :, dx] += d.sum(axis=-1)
        acc["s2"][ell, :, dx] += (d ** 2).sum(axis=-1)
        acc["s3"][ell, :, dx] += (d ** 3).sum(axis=-1)
        acc["s4"][ell, :, dx] += (d ** 4).sum(axis=-1)


def run_condition(model, cfg, inputs_iter, device, n_layer, n_head):
    profA = np.zeros((n_layer, n_head, SEQ_LEN))
    acc = {k: np.zeros((n_layer, n_head, SEQ_LEN)) for k in
           ("n", "s1", "s2", "s3", "s4", "n_zero")}
    n_done = 0
    for ids in inputs_iter:
        with torch.no_grad():
            out = model(ids.to(device), output_attentions=True)
        for ell in range(n_layer):
            A = out.attentions[ell][0].float().cpu().numpy()
            profA[ell] += kit.lag_profile(A)
            log_moments_accumulate(A, acc, ell)
        del out
        n_done += 1
        if n_done % 10 == 0:
            print(f"    {n_done}/{N_INPUTS} inputs", flush=True)
    profA /= n_done
    return profA, acc


def ols_slope(y: np.ndarray, lags: np.ndarray):
    lx = np.log(lags.astype(float))
    X = np.column_stack([np.ones_like(lx), lx])
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ c
    ss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - float(np.sum(resid ** 2)) / ss if ss > 1e-30 else 0.0
    return float(c[1]), r2


def analyze(profA, acc, ell, h):
    lags = np.arange(FIT_LO, FIT_HI + 1)
    n = acc["n"][ell, h, FIT_LO:FIT_HI + 1]
    s1 = acc["s1"][ell, h, FIT_LO:FIT_HI + 1]
    s2 = acc["s2"][ell, h, FIT_LO:FIT_HI + 1]
    s3 = acc["s3"][ell, h, FIT_LO:FIT_HI + 1]
    s4 = acc["s4"][ell, h, FIT_LO:FIT_HI + 1]
    mean = s1 / n
    var = s2 / n - mean ** 2
    mu3 = s3 / n - 3 * mean * s2 / n + 2 * mean ** 3
    mu4 = s4 / n - 4 * mean * s3 / n + 6 * mean ** 2 * s2 / n - 3 * mean ** 4
    skew = mu3 / np.maximum(var, 1e-30) ** 1.5
    kurt = mu4 / np.maximum(var, 1e-30) ** 2 - 3.0

    pA = profA[ell, h, FIT_LO:FIT_HI + 1]
    sl_ann, r2_ann = ols_slope(np.log(pA), lags)
    sl_q, r2_q = ols_slope(mean, lags)
    sl_pred, r2_pred = ols_slope(mean + var / 2, lags)

    two_delta_meas = -sl_ann
    s_quenched = -sl_q
    two_delta_pred = -sl_pred
    diag_idx = [int(dx - FIT_LO) for dx in DIAG_LAGS]
    return {
        "two_delta_measured": two_delta_meas,
        "s_quenched": s_quenched,
        "two_delta_pred_gauss": two_delta_pred,
        "delta_gap": two_delta_meas - two_delta_pred,
        "r2_annealed_fit": r2_ann, "r2_quenched_fit": r2_q,
        "r2_pred_fit": r2_pred,
        "var_slope_over2": s_quenched - two_delta_pred,  # −d(κ2/2)/dlogdx contribution
        "diagnostics": {str(dx): {"skew": float(skew[i]), "excess_kurtosis": float(kurt[i]),
                                  "var": float(var[i]), "mean": float(mean[i])}
                        for dx, i in zip(DIAG_LAGS, diag_idx)},
        "pooled_zeros_in_window": float(acc["n_zero"][ell, h, FIT_LO:FIT_HI + 1].sum()),
        "pooled_count_in_window": float(n.sum()),
    }


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = AutoConfig.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2", dtype=torch.float32, attn_implementation="eager").to(device).eval()
    n_layer, n_head = cfg.num_hidden_layers, cfg.num_attention_heads

    # ---- inputs, bit-identical to exp-107 (sha256-gated) ---------------------
    ts_windows, ts_meta = m107.build_text_windows(tokenizer)
    rec_ts = json.loads((EXP107 / "applied_text.json").read_text())
    assert ts_meta["ids_sha256"] == rec_ts["ids_sha256"], "TinyStories windows drifted"
    wt_windows, wt_meta = w107.build_wikitext_windows(tokenizer)
    rec_wt = json.loads((EXP107 / "exploratory_wikitext.json").read_text())["text_source"]
    assert wt_meta["ids_sha256"] == rec_wt["ids_sha256"], "WikiText windows drifted"

    def random_iter():
        rng = np.random.default_rng(SEED)
        torch.manual_seed(SEED)
        for _ in range(N_INPUTS):
            yield torch.from_numpy(
                rng.integers(0, cfg.vocab_size, size=(1, SEQ_LEN)).astype(np.int64))

    def win_iter(w):
        def it():
            for i in range(N_INPUTS):
                yield torch.from_numpy(w[i:i + 1])
        return it

    saved107 = np.load(EXP107 / "profiles_gpt2.npz")
    saved_wt = np.load(EXP107 / "profiles_wikitext.npz")
    ref_profiles = {"random": saved107["A_random"], "tinystories": saved107["A_text"],
                    "wikitext": saved_wt["A"]}
    iters = {"random": random_iter, "tinystories": win_iter(ts_windows),
             "wikitext": win_iter(wt_windows)}

    results = {"protocol": {"seq_len": SEQ_LEN, "n_inputs": N_INPUTS,
                            "deep_lo": DEEP_LO, "fit_lags": [FIT_LO, FIT_HI],
                            "seed": SEED, "registered_heads": REGISTERED_HEADS},
               "conditions": {}}
    moments_out = {}

    for cond in CONDITIONS:
        print(f"condition {cond.upper()}")
        profA, acc = run_condition(model, cfg, iters[cond](), device, n_layer, n_head)

        k1_drift = float(np.max(np.abs(profA - ref_profiles[cond])))
        k1_ok = k1_drift <= 1e-10
        print(f"  K1: max |P_A - exp-107 saved| = {k1_drift:.3e} -> "
              f"{'OK' if k1_ok else 'FAIL'}")
        if not k1_ok:
            results["verdict"] = f"STOPPED: K1 failed on {cond}"
            (HERE / "results_gpt2.json").write_text(json.dumps(results, indent=1))
            raise SystemExit("K1 pipeline gate failed; nothing is readable")

        cond_res = {"k1_max_abs_drift": k1_drift, "heads": {}}
        for ell in range(n_layer):
            for h in range(n_head):
                rec = analyze(profA, acc, ell, h)
                cond_res["heads"][f"L{ell}H{h}"] = rec
        results["conditions"][cond] = cond_res
        for k, v in acc.items():
            moments_out[f"{k}_{cond}"] = v
        moments_out[f"profA_{cond}"] = profA

    # ---- registered verdicts --------------------------------------------------
    pairs = []
    for (ell, h) in REGISTERED_HEADS:
        for cond in CONDITIONS:
            rec = results["conditions"][cond]["heads"][f"L{ell}H{h}"]
            zero_frac = rec["pooled_zeros_in_window"] / rec["pooled_count_in_window"]
            pairs.append({"head": f"L{ell}H{h}", "cond": cond,
                          "two_delta_measured": rec["two_delta_measured"],
                          "s_quenched": rec["s_quenched"],
                          "two_delta_pred_gauss": rec["two_delta_pred_gauss"],
                          "delta_gap": rec["delta_gap"],
                          "zero_frac": zero_frac,
                          "k2_contaminated": zero_frac > 0.01})

    clean = [p for p in pairs if not p["k2_contaminated"]]
    gaps = np.array([abs(p["delta_gap"]) for p in clean])
    med_gap = float(np.median(gaps))
    p1 = ("CONFIRMED" if med_gap <= 0.10 else
          "DEAD" if med_gap > 0.20 else "AMBIGUOUS")

    p2_votes = []
    for (ell, h) in REGISTERED_HEADS:
        hp = [p for p in clean if p["head"] == f"L{ell}H{h}"]
        if len(hp) < 3:
            continue
        rng_q = max(p["s_quenched"] for p in hp) - min(p["s_quenched"] for p in hp)
        rng_m = (max(p["two_delta_measured"] for p in hp)
                 - min(p["two_delta_measured"] for p in hp))
        p2_votes.append({"head": f"L{ell}H{h}", "range_quenched": rng_q,
                         "range_measured": rng_m, "quenched_stabler": rng_q < rng_m})
    n_stable = sum(v["quenched_stabler"] for v in p2_votes)
    p2 = ("CONFIRMED" if n_stable >= 4 else
          "DEAD" if (len(p2_votes) - n_stable) >= 3 else "AMBIGUOUS")

    print("\n=== registered verdicts ===")
    print(f"  pairs: {len(pairs)} ({len(pairs) - len(clean)} K2-contaminated, dropped)")
    for p in pairs:
        flag = " [K2-DROPPED]" if p["k2_contaminated"] else ""
        print(f"    {p['head']:8s} {p['cond']:12s} 2D_meas={p['two_delta_measured']:+.3f} "
              f"s_q={p['s_quenched']:+.3f} 2D_pred={p['two_delta_pred_gauss']:+.3f} "
              f"gap={p['delta_gap']:+.4f}{flag}")
    print(f"  P1 (Gaussian truncation): median |gap| = {med_gap:.4f} -> {p1} "
          f"(prediction on record: confirmed at ~0.05)")
    for v in p2_votes:
        print(f"    {v['head']:8s} range(s_q)={v['range_quenched']:.3f} "
              f"range(2D_meas)={v['range_measured']:.3f} "
              f"quenched stabler: {v['quenched_stabler']}")
    print(f"  P2 (quenched slope more input-stable): {n_stable}/{len(p2_votes)} -> {p2} "
          f"(prediction on record: confirmed)")

    results["registered"] = {"pairs": pairs, "P1": {"median_abs_gap": med_gap,
                                                    "verdict": p1},
                             "P2": {"votes": p2_votes, "verdict": p2}}
    (HERE / "results_gpt2.json").write_text(json.dumps(results, indent=1))
    np.savez_compressed(HERE / "moments_gpt2.npz", **moments_out)
    print(f"\nwrote {HERE / 'results_gpt2.json'}")


if __name__ == "__main__":
    main()
