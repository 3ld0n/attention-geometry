"""exp-112 — score-level carrier of the quenched drift: positional mean vs
token covariance.

Pre-registration: notes.md in this folder, written before this file existed.
Inputs bit-identical to exp-107's three conditions (sha256-gated). Captures
q, k from c_attn per layer, computes pre-softmax scores s = q.k^T * scaling
exactly as the model does, and decomposes the pooled ensemble-mean score
profile into positional-mean and token-covariance components.

Gates: K1a (per-entry softmax reconstruction vs output_attentions <= 1e-5),
K1b (kit lag-profile of output_attentions vs exp-107 saved <= 1e-10),
K2 (pooled mean-score slope == exp-110 quenched slope to 5e-3, exact identity).

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
EXP110 = HERE.parent / "exp-110_annealed_quenched_decomposition"
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

STRUCTURAL = [(2, 1), (3, 4), (5, 0), (7, 11), (10, 8)]
SEMANTIC = [(4, 10), (7, 1), (8, 2), (9, 4), (9, 6), (10, 1), (10, 2), (10, 10),
            (11, 0), (11, 1), (11, 2), (11, 4), (11, 5), (11, 6), (11, 7), (11, 9)]
CONDITIONS = ("random", "tinystories", "wikitext")
REGISTERED = {"random": STRUCTURAL, "tinystories": STRUCTURAL,
              "wikitext": STRUCTURAL + SEMANTIC}
WINDOW = np.arange(FIT_LO, FIT_HI + 1)          # dx in [8, 256]
NW = len(WINDOW)
DIAG_LAGS = (8, 32, 128, 256)

K1A_TOL = 1e-5
K1B_TOL = 1e-10
K2_TOL = 5e-3


def ols_slope(y: np.ndarray, lags: np.ndarray) -> float:
    lx = np.log(lags.astype(float))
    X = np.column_stack([np.ones_like(lx), lx])
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(c[1])


def pooled_window_profile(mat: np.ndarray) -> np.ndarray:
    """mat: (..., L, L). Mean over pool i >= 256 at each window dx (identical
    query set i in [256, 511] for every dx <= 256)."""
    out = np.empty(mat.shape[:-2] + (NW,))
    for w, dx in enumerate(WINDOW):
        diag = np.diagonal(mat, offset=-dx, axis1=-2, axis2=-1)
        k_lo = max(DEEP_LO, dx) - dx
        out[..., w] = diag[..., k_lo:].mean(axis=-1)
    return out


def accumulate_scores(s: np.ndarray, acc: dict, ell: int) -> None:
    """s: (n_head, L, L) fp32 scores. Pooled sums per (head, dx), full range."""
    s64 = s.astype(np.float64)
    for dx in range(SEQ_LEN):
        diag = np.diagonal(s64, offset=-dx, axis1=-2, axis2=-1)
        k_lo = max(DEEP_LO, dx) - dx
        if k_lo >= diag.shape[-1]:
            continue
        d = diag[:, k_lo:]
        acc["n"][ell, :, dx] += d.shape[-1]
        acc["s1"][ell, :, dx] += d.sum(axis=-1)
        acc["s2"][ell, :, dx] += (d ** 2).sum(axis=-1)


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = AutoConfig.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2", dtype=torch.float32, attn_implementation="eager").to(device).eval()
    n_layer, n_head = cfg.num_hidden_layers, cfg.num_attention_heads
    d_head = cfg.hidden_size // n_head
    scaling = d_head ** -0.5
    assert not cfg.scale_attn_by_inverse_layer_idx
    tril = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool, device=device))

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

    # exp-110 quenched slopes for K2
    r110 = json.loads((EXP110 / "results_gpt2.json").read_text())

    # ---- hooks: capture c_attn output per layer ------------------------------
    captured: dict[int, torch.Tensor] = {}

    def make_hook(ell):
        def hook(_mod, _inp, out):
            captured[ell] = out
        return hook

    for ell in range(n_layer):
        model.transformer.h[ell].attn.c_attn.register_forward_hook(make_hook(ell))

    results = {"protocol": {"seq_len": SEQ_LEN, "n_inputs": N_INPUTS,
                            "deep_lo": DEEP_LO, "fit_lags": [FIT_LO, FIT_HI],
                            "seed": SEED, "scaling": scaling,
                            "structural": [f"L{l}H{h}" for l, h in STRUCTURAL],
                            "semantic": [f"L{l}H{h}" for l, h in SEMANTIC]},
               "gates": {}, "conditions": {}}
    npz_out = {}

    for cond in CONDITIONS:
        print(f"condition {cond.upper()}", flush=True)
        reg_pairs = REGISTERED[cond]
        reg_layers = sorted({l for l, _ in reg_pairs})

        acc = {k: np.zeros((n_layer, n_head, SEQ_LEN)) for k in ("n", "s1", "s2")}
        sum_q = np.zeros((n_layer, n_head, SEQ_LEN, d_head), dtype=np.float64)
        sum_k = np.zeros((n_layer, n_head, SEQ_LEN, d_head), dtype=np.float64)
        profA = np.zeros((n_layer, n_head, SEQ_LEN))
        # per-input storage for jackknife (registered pairs only)
        prof_m = np.zeros((len(reg_pairs), N_INPUTS, NW))
        q_store = np.zeros((len(reg_pairs), N_INPUTS, SEQ_LEN, d_head), dtype=np.float32)
        k_store = np.zeros((len(reg_pairs), N_INPUTS, SEQ_LEN, d_head), dtype=np.float32)
        k1a_max = 0.0

        for m_idx, ids in enumerate(iters[cond]()):
            with torch.no_grad():
                out = model(ids.to(device), output_attentions=True)
                for ell in range(n_layer):
                    qkv = captured[ell][0]                       # (L, 3*768)
                    q, k, _v = qkv.split(cfg.hidden_size, dim=-1)
                    q = q.view(SEQ_LEN, n_head, d_head).permute(1, 0, 2)
                    k = k.view(SEQ_LEN, n_head, d_head).permute(1, 0, 2)
                    s_t = torch.matmul(q, k.transpose(-1, -2)) * scaling
                    # K1a: reconstruct attention from these scores
                    rec = torch.softmax(
                        s_t.masked_fill(~tril, float("-inf")), dim=-1)
                    diff = float((rec - out.attentions[ell][0]).abs().max())
                    k1a_max = max(k1a_max, diff)

                    s_np = s_t.cpu().numpy()
                    A_np = out.attentions[ell][0].float().cpu().numpy()
                    profA[ell] += kit.lag_profile(A_np)
                    accumulate_scores(s_np, acc, ell)
                    if ell in reg_layers:
                        q_np = q.cpu().numpy()
                        k_np = k.cpu().numpy()
                        sum_q[ell] += q_np
                        sum_k[ell] += k_np
                        for p_idx, (rl, rh) in enumerate(reg_pairs):
                            if rl != ell:
                                continue
                            prof_m[p_idx, m_idx] = pooled_window_profile(
                                s_np[rh].astype(np.float64))
                            q_store[p_idx, m_idx] = q_np[rh]
                            k_store[p_idx, m_idx] = k_np[rh]
                    else:
                        sum_q[ell] += q.cpu().numpy()
                        sum_k[ell] += k.cpu().numpy()
                del out
            captured.clear()
            if (m_idx + 1) % 10 == 0:
                print(f"    {m_idx + 1}/{N_INPUTS} inputs", flush=True)
        profA /= N_INPUTS

        # ---- gates ----------------------------------------------------------
        k1b_drift = float(np.max(np.abs(profA - ref_profiles[cond])))
        k1a_ok, k1b_ok = k1a_max <= K1A_TOL, k1b_drift <= K1B_TOL
        print(f"  K1a: max |A_rec - A_model| = {k1a_max:.3e} -> "
              f"{'OK' if k1a_ok else 'FAIL'}", flush=True)
        print(f"  K1b: max |P_A - exp-107 saved| = {k1b_drift:.3e} -> "
              f"{'OK' if k1b_ok else 'FAIL'}", flush=True)
        results["gates"][cond] = {"k1a_max_abs": k1a_max, "k1b_max_abs": k1b_drift}
        if not (k1a_ok and k1b_ok):
            results["verdict"] = f"STOPPED: K1 failed on {cond}"
            (HERE / "results_gpt2.json").write_text(json.dumps(results, indent=1))
            raise SystemExit("K1 gate failed; nothing is readable")

        # ---- per-head profiles and decomposition (all 144, exploratory) ------
        qbar = sum_q / N_INPUTS
        kbar = sum_k / N_INPUTS
        wsl = slice(FIT_LO, FIT_HI + 1)
        S_full = (acc["s1"] / acc["n"])[:, :, wsl]                    # (12,12,NW)
        v_s = (acc["s2"] / acc["n"])[:, :, wsl] - S_full ** 2
        S_pos = np.zeros_like(S_full)
        for ell in range(n_layer):
            for h in range(n_head):
                M = (qbar[ell, h] @ kbar[ell, h].T) * scaling
                S_pos[ell, h] = pooled_window_profile(M)

        cond_res = {"heads_exploratory": {}, "registered": {}}
        corr = N_INPUTS / (N_INPUTS - 1)
        for ell in range(n_layer):
            for h in range(n_head):
                sig_full = -ols_slope(S_full[ell, h], WINDOW)
                sig_pos_raw = -ols_slope(S_pos[ell, h], WINDOW)
                sig_cov_raw = sig_full - sig_pos_raw
                sig_cov = sig_cov_raw * corr
                cond_res["heads_exploratory"][f"L{ell}H{h}"] = {
                    "sigma_full": sig_full, "sigma_pos_raw": sig_pos_raw,
                    "sigma_cov_raw": sig_cov_raw, "sigma_cov_corrected": sig_cov,
                    "sigma_pos_corrected": sig_full - sig_cov,
                    "vs_slope": ols_slope(v_s[ell, h], WINDOW),
                    "vs_at_lags": {str(dx): float(v_s[ell, h, int(dx - FIT_LO)])
                                   for dx in DIAG_LAGS}}

        # ---- registered pairs: K2 + jackknife --------------------------------
        for p_idx, (rl, rh) in enumerate(reg_pairs):
            name = f"L{rl}H{rh}"
            rec = cond_res["heads_exploratory"][name]
            s_q_110 = r110["conditions"][cond]["heads"][name]["s_quenched"]
            k2_diff = abs(rec["sigma_full"] - s_q_110)
            k2_ok = k2_diff <= K2_TOL

            # jackknife over inputs
            jk = {"full": [], "pos": [], "cov": []}
            tot_prof = prof_m[p_idx].sum(axis=0)
            tot_q = q_store[p_idx].astype(np.float64).sum(axis=0)
            tot_k = k_store[p_idx].astype(np.float64).sum(axis=0)
            n1 = N_INPUTS - 1
            corr_jk = n1 / (n1 - 1)
            for m in range(N_INPUTS):
                Sf = (tot_prof - prof_m[p_idx, m]) / n1
                qb = (tot_q - q_store[p_idx, m]) / n1
                kb = (tot_k - k_store[p_idx, m]) / n1
                Sp = pooled_window_profile((qb @ kb.T) * scaling)
                sf = -ols_slope(Sf, WINDOW)
                sp = -ols_slope(Sp, WINDOW)
                sc = (sf - sp) * corr_jk
                jk["full"].append(sf); jk["pos"].append(sp); jk["cov"].append(sc)
            se = {key: float(np.sqrt((N_INPUTS - 1) / N_INPUTS *
                                     np.sum((np.array(v) - np.mean(v)) ** 2)))
                  for key, v in jk.items()}

            cond_res["registered"][name] = {
                "sigma_full": rec["sigma_full"],
                "sigma_pos_corrected": rec["sigma_pos_corrected"],
                "sigma_cov_corrected": rec["sigma_cov_corrected"],
                "s_quenched_exp110": s_q_110,
                "k2_abs_diff": k2_diff, "k2_ok": k2_ok,
                "jackknife_se": se,
                "vs_slope": rec["vs_slope"], "vs_at_lags": rec["vs_at_lags"],
                "population": ("structural" if (rl, rh) in STRUCTURAL
                               else "semantic")}
            if not k2_ok:
                results["verdict"] = f"STOPPED: K2 failed on {name}/{cond} ({k2_diff:.4f})"
                results["conditions"][cond] = cond_res
                (HERE / "results_gpt2.json").write_text(json.dumps(results, indent=1))
                raise SystemExit("K2 identity gate failed; pooling drifted")

        results["conditions"][cond] = cond_res
        npz_out[f"S_full_{cond}"] = S_full
        npz_out[f"S_pos_{cond}"] = S_pos
        npz_out[f"v_s_{cond}"] = v_s
        for ell in reg_layers:
            npz_out[f"qbar_{cond}_L{ell}"] = qbar[ell].astype(np.float32)
            npz_out[f"kbar_{cond}_L{ell}"] = kbar[ell].astype(np.float32)

    # ---- registered verdicts ---------------------------------------------------
    print("\n=== registered verdicts ===", flush=True)

    def stable_verdict(rows, confirmed_rule, dead_rule):
        """Ambiguous rows counted both ways; verdict only if stable."""
        def resolve(assume_confirm):
            statuses = [("confirm" if assume_confirm else "deny")
                        if r["status"] == "ambiguous" else r["status"]
                        for r in rows]
            if confirmed_rule(statuses):
                return "CONFIRMED"
            if dead_rule(statuses):
                return "DEAD"
            return "AMBIGUOUS"
        v_hi, v_lo = resolve(True), resolve(False)
        return v_hi if v_hi == v_lo else "AMBIGUOUS"

    # P1: structural under random — positional mean carries the drift
    p1_rows = []
    for (rl, rh) in STRUCTURAL:
        r = results["conditions"]["random"]["registered"][f"L{rl}H{rh}"]
        sc, se = r["sigma_cov_corrected"], r["jackknife_se"]["cov"]
        status = ("ambiguous" if abs(abs(sc) - 0.10) <= se else
                  "confirm" if abs(sc) <= 0.10 else
                  "deny" if abs(sc) > 0.25 else "between")
        p1_rows.append({"head": f"L{rl}H{rh}", "sigma_full": r["sigma_full"],
                        "sigma_pos": r["sigma_pos_corrected"], "sigma_cov": sc,
                        "se_cov": se, "status": status})
        print(f"  P1 {p1_rows[-1]['head']:8s} full={r['sigma_full']:+.3f} "
              f"pos={r['sigma_pos_corrected']:+.3f} cov={sc:+.3f} "
              f"(SE {se:.3f}) -> {status}", flush=True)
    p1 = stable_verdict(
        p1_rows,
        confirmed_rule=lambda st: st.count("confirm") >= 4,
        dead_rule=lambda st: st.count("deny") >= 3)
    n_conf = sum(r["status"] == "confirm" for r in p1_rows)
    n_deny = sum(r["status"] == "deny" for r in p1_rows)
    print(f"  P1 -> {p1} ({n_conf}/5 confirm, {n_deny}/5 deny; "
          f"prediction on record: CONFIRMED)", flush=True)

    # P2: semantic under wikitext — covariance component load-bearing
    p2_rows = []
    n_guard = 0
    for (rl, rh) in SEMANTIC:
        r = results["conditions"]["wikitext"]["registered"][f"L{rl}H{rh}"]
        sf, sc = r["sigma_full"], r["sigma_cov_corrected"]
        se_c = r["jackknife_se"]["cov"]
        if sf <= 0.2:
            n_guard += 1
            status = "guard_fired"
        else:
            share = sc / sf
            se_share = se_c / abs(sf)          # first-order; sf SE is small
            status = ("ambiguous" if abs(share - 0.5) <= se_share else
                      "confirm" if share >= 0.5 else "deny")
        p2_rows.append({"head": f"L{rl}H{rh}", "sigma_full": sf,
                        "sigma_pos": r["sigma_pos_corrected"], "sigma_cov": sc,
                        "se_cov": se_c, "status": status})
        print(f"  P2 {p2_rows[-1]['head']:8s} full={sf:+.3f} "
              f"pos={r['sigma_pos_corrected']:+.3f} cov={sc:+.3f} "
              f"(SE {se_c:.3f}) -> {status}", flush=True)
    n_conf2 = sum(r["status"] == "confirm" for r in p2_rows)
    n_deny2 = sum(r["status"] == "deny" for r in p2_rows)
    if n_guard > 4:
        p2 = "AMBIGUOUS (guard fired on >4 heads)"
    else:
        p2 = stable_verdict(
            [r for r in p2_rows if r["status"] != "guard_fired"],
            confirmed_rule=lambda st: st.count("confirm") >= 12,
            dead_rule=lambda st: st.count("confirm") <= 4)
    print(f"  P2 -> {p2} ({n_conf2}/16 confirm, {n_deny2}/16 deny, "
          f"{n_guard} guard; prediction on record: CONFIRMED)", flush=True)

    results["registered_verdicts"] = {
        "P1": {"rows": p1_rows, "verdict": p1},
        "P2": {"rows": p2_rows, "verdict": p2}}
    (HERE / "results_gpt2.json").write_text(json.dumps(results, indent=1))
    np.savez_compressed(HERE / "scores_gpt2.npz", **npz_out)
    print(f"\nwrote {HERE / 'results_gpt2.json'}", flush=True)


if __name__ == "__main__":
    main()
