"""exp-137 — The subspace gap: where the positional field is read.

Pre-registration: prereg.md in this folder (= notes/2026-09-08_subspace_gap_map.md §5),
committed to attention-geometry at 3f5152e before this file existed.

One field x_i = ln_1(h_i) at each attention block input; three read maps R_Q, R_K,
R_V. Position-mean field x̄_i over the frozen census inputs; positional field
δ_i = x̄_i − mean_i x̄_i (exp-064's object). Measures, per registered head:
  κ̃(R)      positional capture, isotropic-normalized (random R → 1)
  Ĉ_R(dx)    cosine lag profile of R δ_i for R ∈ {Q, K, V, P2, P8, I, rand}
  S^(k)(dx)  exp-112's carrier recomputed with δ truncated to top-k PCs
Gates: K1 (S_pos recomputed from x̄ matches exp-112's saved profile ≤ 1e-3),
K2 (its slope matches exp-112's sigma_pos_raw ≤ 5e-3).

Ariel — September 8, 2026, ~1:30 AM.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EXP112 = HERE.parent / "exp-112_score_drift_decomposition"
EXP107 = HERE.parent / "exp-107_natural_text_bilocal"
sys.path.insert(0, str(EXP107))
spec = importlib.util.spec_from_file_location("exp112", EXP112 / "measure_scores.py")
exp112 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp112)

pooled_window_profile = exp112.pooled_window_profile
ols_slope = exp112.ols_slope
WINDOW = exp112.WINDOW
SEQ_LEN, N_INPUTS, SEED = exp112.SEQ_LEN, exp112.N_INPUTS, exp112.SEED
m107, w107 = exp112.m107, exp112.w107

PREREG_COMMIT = "3f5152e"
STRUCTURAL = [(2, 1), (3, 4), (5, 0), (7, 11), (10, 8)]
SEMANTIC = [(4, 10), (7, 1), (8, 2), (9, 4), (9, 6), (10, 1), (10, 2), (10, 10),
            (11, 0), (11, 1), (11, 2), (11, 4), (11, 5), (11, 6), (11, 7), (11, 9)]
CONTROL = [(0, 10), (0, 11), (2, 0), (4, 2), (5, 5), (6, 4), (7, 0), (7, 6),
           (7, 9), (8, 1), (8, 7), (8, 8), (10, 3), (10, 5), (10, 7), (11, 10)]
KS = (1, 2, 4, 8, 16, 64, 768)
N_RAND = 20
K1_TOL, K2_TOL = 1e-3, 5e-3
LX = np.log(WINDOW.astype(float))


# ----------------------------------------------------------------- fits ----
def fit_lin(y: np.ndarray, x: np.ndarray) -> tuple[float, float, float]:
    """OLS y = a + b x. Returns (a, b, R²)."""
    X = np.column_stack([np.ones_like(x), x])
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ c
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float(((y - pred) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-14 else 0.0
    return float(c[0]), float(c[1]), float(r2)


def profile_stats(prof: np.ndarray) -> dict:
    """Both registered forms on a window profile."""
    out = {"values_at": {str(dx): float(prof[list(WINDOW).index(dx)]) for dx in (8, 32, 128, 256)},
           "n_increases": int((np.diff(prof) > 0).sum())}
    a, b, r2 = fit_lin(prof, LX)
    out["loglin"] = {"a": a, "b": -b, "R2": r2}            # b reported as decay coefficient
    if np.all(prof > 0):
        _, s, r2l = fit_lin(np.log(prof), LX)
        out["loglog"] = {"sigma": -s, "R2": r2l}
    else:
        out["loglog"] = {"sigma": None, "R2": None, "note": "non-positive values in window"}
    r2s = [out["loglin"]["R2"]] + ([out["loglog"]["R2"]] if out["loglog"]["R2"] is not None else [])
    out["best_R2"] = float(max(r2s))
    out["best_form"] = "loglog" if (out["loglog"]["R2"] is not None and out["loglog"]["R2"] >= out["loglin"]["R2"]) else "loglin"
    out["best_slope"] = out["loglog"]["sigma"] if out["best_form"] == "loglog" else out["loglin"]["b"]
    return out


def cosine_profile(Y: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(Y, axis=-1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    M = Y / n
    return pooled_window_profile(M @ M.T)


def raw_profile(Y: np.ndarray) -> np.ndarray:
    return pooled_window_profile(Y @ Y.T)


def kappa(R: np.ndarray, delta: np.ndarray) -> float:
    """R: (768, r) map applied as delta @ R. Isotropic-normalized positional capture."""
    d = delta.shape[1]
    cap = float((np.linalg.norm(delta @ R, axis=1) ** 2).sum() / (np.linalg.norm(delta, axis=1) ** 2).sum())
    return cap / (float((R ** 2).sum()) / d)


# ------------------------------------------------------------- forward -----
def mean_fields(model, cfg, device, it):
    """x̄^(ℓ) = mean over inputs of ln_1 output at every layer; also mean c_attn out."""
    n_layer, d = cfg.n_layer, cfg.n_embd
    xbar = np.zeros((n_layer, SEQ_LEN, d))
    qkvbar = np.zeros((n_layer, SEQ_LEN, 3 * d))
    cap_ln, cap_qkv = {}, {}

    def mk(store, ell):
        def hook(_m, _i, out):
            store[ell] = out
        return hook

    hs = []
    for ell in range(n_layer):
        hs.append(model.transformer.h[ell].ln_1.register_forward_hook(mk(cap_ln, ell)))
        hs.append(model.transformer.h[ell].attn.c_attn.register_forward_hook(mk(cap_qkv, ell)))
    n = 0
    for ids in it():
        with torch.no_grad():
            model(ids.to(device))
        for ell in range(n_layer):
            xbar[ell] += cap_ln[ell][0].cpu().double().numpy()
            qkvbar[ell] += cap_qkv[ell][0].cpu().double().numpy()
        n += 1
    for h in hs:
        h.remove()
    return xbar / n, qkvbar / n


def head_maps(model, ell, h, d_head):
    W = model.transformer.h[ell].attn.c_attn.weight.detach().cpu().double().numpy()   # (768, 2304)
    b = model.transformer.h[ell].attn.c_attn.bias.detach().cpu().double().numpy()
    d = W.shape[0]
    sl = lambda blk: slice(blk * d + h * d_head, blk * d + (h + 1) * d_head)
    return {"Q": (W[:, sl(0)], b[sl(0)]), "K": (W[:, sl(1)], b[sl(1)]), "V": (W[:, sl(2)], b[sl(2)])}


def score_profile(x: np.ndarray, maps: dict, scaling: float) -> np.ndarray:
    WQ, bQ = maps["Q"]; WK, bK = maps["K"]
    q = x @ WQ + bQ
    k = x @ WK + bK
    return pooled_window_profile((q @ k.T) * scaling)


# ------------------------------------------------------------ per head -----
def analyze_head(model, ell, h, xbar_l, saved_Spos, sigma_pos_ref, scaling, rng):
    d = xbar_l.shape[1]
    d_head = model.config.n_embd // model.config.n_head
    maps = head_maps(model, ell, h, d_head)
    m = xbar_l.mean(0, keepdims=True)
    delta = xbar_l - m
    U, S, Vt = np.linalg.svd(delta, full_matrices=False)
    var = S ** 2 / (S ** 2).sum()
    cum = np.cumsum(var)
    out = {"head": f"L{ell}H{h}", "gates": {}, "kappa": {}, "profiles": {}, "reconstruction": {},
           "positional_field": {"cum_var_top": [float(c) for c in cum[:16]],
                                "k50": int(np.searchsorted(cum, 0.5) + 1),
                                "k90": int(np.searchsorted(cum, 0.9) + 1)}}

    # ---- gates: recompute exp-112's S_pos from x̄ through the head's own maps
    Sp = score_profile(xbar_l, maps, scaling)
    k1 = float(np.max(np.abs(Sp - saved_Spos)))
    sig_pos = -ols_slope(Sp, WINDOW)
    out["gates"] = {"K1_max_abs_diff": k1, "K1_pass": bool(k1 <= K1_TOL),
                    "sigma_pos_recomputed": float(sig_pos), "sigma_pos_exp112_raw": float(sigma_pos_ref),
                    "K2_abs_diff": float(abs(sig_pos - sigma_pos_ref)),
                    "K2_pass": bool(abs(sig_pos - sigma_pos_ref) <= K2_TOL)}

    # ---- κ̃
    for name in ("Q", "K", "V"):
        out["kappa"][name] = kappa(maps[name][0], delta)
    kr = [kappa(rng.standard_normal((d, d_head)), delta) for _ in range(N_RAND)]
    out["kappa"]["rand_mean"], out["kappa"]["rand_std"] = float(np.mean(kr)), float(np.std(kr))
    # projected-energy share in the handle's subspace, for reference
    P8 = Vt[:8].T
    for name in ("Q", "K", "V"):
        W = maps[name][0]
        out["kappa"][f"share_in_P8_{name}"] = float(((P8.T @ W) ** 2).sum() / (W ** 2).sum())
    out["kappa"]["share_in_P8_isotropic"] = 8.0 / d

    # ---- cosine (and raw) lag profiles of the read positional field
    reads = {"Q": delta @ maps["Q"][0], "K": delta @ maps["K"][0], "V": delta @ maps["V"][0],
             "P2": delta @ Vt[:2].T, "P8": delta @ Vt[:8].T, "I": delta}
    for name, Y in reads.items():
        out["profiles"][name] = {"cosine": profile_stats(cosine_profile(Y)),
                                 "raw": profile_stats(raw_profile(Y))}
    rand_stats = []
    for _ in range(N_RAND):
        Y = delta @ rng.standard_normal((d, d_head))
        rand_stats.append(profile_stats(cosine_profile(Y)))
    out["profiles"]["rand_cosine"] = {
        "best_slope_mean": float(np.mean([s["best_slope"] for s in rand_stats])),
        "best_slope_std": float(np.std([s["best_slope"] for s in rand_stats])),
        "best_R2_mean": float(np.mean([s["best_R2"] for s in rand_stats])),
        "loglin_b_mean": float(np.mean([s["loglin"]["b"] for s in rand_stats])),
        "loglog_sigma_mean": float(np.mean([s["loglog"]["sigma"] for s in rand_stats if s["loglog"]["sigma"] is not None])) if any(s["loglog"]["sigma"] is not None for s in rand_stats) else None}

    # ---- low-rank reconstruction of the carrier
    for k in KS:
        Pk = Vt[:k].T
        xk = m + (delta @ Pk) @ Pk.T
        Sk = score_profile(xk, maps, scaling)
        ss_tot = float(((Sp - Sp.mean()) ** 2).sum())
        r2_direct = 1.0 - float(((Sp - Sk) ** 2).sum()) / ss_tot
        r2_offset = 1.0 - float((((Sp - Sp.mean()) - (Sk - Sk.mean())) ** 2).sum()) / ss_tot
        out["reconstruction"][str(k)] = {"sigma_k": float(-ols_slope(Sk, WINDOW)),
                                         "R2_direct": r2_direct, "R2_offset_removed": r2_offset,
                                         "rel_slope_err": float(abs(-ols_slope(Sk, WINDOW) - sig_pos) / abs(sig_pos))}
    # random rank-8 subspace control (not discriminating; sanity)
    G = rng.standard_normal((d, 8)); Qr, _ = np.linalg.qr(G)
    xk = m + (delta @ Qr) @ Qr.T
    Sk = score_profile(xk, maps, scaling)
    out["reconstruction"]["rand8"] = {"sigma_k": float(-ols_slope(Sk, WINDOW)),
                                      "rel_slope_err": float(abs(-ols_slope(Sk, WINDOW) - sig_pos) / abs(sig_pos))}
    return out


# ------------------------------------------------------------- verdicts ----
def p1_head(r):
    kq, kk, kv = r["kappa"]["Q"], r["kappa"]["K"], r["kappa"]["V"]
    return kq > 2 * kv and kk > 2 * kv


def p2_head(r):
    def carries(p):
        return p["n_increases"] <= 12 and p["best_R2"] >= 0.80
    Q, K, V = (r["profiles"][n]["cosine"] for n in ("Q", "K", "V"))
    if not (carries(Q) and carries(K)):
        return False
    vfail = (V["best_R2"] < 0.70) or (V["n_increases"] > 40) or \
            (abs(V["best_slope"]) < min(abs(Q["best_slope"]), abs(K["best_slope"])) / 3)
    return bool(vfail)


def p3_head(r, name):
    p = r["profiles"][name]["cosine"]
    return p["loglog"]["R2"] is None or p["loglin"]["R2"] >= p["loglog"]["R2"]


def p4_head(r):
    k8 = r["reconstruction"]["8"]
    return k8["rel_slope_err"] <= 0.15 and k8["R2_direct"] >= 0.90


def p4_dead_head(r):
    k8 = r["reconstruction"]["8"]
    return k8["rel_slope_err"] > 0.5 or k8["R2_direct"] < 0.70


def verdict(n_ok, n, confirm_at, dead_at):
    if n_ok >= confirm_at:
        return "CONFIRMED"
    if n_ok <= dead_at:
        return "DEAD"
    return "AMBIGUOUS"


# ----------------------------------------------------------------- main ----
def main() -> None:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = AutoConfig.from_pretrained("gpt2")
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32,
                                                 attn_implementation="eager").to(device).eval()
    d_head = cfg.n_embd // cfg.n_head
    scaling = d_head ** -0.5
    rng = np.random.default_rng(137)

    # inputs, bit-identical to exp-107/112
    wt_windows, wt_meta = w107.build_wikitext_windows(tok)
    rec_wt = json.loads((EXP107 / "exploratory_wikitext.json").read_text())["text_source"]
    assert wt_meta["ids_sha256"] == rec_wt["ids_sha256"], "WikiText windows drifted"

    def random_iter():
        r = np.random.default_rng(SEED)
        torch.manual_seed(SEED)
        for _ in range(N_INPUTS):
            yield torch.from_numpy(r.integers(0, cfg.vocab_size, size=(1, SEQ_LEN)).astype(np.int64))

    def wt_iter():
        for i in range(N_INPUTS):
            yield torch.from_numpy(wt_windows[i:i + 1])

    saved = np.load(EXP112 / "scores_gpt2.npz")
    r112 = json.loads((EXP112 / "results_gpt2.json").read_text())["conditions"]

    def ref_sigma(cond, ell, h):
        # exp-112's plug-in positional-mean slope, taken from its saved S_pos profile
        # (registered entries in results_gpt2.json store only the n/(n-1)-corrected value).
        return float(-ols_slope(saved[f"S_pos_{cond}"][ell, h], WINDOW))

    results = {"exp": "exp-137", "prereg_commit": PREREG_COMMIT, "device": device,
               "protocol": {"seq_len": SEQ_LEN, "n_inputs": N_INPUTS, "seed": SEED,
                            "window": [int(WINDOW[0]), int(WINDOW[-1])], "n_rand": N_RAND, "ks": list(KS)},
               "conditions": {}}
    plan = {"random": [("structural", STRUCTURAL), ("control", CONTROL)],
            "wikitext": [("semantic", SEMANTIC)]}
    iters = {"random": random_iter, "wikitext": wt_iter}

    for cond, groups in plan.items():
        print(f"== condition {cond}", flush=True)
        xbar, qkvbar = mean_fields(model, cfg, device, iters[cond])
        # linearity check: q̄ from x̄ vs captured mean q (should agree to fp32 roundoff)
        lin = []
        for ell in range(cfg.n_layer):
            W = model.transformer.h[ell].attn.c_attn.weight.detach().cpu().double().numpy()
            b = model.transformer.h[ell].attn.c_attn.bias.detach().cpu().double().numpy()
            lin.append(float(np.max(np.abs(xbar[ell] @ W + b - qkvbar[ell]))))
        results["conditions"][cond] = {"linearity_max_abs": lin, "groups": {}}
        for gname, heads in groups:
            rows = []
            for ell, h in heads:
                r = analyze_head(model, ell, h, xbar[ell], saved[f"S_pos_{cond}"][ell, h], ref_sigma(cond, ell, h), scaling, rng)
                rows.append(r)
                g = r["gates"]
                print(f"  {r['head']:7s} K1 {g['K1_max_abs_diff']:.2e} K2 {g['K2_abs_diff']:.2e} | "
                      f"κ Q {r['kappa']['Q']:.2f} K {r['kappa']['K']:.2f} V {r['kappa']['V']:.2f} "
                      f"rand {r['kappa']['rand_mean']:.2f}±{r['kappa']['rand_std']:.2f} | "
                      f"k8 σ {r['reconstruction']['8']['sigma_k']:.3f}/{g['sigma_pos_recomputed']:.3f} "
                      f"R² {r['reconstruction']['8']['R2_direct']:.3f} | k50 {r['positional_field']['k50']}", flush=True)
            results["conditions"][cond]["groups"][gname] = rows

    # ---- gates overall
    all_rows = [r for c in results["conditions"].values() for g in c["groups"].values() for r in g]
    reg_rows = results["conditions"]["random"]["groups"]["structural"] + results["conditions"]["wikitext"]["groups"]["semantic"]
    results["gates"] = {"K1_all_pass": all(r["gates"]["K1_pass"] for r in reg_rows),
                        "K1_worst": max(r["gates"]["K1_max_abs_diff"] for r in reg_rows),
                        "K2_all_pass": all(r["gates"]["K2_pass"] for r in reg_rows),
                        "K2_worst": max(r["gates"]["K2_abs_diff"] for r in reg_rows)}

    # ---- registered verdicts
    S = results["conditions"]["random"]["groups"]["structural"]
    Sem = results["conditions"]["wikitext"]["groups"]["semantic"]
    n1 = sum(p1_head(r) for r in S)
    med = lambda k: float(np.median([r["kappa"][k] for r in S]))
    p1 = verdict(n1, 5, 4, 2)
    if med("V") >= med("Q"):
        p1 = "DEAD"
    n2 = sum(p2_head(r) for r in S)
    p3 = {name: sum(p3_head(r, name) for r in S) for name in ("I", "Q", "K")}
    n4 = sum(p4_head(r) for r in S)
    n4d = sum(p4_dead_head(r) for r in S)
    p4 = "CONFIRMED" if n4 >= 4 else ("DEAD" if n4d >= 3 else "AMBIGUOUS")
    n5a = sum(p1_head(r) for r in Sem)
    n5b = sum(p4_head(r) for r in Sem)
    results["registered_verdicts"] = {
        "P1": {"n_ok": n1, "of": 5, "verdict": p1, "median_kappa": {k: med(k) for k in ("Q", "K", "V", "rand_mean")}},
        "P2": {"n_ok": n2, "of": 5, "verdict": verdict(n2, 5, 4, 2)},
        "P3": {name: {"n_loglin_wins": p3[name], "of": 5, "verdict": "CONFIRMED" if p3[name] >= 3 else "NOT CONFIRMED"} for name in p3},
        "P4": {"n_ok": n4, "n_dead": n4d, "of": 5, "verdict": p4},
        "P5": {"P1_criterion": {"n_ok": n5a, "of": 16, "verdict": verdict(n5a, 16, 10, 5)},
               "P4_criterion": {"n_ok": n5b, "of": 16, "verdict": verdict(n5b, 16, 10, 5)}},
    }
    print(json.dumps(results["gates"], indent=1))
    print(json.dumps(results["registered_verdicts"], indent=1))
    (HERE / "results.json").write_text(json.dumps(results, indent=1))
    print("wrote results.json")


if __name__ == "__main__":
    main()
