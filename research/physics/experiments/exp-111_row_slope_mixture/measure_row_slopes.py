"""exp-111 — row-slope mixture: per-(row, input) window fits under three inputs.

Pre-registration: notes.md in this folder, written before this file. Inputs
bit-identical to exp-107/110 (sha256-gated). Per (head, condition, input,
row i in [256, 512)): OLS of log A(i, i-dx) on log dx over dx in [8, 256];
records slope beta, R2, log in-window mass. K1 gate: pooled window profile
rebuilt from rows must match exp-107's saved profiles (rel <= 1e-5).

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
KIT = HERE.parent.parent / "replication" / "measure_conformal_heads.py"

spec = importlib.util.spec_from_file_location("census_kit", KIT)
kit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kit)
SEQ_LEN, N_INPUTS, DEEP_LO, FIT_LO, FIT_HI, SEED = (
    kit.SEQ_LEN, kit.N_INPUTS, kit.DEEP_LO, kit.FIT_LO, kit.FIT_HI, kit.SEED)

sys.path.insert(0, str(EXP107))
import measure_natural_bilocal as m107          # noqa: E402
import exploratory_wikitext as w107             # noqa: E402

STRUCTURAL = [(2, 1), (3, 4), (5, 0), (7, 11), (10, 8)]
SEMANTIC = [(4, 10), (7, 1), (8, 2), (9, 4), (9, 6), (10, 1), (10, 2), (10, 10),
            (11, 0), (11, 1), (11, 2), (11, 4), (11, 5), (11, 6), (11, 7), (11, 9)]
CONDITIONS = ("random", "tinystories", "wikitext")
N_ROWS = SEQ_LEN - DEEP_LO                       # 256
LAGS = np.arange(FIT_LO, FIT_HI + 1)             # 249 lags
LX = np.log(LAGS.astype(float))
XD = np.column_stack([np.ones_like(LX), LX])
PINV = np.linalg.pinv(XD)                        # (2, 249)
SXX = float(np.sum((LX - LX.mean()) ** 2))


def window_rows(A: np.ndarray) -> np.ndarray:
    """A: (n_head, L, L) -> Y: (n_head, n_lags, N_ROWS), Y[:, k, r] = A(256+r, 256+r-dx_k)."""
    n_head = A.shape[0]
    Y = np.empty((n_head, len(LAGS), N_ROWS), dtype=np.float64)
    for k, dx in enumerate(LAGS):
        diag = np.diagonal(A, offset=-dx, axis1=-2, axis2=-1)
        Y[:, k, :] = diag[:, DEEP_LO - dx:]
    return Y


def fit_rows(Y: np.ndarray):
    """Y: (n_head, n_lags, N_ROWS) positive. Returns beta, r2, logM, se2 per (head, row)."""
    lY = np.log(np.clip(Y, 1e-45, None))
    coef = np.einsum("ck,hkr->hcr", PINV, lY)     # (n_head, 2, N_ROWS)
    beta = -coef[:, 1, :]
    fit = np.einsum("kc,hcr->hkr", XD, coef)
    resid = lY - fit
    ss_res = (resid ** 2).sum(axis=1)
    ss_tot = ((lY - lY.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-30)
    logM = np.log(Y.sum(axis=1))
    se2 = (ss_res / (len(LAGS) - 2)) / SXX        # per-fit slope-error variance
    return beta, r2, logM, se2


def ols_slope(y: np.ndarray) -> float:
    c = PINV @ y
    return float(c[1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = AutoConfig.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2", dtype=torch.float32, attn_implementation="eager").to(device).eval()
    n_layer, n_head = cfg.num_hidden_layers, cfg.num_attention_heads

    ts_windows, ts_meta = m107.build_text_windows(tokenizer)
    assert ts_meta["ids_sha256"] == json.loads(
        (EXP107 / "applied_text.json").read_text())["ids_sha256"]
    wt_windows, wt_meta = w107.build_wikitext_windows(tokenizer)
    assert wt_meta["ids_sha256"] == json.loads(
        (EXP107 / "exploratory_wikitext.json").read_text())["text_source"]["ids_sha256"]

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
    ref = {"random": saved107["A_random"], "tinystories": saved107["A_text"],
           "wikitext": saved_wt["A"]}
    iters = {"random": random_iter, "tinystories": win_iter(ts_windows),
             "wikitext": win_iter(wt_windows)}

    store = {}   # arrays per condition: beta/r2/logM/se2 (n_layer, n_head, N_INPUTS, N_ROWS)
    prof_win = {}  # pooled window profile rebuilt from rows

    for cond in CONDITIONS:
        print(f"condition {cond.upper()}", flush=True)
        beta = np.empty((n_layer, n_head, N_INPUTS, N_ROWS), dtype=np.float32)
        r2 = np.empty_like(beta); logM = np.empty_like(beta); se2 = np.empty_like(beta)
        pw = np.zeros((n_layer, n_head, len(LAGS)))
        for idx, ids in enumerate(iters[cond]()):
            with torch.no_grad():
                out = model(ids.to(device), output_attentions=True)
            for ell in range(n_layer):
                A = out.attentions[ell][0].float().cpu().numpy()
                Y = window_rows(A)
                pw[ell] += Y.mean(axis=-1)
                b, r, lm, s2 = fit_rows(Y)
                beta[ell, :, idx], r2[ell, :, idx] = b, r
                logM[ell, :, idx], se2[ell, :, idx] = lm, s2
            del out
            if (idx + 1) % 10 == 0:
                print(f"    {idx + 1}/{N_INPUTS} inputs", flush=True)
        pw /= N_INPUTS
        drift = float(np.max(np.abs(pw - ref[cond][:, :, FIT_LO:FIT_HI + 1])
                             / np.maximum(ref[cond][:, :, FIT_LO:FIT_HI + 1], 1e-30)))
        print(f"  K1: max rel |P_win(rows) - exp-107| = {drift:.3e} -> "
              f"{'OK' if drift <= 1e-5 else 'FAIL'}")
        if drift > 1e-5:
            raise SystemExit("K1 pipeline gate failed")
        store[cond] = dict(beta=beta, r2=r2, logM=logM, se2=se2)
        prof_win[cond] = pw

    # ---------------- per-pair analysis ----------------
    registered = [(l, h, c) for (l, h) in STRUCTURAL for c in CONDITIONS]
    registered += [(l, h, "wikitext") for (l, h) in SEMANTIC]
    all_pairs = [(l, h, c) for (l, h) in STRUCTURAL + SEMANTIC for c in CONDITIONS]

    pair_out = []
    for (l, h, c) in all_pairs:
        s = store[c]
        b = s["beta"][l, h].ravel().astype(np.float64)
        r = s["r2"][l, h].ravel().astype(np.float64)
        lm = s["logM"][l, h].ravel().astype(np.float64)
        e2 = s["se2"][l, h].ravel().astype(np.float64)
        med_r2 = float(np.median(r))
        var_raw = float(np.var(b))
        var_shrunk = max(var_raw - float(np.mean(e2)), 0.0)

        # measured annealed & quenched from this run's rows
        pw = prof_win[c][l, h]
        s_ann = -ols_slope(np.log(pw))
        # quenched: mean log A over pool
        # (recompute from row data: mean of log Y across pool = handled during run?
        #  reconstruct: E[log A] = E over rows of (fit + resid) — we did not store
        #  full logY; use exact identity E[log A] via stored coef? Not stored.
        #  Instead use exp-110 moments (same pool, same forwards, bit-identical).)
        pair_out.append(dict(layer=l, head=h, cond=c,
                             registered=(l, h, c) in registered,
                             population=("structural" if (l, h) in STRUCTURAL
                                         else "semantic"),
                             median_row_r2=med_r2,
                             eligible=med_r2 >= 0.5,
                             mean_beta=float(np.mean(b)),
                             var_beta_raw=var_raw, var_beta_shrunk=var_shrunk,
                             s_annealed_measured=s_ann))

    # quenched slopes from exp-110 moments (same pool, bit-identical forwards)
    mom = np.load(HERE.parent / "exp-110_annealed_quenched_decomposition"
                  / "moments_gpt2.npz")
    for p in pair_out:
        l, h, c = p["layer"], p["head"], p["cond"]
        n = mom[f"n_{c}"][l, h][FIT_LO:FIT_HI + 1]
        m = mom[f"s1_{c}"][l, h][FIT_LO:FIT_HI + 1] / n
        p["s_quenched_measured"] = -ols_slope(m)
        p["damping_gap"] = p["s_quenched_measured"] - p["s_annealed_measured"]

    # P2: zero-free-parameter reconstruction on eligible registered pairs
    for p in pair_out:
        l, h, c = p["layer"], p["head"], p["cond"]
        s = store[c]
        b = s["beta"][l, h].ravel().astype(np.float64)
        M = np.exp(s["logM"][l, h].ravel().astype(np.float64))
        zeta = (LAGS[None, :].astype(float) ** (-b[:, None])).sum(axis=1)
        P_pred = (M[:, None] * LAGS[None, :].astype(float) ** (-b[:, None])
                  / zeta[:, None]).mean(axis=0)
        p["s_pred_mixture"] = -ols_slope(np.log(P_pred))
        p["p2_slope_diff"] = p["s_pred_mixture"] - p["s_annealed_measured"]

    # ---------------- verdicts ----------------
    reg = [p for p in pair_out if p["registered"]]
    elig = [p for p in reg if p["eligible"]]
    native = [p for p in reg if
              (p["population"] == "structural" and p["cond"] == "random") or
              (p["population"] == "semantic" and p["cond"] == "wikitext")]

    p1_native = {f"L{p['layer']}H{p['head']}-{p['cond']}": p["median_row_r2"]
                 for p in native}
    ka_fired = any(p["median_row_r2"] < 0.5 for p in native)

    p2_diffs = np.array([abs(p["p2_slope_diff"]) for p in elig])
    p2_med = float(np.median(p2_diffs)) if len(elig) else float("nan")
    p2 = ("CONFIRMED" if p2_med <= 0.10 else
          "DEAD" if p2_med > 0.20 else "AMBIGUOUS")

    p3a_votes = []
    for (l, h) in STRUCTURAL:
        pr = {p["cond"]: p for p in pair_out
              if p["layer"] == l and p["head"] == h}
        if pr["random"]["eligible"] and pr["tinystories"]["eligible"]:
            p3a_votes.append(pr["tinystories"]["var_beta_shrunk"]
                             > pr["random"]["var_beta_shrunk"])
    p3a_n = sum(p3a_votes)
    varr = np.array([p["var_beta_shrunk"] for p in elig])
    gaps = np.array([p["damping_gap"] for p in elig])
    p3b_rho = spearman(varr, gaps) if len(elig) >= 3 else float("nan")
    p3 = ("CONFIRMED" if (p3a_n >= 4 and p3b_rho >= 0.5) else
          "DEAD" if (p3a_n <= 2 and p3b_rho < 0.2) else "AMBIGUOUS")

    print("\n=== registered verdicts ===")
    for p in reg:
        tag = "" if p["eligible"] else " [INELIGIBLE]"
        print(f"  L{p['layer']}H{p['head']:<3d} {p['cond']:<12s} {p['population'][:6]}"
              f" med_row_R2={p['median_row_r2']:.3f} E[b]={p['mean_beta']:+.3f}"
              f" Var_b={p['var_beta_shrunk']:.4f} s_q={p['s_quenched_measured']:+.3f}"
              f" s_ann={p['s_annealed_measured']:+.3f}"
              f" s_pred={p['s_pred_mixture']:+.3f}"
              f" P2diff={p['p2_slope_diff']:+.4f}{tag}")
    print(f"  P1 native median row R2: { {k: round(v, 3) for k, v in p1_native.items()} }")
    print(f"  K-a fired: {ka_fired}")
    print(f"  P2 (mixture reconstruction): median |diff| = {p2_med:.4f} over "
          f"{len(elig)} eligible -> {p2} (prediction: confirmed ~0.05 native)")
    print(f"  P3a Var(b) TinyStories > random: {p3a_n}/{len(p3a_votes)}")
    print(f"  P3b Spearman(Var b, damping gap) = {p3b_rho:.3f}")
    print(f"  P3 -> {p3} (prediction: confirmed)")

    out = {"protocol": {"rows": [DEEP_LO, SEQ_LEN - 1], "fit_lags": [FIT_LO, FIT_HI],
                        "n_inputs": N_INPUTS, "seed": SEED,
                        "structural": STRUCTURAL, "semantic": SEMANTIC},
           "pairs": pair_out,
           "verdicts": {"P1_native_median_row_r2": p1_native, "Ka_fired": ka_fired,
                        "P2": {"median_abs_diff": p2_med, "verdict": p2},
                        "P3": {"a_votes": p3a_n, "a_total": len(p3a_votes),
                               "b_spearman": p3b_rho, "verdict": p3}}}
    (HERE / "results_gpt2.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(
        HERE / "rowfits_gpt2.npz",
        **{f"{k}_{c}": store[c][k] for c in CONDITIONS
           for k in ("beta", "r2", "logM", "se2")},
        **{f"profwin_{c}": prof_win[c] for c in CONDITIONS})
    print(f"\nwrote {HERE / 'results_gpt2.json'}")


if __name__ == "__main__":
    main()
