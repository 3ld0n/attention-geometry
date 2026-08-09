"""exp-106 — apply the forward model to GPT-2. Arms 1a, 1b, 2 as pre-registered.

G_out = o o^T with o = A v is EXACTLY A K_V A^T for K_V = v v^T, the head's value
Gram matrix. So the forward model can be evaluated at three levels of input:

  arm 1a   P_pred = lag_profile(A A^T)             real A, K = I
  arm 1b   P_pred = lag_profile(A Kbar_V A^T)      real A, K_V replaced by its own
                                                   TI lag-profile reconstruction
  arm 2    P_pred = lag_profile(Abar Abar^T)       A replaced by its TI
                                                   reconstruction, K = I

Each prediction has NO free exponent. The measured profile is compared by
  P_meas(s) ~ alpha * P_pred(s) + beta   over s in [8, 256]
in two linear parameters, on the raw profile and on log P.

Also recorded (free, and an independent check of Proposition 1):
  - ||vbar||^2 / mean ||v||^2 per head, which Proposition 1 says sets G's floor
  - the lag exponent q of K_V itself, which enters eq. (3.2)

Reproduction check: this script's A and G_out profiles must equal exp-104's saved
profiles to floating-point equality, or it aborts.

Run: python apply_forward_model.py gpt2
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

HERE = Path(__file__).resolve().parent
EXP104 = HERE.parent / "exp-104_bilocal_object_identification"
KIT = HERE.parent.parent / "replication" / "measure_conformal_heads.py"


def load_kit():
    spec = importlib.util.spec_from_file_location("census_kit", KIT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


kit = load_kit()
SEQ_LEN, N_INPUTS, DEEP_LO = kit.SEQ_LEN, kit.N_INPUTS, kit.DEEP_LO
FIT_LO, FIT_HI, SEED = kit.FIT_LO, kit.FIT_HI, kit.SEED
R2_MIN, DELTA_MIN = kit.R2_MIN, kit.DELTA_MIN
assert (SEQ_LEN, N_INPUTS, DEEP_LO, FIT_LO, FIT_HI, SEED) == (512, 50, 256, 8, 256, 42)

W = slice(FIT_LO, FIT_HI + 1)


def head_split(t: torch.Tensor, n_head: int) -> torch.Tensor:
    b, L, d = t.shape
    return t.view(b, L, n_head, d // n_head).permute(0, 2, 1, 3)


def causal_from_profile(prof: np.ndarray) -> np.ndarray:
    n = len(prof)
    idx = np.arange(n)
    lag = idx[:, None] - idx[None, :]
    A = np.where(lag >= 0, prof[np.clip(lag, 0, n - 1)], 0.0)
    s = A.sum(axis=1, keepdims=True)
    return A / np.where(s > 0, s, 1.0)


def symmetric_from_profile(prof: np.ndarray) -> np.ndarray:
    """TI reconstruction of a symmetric matrix from its lag profile."""
    n = len(prof)
    idx = np.arange(n)
    return prof[np.abs(idx[:, None] - idx[None, :])]


def linear_fit_r2(y: np.ndarray, x: np.ndarray):
    """y ~ alpha*x + beta. Returns (alpha, beta, R2, longest same-sign residual run)."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5:
        return None, None, None, None
    X = np.column_stack([x[ok], np.ones(ok.sum())])
    coef, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
    resid = y[ok] - X @ coef
    ss_tot = float(np.sum((y[ok] - y[ok].mean()) ** 2))
    r2 = 1 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 1e-30 else 0.0
    sign = np.sign(resid)
    run = best = 1
    for i in range(1, len(sign)):
        run = run + 1 if sign[i] == sign[i - 1] else 1
        best = max(best, run)
    return float(coef[0]), float(coef[1]), float(r2), int(best)


def compare(meas: np.ndarray, pred: np.ndarray) -> dict:
    out = {}
    a, b, r2, run = linear_fit_r2(meas[W], pred[W])
    out.update(alpha=a, beta=b, r2_raw=r2, resid_run_raw=run)
    pos = (meas[W] > 0) & (pred[W] > 0)
    if pos.sum() >= 5:
        la, lb, lr2, lrun = linear_fit_r2(np.log(meas[W][pos]), np.log(pred[W][pos]))
        out.update(r2_log=lr2, log_slope=la, resid_run_log=lrun)
    else:
        out.update(r2_log=None, log_slope=None, resid_run_log=None)
    mono = bool(np.all(np.diff(pred[W]) <= 1e-15))
    out["pred_monotone_decreasing"] = mono
    out["pred_all_positive"] = bool(np.all(pred[W] > 0))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="gpt2")
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-inputs", type=int, default=N_INPUTS)
    args = ap.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    cfg = AutoConfig.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager").to(device).eval()
    n_layer, n_head, vocab = (cfg.num_hidden_layers, cfg.num_attention_heads,
                              cfg.vocab_size)
    blocks = model.transformer.h
    cap: dict[int, torch.Tensor] = {}

    def post_hook_factory(ell: int):
        def post(mod, inputs, output):
            d = cfg.hidden_size
            cap[ell] = head_split(output.detach()[..., 2 * d:3 * d], n_head)
        return post

    handles = [blocks[ell].attn.c_attn.register_forward_hook(post_hook_factory(ell))
               for ell in range(n_layer)]

    names = ("A", "G_out", "AAt", "K_V")
    prof = {k: np.zeros((n_layer, n_head, SEQ_LEN)) for k in names}
    vbar_sq = np.zeros((n_layer, n_head))
    vnorm_sq = np.zeros((n_layer, n_head))

    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    for n in range(args.n_inputs):
        ids = torch.from_numpy(
            rng.integers(0, vocab, size=(1, SEQ_LEN)).astype(np.int64)).to(device)
        cap.clear()
        with torch.no_grad():
            out = model(ids, output_attentions=True)
        for ell in range(n_layer):
            A = out.attentions[ell][0].float()
            assert not torch.isnan(A).any()
            v = cap[ell][0].float()                       # (n_head, L, d_head)
            K_V = v @ v.transpose(-1, -2)                 # (n_head, L, L)
            o = A @ v
            G_out = o @ o.transpose(-1, -2)
            AAt = A @ A.transpose(-1, -2)
            for key, M in (("A", A), ("G_out", G_out), ("AAt", AAt), ("K_V", K_V)):
                prof[key][ell] += kit.lag_profile(M.cpu().numpy())
            vb = v.mean(dim=1)                            # (n_head, d_head)
            vbar_sq[ell] += (vb * vb).sum(-1).cpu().numpy()
            vnorm_sq[ell] += (v * v).sum(-1).mean(-1).cpu().numpy()
        del out
        if (n + 1) % 10 == 0:
            print(f"  {n + 1}/{args.n_inputs} inputs", flush=True)
    for h in handles:
        h.remove()
    vbar_sq /= args.n_inputs
    vnorm_sq /= args.n_inputs

    # ---- reproduction check against exp-104 --------------------------------
    saved = np.load(EXP104 / "profiles_gpt2.npz")
    for key in ("A", "G_out"):
        d = float(np.max(np.abs(prof[key] - saved[key])))
        print(f"reproduction check {key}: max abs diff vs exp-104 = {d:.3e}")
        assert d < 1e-6, f"{key} does not reproduce exp-104; comparability broken"

    # ---- per-head comparison ------------------------------------------------
    heads = []
    for ell in range(n_layer):
        for h in range(n_head):
            dA, r2A = kit.fit_head(prof["A"][ell, h])
            dG, r2G = kit.fit_head(prof["G_out"][ell, h])
            dAAt, r2AAt = kit.fit_head(prof["AAt"][ell, h])
            dKV, r2KV = kit.fit_head(prof["K_V"][ell, h])
            rec = {"layer": ell, "head": h,
                   "delta_A": dA, "r2_A": r2A,
                   "delta_G_out": dG, "r2_G_out_census": r2G,
                   "delta_AAt": dAAt, "r2_AAt": r2AAt,
                   "delta_K_V": dKV, "r2_K_V": r2KV,
                   "vbar_sq_over_vnorm_sq": float(vbar_sq[ell, h] / vnorm_sq[ell, h])}

            meas = prof["G_out"][ell, h]
            rec["arm1a"] = compare(meas, prof["AAt"][ell, h])

            # arm 1b: real A, K_V replaced by its TI reconstruction
            A_real_prof = prof["A"][ell, h]
            Kbar = symmetric_from_profile(prof["K_V"][ell, h])
            A_ti = causal_from_profile(A_real_prof)
            rec["arm1b"] = compare(meas, kit.lag_profile(
                (A_ti @ Kbar @ A_ti.T)[None, ...])[0])

            # arm 2: TI A from its own profile, K = I
            rec["arm2"] = compare(meas, kit.lag_profile((A_ti @ A_ti.T)[None, ...])[0])
            heads.append(rec)

    payload = {"model": args.model, "device": device, "n_inputs": args.n_inputs,
               "protocol": {"seq_len": SEQ_LEN, "deep_lo": DEEP_LO, "seed": SEED,
                            "fit_lags": [FIT_LO, FIT_HI],
                            "estimator": "imported verbatim from replication kit"},
               "heads": heads}
    (HERE / "applied_gpt2.json").write_text(json.dumps(payload, indent=1))
    np.savez_compressed(HERE / "profiles_forward_gpt2.npz", **prof,
                        vbar_sq=vbar_sq, vnorm_sq=vnorm_sq)

    summarize(heads)


def summarize(heads: list[dict]) -> None:
    conformal = [h for h in heads if h["r2_A"] and h["r2_A"] >= R2_MIN
                 and h["delta_A"] and h["delta_A"] >= DELTA_MIN]
    syk = [h for h in conformal if abs(h["delta_A"] - 0.25) <= 0.05]
    print(f"\nheads {len(heads)}  conformal {len(conformal)}  SYK-near {len(syk)}")

    def med(subset, path):
        vals = []
        for h in subset:
            v = h
            for p in path.split("."):
                v = v[p] if v is not None else None
            if v is not None:
                vals.append(v)
        return float(np.median(vals)) if vals else None

    for label, subset in (("ALL", heads), ("CONFORMAL", conformal), ("SYK-NEAR", syk)):
        if not subset:
            continue
        print(f"\n[{label}] n={len(subset)}")
        print(f"  census 2-param R2 on G_out          : {med(subset,'r2_G_out_census'):.4f}")
        for arm in ("arm1a", "arm1b", "arm2"):
            r2r, r2l = med(subset, f"{arm}.r2_raw"), med(subset, f"{arm}.r2_log")
            al = med(subset, f"{arm}.alpha")
            run = med(subset, f"{arm}.resid_run_log")
            print(f"  {arm}: R2_raw={r2r:.4f}  R2_log={r2l:.4f}  "
                  f"alpha_med={al:.4g}  resid_run_log={run:.0f}")
        print(f"  median delta_A={med(subset,'delta_A'):.4f}  "
              f"delta_G_out={med(subset,'delta_G_out'):.4f}  "
              f"delta_AAt={med(subset,'delta_AAt'):.4f}  "
              f"delta_K_V={med(subset,'delta_K_V'):.4f}")
        print(f"  median ||vbar||^2/||v||^2 = {med(subset,'vbar_sq_over_vnorm_sq'):.4f}")


if __name__ == "__main__":
    main()
