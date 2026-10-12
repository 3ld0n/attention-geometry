#!/usr/bin/env python3
"""
exp-172 — Gap or position? Rotary-aware score decomposition on the formation-ladder checkpoints.

Protocol of record: prereg.md in this folder (commit f30fb44). This script implements it.
Analysis-only: forward passes on existing 70m GPT-NeoX checkpoints (step_2000).

Usage:  python run.py            (expects checkpoints under /tmp/ladder_ckpt/<TAG>/)
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import GPTNeoXForCausalLM
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
CK = Path("/tmp/ladder_ckpt")

# frozen-protocol helpers (lag_profile, fit_power_law) imported from the ladder's own measure.py
_spec = importlib.util.spec_from_file_location("measure062", EXP / "exp-062_corpus_statistics" / "measure.py")
meas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(meas)

SEQ_LEN, DEEP_LO, N_INPUTS, SEED = 512, 256, 50, 42
FIT_LO, FIT_HI = 8, 256
LAGS = np.arange(FIT_LO, FIT_HI + 1)
LNLAG = np.log(LAGS.astype(float))

M062 = EXP / "exp-062_corpus_statistics" / "measurements"
M091 = EXP / "exp-091_narrative_decomposition_sentence_order"
MODELS = [
    ("CNAT_s0", "natural", M062 / "run_CNAT_s0_final.json"),
    ("CNAT_s1", "natural", M062 / "run_CNAT_s1_final.json"),
    ("CNAT_s2", "natural", M062 / "run_CNAT_s2_final.json"),
    ("CNATshuf_s0", "shuffled", M091 / "results_s0.json"),
    ("CNATshuf_s1", "shuffled", M091 / "results_s1.json"),
    ("CNATshuf_s2", "shuffled", M091 / "results_s2.json"),
]
TERMS = ["gap", "abskey", "absquery", "mixed", "full"]


def sigma_of(profile: np.ndarray) -> float:
    """sigma = -(OLS slope of profile vs ln lag) over lags 8..256 (prereg: used for every term)."""
    y = profile[LAGS].astype(np.float64)
    A = np.column_stack([np.ones_like(LNLAG), LNLAG])
    c = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(-c[1])


def layer_class(l: int) -> str:
    return "L0" if l == 0 else ("L12" if l <= 2 else "deep")


def run_model(tag: str, path_stored: Path) -> dict:
    stored = json.loads(path_stored.read_text())
    sel = [(h["layer"], h["head"], h["delta"]) for h in stored["heads"] if h["conformal"]]

    model = GPTNeoXForCausalLM.from_pretrained(
        CK / tag, torch_dtype=torch.float32, attn_implementation="eager").eval()
    nl, nh = model.config.num_hidden_layers, model.config.num_attention_heads
    hid = model.config.hidden_size
    hs = hid // nh
    L = SEQ_LEN

    cur: dict[int, torch.Tensor] = {}
    hooks = []
    for l in range(nl):
        hooks.append(model.gpt_neox.layers[l].attention.query_key_value.register_forward_hook(
            lambda mod, inp, out, l=l: cur.__setitem__(l, out.detach())))

    sums = {k: [np.zeros((nh, L, hs)) for _ in range(nl)] for k in ("q", "k", "q1", "k1", "q2", "k2")}
    mean_prof = np.zeros((nl, nh, L))
    first_qk, first_att = {}, {}

    rng = np.random.default_rng(SEED)
    pool = np.arange(model.config.vocab_size)
    for inp in range(N_INPUTS):
        ids = rng.choice(pool, size=L, replace=True)
        x = torch.tensor(ids[None, :], dtype=torch.long)
        with torch.no_grad():
            out = model(x, output_attentions=True)
        for l in range(nl):
            qkv = cur[l][0].view(L, nh, 3 * hs).permute(1, 0, 2).double().numpy()  # (nh, L, 3hs)
            q, k = qkv[..., :hs], qkv[..., hs:2 * hs]
            sums["q"][l] += q
            sums["k"][l] += k
            half = "1" if inp < N_INPUTS // 2 else "2"
            sums["q" + half][l] += q
            sums["k" + half][l] += k
            att = out.attentions[l][0].float().numpy()
            mean_prof[l] += meas.lag_profile(att, DEEP_LO)
            if inp == 0:
                first_qk[l] = (q.copy(), k.copy())
                first_att[l] = att.copy()
        del out
    mean_prof /= N_INPUTS
    for h in hooks:
        h.remove()

    # rotary (the model's own cos/sin, applied analytically to mean vectors)
    pos = torch.arange(L)[None]
    cos, sin = model.gpt_neox.rotary_emb(torch.zeros(1, L, hid), position_ids=pos)
    cos, sin = cos.double(), sin.double()

    def rot(x: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(np.ascontiguousarray(x)).double()[None]  # (1, nh, L, hs)
        r, _ = apply_rotary_pos_emb(t, t, cos, sin)
        return r[0]  # (nh, L, hs)

    scale = hs ** -0.5

    # ---- gate K1: reconstruct attention for input 0 from q, k, rotary, causal softmax
    k1_max = 0.0
    mask = torch.tril(torch.ones(L, L, dtype=torch.bool))
    for l in range(nl):
        q0, k0 = first_qk[l]
        s = (rot(q0) @ rot(k0).transpose(-1, -2)) * scale
        s = s.masked_fill(~mask, float("-inf"))
        a = torch.softmax(s, dim=-1).numpy()
        k1_max = max(k1_max, float(np.abs(a - first_att[l]).max()))

    # ---- decomposition of the position-mean score, per layer
    def decompose(qs: list[np.ndarray], ks: list[np.ndarray], n: int) -> dict:
        res = {}
        for l in range(nl):
            q, k = qs[l] / n, ks[l] / n
            mq, mk = q.mean(1, keepdims=True), k.mean(1, keepdims=True)
            dq, dk = q - mq, k - mk
            Rmq, Rmk = rot(np.broadcast_to(mq, q.shape)), rot(np.broadcast_to(mk, k.shape))
            Rdq, Rdk = rot(dq), rot(dk)
            Rq, Rk = rot(q), rot(k)
            mats = {
                "gap": Rmq @ Rmk.transpose(-1, -2),
                "abskey": Rmq @ Rdk.transpose(-1, -2),
                "absquery": Rdq @ Rmk.transpose(-1, -2),
                "mixed": Rdq @ Rdk.transpose(-1, -2),
                "full": Rq @ Rk.transpose(-1, -2),
            }
            prof = {t: meas.lag_profile((m * scale).numpy(), DEEP_LO) for t, m in mats.items()}
            res[l] = {t: np.array([sigma_of(prof[t][h]) for h in range(nh)]) for t in TERMS}
        return res

    full = decompose(sums["q"], sums["k"], N_INPUTS)
    half1 = decompose(sums["q1"], sums["k1"], N_INPUTS // 2)
    half2 = decompose(sums["q2"], sums["k2"], N_INPUTS // 2)

    def shares(sig: dict, l: int, h: int) -> dict:
        f = sig[l]["full"][h]
        ok = abs(f) > 1e-3
        return {t: (float(sig[l][t][h] / f) if ok else float("nan")) for t in TERMS if t != "full"}

    heads = []
    k3_max = 0.0
    for (l, h, d_stored) in sel:
        fit = meas.fit_power_law(mean_prof[l, h], FIT_LO, FIT_HI)
        d_new = fit["delta"]
        s_full = {t: float(full[l][t][h]) for t in TERMS}
        k3 = abs(s_full["gap"] + s_full["abskey"] + s_full["absquery"] + s_full["mixed"] - s_full["full"])
        k3_max = max(k3_max, k3)
        heads.append({
            "layer": l, "head": h, "class": layer_class(l),
            "delta_stored": d_stored, "delta_recomputed": d_new,
            "k2_ok": bool(d_new is not None and abs(d_new - d_stored) <= 0.02),
            "sigma_census_attention": None if d_new is None else 2 * d_new,
            "sigma": s_full, "shares": shares(full, l, h),
            "shares_half1": shares(half1, l, h), "shares_half2": shares(half2, l, h),
        })
    return {"tag": tag, "n_selected": len(sel), "K1_max_abs_diff": k1_max,
            "K3_max_additivity_err": k3_max, "heads": heads,
            "rotary": {"rotary_dim": int(cos.shape[-1]), "head_size": hs}}


def med(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


def verdicts(models: list[dict]) -> dict:
    kinds = {t: k for t, k, _ in MODELS}
    out = {"per_model": {}}
    nat_A, nat_G, shuf_A = [], [], []
    for m in models:
        deep = [h for h in m["heads"] if h["class"] == "deep" and h["k2_ok"]]
        l0 = [h for h in m["heads"] if h["class"] == "L0" and h["k2_ok"]]
        A, G = [h["shares"]["abskey"] for h in deep], [h["shares"]["gap"] for h in deep]
        # K4: split-half agreement for deep heads
        agree = [abs(h["shares_half1"]["abskey"] - h["shares_half2"]["abskey"]) <= 0.15
                 and abs(h["shares_half1"]["gap"] - h["shares_half2"]["gap"]) <= 0.15 for h in deep
                 if np.isfinite(h["shares_half1"]["abskey"]) and np.isfinite(h["shares_half2"]["abskey"])]
        out["per_model"][m["tag"]] = {
            "kind": kinds[m["tag"]], "n_deep": len(deep), "deep_median_A": med(A), "deep_median_G": med(G),
            "deep_median_mixed": med([h["shares"]["mixed"] for h in deep]),
            "deep_median_absquery": med([h["shares"]["absquery"] for h in deep]),
            "L0_n": len(l0), "L0_median_G": med([h["shares"]["gap"] for h in l0]),
            "L0_frac_G_ge_0.8": (float(np.mean([h["shares"]["gap"] >= 0.8 for h in l0 if np.isfinite(h["shares"]["gap"])]))
                                 if l0 else float("nan")),
            "K4_halves_agree_frac": float(np.mean(agree)) if agree else float("nan"),
            "k2_excluded": [(h["layer"], h["head"]) for h in m["heads"] if not h["k2_ok"]],
        }
        if kinds[m["tag"]] == "natural":
            nat_A += A
            nat_G += G
        else:
            shuf_A += A
    nat = [out["per_model"][t] for t, k, _ in MODELS if k == "natural"]
    out["H1_position_driven"] = sum(1 for p in nat if p["deep_median_A"] >= 0.7) >= 2
    out["H2_gap_driven"] = sum(1 for p in nat if p["deep_median_G"] >= 0.7) >= 2
    out["H3_layer0_control_all_heads"] = all(p["L0_frac_G_ge_0.8"] == 1.0 for p in out["per_model"].values()
                                              if p["L0_n"] > 0)
    out["H3_layer0_control_model_medians"] = all(p["L0_median_G"] >= 0.8 for p in out["per_model"].values()
                                                  if p["L0_n"] > 0)
    out["H4_pooled_deep_median_A"] = {"natural": med(nat_A), "shuffled": med(shuf_A),
                                      "n_natural": len(nat_A), "n_shuffled": len(shuf_A)}
    if np.isfinite(med(nat_A)) and np.isfinite(med(shuf_A)):
        out["H4_contrast_ge_0.25"] = abs(med(nat_A) - med(shuf_A)) >= 0.25
    return out


def main() -> None:
    results = []
    for tag, kind, p in MODELS:
        print(f"=== {tag} ({kind})", flush=True)
        r = run_model(tag, p)
        print(f"  K1 max|Δattn| = {r['K1_max_abs_diff']:.2e}   K3 additivity err = {r['K3_max_additivity_err']:.2e}   "
              f"selected heads: {r['n_selected']}   rotary dims: {r['rotary']['rotary_dim']}/{r['rotary']['head_size']}", flush=True)
        n_bad = sum(1 for h in r["heads"] if not h["k2_ok"])
        print(f"  K2: {n_bad} of {r['n_selected']} heads do not replicate their stored census Δ (±0.02)", flush=True)
        for h in r["heads"]:
            s = h["shares"]
            print(f"   L{h['layer']}H{h['head']:<2} {h['class']:<4} Δ={h['delta_recomputed']:.3f}  "
                  f"σ_full={h['sigma']['full']:+.3f}  G={s['gap']:+.2f} A={s['abskey']:+.2f} "
                  f"Aq={s['absquery']:+.2f} M={s['mixed']:+.2f}  k2={'ok' if h['k2_ok'] else 'NO'}", flush=True)
        results.append(r)
    v = verdicts(results)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "results.json").write_text(json.dumps({"models": results, "verdicts": v}, indent=1, default=float))
    print("\n=== VERDICTS ===")
    print(json.dumps(v, indent=1, default=float))


if __name__ == "__main__":
    sys.exit(main())
