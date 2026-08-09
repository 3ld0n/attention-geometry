"""exp-107 — Does natural text give the bilocal a positive connected profile?

Pre-registration: notes.md in this folder (Aug 8, 2026), committed before this
file existed. One variable changes from exp-104/106: the inputs (random token
ids -> natural text). Both conditions run in this one script, random first, so
the comparison is within-run (registration section 2).

Operationalizations fixed at write time, BEFORE any natural-text forward pass
(the registration left these at the level of words; they are pinned here):

  - floor: ||vbar||^2 with vbar = v.mean over all 512 positions, averaged over
    inputs — identical to exp-106's apply_forward_model.py.
  - "below floor" head-count for K1: conn(8) < 0, i.e. the floor exceeds the
    profile at the shortest fitted lag — the definition that gives exp-106's
    published 116/144 (verified against exp-106's saved npz before this run).
  - H1 (strict, all SYK-near heads): max of P_conn over lags [8,256] < 0 on
    every SYK-near head of the text condition.
  - H2: min of P_conn over the window > 0 on a majority of SYK-near heads.
  - H4: P_conn takes both signs inside the window (min < 0 < max) on a
    majority of SYK-near heads.
  - H3 primary statistic: sign of the WINDOW MEAN of P_conn vs sign of the
    window mean of P_Ktilde, agreement fraction over all head-condition
    pairs (144 x 2); the majority-lag-sign version is reported alongside.
  - SYK-near set, per condition: R2_A >= 0.90, Delta_A >= 0.05,
    |Delta_A - 0.25| <= 0.05, from that condition's own A profiles.

Text source: TinyStories (roneneldan/TinyStories), validation split, local HF
cache — the corpus family of the program's natural-text formation condition
(exp-085/091/092). Windows are consecutive non-overlapping 512-token spans of
the EOS-joined validation stories in dataset order, starting at token 0.
Recorded in applied_text.json with a checksum. (The census itself has no
natural-text condition on GPT-2; TinyStories is the program's natural family.
This is the registration's "record the exact source" case, noted per its
outcome-limit 1.)

Ariel — August 9, 2026.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
KIT = HERE.parent.parent / "replication" / "measure_conformal_heads.py"
EXP104 = HERE.parent / "exp-104_bilocal_object_identification"
EXP106 = HERE.parent / "exp-106_bilocal_profile_shape"

spec = importlib.util.spec_from_file_location("census_kit", KIT)
kit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kit)

SEQ_LEN, N_INPUTS, DEEP_LO = kit.SEQ_LEN, kit.N_INPUTS, kit.DEEP_LO
FIT_LO, FIT_HI, SEED = kit.FIT_LO, kit.FIT_HI, kit.SEED
R2_MIN, DELTA_MIN = kit.R2_MIN, kit.DELTA_MIN
assert (SEQ_LEN, N_INPUTS, DEEP_LO, FIT_LO, FIT_HI, SEED) == (512, 50, 256, 8, 256, 42), \
    "kit protocol constants changed; exp-107 comparability broken"
W = slice(FIT_LO, FIT_HI + 1)

MODEL = "gpt2"
EXP106_BELOW_AT_8 = 116  # published count; K1 gate is +/- 3


def head_split(t: torch.Tensor, n_head: int) -> torch.Tensor:
    b, L, d = t.shape
    return t.view(b, L, n_head, d // n_head).permute(0, 2, 1, 3)


def build_text_windows(tokenizer) -> tuple[np.ndarray, dict]:
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="validation")
    eos = tokenizer.eos_token_id
    ids: list[int] = []
    n_stories = 0
    need = N_INPUTS * SEQ_LEN
    for story in ds["text"]:
        ids.extend(tokenizer.encode(story))
        ids.append(eos)
        n_stories += 1
        if len(ids) >= need:
            break
    ids = ids[:need]
    windows = np.array(ids, dtype=np.int64).reshape(N_INPUTS, SEQ_LEN)
    meta = {
        "dataset": "roneneldan/TinyStories", "split": "validation",
        "construction": "stories in dataset order, EOS-joined, consecutive "
                        "non-overlapping 512-token windows from token 0",
        "n_stories_consumed": n_stories,
        "window_starts": [int(i * SEQ_LEN) for i in range(N_INPUTS)],
        "ids_sha256": hashlib.sha256(windows.tobytes()).hexdigest(),
    }
    return windows, meta


def run_condition(model, cfg, inputs_iter, device, n_layer, n_head):
    """inputs_iter yields (1, SEQ_LEN) int64 tensors. Returns profiles + floor + checks."""
    blocks = model.transformer.h
    cap: dict[int, torch.Tensor] = {}

    def post_hook_factory(ell):
        def post_hook(mod, inp, output):
            d = cfg.hidden_size
            v = output.detach()[..., 2 * d:3 * d]
            cap[ell] = head_split(v, n_head)
        return post_hook

    handles = [blocks[ell].attn.c_attn.register_forward_hook(post_hook_factory(ell))
               for ell in range(n_layer)]

    prof = {k: np.zeros((n_layer, n_head, SEQ_LEN)) for k in ("A", "G_out", "Ktilde")}
    vbar_sq = np.zeros((n_layer, n_head))
    c2_max_rel_err = 0.0

    n_done = 0
    for ids in inputs_iter:
        cap.clear()
        with torch.no_grad():
            out = model(ids.to(device), output_attentions=True)
        for ell in range(n_layer):
            A = out.attentions[ell][0].float()            # (n_head, L, L)
            assert not torch.isnan(A).any(), f"layer {ell}: NaN attention"
            v = cap[ell][0].float()                        # (n_head, L, d_head)
            o = A @ v
            G_out = o @ o.transpose(-1, -2)
            K_V = v @ v.transpose(-1, -2)                  # (n_head, L, L)
            Ktilde = K_V - K_V.mean(dim=(-2, -1), keepdim=True)

            # C2 identity per head: sum_{a!=b} Ktilde_ab == -sum_a ||v_a - vbar||^2
            # Checked at float64 (same code path): the identity is a 512^2-term
            # near-cancelling sum and fp32 accumulation alone exceeds the 1e-4
            # gate (see notes.md run log, Aug 9). Measurements stay fp32.
            v64 = v.cpu().double()
            K64 = v64 @ v64.transpose(-1, -2)
            Kt64 = K64 - K64.mean(dim=(-2, -1), keepdim=True)
            vb64 = v64.mean(dim=1, keepdim=True)
            rhs = -((v64 - vb64) ** 2).sum(dim=(-2, -1))
            lhs = Kt64.sum(dim=(-2, -1)) - torch.diagonal(
                Kt64, dim1=-2, dim2=-1).sum(-1)
            rel = ((lhs - rhs).abs() / rhs.abs().clamp_min(1e-12)).max().item()
            c2_max_rel_err = max(c2_max_rel_err, rel)
            vb = v.mean(dim=1, keepdim=True)               # (n_head, 1, d_head)

            for key, M in (("A", A), ("G_out", G_out), ("Ktilde", Ktilde)):
                prof[key][ell] += kit.lag_profile(M.cpu().numpy())
            vbar_sq[ell] += (vb[:, 0] ** 2).sum(-1).cpu().numpy()
        del out
        n_done += 1
        if n_done % 10 == 0:
            print(f"    {n_done}/{N_INPUTS} inputs", flush=True)

    for h in handles:
        h.remove()
    for k in prof:
        prof[k] /= n_done
    vbar_sq /= n_done
    return prof, vbar_sq, c2_max_rel_err


def analyze_condition(name, prof, vbar_sq, n_layer, n_head):
    heads = []
    for ell in range(n_layer):
        for h in range(n_head):
            dA, r2A = kit.fit_head(prof["A"][ell, h])
            conn = prof["G_out"][ell, h] - vbar_sq[ell, h]
            kt = prof["Ktilde"][ell, h]
            rec = {
                "layer": ell, "head": h,
                "delta_A": dA, "r2_A": r2A,
                "vbar_sq": float(vbar_sq[ell, h]),
                "conn_at_8": float(conn[FIT_LO]),
                "conn_at_256": float(conn[FIT_HI]),
                "conn_window_max": float(conn[W].max()),
                "conn_window_min": float(conn[W].min()),
                "conn_window_mean": float(conn[W].mean()),
                "conn_frac_lags_positive": float((conn[W] > 0).mean()),
                "ktilde_window_mean": float(kt[W].mean()),
                "ktilde_frac_lags_positive": float((kt[W] > 0).mean()),
            }
            rec["conformal"] = bool(r2A is not None and r2A >= R2_MIN
                                    and dA is not None and dA >= DELTA_MIN)
            rec["syk_near"] = bool(rec["conformal"] and abs(dA - 0.25) <= 0.05)
            heads.append(rec)

    below8 = sum(1 for r in heads if r["conn_at_8"] < 0)
    all_neg = sum(1 for r in heads if r["conn_window_max"] < 0)
    some_neg = sum(1 for r in heads if r["conn_window_min"] < 0)
    syk = [r for r in heads if r["syk_near"]]
    conf = [r for r in heads if r["conformal"]]
    print(f"\n  [{name}] conformal {len(conf)}/144, SYK-near {len(syk)}"
          f" | below floor at lag 8: {below8}/144, entire window: {all_neg}/144,"
          f" somewhere: {some_neg}/144")
    if conf:
        print(f"  median Delta_A (conformal): "
              f"{np.median([r['delta_A'] for r in conf]):.4f}")
    for r in syk:
        print(f"    SYK L{r['layer']}H{r['head']} dA={r['delta_A']:.3f} "
              f"conn[min,max]=[{r['conn_window_min']:+.3e},{r['conn_window_max']:+.3e}] "
              f"frac+={r['conn_frac_lags_positive']:.2f} "
              f"Kt_mean={r['ktilde_window_mean']:+.3e}")
    return heads, {"below_floor_at_8": below8, "below_entire_window": all_neg,
                   "below_somewhere": some_neg}


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = AutoConfig.from_pretrained(MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation="eager").to(device).eval()
    n_layer, n_head = cfg.num_hidden_layers, cfg.num_attention_heads

    results = {"model": MODEL, "device": device,
               "protocol": {"seq_len": SEQ_LEN, "n_inputs": N_INPUTS,
                            "deep_lo": DEEP_LO, "fit_lags": [FIT_LO, FIT_HI],
                            "seed": SEED,
                            "estimator": "imported verbatim from replication kit"}}

    # ---- condition 1: random tokens (exp-104 protocol, rerun) ----------------
    print("condition RANDOM (frozen census protocol, within-run rerun)")
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    def random_iter():
        for _ in range(N_INPUTS):
            yield torch.from_numpy(
                rng.integers(0, cfg.vocab_size, size=(1, SEQ_LEN)).astype(np.int64))

    prof_r, vbar_r, c2_r = run_condition(model, cfg, random_iter(), device,
                                         n_layer, n_head)
    heads_r, counts_r = analyze_condition("random", prof_r, vbar_r, n_layer, n_head)

    # K1: reproduce exp-106's published 116/144 within +/-3.
    k1_ok = abs(counts_r["below_floor_at_8"] - EXP106_BELOW_AT_8) <= 3
    # stronger within-run check: bitwise-comparable profile agreement with exp-104
    saved = np.load(EXP104 / "profiles_gpt2.npz")
    prof_drift = {k: float(np.max(np.abs(prof_r[k] * N_INPUTS - saved[k])))
                  for k in ("A", "G_out")}
    print(f"\n  K1: below-floor-at-8 {counts_r['below_floor_at_8']} vs exp-106's "
          f"{EXP106_BELOW_AT_8} -> {'OK' if k1_ok else 'FAIL'}")
    print(f"  profile drift vs exp-104 saved: {prof_drift}")
    k2_r_ok = c2_r <= 1e-4
    print(f"  K2 (random): C2 identity max rel err {c2_r:.2e} -> "
          f"{'OK' if k2_r_ok else 'FAIL'}")
    if not (k1_ok and k2_r_ok):
        results.update({"verdict": "STOPPED at kill condition",
                        "k1_ok": k1_ok, "k2_random_ok": k2_r_ok,
                        "counts_random": counts_r, "heads_random": heads_r,
                        "c2_max_rel_err_random": c2_r})
        (HERE / "results_gpt2.json").write_text(json.dumps(results, indent=1))
        raise SystemExit("kill condition fired before text condition; see notes.md")

    # ---- condition 2: natural text -------------------------------------------
    print("\ncondition TEXT (TinyStories validation, EOS-joined windows)")
    windows, text_meta = build_text_windows(tokenizer)
    (HERE / "applied_text.json").write_text(json.dumps(text_meta, indent=1))

    def text_iter():
        for i in range(N_INPUTS):
            yield torch.from_numpy(windows[i:i + 1])

    prof_t, vbar_t, c2_t = run_condition(model, cfg, text_iter(), device,
                                         n_layer, n_head)
    heads_t, counts_t = analyze_condition("text", prof_t, vbar_t, n_layer, n_head)
    k2_t_ok = c2_t <= 1e-4
    print(f"  K2 (text): C2 identity max rel err {c2_t:.2e} -> "
          f"{'OK' if k2_t_ok else 'FAIL'}")

    # ---- hypothesis evaluation ------------------------------------------------
    syk_t = [r for r in heads_t if r["syk_near"]]
    k3_fired = len(syk_t) == 0
    k4_fired = (len(syk_t) < 3
                or len([r for r in heads_r if r["syk_near"]]) < 3)

    h1 = (len(syk_t) > 0 and all(r["conn_window_max"] < 0 for r in syk_t))
    h2 = (len(syk_t) > 0
          and sum(r["conn_window_min"] > 0 for r in syk_t) > len(syk_t) / 2)
    h4 = (len(syk_t) > 0
          and sum(r["conn_window_min"] < 0 < r["conn_window_max"] for r in syk_t)
          > len(syk_t) / 2)

    # H3 over all head-condition pairs; both statistics, mean-sign primary.
    def h3_stats(pairs):
        mean_agree = sum(1 for r in pairs
                         if np.sign(r["conn_window_mean"])
                         == np.sign(r["ktilde_window_mean"]))
        lag_agree = sum(1 for r in pairs
                        if (r["conn_frac_lags_positive"] > 0.5)
                        == (r["ktilde_frac_lags_positive"] > 0.5))
        return mean_agree / len(pairs), lag_agree / len(pairs)

    all_pairs = heads_r + heads_t
    h3_mean, h3_lag = h3_stats(all_pairs)
    h3 = h3_mean >= 0.80

    print("\n=== verdicts ===")
    print(f"  K3 (no SYK-near population under text): {'FIRED' if k3_fired else 'no'}")
    print(f"  K4 (<3 SYK-near heads in either condition; per-head only): "
          f"{'FIRED' if k4_fired else 'no'}")
    print(f"  H1 (negative connected profile survives text, all SYK heads): {h1}")
    print(f"  H2 (positive over window, majority): {h2}")
    print(f"  H4 (sign change inside window, majority): {h4}")
    print(f"  H3 (sign inherited from Ktilde): mean-sign agreement {h3_mean:.3f}, "
          f"majority-lag agreement {h3_lag:.3f} -> {'CONFIRMED' if h3 else 'not'}"
          f" (threshold 0.80)")
    print("  registered prediction was: H4, then H1")

    results.update({
        "k1_ok": k1_ok, "k2_random_ok": k2_r_ok, "k2_text_ok": k2_t_ok,
        "c2_max_rel_err": {"random": c2_r, "text": c2_t},
        "profile_drift_vs_exp104": prof_drift,
        "counts_random": counts_r, "counts_text": counts_t,
        "k3_fired": k3_fired, "k4_fired": k4_fired,
        "H1": h1, "H2": h2, "H4": h4,
        "H3": {"mean_sign_agreement": h3_mean,
               "majority_lag_agreement": h3_lag, "confirmed": h3},
        "text_source": text_meta,
        "heads_random": heads_r, "heads_text": heads_t,
    })
    (HERE / "results_gpt2.json").write_text(json.dumps(results, indent=1))
    np.savez_compressed(
        HERE / "profiles_gpt2.npz",
        **{f"{k}_random": prof_r[k] for k in prof_r},
        **{f"{k}_text": prof_t[k] for k in prof_t},
        vbar_sq_random=vbar_r, vbar_sq_text=vbar_t)
    print(f"\nwrote {HERE / 'results_gpt2.json'}")


if __name__ == "__main__":
    main()
