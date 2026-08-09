"""exp-104 — Is the census's Delta the theory's Delta?

Measures, per head, the lag-profile exponent of four objects under the frozen
census protocol:

  A       the softmax attention matrix (query, key)   <- what the census fits
  G_out   <o_i, o_j>, o_i = sum_a alpha_ia v_a        <- trained bilocal, PRIMARY
  G_K     [A K A^T]_ij, K_ab = x_a . x_b              <- melonic note eq. (2.1)
  G_cos   G_out normalized to cosine                  <- amplitude control

The lag_profile and fit_head functions are IMPORTED from the published
replication kit rather than reimplemented, so the estimator is provably
identical to the one that produced every published Delta.

Pre-registration: notes.md in this folder, commit 4bb825c, before this file
existed.

Ariel — August 8, 2026.
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
KIT = HERE.parent.parent / "replication" / "measure_conformal_heads.py"


def load_kit():
    """Import the published census estimator verbatim."""
    spec = importlib.util.spec_from_file_location("census_kit", KIT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


kit = load_kit()

# Frozen protocol constants, asserted against the kit so drift is impossible.
SEQ_LEN, N_INPUTS, DEEP_LO = kit.SEQ_LEN, kit.N_INPUTS, kit.DEEP_LO
FIT_LO, FIT_HI, SEED = kit.FIT_LO, kit.FIT_HI, kit.SEED
R2_MIN, DELTA_MIN = kit.R2_MIN, kit.DELTA_MIN
assert (SEQ_LEN, N_INPUTS, DEEP_LO, FIT_LO, FIT_HI, SEED) == (512, 50, 256, 8, 256, 42), \
    "kit protocol constants changed; exp-104 comparability broken"

OBJECTS = ("A", "G_out", "G_K", "G_cos")


def head_split(t: torch.Tensor, n_head: int) -> torch.Tensor:
    """(B, L, d) -> (B, n_head, L, d_head)."""
    b, L, d = t.shape
    return t.view(b, L, n_head, d // n_head).permute(0, 2, 1, 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-inputs", type=int, default=N_INPUTS)
    ap.add_argument("--layers", default=None, help="comma list, default all")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    cfg = AutoConfig.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager")
    model = model.to(device).eval()

    n_layer = cfg.num_hidden_layers
    n_head = cfg.num_attention_heads
    vocab = cfg.vocab_size
    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else list(range(n_layer)))

    blocks = model.transformer.h if hasattr(model, "transformer") else model.gpt_neox.layers
    is_gpt2 = hasattr(model, "transformer")

    # Captured per forward pass: layer-input Gram basis and per-head value vectors.
    cap: dict[int, dict[str, torch.Tensor]] = {}

    def make_hook(ell: int):
        def pre_hook(mod, inputs):
            cap.setdefault(ell, {})["x_in"] = inputs[0].detach()

        def post_hook(mod, inputs, output):
            d = cfg.hidden_size
            v = output.detach()[..., 2 * d:3 * d]   # qkv concat -> value block
            cap.setdefault(ell, {})["v"] = head_split(v, n_head)
        return pre_hook, post_hook

    handles = []
    for ell in layers:
        attn = blocks[ell].attn if is_gpt2 else blocks[ell].attention
        qkv = attn.c_attn if is_gpt2 else attn.query_key_value
        pre, post = make_hook(ell)
        handles.append(qkv.register_forward_pre_hook(pre))
        handles.append(qkv.register_forward_hook(post))

    # Accumulated lag profiles: obj -> (n_layer, n_head, L)
    prof = {o: np.zeros((n_layer, n_head, SEQ_LEN)) for o in OBJECTS}

    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    for n in range(args.n_inputs):
        ids = torch.from_numpy(
            rng.integers(0, vocab, size=(1, SEQ_LEN)).astype(np.int64)).to(device)
        cap.clear()
        with torch.no_grad():
            out = model(ids, output_attentions=True)

        for ell in layers:
            A = out.attentions[ell][0].float()          # (n_head, L, L)
            assert not torch.isnan(A).any(), f"layer {ell}: NaN attention"

            v = cap[ell]["v"][0].float()                 # (n_head, L, d_head)
            o = A @ v                                    # (n_head, L, d_head)
            G_out = o @ o.transpose(-1, -2)              # (n_head, L, L)

            x = cap[ell]["x_in"][0].float()              # (L, d)
            K = x @ x.T                                  # (L, L)
            G_K = A @ (K.unsqueeze(0) @ A.transpose(-1, -2))

            nrm = torch.sqrt(torch.diagonal(G_out, dim1=-2, dim2=-1)).clamp_min(1e-12)
            G_cos = G_out / (nrm.unsqueeze(-1) * nrm.unsqueeze(-2))

            for name, M in (("A", A), ("G_out", G_out), ("G_K", G_K), ("G_cos", G_cos)):
                prof[name][ell] += kit.lag_profile(M.cpu().numpy())

        del out
        if (n + 1) % 10 == 0:
            print(f"  {n + 1}/{args.n_inputs} inputs", flush=True)

    for h in handles:
        h.remove()

    # Fit every object with the kit's own estimator.
    heads = []
    for ell in layers:
        for h in range(n_head):
            rec = {"layer": ell, "head": h}
            for name in OBJECTS:
                d, r2 = kit.fit_head(prof[name][ell, h])
                rec[f"delta_{name}"] = None if d is None else float(d)
                rec[f"r2_{name}"] = None if r2 is None else float(r2)
            heads.append(rec)

    tag = args.tag or args.model.replace("/", "_")
    out_path = HERE / f"results_{tag}.json"
    payload = {
        "model": args.model, "device": device,
        "n_inputs": args.n_inputs, "layers": layers,
        "protocol": {"seq_len": SEQ_LEN, "deep_lo": DEEP_LO, "seed": SEED,
                     "fit_lags": [FIT_LO, FIT_HI],
                     "estimator": "imported verbatim from replication kit"},
        "heads": heads,
    }
    out_path.write_text(json.dumps(payload, indent=1))
    np.savez_compressed(HERE / f"profiles_{tag}.npz",
                        **{name: prof[name] for name in OBJECTS})
    print(f"\nwrote {out_path}")

    summarize(heads)


def summarize(heads: list[dict]) -> None:
    def arr(key, subset):
        return np.array([h[key] for h in subset if h[key] is not None], dtype=float)

    conformal = [h for h in heads
                 if h["r2_A"] is not None and h["r2_A"] >= R2_MIN
                 and h["delta_A"] is not None and h["delta_A"] >= DELTA_MIN]
    syk_near = [h for h in conformal if abs(h["delta_A"] - 0.25) <= 0.05]

    print(f"\nheads: {len(heads)}  conformal (census criterion): {len(conformal)}"
          f"  SYK-near: {len(syk_near)}")

    for label, subset in (("ALL", heads), ("CONFORMAL", conformal), ("SYK-NEAR", syk_near)):
        if not subset:
            print(f"\n[{label}] empty")
            continue
        print(f"\n[{label}] n={len(subset)}")
        print(f"  {'object':8s} {'median Δ':>10s} {'median R²':>10s}")
        for name in OBJECTS:
            d, r = arr(f"delta_{name}", subset), arr(f"r2_{name}", subset)
            dm = f"{np.median(d):.4f}" if len(d) else "—"
            rm = f"{np.median(r):.3f}" if len(r) else "—"
            print(f"  {name:8s} {dm:>10s} {rm:>10s}")
        paired = [(h["delta_G_out"], h["delta_A"]) for h in subset
                  if h["delta_G_out"] is not None and h["delta_A"] is not None]
        if paired:
            diff = np.array([g - a for g, a in paired])
            print(f"  median(Δ_G_out − Δ_A) = {np.median(diff):+.4f}   "
                  f"IQR [{np.percentile(diff, 25):+.4f}, {np.percentile(diff, 75):+.4f}]")


if __name__ == "__main__":
    main()
