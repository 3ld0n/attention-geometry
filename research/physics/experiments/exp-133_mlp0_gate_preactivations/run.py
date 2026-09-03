"""exp-133 — MLP block-0 gate pre-activations mechanism.

Pre-registration committed to attention-geometry at 3694062 before this script
was written or any forward passes run.

Question: is MLP block-0's amplification (σ input 0.144 → σ output 0.313,
exp-132) gate-driven — i.e., does W_fc's linear projection already carry
σ ≥ 0.313 before GeLU? Or does the amplification occur at GeLU or W_proj?

GPT-2 MLP block structure:
    pre_act = c_fc(h^(0.5))        # linear up-projection: 768 → 3072
    h_gelu  = GeLU(pre_act)        # element-wise nonlinearity
    mlp_out = c_proj(h_gelu)       # linear down-projection: 3072 → 768

Ariel — 2026-09-03.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

HERE = Path(__file__).resolve().parent
EXP112 = HERE.parent / "exp-112_score_drift_decomposition"

spec = importlib.util.spec_from_file_location("exp112", EXP112 / "measure_scores.py")
sys.path.insert(0, str(EXP112.parent / "exp-107_natural_text_bilocal"))
exp112 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp112)

pooled_window_profile = exp112.pooled_window_profile
ols_slope = exp112.ols_slope
WINDOW = exp112.WINDOW    # lags 8..256, 249 elements

SEQ_LEN = 512
N_SEQS = 50
SEED = 42
VOCAB_SIZE = 50257
PREREG_COMMIT = "3694062"


def r2_of_slope(profile: np.ndarray, lags: np.ndarray) -> float:
    lx = np.log(lags.astype(float))
    X = np.column_stack([np.ones_like(lx), lx])
    c, *_ = np.linalg.lstsq(X, profile, rcond=None)
    y_pred = X @ c
    ss_tot = float(((profile - profile.mean()) ** 2).sum())
    ss_res = float(((profile - y_pred) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


def sigma_meanfirst(mean_arr: np.ndarray) -> tuple[float, float, float]:
    """σ from mean-over-sequences position array (mean-first protocol)."""
    norms = np.linalg.norm(mean_arr, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    M = mean_arr / norms
    C = M @ M.T
    profile = pooled_window_profile(C)
    sigma = -ols_slope(profile, WINDOW)
    r2 = r2_of_slope(profile, WINDOW)
    c8 = float(profile[0])
    return float(sigma), float(r2), c8


def main() -> None:
    print("exp-133: MLP Block-0 Gate Pre-Activations Mechanism", flush=True)
    print(f"Pre-registration commit: {PREREG_COMMIT}\n", flush=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        "gpt2", dtype=torch.float32,
        attn_implementation="eager").to(device).eval()

    d_model = 768
    d_inner = 3072  # 4 * d_model

    # Random-token census — identical protocol to exp-131/exp-132
    rng = np.random.RandomState(SEED)
    seqs_np = rng.randint(0, VOCAB_SIZE, size=(N_SEQS, SEQ_LEN))
    seqs = torch.tensor(seqs_np, dtype=torch.long, device=device)
    print(f"Random-token sequences: {N_SEQS} × {SEQ_LEN}  seed={SEED}", flush=True)

    # Mean-first accumulators
    # h0p5: h^(0.5) = h^(0) + attn_out^(0), shape (SEQ_LEN, d_model)
    # pre_act: c_fc output before GeLU, shape (SEQ_LEN, d_inner)
    # h_gelu: after GeLU = input to c_proj, shape (SEQ_LEN, d_inner)
    # mlp_out: c_proj output = block-0 MLP write, shape (SEQ_LEN, d_model)
    acc_h0     = np.zeros((SEQ_LEN, d_model), dtype=np.float64)
    acc_attn0  = np.zeros((SEQ_LEN, d_model), dtype=np.float64)
    acc_preact = np.zeros((SEQ_LEN, d_inner), dtype=np.float64)
    acc_hgelu  = np.zeros((SEQ_LEN, d_inner), dtype=np.float64)
    acc_mlp0   = np.zeros((SEQ_LEN, d_model), dtype=np.float64)

    handles = []
    _buf: dict[str, np.ndarray] = {}

    def hook_h0(module, args):
        """Capture block-0 input (embedding = token + position embeddings)."""
        _buf["h0"] = args[0].detach().float().cpu().numpy()[0]

    def hook_attn0(module, args, output):
        """Capture block-0 attention output write."""
        out = output[0] if isinstance(output, tuple) else output
        _buf["attn0"] = out.detach().float().cpu().numpy()[0]

    def hook_preact(module, args, output):
        """Capture c_fc output = pre-GeLU activations."""
        out = output if not isinstance(output, tuple) else output[0]
        _buf["preact"] = out.detach().float().cpu().numpy()[0]

    def hook_hgelu(module, args):
        """Capture c_proj input = post-GeLU activations (h_gelu)."""
        _buf["hgelu"] = args[0].detach().float().cpu().numpy()[0]

    def hook_mlp0(module, args, output):
        """Capture block-0 MLP output write."""
        out = output if not isinstance(output, tuple) else output[0]
        _buf["mlp0"] = out.detach().float().cpu().numpy()[0]

    handles.append(model.transformer.h[0].register_forward_pre_hook(hook_h0))
    handles.append(model.transformer.h[0].attn.register_forward_hook(hook_attn0))
    handles.append(model.transformer.h[0].mlp.c_fc.register_forward_hook(hook_preact))
    handles.append(model.transformer.h[0].mlp.c_proj.register_forward_pre_hook(hook_hgelu))
    handles.append(model.transformer.h[0].mlp.register_forward_hook(hook_mlp0))

    with torch.no_grad():
        for i, seq in enumerate(seqs):
            if (i + 1) % 10 == 0:
                print(f"  Sequence {i+1}/{N_SEQS}...", flush=True)
            _buf.clear()
            model(seq.unsqueeze(0))

            acc_h0    += _buf["h0"].astype(np.float64)    / N_SEQS
            acc_attn0 += _buf["attn0"].astype(np.float64) / N_SEQS
            acc_preact += _buf["preact"].astype(np.float64) / N_SEQS
            acc_hgelu  += _buf["hgelu"].astype(np.float64)  / N_SEQS
            acc_mlp0   += _buf["mlp0"].astype(np.float64)   / N_SEQS

    for h in handles:
        h.remove()

    # h^(0.5) = h^(0) + attn_out^(0)
    acc_h0p5 = acc_h0 + acc_attn0

    print("\n=== Mean-first cosine protocol (σ measurements) ===", flush=True)

    def report(label: str, arr: np.ndarray) -> tuple[float, float, float]:
        sigma, r2, c8 = sigma_meanfirst(arr)
        print(f"  {label:52s}  σ={sigma:+.4f}  R²={r2:.4f}  C[8]={c8:.3f}",
              flush=True)
        return sigma, r2, c8

    sig_h0,    r2_h0,    c8_h0    = report("h^(0)      [embedding]", acc_h0)
    sig_attn0, r2_attn0, c8_attn0 = report("attn_out^(0) [block-0 attn write]", acc_attn0)
    sig_h0p5,  r2_h0p5,  c8_h0p5  = report("h^(0.5)    [MLP input = h^(0)+attn0]", acc_h0p5)
    sig_preact, r2_preact, c8_preact = report("pre_act    [c_fc output, pre-GeLU, 3072-dim]", acc_preact)
    sig_hgelu,  r2_hgelu,  c8_hgelu  = report("h_gelu     [post-GeLU, c_proj input, 3072-dim]", acc_hgelu)
    sig_mlp0,  r2_mlp0,  c8_mlp0   = report("mlp_out^(0) [c_proj output, MLP write]", acc_mlp0)

    # Registered predictions
    exp132_sigma_h0p5 = 0.1441
    exp132_sigma_mlp0 = 0.3125
    threshold_gate    = 0.313   # P1 criterion

    P1_ok = sig_preact >= threshold_gate
    K1a   = (sig_preact < threshold_gate) and (sig_hgelu >= threshold_gate)
    K1b   = (sig_preact < threshold_gate) and (sig_hgelu < threshold_gate)
    K1    = sig_preact < threshold_gate

    # Reproduction check (not a registered prediction, diagnostic)
    h0p5_match = abs(sig_h0p5 - exp132_sigma_h0p5) < 0.03
    mlp0_match = abs(sig_mlp0 - exp132_sigma_mlp0) < 0.04

    print(f"\n=== Registered predictions ===", flush=True)
    print(f"  P1 (σ(pre_act) ≥ {threshold_gate}):  σ={sig_preact:.4f}  → {'CONFIRMED' if P1_ok else 'FALSIFIED (K1)'}", flush=True)
    if K1a:
        print(f"  K1a: GeLU amplifier — σ(h_gelu)={sig_hgelu:.4f} ≥ {threshold_gate}", flush=True)
    elif K1b:
        print(f"  K1b: W_proj amplifier — σ(h_gelu)={sig_hgelu:.4f} < {threshold_gate}; "
              f"σ jumps from {sig_hgelu:.4f} → {sig_mlp0:.4f} in W_proj", flush=True)

    print(f"\n  [diagnostic] h^(0.5) σ≈0.144 (exp-132): {sig_h0p5:.4f}  → {'match' if h0p5_match else 'MISMATCH'}", flush=True)
    print(f"  [diagnostic] mlp_out σ≈0.313 (exp-132): {sig_mlp0:.4f}  → {'match' if mlp0_match else 'MISMATCH'}", flush=True)

    # Verdict
    if not (h0p5_match and mlp0_match):
        overall = "inconclusive"
        note = "exp-132 not reproduced; protocol issue"
    elif P1_ok:
        if sig_hgelu >= sig_preact - 0.02:
            overall = "confirmed"
            note = f"Gate-driven: W_fc linear projection delivers σ(pre_act)={sig_preact:.4f} ≥ threshold; GeLU passes through"
        else:
            overall = "partial"
            note = f"W_fc partially gate-driven (σ={sig_preact:.4f} ≥ threshold), GeLU attenuates to {sig_hgelu:.4f}"
    elif K1a:
        overall = "inconclusive"
        note = f"K1a: GeLU nonlinear selection — σ(pre_act)={sig_preact:.4f} < threshold but σ(h_gelu)={sig_hgelu:.4f} ≥ threshold"
    elif K1b:
        overall = "inconclusive"
        note = f"K1b: W_proj amplifier — σ jumps at c_proj from {sig_hgelu:.4f} to {sig_mlp0:.4f}"
    else:
        overall = "inconclusive"
        note = f"K1 fired; sub-case unclear: σ(pre_act)={sig_preact:.4f}, σ(h_gelu)={sig_hgelu:.4f}, σ(mlp_out)={sig_mlp0:.4f}"

    print(f"\n  Overall verdict: {overall.upper()} — {note}", flush=True)

    results = {
        "exp": "exp-133",
        "date": "2026-09-03",
        "prereg_commit": PREREG_COMMIT,
        "model": "gpt2",
        "n_seqs": N_SEQS,
        "seq_len": SEQ_LEN,
        "seed": SEED,
        "data": "random-token census (uniform [0, 50257), seed=42)",
        "protocol": "mean-first cosine, same as exp-131/exp-132",
        "sigma_results": {
            "h0":     {"sigma": sig_h0,    "r2": r2_h0,    "c8": c8_h0,    "dim": d_model},
            "attn0":  {"sigma": sig_attn0, "r2": r2_attn0, "c8": c8_attn0, "dim": d_model},
            "h0p5":   {"sigma": sig_h0p5,  "r2": r2_h0p5,  "c8": c8_h0p5,  "dim": d_model,
                       "note": "MLP input = h^(0) + attn_out^(0)"},
            "pre_act":{"sigma": sig_preact,"r2": r2_preact,"c8": c8_preact,"dim": d_inner,
                       "note": "c_fc output, before GeLU"},
            "h_gelu": {"sigma": sig_hgelu, "r2": r2_hgelu, "c8": c8_hgelu, "dim": d_inner,
                       "note": "post-GeLU activations = c_proj input"},
            "mlp0":   {"sigma": sig_mlp0,  "r2": r2_mlp0,  "c8": c8_mlp0,  "dim": d_model,
                       "note": "c_proj output = block-0 MLP write"},
        },
        "exp132_reference": {
            "sigma_h0p5": exp132_sigma_h0p5,
            "sigma_mlp0": exp132_sigma_mlp0,
        },
        "registered_predictions": {
            "P1": {
                "criterion": f"sigma(pre_act) >= {threshold_gate}",
                "sigma_preact": float(sig_preact),
                "result": "confirmed" if P1_ok else "falsified",
            },
        },
        "kill_conditions": {
            "K1":  {"fired": K1,  "criterion": f"sigma(pre_act) < {threshold_gate}"},
            "K1a": {"fired": K1a, "criterion": "K1 + sigma(h_gelu) >= threshold → GeLU amplifier"},
            "K1b": {"fired": K1b, "criterion": "K1 + sigma(h_gelu) < threshold → W_proj amplifier"},
        },
        "diagnostic": {
            "h0p5_matches_exp132": h0p5_match,
            "mlp0_matches_exp132": mlp0_match,
        },
        "overall_verdict": overall,
        "verdict_note": note,
    }

    out = HERE / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()
