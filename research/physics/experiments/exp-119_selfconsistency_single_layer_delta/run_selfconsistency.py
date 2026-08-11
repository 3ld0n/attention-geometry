"""exp-119 — Theory-of-A Level 3: self-consistency of the single-layer attention delta.

Pre-registration committed to attention-geometry at c800adc before this script
was written or run.

Measures position-correlation structure of single-layer head output (pre-W_O)
for structural and semantic heads under random-token census protocol.

Two measures:
  A (within-input): C_within(dx) = mean over inputs of within-sequence corr
  B (mean-field):   C_mf(dx)     = correlation of the mean-over-inputs output

Ariel — 2026-08-11.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

HERE = Path(__file__).resolve().parent

SEED = 42
N_INPUTS = 50
SEQ_LEN = 512
VOCAB_SIZE = 50257  # GPT-2 vocab
FIT_LO, FIT_HI = 8, 256
QUERY_MIN = 256  # require query position > QUERY_MIN for lag profiles

# Structural heads (random-token census, exp-107/109)
STRUCTURAL = [(2, 1), (3, 4), (5, 0), (7, 11), (10, 8)]

# 5 representative semantic heads (text-native census, exp-109: 16 total)
# Using the 5 deepest (highest-layer) semantic heads from exp-109
SEMANTIC = [(10, 3), (10, 7), (11, 0), (11, 5), (11, 8)]

ALL_HEADS = [("structural", l, h) for (l, h) in STRUCTURAL] + \
            [("semantic", l, h) for (l, h) in SEMANTIC]


def ols_loglog(y: np.ndarray, x: np.ndarray) -> tuple[float, float, float]:
    """OLS slope in log-log space. Returns (slope, intercept, R2)."""
    lx = np.log(x.astype(float))
    ly = np.log(np.abs(y.astype(float)) + 1e-30)
    X = np.column_stack([np.ones_like(lx), lx])
    c, *_ = np.linalg.lstsq(X, ly, rcond=None)
    ly_pred = X @ c
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    ss_res = float(((ly - ly_pred) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return float(c[1]), float(c[0]), r2


def main() -> None:
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    print("exp-119: single-layer attention delta self-consistency", flush=True)
    print(f"Pre-reg commit: c800adc  |  N={N_INPUTS} × {SEQ_LEN} tokens", flush=True)
    print(flush=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)
    model.eval()
    model = model.to(device)

    # Collect (layer, head) pairs we need
    needed = {(l, h) for _, l, h in ALL_HEADS}
    # {(l,h): list of head_out arrays per input, each shape (seq, d_head)}
    head_outs: dict[tuple, list] = {key: [] for key in needed}

    print("Running forward passes...", flush=True)
    with torch.no_grad():
        for n in range(N_INPUTS):
            ids = torch.from_numpy(
                rng.integers(0, VOCAB_SIZE, size=(1, SEQ_LEN), dtype=np.int64)
            ).to(device)

            # Hook to capture attention weights
            captured: dict[int, torch.Tensor] = {}

            def make_hook(layer_idx: int):
                def hook(module, inp, out):
                    # out[1] is the attention weights if output_attentions=True
                    if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                        captured[layer_idx] = out[1].detach().cpu()
                return hook

            hooks = []
            for l_idx in set(l for l, _ in needed):
                layer = model.transformer.h[l_idx].attn
                hooks.append(layer.register_forward_hook(make_hook(l_idx)))

            out = model(ids, output_attentions=True)

            for hook in hooks:
                hook.remove()

            # Compute head_out from attention weights and value projections
            # We need access to the value vectors. Re-run with hooks on the attn module's
            # value computation. Easier: use attention weight matrices from captured.
            #
            # GPT-2: attn module has c_attn (QKV projection) and c_proj (W_O).
            # We extract V directly from the module's last computed value.
            # Since GPT-2 computes Q,K,V in one matmul, we do it manually here.

            # Layer-by-layer forward, collecting per-head outputs.
            h_state_manual = (
                model.transformer.wte(ids) +
                model.transformer.wpe(torch.arange(SEQ_LEN, device=device).unsqueeze(0))
            )

            if n == 0:
                d_head = 768 // 12  # 64
                n_head = 12

            for l_idx in range(model.config.n_layer):
                block = model.transformer.h[l_idx]
                # Compute QKV
                residual = h_state_manual
                h_ln = block.ln_1(h_state_manual)
                qkv = block.attn.c_attn(h_ln)  # (1, seq, 3*d_model)
                q, k, v = qkv.split(768, dim=-1)  # each (1, seq, d_model)

                # Reshape to (1, n_head, seq, d_head)
                q = q.view(1, SEQ_LEN, 12, 64).transpose(1, 2)
                k = k.view(1, SEQ_LEN, 12, 64).transpose(1, 2)
                v = v.view(1, SEQ_LEN, 12, 64).transpose(1, 2)

                # Attention weights (causal mask applied by GPT-2 internals)
                scale = 64 ** -0.5
                attn = torch.matmul(q, k.transpose(-1, -2)) * scale
                # Apply causal mask
                mask = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN, device=device)).bool()
                attn = attn.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                attn = torch.softmax(attn, dim=-1)
                attn = torch.nan_to_num(attn, nan=0.0)

                # Compute head outputs: (1, n_head, seq, d_head)
                head_out_tensor = torch.matmul(attn, v)

                if l_idx in [l for l, _ in needed]:
                    for (l_need, h_need) in needed:
                        if l_need == l_idx:
                            # head_out_tensor[0, h_need, :, :] shape (seq, d_head)
                            head_outs[(l_need, h_need)].append(
                                head_out_tensor[0, h_need, :, :].float().cpu().numpy()
                            )

                # Continue forward: project heads, add residual, MLP
                # head_out: (1, seq, d_model) — concat and project
                head_cat = head_out_tensor.transpose(1, 2).reshape(1, SEQ_LEN, 768)
                attn_out = block.attn.c_proj(head_cat)
                h_state_manual = residual + attn_out
                # MLP
                h_state_manual = h_state_manual + block.mlp(block.ln_2(h_state_manual))

            if (n + 1) % 10 == 0:
                print(f"  input {n+1}/{N_INPUTS}", flush=True)

    print("\nComputing correlations...", flush=True)
    lags = np.arange(FIT_LO, FIT_HI + 1)

    results_by_head = {}

    for pop, l_idx, h_idx in ALL_HEADS:
        arr = np.stack(head_outs[(l_idx, h_idx)], axis=0)  # (N, seq, d_head)
        # Queries: positions > QUERY_MIN
        q_positions = np.arange(QUERY_MIN, SEQ_LEN)

        # Measure A: within-input correlations.
        # Use backward lags (consistent with census): A(i, i-dx) where i >= QUERY_MIN.
        # So valid queries at lag dx: q in [QUERY_MIN, SEQ_LEN) with q-dx >= 0.
        c_within = np.zeros(len(lags))
        for dx_i, dx in enumerate(lags):
            valid_q = q_positions[q_positions - dx >= 0]
            if len(valid_q) == 0:
                continue
            # For each input: mean_i(head_out[i] · head_out[i-dx])
            # arr: (N, seq, d_head)
            per_input = np.mean(
                np.sum(arr[:, valid_q, :] * arr[:, valid_q - dx, :], axis=-1),
                axis=1
            )  # shape (N,)
            c_within[dx_i] = float(per_input.mean())

        sigma_within, intercept_within, r2_within = ols_loglog(c_within, lags)

        # Measure B: mean-field (mean over inputs first, then correlation)
        head_mean = arr.mean(axis=0)  # (seq, d_head)
        c_mf = np.zeros(len(lags))
        for dx_i, dx in enumerate(lags):
            valid_q = q_positions[q_positions - dx >= 0]
            if len(valid_q) == 0:
                continue
            c_mf[dx_i] = float(np.mean(
                np.sum(head_mean[valid_q, :] * head_mean[valid_q - dx, :], axis=-1)
            ))

        # Check if mf has any meaningful signal (may be near-zero for random tokens)
        mf_range = float(c_mf.max() - c_mf.min())
        if mf_range > 1e-6 and c_mf.min() > 0:
            sigma_mf, intercept_mf, r2_mf = ols_loglog(c_mf, lags)
        else:
            sigma_mf, intercept_mf, r2_mf = float('nan'), float('nan'), 0.0

        key = f"L{l_idx}H{h_idx}"
        results_by_head[key] = {
            "population": pop,
            "layer": l_idx,
            "head": h_idx,
            "sigma_within": round(sigma_within, 4),
            "r2_within": round(r2_within, 4),
            "sigma_mf_single": round(sigma_mf, 4) if not np.isnan(sigma_mf) else None,
            "r2_mf_single": round(r2_mf, 4),
            "c_within_profile": [round(float(v), 6) for v in c_within],
            "c_mf_profile": [round(float(v), 6) for v in c_mf],
            "mf_mean_magnitude": round(float(np.abs(head_mean).mean()), 6),
        }

        print(
            f"  {key} ({pop:10s}): σ_within={sigma_within:+.3f} R²={r2_within:.2f} | "
            f"σ_mf={sigma_mf:+.3f} R²={r2_mf:.2f}",
            flush=True
        )

    elapsed = time.time() - t0

    # Summary statistics
    struct_sigmas = [results_by_head[f"L{l}H{h}"]["sigma_within"]
                     for _, l, h in ALL_HEADS if _ == "structural"]
    sem_sigmas = [results_by_head[f"L{l}H{h}"]["sigma_within"]
                  for _, l, h in ALL_HEADS if _ == "semantic"]
    struct_r2s = [results_by_head[f"L{l}H{h}"]["r2_within"]
                  for _, l, h in ALL_HEADS if _ == "structural"]
    n_powerlaw_structural = sum(1 for r2 in struct_r2s if r2 >= 0.70)

    print(f"\n=== SUMMARY ===", flush=True)
    print(f"Structural σ_within: {struct_sigmas}", flush=True)
    print(f"Structural R²: {struct_r2s}", flush=True)
    print(f"Semantic σ_within: {sem_sigmas}", flush=True)
    print(f"n_powerlaw_structural (R²≥0.70): {n_powerlaw_structural}/5", flush=True)
    print(f"Elapsed: {elapsed:.1f}s", flush=True)

    # P1 verdict
    p1_confirmed = n_powerlaw_structural >= 3
    p1_verdict = "CONFIRMED" if p1_confirmed else "DEAD"
    # P2: majority with σ ≈ 0.25
    p2_close = sum(1 for s in struct_sigmas if 0.15 < s < 0.35)
    # P3: mean-field flat
    mf_sigmas = [results_by_head[f"L{l}H{h}"]["sigma_mf_single"]
                 for _, l, h in ALL_HEADS if _ == "structural"
                 and results_by_head[f"L{l}H{h}"]["sigma_mf_single"] is not None]
    p3_flat = all(abs(s) < 0.10 for s in mf_sigmas) if mf_sigmas else True

    print(f"\nP1 ({p1_verdict}): n_powerlaw_structural={n_powerlaw_structural}/5 (need ≥3 with R²≥0.70)", flush=True)
    print(f"P2 (exploratory): {p2_close}/5 structural heads with σ_within ∈ (0.15, 0.35)", flush=True)
    print(f"P3 (mean-field flat ≤0.10): {'CONFIRMED' if p3_flat else 'FAILS'} — mf σ={mf_sigmas}", flush=True)

    output = {
        "exp": "exp-119",
        "prereg_commit": "c800adc",
        "date": "2026-08-11",
        "protocol": {
            "N_inputs": N_INPUTS,
            "seq_len": SEQ_LEN,
            "fit_range": [FIT_LO, FIT_HI],
            "query_min": QUERY_MIN,
            "random_seed": SEED,
        },
        "heads": results_by_head,
        "summary": {
            "structural_sigma_within": struct_sigmas,
            "structural_r2_within": struct_r2s,
            "semantic_sigma_within": sem_sigmas,
            "n_powerlaw_structural": n_powerlaw_structural,
            "P1_verdict": p1_verdict,
            "P2_close_to_delta": p2_close,
            "P3_mf_flat": p3_flat,
        },
        "elapsed_s": round(elapsed, 1),
    }

    out_path = HERE / "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults → {out_path}", flush=True)


if __name__ == "__main__":
    main()
