"""exp-108 — row-resolved census on GPT-2. See notes.md (pre-registered).

Frozen census input stream: 50 random-token sequences, L=512, seed 42, fp32.
Per head, per row i in [256, 511], averaged over inputs:
  near(i), sink(i), tail(i), amp(i)  as defined in notes.md.
Then evaluates registered predictions P1 (amplitude slope) and P2 (localized
mass) for the SYK-window and conformal populations defined by exp-104's
results_gpt2.json.
"""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

HERE = Path(__file__).resolve().parent
E104 = HERE.parent / "exp-104_bilocal_object_identification"

SEQ_LEN, N_INPUTS, SEED = 512, 50, 42
ROW_LO = 256
NEAR, SINKW = 8, 8
AMP_LO, AMP_HI = 8, 64

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    "gpt2", dtype=torch.float32, attn_implementation="eager").to(device).eval()
n_layer, n_head = model.config.n_layer, model.config.n_head
vocab = model.config.vocab_size

rows = np.arange(ROW_LO, SEQ_LEN)                       # (256,)
acc = {k: np.zeros((n_layer, n_head, len(rows))) for k in
       ("near", "sink", "tail", "amp")}

rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
for n in range(N_INPUTS):
    ids = torch.from_numpy(
        rng.integers(0, vocab, size=(1, SEQ_LEN)).astype(np.int64)).to(device)
    with torch.no_grad():
        out = model(ids, output_attentions=True)
    for ell in range(n_layer):
        A = out.attentions[ell][0].float().cpu().numpy()      # (H, L, L)
        assert not np.isnan(A).any()
        sub = A[:, ROW_LO:, :]                                # (H, 256, L)
        for ri, i in enumerate(rows):
            r = sub[:, ri, :]                                 # (H, L)
            near = r[:, i - NEAR + 1: i + 1].sum(axis=1)      # dx in [0, 8)
            sink = r[:, :SINKW].sum(axis=1)                   # j in [0, 8)
            tail = r[:, SINKW: i - NEAR + 1].sum(axis=1)      # everything between
            amp = np.median(r[:, i - AMP_HI: i - AMP_LO + 1], axis=1)
            acc["near"][ell, :, ri] += near
            acc["sink"][ell, :, ri] += sink
            acc["tail"][ell, :, ri] += tail
            acc["amp"][ell, :, ri] += amp
    del out
    if (n + 1) % 10 == 0:
        print(f"  {n + 1}/{N_INPUTS} inputs", flush=True)

for k in acc:
    acc[k] /= N_INPUTS

np.savez_compressed(HERE / "rows_gpt2.npz", rows=rows, **acc)

# --- evaluate registered predictions -----------------------------------------
res = json.loads((E104 / "results_gpt2.json").read_text())
heads_meta = {(h["layer"], h["head"]): h for h in res["heads"]}
syk = [(l, h) for (l, h), m in heads_meta.items()
       if m["r2_A"] is not None and m["r2_A"] >= 0.90
       and m["delta_A"] is not None and 0.20 <= m["delta_A"] <= 0.30]
conformal = [(l, h) for (l, h), m in heads_meta.items()
             if m["r2_A"] is not None and m["r2_A"] >= 0.90
             and m["delta_A"] is not None and m["delta_A"] >= 0.05]

lx = np.log(rows.astype(float))
X = np.column_stack([np.ones_like(lx), lx])

def head_report(l, h):
    m = heads_meta[(l, h)]
    s = 2.0 * m["delta_A"]
    amp = acc["amp"][l, h]
    ok = amp > 1e-15
    coef, *_ = np.linalg.lstsq(X[ok], np.log(amp[ok]), rcond=None)
    slope = float(coef[1])
    tailfrac = acc["tail"][l, h]
    loc = acc["near"][l, h] + acc["sink"][l, h]
    pred = -(1.0 - s) * float(tailfrac.mean())
    ratio_loc = float(loc.max() / max(loc.min(), 1e-12))
    return {
        "layer": l, "head": h, "delta_A": m["delta_A"], "s": s,
        "amp_slope_measured": slope, "amp_slope_predicted": pred,
        "P1_pass": bool(abs(slope - pred) <= 0.10),
        "tail_frac_mean": float(tailfrac.mean()),
        "near_frac_mean": float(acc["near"][l, h].mean()),
        "sink_frac_mean": float(acc["sink"][l, h].mean()),
        "localized_mass_median": float(np.median(loc)),
        "localized_mass_max_over_min": ratio_loc,
        "row_sum_check": float((acc["near"][l, h] + acc["sink"][l, h]
                                + acc["tail"][l, h]).mean()),
    }

out = {"protocol": {"seq_len": SEQ_LEN, "n_inputs": N_INPUTS, "seed": SEED,
                    "rows": [ROW_LO, SEQ_LEN - 1], "near": NEAR,
                    "sink_width": SINKW, "amp_window": [AMP_LO, AMP_HI]},
       "syk_window": [head_report(l, h) for l, h in sorted(syk)],
       "conformal": [head_report(l, h) for l, h in sorted(conformal)]}

p1_pass = sum(r["P1_pass"] for r in out["syk_window"])
loc_med = float(np.median([r["localized_mass_median"] for r in out["syk_window"]]))
loc_var = max(r["localized_mass_max_over_min"] for r in out["syk_window"])
out["verdicts"] = {
    "P1": f"{p1_pass}/5 SYK-window heads within ±0.10"
          + ("  -> K3-a KILL (TI model dead)" if 5 - p1_pass >= 3 else "  -> P1 PASS"),
    "P2_localized_median": loc_med,
    "P2": ("K3-b KILL (localized-mass account dead)" if loc_med < 0.05 else
           "P2 PASS" if 0.15 <= loc_med <= 0.6 and loc_var < 2.0 else
           "P2 outside band — report honestly"),
}

print(json.dumps(out["verdicts"], indent=1))
print(f"\n{'head':8s} {'dA':>6s} {'slope':>8s} {'pred':>8s} {'pass':>5s} "
      f"{'tail':>6s} {'near':>6s} {'sink':>6s} {'rowsum':>7s}")
for r in out["syk_window"]:
    print(f"L{r['layer']}H{r['head']:<4d} {r['delta_A']:6.3f} "
          f"{r['amp_slope_measured']:8.3f} {r['amp_slope_predicted']:8.3f} "
          f"{str(r['P1_pass']):>5s} {r['tail_frac_mean']:6.3f} "
          f"{r['near_frac_mean']:6.3f} {r['sink_frac_mean']:6.3f} "
          f"{r['row_sum_check']:7.4f}")

(HERE / "results_gpt2.json").write_text(json.dumps(out, indent=1))
print(f"\nwrote {HERE / 'results_gpt2.json'} and rows_gpt2.npz")
