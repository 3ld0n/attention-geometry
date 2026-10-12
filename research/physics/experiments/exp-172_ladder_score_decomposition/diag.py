"""exp-172 diagnostics (post-hoc, exploratory): how big is the position-dependent part of q,k
relative to the constant part, and does the mean-field slope match the census slope?"""
import json, numpy as np, torch
from pathlib import Path
from transformers import GPTNeoXForCausalLM

res = json.loads(Path("results/results.json").read_text())
# (1) mean-field sigma vs census attention sigma (=2*Delta)
rat = []
for m in res["models"]:
    for h in m["heads"]:
        if h["sigma_census_attention"]:
            rat.append((m["tag"], h["layer"], h["head"], h["sigma"]["full"] / h["sigma_census_attention"]))
r = np.array([x[3] for x in rat])
print(f"mean-field / census slope ratio over {len(r)} heads: median {np.median(r):.3f}  range {r.min():.3f}-{r.max():.3f}")

# (2) magnitude of position-dependent part of q,k
L, N = 512, 20
for tag, layers in (("CNAT_s0", (0, 3, 4, 5)), ("CNATshuf_s0", (0, 3, 4))):
    model = GPTNeoXForCausalLM.from_pretrained(f"/tmp/ladder_ckpt/{tag}", torch_dtype=torch.float32).eval()
    nh, hs = model.config.num_attention_heads, model.config.hidden_size // model.config.num_attention_heads
    cur, hooks = {}, []
    for l in layers:
        hooks.append(model.gpt_neox.layers[l].attention.query_key_value.register_forward_hook(
            lambda mod, i, o, l=l: cur.__setitem__(l, o.detach())))
    acc = {l: np.zeros((nh, L, 2 * hs)) for l in layers}
    rng = np.random.default_rng(42)
    for _ in range(N):
        ids = torch.tensor(rng.choice(model.config.vocab_size, size=L)[None])
        with torch.no_grad():
            model(ids)
        for l in layers:
            qkv = cur[l][0].view(L, nh, 3 * hs).permute(1, 0, 2).double().numpy()
            acc[l] += qkv[..., :2 * hs]
    for l in layers:
        a = acc[l] / N
        q, k = a[..., :hs], a[..., hs:]
        # exclude first 8 positions (sink / start effects) for the 'bulk' number
        for name, x in (("q", q), ("k", k)):
            m = x.mean(1, keepdims=True); d = x - m
            tot = np.linalg.norm(d, axis=-1).mean(1) / np.linalg.norm(m[:, 0], axis=-1)
            bulk = np.linalg.norm(d[:, 8:], axis=-1).mean(1) / np.linalg.norm(m[:, 0], axis=-1)
            print(f"{tag} L{l} {name}: |delta|/|mean|  all-pos median over heads {np.median(tot):.3f}; pos>=8 {np.median(bulk):.3f}; max-head {tot.max():.3f}")
    for h in hooks: h.remove()
