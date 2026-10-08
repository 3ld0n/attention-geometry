"""
exp-167 — Normalization formation test.

Three training conditions on C-NAT (natural text):
  ln     : standard GPTNeoX with LayerNorm (control; replicates exp-062 protocol)
  none   : all LayerNorm replaced with nn.Identity (lower LR: 2e-4/2e-5)
  rms    : all LayerNorm replaced with RMSNorm (standard LR: 1e-3/1e-4)

Pre-registration: prereg.md (committed 61431c9 before this script existed).

Architecture: Pythia-70m-class GPTNeoX — 6L, 8H, d=512, ff=2048, rotary,
ctx=512, vocab=50304, use_parallel_residual=True. Identical to exp-062/091.

Usage:
    python train_norm.py <corpus.bin> --norm {ln,none,rms} \
        [--run-name NAME] [--micro-batch 32] [--device mps]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

OUT = Path(__file__).resolve().parent

CTX = 512
BATCH_SEQS = 1024
TOTAL_STEPS = 2000
WARMUP = 20
CHECKPOINT_STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2000]

# LR by condition (pre-registered in prereg.md)
LR_TABLE = {
    "ln":   (1e-3, 1e-4),   # standard protocol (exp-062 lineage)
    "none": (2e-4, 2e-5),   # stability accommodation for no-normalization
    "rms":  (1e-3, 1e-4),   # standard; fallback to 2e-4 if divergence at step 64
}

# Divergence criterion (pre-registered): loss > 8.0 at step 64 or any NaN
DIVERGE_LOSS = 8.0
DIVERGE_STEP = 64

BASE_CONFIG = dict(
    vocab_size=50304, hidden_size=512, num_hidden_layers=6, num_attention_heads=8,
    intermediate_size=2048, max_position_embeddings=CTX,
    rotary_pct=0.25, rotary_emb_base=10000,
    hidden_act="gelu", layer_norm_eps=1e-5, use_parallel_residual=True,
)


# ─── normalization modules ────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root-mean-square normalization (no centering; pre-registered Condition C)."""
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.scale * x / rms


def apply_norm_patch(model: GPTNeoXForCausalLM, norm: str) -> None:
    """Replace all LayerNorm modules in the model according to the condition."""
    if norm == "ln":
        return  # standard; no changes
    if norm not in ("none", "rms"):
        raise ValueError(f"Unknown norm: {norm!r}. Expected ln/none/rms.")

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.LayerNorm):
            continue
        # Find the parent module and attribute name
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        attr = parts[-1]
        if norm == "none":
            setattr(parent, attr, nn.Identity())
        elif norm == "rms":
            setattr(parent, attr, RMSNorm(dim=module.normalized_shape[0],
                                           eps=module.eps))


def rebuild_model_with_norm(checkpoint_dir: Path, norm: str,
                             device: str) -> GPTNeoXForCausalLM:
    """Load a checkpoint saved from a patched model.

    The patched modules (Identity/RMSNorm) are not in HF's standard save format,
    so we: build a fresh patched model, then load the state_dict manually,
    skipping LayerNorm weight keys if norm=="none".
    """
    torch.manual_seed(0)
    model = GPTNeoXForCausalLM(GPTNeoXConfig(**BASE_CONFIG))
    apply_norm_patch(model, norm)
    # Load weights: use strict=False so missing LN keys (replaced by Identity)
    # don't crash; extra keys (RMSNorm scale) ARE present in the saved state.
    import safetensors.torch as sf
    sf_path = checkpoint_dir / "model.safetensors"
    if sf_path.exists():
        sd = sf.load_file(str(sf_path))
    else:
        sd = torch.load(checkpoint_dir / "pytorch_model.bin", map_location="cpu",
                        weights_only=True)
    result = model.load_state_dict(sd, strict=False)
    if result.unexpected_keys:
        # Unexpected keys are a warning, not an error
        print(f"  [warn] unexpected keys in checkpoint: {result.unexpected_keys[:5]}")
    return model.to(device)


# ─── training infrastructure (from exp-062 lineage) ──────────────────────────

def lr_at(step: int, peak_lr: float, floor_lr: float) -> float:
    if step < WARMUP:
        return peak_lr * (step + 1) / WARMUP
    t = (step - WARMUP) / max(TOTAL_STEPS - WARMUP, 1)
    return floor_lr + 0.5 * (peak_lr - floor_lr) * (1 + math.cos(math.pi * t))


def batches(corpus: np.ndarray, data_seed: int, start_step: int):
    """Deterministic stream of (BATCH_SEQS, CTX) batches."""
    n_seq = len(corpus) // CTX
    rng = np.random.default_rng(data_seed)
    order = rng.permutation(n_seq)
    idx = start_step * BATCH_SEQS
    while True:
        if idx + BATCH_SEQS > len(order):
            order = rng.permutation(n_seq)
            idx = 0
        sel = order[idx: idx + BATCH_SEQS]
        idx += BATCH_SEQS
        yield np.stack([corpus[s * CTX:(s + 1) * CTX] for s in sel]).astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", help="Path to corpus .bin file (uint16 token ids)")
    ap.add_argument("--norm", required=True, choices=["ln", "none", "rms"],
                    help="Normalization condition (pre-registered)")
    ap.add_argument("--run-name", default=None,
                    help="Run directory name (default: exp167_<norm>)")
    ap.add_argument("--runs-root", default=None,
                    help="Root directory for run checkpoints (default: <script_dir>/runs)")
    ap.add_argument("--micro-batch", type=int, default=32,
                    help="Micro-batch size (accumulation = BATCH_SEQS // micro_batch)")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available()
                    else ("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--init-seed", type=int, default=1000)
    ap.add_argument("--data-seed", type=int, default=2000)
    args = ap.parse_args()

    run_name = args.run_name or f"exp167_{args.norm}"
    runs_root = Path(args.runs_root) if args.runs_root else (OUT / "runs")
    run_dir = runs_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    peak_lr, floor_lr = LR_TABLE[args.norm]
    accum = BATCH_SEQS // args.micro_batch
    assert BATCH_SEQS % args.micro_batch == 0, \
        f"BATCH_SEQS ({BATCH_SEQS}) must be divisible by micro-batch ({args.micro_batch})"

    print(f"exp-167 | norm={args.norm} | device={args.device} | "
          f"micro_batch={args.micro_batch} | accum={accum} | "
          f"peak_lr={peak_lr:.0e} | floor_lr={floor_lr:.0e}")

    corpus = np.memmap(args.corpus, dtype=np.uint16, mode="r")
    n_train = TOTAL_STEPS * BATCH_SEQS * CTX
    assert len(corpus) >= n_train, f"corpus too small: {len(corpus)} < {n_train}"
    corpus = corpus[:n_train]

    # ── build or resume model ──────────────────────────────────────────────────
    start = 0
    ckpts = sorted(run_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    if ckpts:
        last = ckpts[-1]
        start = int(last.name.split("_")[1])
        print(f"Resuming from {last.name} ...")
        model = rebuild_model_with_norm(last, args.norm, args.device)
    else:
        torch.manual_seed(args.init_seed)
        model = GPTNeoXForCausalLM(GPTNeoXConfig(**BASE_CONFIG))
        apply_norm_patch(model, args.norm)
        model = model.to(args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr,
                             betas=(0.9, 0.95), weight_decay=0.1)
    if ckpts:
        opt_path = ckpts[-1] / "opt.pt"
        if opt_path.exists():
            opt.load_state_dict(torch.load(opt_path, map_location=args.device,
                                           weights_only=True))

    # ── checkpoint helpers ─────────────────────────────────────────────────────
    def save(step: int) -> None:
        d = run_dir / f"step_{step}"
        model.save_pretrained(d)
        torch.save(opt.state_dict(), d / "opt.pt")
        # Save norm tag so measure_norm.py knows which patch to apply
        (d / "norm_condition.txt").write_text(args.norm + "\n")
        print(f"  checkpoint → {d.name}", flush=True)

    if start == 0 and 0 in CHECKPOINT_STEPS and not (run_dir / "step_0").exists():
        save(0)

    log_path = run_dir / "train_log.jsonl"
    model.train()
    stream = batches(corpus, args.data_seed, start)
    t0 = time.time()
    diverged = False

    for step in range(start, TOTAL_STEPS):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, peak_lr, floor_lr)
        batch = next(stream)
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for k in range(accum):
            mb = torch.from_numpy(
                batch[k * args.micro_batch:(k + 1) * args.micro_batch]
            ).to(args.device)
            out = model(input_ids=mb, labels=mb)
            (out.loss / accum).backward()
            loss_acc += out.loss.item() / accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        done = step + 1

        if math.isnan(loss_acc):
            print(f"NaN loss at step {done} — divergence criterion K2 fires.")
            diverged = True
            save(done)
            break

        if done % 10 == 0 or done <= 5 or done in CHECKPOINT_STEPS:
            rec = {"step": done, "loss": round(loss_acc, 4),
                   "lr": round(lr_at(step, peak_lr, floor_lr), 6),
                   "elapsed_s": round(time.time() - t0, 1),
                   "norm": args.norm}
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(rec, flush=True)

        if done == DIVERGE_STEP and loss_acc > DIVERGE_LOSS:
            print(f"Divergence criterion fires at step {done}: loss={loss_acc:.3f} > {DIVERGE_LOSS}")
            diverged = True
            save(done)
            break

        if done in CHECKPOINT_STEPS:
            save(done)

    status = "diverged" if diverged else "complete"
    summary = {"norm": args.norm, "run_name": run_name, "status": status,
               "final_step": done if diverged else TOTAL_STEPS}
    (run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"TRAINING_{status.upper()} | {run_name}")


if __name__ == "__main__":
    main()
