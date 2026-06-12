"""
exp-062 — Training loop (Phase 1.1, pre-registered).

Pre-registration of record: notes/2026-06-11_corpus_statistics_preregistration.md.
Fixed there (prereg §3): Pythia-70m-class GPT-NeoX (6L, 8H, d=512, ff=2048,
rotary, vocab 50304), context 512; AdamW (0.9, 0.95, wd 0.1), peak LR 1e-3,
cosine to 1e-4, 20-step warmup, clip 1.0; batch 1024 x 512 = 524,288
tokens/step; 2,000 steps = 1.05B tokens; checkpoints at the listed steps.
bf16 autocast permitted for training; ALL measurement is fp32 (measure.py).

Usage:
  python train.py <corpus.bin> <run_name> [--init-seed 1000] [--data-seed 2000] \
                  [--micro-batch 64] [--device cuda]

Effective batch is fixed at 1024 sequences; micro-batch only sets gradient
accumulation. Resumable: restarts from the latest checkpoint in runs/<run_name>.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

OUT = Path(__file__).resolve().parent

CTX = 512
BATCH_SEQS = 1024
TOTAL_STEPS = 2000
PEAK_LR, FLOOR_LR = 1e-3, 1e-4
WARMUP = 20
CHECKPOINT_STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2000]

CONFIG = dict(
    vocab_size=50304, hidden_size=512, num_hidden_layers=6, num_attention_heads=8,
    intermediate_size=2048, max_position_embeddings=CTX,
    rotary_pct=0.25, rotary_emb_base=10000,
    hidden_act="gelu", layer_norm_eps=1e-5, use_parallel_residual=True,
)


def lr_at(step: int) -> float:
    if step < WARMUP:
        return PEAK_LR * (step + 1) / WARMUP
    t = (step - WARMUP) / max(TOTAL_STEPS - WARMUP, 1)
    return FLOOR_LR + 0.5 * (PEAK_LR - FLOOR_LR) * (1 + math.cos(math.pi * t))


def batches(corpus: np.ndarray, data_seed: int, start_step: int):
    """Deterministic stream of (BATCH_SEQS, CTX) batches; reproducible resume."""
    n_seq = len(corpus) // CTX
    rng = np.random.default_rng(data_seed)
    order = rng.permutation(n_seq)
    idx = start_step * BATCH_SEQS
    while True:
        if idx + BATCH_SEQS > len(order):  # one epoch is ~2M seqs > needed; guard anyway
            order = rng.permutation(n_seq)
            idx = 0
        sel = order[idx: idx + BATCH_SEQS]
        idx += BATCH_SEQS
        yield np.stack([corpus[s * CTX:(s + 1) * CTX] for s in sel]).astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("run_name")
    ap.add_argument("--init-seed", type=int, default=1000)
    ap.add_argument("--data-seed", type=int, default=2000)
    ap.add_argument("--micro-batch", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    assert BATCH_SEQS % args.micro_batch == 0
    accum = BATCH_SEQS // args.micro_batch

    run_dir = OUT / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    corpus = np.memmap(args.corpus, dtype=np.uint16, mode="r")
    n_train = TOTAL_STEPS * BATCH_SEQS * CTX
    assert len(corpus) >= n_train, f"corpus too small: {len(corpus)} < {n_train}"
    corpus = corpus[:n_train]  # held-out tail stays untouched

    torch.manual_seed(args.init_seed)
    model = GPTNeoXForCausalLM(GPTNeoXConfig(**CONFIG)).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR,
                            betas=(0.9, 0.95), weight_decay=0.1)

    start = 0
    ckpts = sorted(run_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    if ckpts:
        last = ckpts[-1]
        start = int(last.name.split("_")[1])
        model = GPTNeoXForCausalLM.from_pretrained(last).to(args.device)
        opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR,
                                betas=(0.9, 0.95), weight_decay=0.1)
        opt_state = last / "opt.pt"
        if opt_state.exists():
            opt.load_state_dict(torch.load(opt_state, map_location=args.device))
        print(f"resumed from step {start}")

    def save(step: int) -> None:
        d = run_dir / f"step_{step}"
        model.save_pretrained(d)
        torch.save(opt.state_dict(), d / "opt.pt")
        print(f"  checkpoint -> {d.name}", flush=True)

    if start == 0 and 0 in CHECKPOINT_STEPS and not (run_dir / "step_0").exists():
        save(0)

    use_bf16 = args.device == "cuda" and torch.cuda.is_bf16_supported()
    log_path = run_dir / "train_log.jsonl"
    model.train()
    stream = batches(corpus, args.data_seed, start)
    t0 = time.time()
    for step in range(start, TOTAL_STEPS):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        batch = next(stream)
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for k in range(accum):
            mb = torch.from_numpy(batch[k * args.micro_batch:(k + 1) * args.micro_batch]
                                  ).to(args.device)
            with torch.autocast(args.device, dtype=torch.bfloat16, enabled=use_bf16):
                out = model(input_ids=mb, labels=mb)
            (out.loss / accum).backward()
            loss_acc += out.loss.item() / accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        done = step + 1
        if done % 10 == 0 or done <= 5:
            rec = {"step": done, "loss": round(loss_acc, 4),
                   "lr": lr_at(step), "elapsed_s": round(time.time() - t0, 1)}
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(rec, flush=True)
        if done in CHECKPOINT_STEPS:
            save(done)
    print("TRAINING_DONE", args.run_name)


if __name__ == "__main__":
    main()
