"""Profile GPU memory/time of one training step over a config grid.

Usage:
    python profile_step.py --reads_dir data/reads --out results.csv
    python profile_step.py --synthetic --out results.csv      # no SRA data needed

Each config runs in a fresh subprocess so an OOM or allocator fragmentation in
one config cannot affect the next. Per config: one warm-up step (allocates
optimizer state, CUDA context, cuBLAS workspaces), then peak stats are reset
and one measured forward+backward+optimizer step is timed.

(Named profile_step.py, not profile.py: a top-level profile.py shadows the
stdlib `profile` module that cProfile imports.)
"""
import argparse
import csv
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from common import LABEL_TO_INT, seed_everything

READS = [64, 128, 256, 512, 1024]
BATCH = [1, 2, 4]
AMP = [False, True]
CKPT = [False, True]
FIELDS = ["reads_per_sample", "batch_size", "amp", "grad_ckpt", "oom", "peak_alloc_mb",
          "peak_reserved_mb", "model_mb", "step_time_s", "max_tokens", "device", "gpu", "gpu_total_mb",
          "error"]


# ---------------------------------------------------------------- batch inputs

def load_reads(reads_dir: Path, batch_size: int, reads: int, seed: int):
    """Pick batch_size samples (seeded) and return their first `reads` reads + labels."""
    with open(reads_dir / "samples.csv") as f:
        rows = list(csv.DictReader(f))
    rng = np.random.default_rng(seed)
    rows = [rows[i] for i in rng.permutation(len(rows))[:batch_size]]
    if len(rows) < batch_size:
        raise ValueError(f"Need {batch_size} samples, found {len(rows)}")
    seqs, labels = [], []
    for row in rows:
        lines = Path(row["reads"]).read_text().split()
        if len(lines) < reads:
            raise ValueError(f"{row['sample']} has {len(lines)} reads < {reads}")
        seqs.append(lines[:reads])
        labels.append(LABEL_TO_INT[row["label"]])
    return seqs, labels


def synthetic_reads(batch_size: int, reads: int, read_len: int, seed: int):
    rng = np.random.default_rng(seed)
    seqs = [["".join(rng.choice(list("ACGT"), read_len)) for _ in range(reads)]
            for _ in range(batch_size)]
    labels = [i % 2 for i in range(batch_size)]
    return seqs, labels


def tokenize(tokenizer, seqs, max_len: int):
    """list[B][R] of str -> (B, R, L) ids and mask, padded to the batch's longest read."""
    b, r = len(seqs), len(seqs[0])
    enc = tokenizer([s for sample in seqs for s in sample], padding="longest",
                    truncation=True, max_length=max_len, return_tensors="pt")
    return enc["input_ids"].view(b, r, -1), enc["attention_mask"].view(b, r, -1)


# ------------------------------------------------------------- single config

def run_one(cfg: dict, args) -> dict:
    import torch
    from model import MetagenomeClassifier, load_dnabert2

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = device.type == "cuda"
    res = {**cfg, "oom": False, "error": "", "device": device.type}  # AMP is CUDA-only
    if cuda:
        props = torch.cuda.get_device_properties(device)
        res.update(gpu=props.name, gpu_total_mb=round(props.total_memory / 2**20))

    tokenizer, encoder = load_dnabert2()
    model = MetagenomeClassifier(encoder, n_trainable_blocks=2,
                                 grad_checkpointing=cfg["grad_ckpt"]).to(device).train()
    if cuda:
        res["model_mb"] = round(torch.cuda.memory_allocated() / 2**20, 1)

    if args.synthetic:
        seqs, labels = synthetic_reads(cfg["batch_size"], cfg["reads_per_sample"],
                                       args.read_len, args.seed)
    else:
        seqs, labels = load_reads(args.reads_dir, cfg["batch_size"],
                                  cfg["reads_per_sample"], args.seed)
    ids, mask = tokenize(tokenizer, seqs, args.max_len)
    res["max_tokens"] = ids.shape[-1]
    ids, mask = ids.to(device), mask.to(device)
    y = torch.tensor(labels, dtype=torch.float32, device=device)

    opt = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"] and cuda)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=cfg["amp"] and cuda):
            logits = model(ids, mask)
        loss = loss_fn(logits.float(), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    def sync():
        if cuda:
            torch.cuda.synchronize()

    try:
        step()  # warm-up
        sync()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        step()
        sync()
        res["step_time_s"] = round(time.perf_counter() - t0, 4)
    except torch.cuda.OutOfMemoryError as e:
        res["oom"] = True
        res["error"] = str(e).splitlines()[0][:200]
    if cuda:
        res["peak_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        res["peak_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
    return res


# ------------------------------------------------------------------- driver

def run_grid(args) -> None:
    grid = [dict(reads_per_sample=r, batch_size=b, amp=a, grad_ckpt=c)
            for r, b, a, c in itertools.product(args.reads, args.batch, AMP, CKPT)]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for i, cfg in enumerate(grid, 1):
            cmd = [sys.executable, __file__, "--one", json.dumps(cfg), *sys.argv[1:]]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
            lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
            if lines:
                res = json.loads(lines[-1][len("RESULT "):])
            else:  # crashed hard (e.g. CUDA error outside the step)
                err = (proc.stderr.strip().splitlines() or ["unknown error"])[-1]
                res = {**cfg, "oom": "out of memory" in proc.stderr.lower(), "error": err[:200]}
            writer.writerow(res)
            f.flush()
            status = "OOM" if res["oom"] else (res.get("error") or f"{res.get('peak_alloc_mb')} MB")
            print(f"[{i}/{len(grid)}] {cfg} -> {status}", flush=True)
    print(f"Wrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reads_dir", type=Path, default=Path("data/reads"))
    ap.add_argument("--synthetic", action="store_true", help="random ACGT reads instead of data")
    ap.add_argument("--read_len", type=int, default=90, help="synthetic read length (bp)")
    ap.add_argument("--reads", type=int, nargs="+", default=READS)
    ap.add_argument("--batch", type=int, nargs="+", default=BATCH)
    ap.add_argument("--max_len", type=int, default=128, help="max tokens per read")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=int, default=900, help="seconds per config")
    ap.add_argument("--out", type=Path, default=Path("results.csv"))
    ap.add_argument("--one", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.one:
        print("RESULT " + json.dumps(run_one(json.loads(args.one), args)), flush=True)
    else:
        run_grid(args)


if __name__ == "__main__":
    main()
