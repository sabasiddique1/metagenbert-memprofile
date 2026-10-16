"""Draw R random reads per sample (seeded) from the downloaded FASTQs.

Usage:
    python -m data.sample_reads --manifest data/raw/manifest.csv --reads 1024 --out data/reads

Writes <out>/<sample>.txt (one read per line) and <out>/samples.csv.
Draw the largest R you will profile; profile.py takes the first r <= R reads,
which is itself a uniform random subset.
"""
import argparse
import csv
import zlib
from pathlib import Path

import numpy as np

from common import seed_everything


def read_sequences(fastq: Path, min_len: int) -> list[str]:
    seqs = []
    with open(fastq) as f:
        for i, line in enumerate(f):
            if i % 4 == 1:
                s = line.strip().upper()
                if len(s) >= min_len:
                    seqs.append(s)
    return seqs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("data/raw/manifest.csv"))
    ap.add_argument("--reads", type=int, default=1024, help="R: reads per sample")
    ap.add_argument("--min_len", type=int, default=50)
    ap.add_argument("--out", type=Path, default=Path("data/reads"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.manifest) as f:
        rows = list(csv.DictReader(f))

    out_rows = []
    for row in rows:
        seqs = read_sequences(Path(row["fastq"]), args.min_len)
        if not seqs:
            print(f"{row['sample']}: no reads >= {args.min_len} bp, skipped")
            continue
        # Per-sample RNG so results don't depend on manifest order.
        rng = np.random.default_rng([args.seed, zlib.crc32(row["sample"].encode())])
        replace = len(seqs) < args.reads
        if replace:
            print(f"{row['sample']}: only {len(seqs)} reads, sampling with replacement")
        idx = rng.choice(len(seqs), size=args.reads, replace=replace)
        path = args.out / f"{row['sample']}.txt"
        path.write_text("\n".join(seqs[i] for i in idx) + "\n")
        out_rows.append({"sample": row["sample"], "label": row["label"],
                         "n_available": len(seqs), "reads": str(path)})

    with open(args.out / "samples.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample", "label", "n_available", "reads"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {len(out_rows)} samples x {args.reads} reads to {args.out}")


if __name__ == "__main__":
    main()
