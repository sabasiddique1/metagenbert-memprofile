"""Download the first K spots (read 1 only) for N label-balanced runs.

Usage:
    python -m data.download --n_runs 20 --spots 200000 --out data/raw

Writes <out>/<RUN>.fastq and <out>/manifest.csv (run, sample, label, fastq).
Note: `-X K` takes the FIRST K spots of each run, not a random subset.
"""
import argparse
import csv
import random
import shutil
import subprocess
import sys
from pathlib import Path

from common import load_run_table, seed_everything


def select_runs(n_runs: int, seed: int, run_prefix: str | None) -> list[dict]:
    """Pick n_runs runs, half per label, reproducibly."""
    table = load_run_table()
    if run_prefix:
        table = [r for r in table if r["run"].startswith(run_prefix)]
    rng = random.Random(seed)
    by_label = {}
    for row in table:
        by_label.setdefault(row["label"], []).append(row)
    picked = []
    for i, label in enumerate(sorted(by_label)):
        rows = by_label[label]
        rng.shuffle(rows)
        k = n_runs // 2 + (n_runs % 2 if i == 0 else 0)
        picked.extend(rows[:k])
    return sorted(picked, key=lambda r: r["run"])


def fetch_read1(run: str, spots: int, out_dir: Path) -> Path:
    """fastq-dump the first `spots` spots; keep read 1 (or the only read)."""
    final = out_dir / f"{run}.fastq"
    if final.exists() and final.stat().st_size > 0:
        return final
    tmp = out_dir / f"tmp_{run}"
    tmp.mkdir(exist_ok=True)
    cmd = ["fastq-dump", "-X", str(spots), "--split-files", "--skip-technical",
           "-O", str(tmp), run]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    # Paired runs give RUN_1/RUN_2; single-end runs give RUN_1 or RUN.
    for name in (f"{run}_1.fastq", f"{run}.fastq"):
        if (tmp / name).exists():
            (tmp / name).replace(final)
            break
    else:
        raise FileNotFoundError(f"No read-1 FASTQ produced for {run}")
    shutil.rmtree(tmp)
    return final


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_runs", type=int, default=20)
    ap.add_argument("--spots", type=int, default=200_000, help="K: first K spots per run")
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    ap.add_argument("--run_prefix", default=None,
                    help="Restrict to one sequencing batch, e.g. SRR413 or SRR341")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if shutil.which("fastq-dump") is None:
        sys.exit("fastq-dump not found; install sra-tools (see README).")
    seed_everything(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = select_runs(args.n_runs, args.seed, args.run_prefix)
    manifest = []
    for i, row in enumerate(rows, 1):
        print(f"[{i}/{len(rows)}] {row['run']} ({row['sample']}, {row['label']})", flush=True)
        try:
            path = fetch_read1(row["run"], args.spots, args.out)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"  skipped: {e}", file=sys.stderr)
            continue
        manifest.append({**row, "fastq": str(path)})

    with open(args.out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "sample", "label", "fastq"])
        w.writeheader()
        w.writerows(manifest)
    print(f"Wrote {len(manifest)} runs to {args.out / 'manifest.csv'}")


if __name__ == "__main__":
    main()
