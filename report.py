"""Summarize results.csv: markdown table + memory-vs-reads chart.

Usage:
    python report.py --results results.csv --out_dir report
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# One fixed color + marker + line style per (amp, ckpt) variant, so identity
# never relies on color alone.
VARIANTS = {
    (False, False): ("fp32", "#2a78d6", "o", "-"),
    (False, True): ("fp32 + ckpt", "#eb6834", "s", "--"),
    (True, False): ("fp16 AMP", "#1baf7a", "^", "-"),
    (True, True): ("fp16 AMP + ckpt", "#eda100", "D", "--"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("amp", "grad_ckpt", "oom"):
        df[col] = df[col].astype(str).str.lower().eq("true")
    df["peak_alloc_gb"] = df["peak_alloc_mb"] / 1024
    return df.sort_values(["batch_size", "amp", "grad_ckpt", "reads_per_sample"])


def write_table(df: pd.DataFrame, out: Path) -> None:
    def cell(row):
        if row.oom:
            return "OOM"
        if pd.isna(row.peak_alloc_gb):
            return "error"
        return f"{row.peak_alloc_gb:.2f} GB / {row.step_time_s:.2f} s"

    df = df.assign(variant=[VARIANTS[(a, c)][0] for a, c in zip(df.amp, df.grad_ckpt)],
                   cell=[cell(r) for r in df.itertuples()])
    table = df.pivot_table(index=["batch_size", "variant"], columns="reads_per_sample",
                           values="cell", aggfunc="first")
    gpu = df["gpu"].dropna().iloc[0] if df["gpu"].notna().any() else "unknown GPU"
    lines = [f"# Memory profile ({gpu})", "",
             "Cell = peak `torch.cuda.max_memory_allocated` / measured step time. "
             "Columns = reads per sample.", "",
             table.reset_index().to_markdown(index=False)]
    out.write_text("\n".join(lines) + "\n")


def plot(df: pd.DataFrame, out: Path) -> None:
    batches = sorted(df.batch_size.unique())
    fig, axes = plt.subplots(1, len(batches), figsize=(4.2 * len(batches), 4),
                             sharey=True, squeeze=False)
    total_gb = df["gpu_total_mb"].dropna().max() / 1024 if "gpu_total_mb" in df else None
    for ax, b in zip(axes[0], batches):
        sub = df[df.batch_size == b]
        for (amp, ckpt), (label, color, marker, ls) in VARIANTS.items():
            v = sub[(sub.amp == amp) & (sub.grad_ckpt == ckpt)]
            ok = v[~v.oom]
            ax.plot(ok.reads_per_sample, ok.peak_alloc_gb, color=color, marker=marker,
                    linestyle=ls, linewidth=2, markersize=7, label=label)
            if total_gb and v.oom.any():  # OOM marked at the GPU limit
                ax.scatter(v[v.oom].reads_per_sample, [total_gb] * int(v.oom.sum()),
                           color=color, marker="x", s=70, linewidths=2, zorder=3)
        if total_gb:
            ax.axhline(total_gb, color=MUTED, linewidth=1, linestyle=":")
            ax.text(sub.reads_per_sample.min(), total_gb, " GPU limit (x = OOM)",
                    va="bottom", fontsize=8, color=MUTED)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(df.reads_per_sample.unique()))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_title(f"batch size = {b} sample(s)", fontsize=10, color=INK)
        ax.set_xlabel("reads per sample", color=MUTED)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.tick_params(colors=MUTED)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0][0].set_ylabel("peak allocated memory (GB)", color=MUTED)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(VARIANTS), frameon=False, fontsize=9)
    fig.suptitle("DNABERT-2 + DeepSets: peak GPU memory per training step",
                 color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=150, facecolor="white")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results.csv"))
    ap.add_argument("--out_dir", type=Path, default=Path("report"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load(args.results)
    write_table(df, args.out_dir / "results.md")
    plot(df, args.out_dir / "memory_vs_reads.png")
    print(f"Wrote {args.out_dir / 'results.md'} and {args.out_dir / 'memory_vs_reads.png'}")


if __name__ == "__main__":
    main()
