"""Summary figure: peak GPU memory vs total reads per training step (Tesla T4).

Usage: python plot_summary.py results results/memory_summary.png
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

res, out = Path(sys.argv[1]), Path(sys.argv[2])
df = pd.concat([pd.read_csv(res / "results_synthetic.csv"), pd.read_csv(res / "results_ceiling.csv")])
for c in ("amp", "grad_ckpt", "oom"):
    df[c] = df[c].astype(str).str.lower().eq("true")
df["N"] = df.reads_per_sample * df.batch_size  # memory depends only on total reads per step
limit_gb = df.gpu_total_mb.dropna().max() / 1024

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
V = {(False, False): ("fp32", "#2a78d6", "o", "-", 0),
     (False, True): ("fp32 + checkpointing", "#eb6834", "s", "--", -130),
     (True, False): ("fp16 AMP", "#1baf7a", "^", "-", 130),
     (True, True): ("fp16 AMP + checkpointing", "#eda100", "D", "--", 0)}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, ax = plt.subplots(figsize=(8.6, 5.4), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.axhline(limit_gb, color=MUTED, lw=1, zorder=1)
ax.text(150, limit_gb + 0.2, f"Tesla T4 memory ({limit_gb:.1f} GB)", color=MUTED, fontsize=9, va="bottom")

for (amp, ck), (name, col, mk, ls, dx) in V.items():
    g = df[(df.amp == amp) & (df.grad_ckpt == ck)]
    ok = g[~g.oom].groupby("N").peak_alloc_mb.mean().sort_index() / 1024
    ax.plot(ok.index, ok.values, color=col, marker=mk, ls=ls, lw=2, ms=7,
            mec=SURFACE, mew=1.5, label=name, zorder=3)
    first_oom = g[g.oom & (g.N > ok.index.max())].N.min()
    if pd.notna(first_oom):
        ax.scatter([first_oom + dx], [limit_gb]  # dx keeps coincident OOM marks visible
                  , marker="x", s=60, color=col, lw=2, zorder=4)

def note(x, y, text):
    ax.annotate(text, (x, y), xytext=(0, 12), textcoords="offset points", ha="center",
                fontsize=8.5, color=INK, va="bottom")

note(2048, 8.26, "fp32\nfits up to 2,048 reads")
note(4096, 9.79, "AMP or checkpointing alone\nfit up to 4,096 reads")
note(8192, 11.51, "AMP + checkpointing\nfits up to 8,192 reads")

ax.set_xlim(0, 10800); ax.set_ylim(0, 16)
ax.set_xticks([0, 2048, 4096, 6144, 8192, 10240])
ax.set_xticklabels([f"{t:,}" for t in [0, 2048, 4096, 6144, 8192, 10240]])
ax.set_xlabel("total reads per training step  (reads per sample × samples per batch)", color=MUTED)
ax.set_ylabel("peak GPU memory (GB)", color=MUTED)
ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.tick_params(colors=MUTED)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.legend(loc="lower right", frameon=False, fontsize=9)
fig.text(0.075, 0.955, "DNABERT-2 + DeepSets: GPU memory grows linearly with reads per step",
         fontsize=12.5, color=INK, fontweight="bold", ha="left")
fig.text(0.075, 0.905, "One forward + backward + AdamW step, last 2 transformer blocks trainable.  "
         "× = out of memory.\nReal SRR413 reads used within 2% of the memory shown (synthetic 90 bp reads).",
         fontsize=8.5, color=MUTED, ha="left", va="top", linespacing=1.4)
fig.tight_layout(rect=(0, 0, 1, 0.85))
fig.savefig(out, dpi=130, facecolor=SURFACE)
print("wrote", out, out.stat().st_size, "bytes")
