# MetagenBERT memory profile (DNABERT-2 + DeepSets, T2D)

A feasibility/memory profile of a MetagenBERT-style training step
(arXiv:2601.03295) on a Kaggle T4. **This is not an accuracy study**: there is
no training loop, no evaluation and no train/test split.

```
run_to_sample.json, sample_to_label.json   # SRA run -> sample -> label (Y = T2D, N = control)
common.py              seeding + mapping helpers
data/download.py       first K spots (read 1) for N label-balanced runs -> data/raw/
data/sample_reads.py   R seeded random reads per sample                -> data/reads/
model.py               DNABERT-2 encoder (frozen except last 2 blocks) + DeepSets head
profile_step.py        grid of single training steps                    -> results.csv
report.py              markdown table + memory-vs-reads chart          -> report/
```

## Model

- **Encoder:** `zhihan1996/DNABERT-2-117M`, pinned to revision `7bce263b`. Each read is
  tokenized (BPE, ≤128 tokens), encoded, then **mean-pooled over non-pad tokens**
  into a 768-d read embedding.
- **Freezing:** embeddings and blocks 0–9 are frozen; blocks 10–11 train. That is
  19.2M of 116.8M parameters. The unused pooler is dropped. Frozen blocks build no
  autograd graph, so their activations are freed right away.
- **DeepSets head:** φ is an MLP (768→256→256) applied to each read. The outputs
  are mean-pooled across reads, then ρ (256→256→1) produces one logit per sample.
  The loss is BCE-with-logits.
- **Triton / flash-attention fix:** the remote code imports a Triton flash-attention
  kernel. Current Triton versions fail to compile it, and it doesn't support the
  T4's sm_75 anyway. `load_dnabert2()` blocks `import triton` during loading and also sets
  the module's `flash_attn_qkvpacked_func = None`. Together these force the plain PyTorch
  attention path. When the fix is active, the remote code logs *"Unable to import
  Triton; defaulting … to pytorch"*. Passing transformers' `BertConfig` explicitly
  avoids the known `config_class` mismatch error.
- **Gradient checkpointing:** wraps only the 2 trainable blocks with
  `torch.utils.checkpoint` (non-reentrant). There is nothing to recompute in frozen
  blocks. This was checked on CPU: gradients are identical with and without it.

## Running on Kaggle (T4)

Settings: Accelerator **GPU T4** (one is enough), **Internet on**.

**One-shot:** `bash kaggle_run.sh` runs everything below: sratoolkit, deps, a synthetic
full-grid profile, then a small real-data profile on batch `SRR413` (10 runs × 5000
spots, 512 reads/sample, reads ∈ {64..512}), and both reports.

```bash
!git clone https://github.com/sabasiddique1/metagenbert-memprofile.git
!cd metagenbert-memprofile && bash kaggle_run.sh
```

Or step by step:

```bash
# 1. Python deps (torch is preinstalled on Kaggle with CUDA; don't reinstall it)
grep -v '^torch' requirements.txt > /tmp/req.txt && pip install -q -r /tmp/req.txt
```

```bash
# 2. sra-tools (fastq-dump)
wget -q https://ftp-trace.ncbi.nlm.nih.gov/sra/sdk/3.1.1/sratoolkit.3.1.1-ubuntu64.tar.gz
tar xzf sratoolkit.3.1.1-ubuntu64.tar.gz && export PATH=$PWD/sratoolkit.3.1.1-ubuntu64/bin:$PATH
mkdir -p ~/.ncbi && printf '/LIBS/GUID = "%s"\n' "$(cat /proc/sys/kernel/random/uuid)" > ~/.ncbi/user-settings.mkfg
```

```bash
# 3. Data: 8 runs x first 100k spots, then 1024 reads per sample
python -m data.download --n_runs 8 --spots 100000 --out data/raw
python -m data.sample_reads --manifest data/raw/manifest.csv --reads 1024 --out data/reads
```

```bash
# 4. Profile the full grid (60 configs, one subprocess each), then report
python profile_step.py --reads_dir data/reads --out results.csv
python report.py --results results.csv --out_dir report
```

In a notebook cell, prefix each line with `!` and set `PATH` with
`os.environ["PATH"]`, since `export` doesn't persist between `!` lines.

Profiling doesn't depend on the download: `python profile_step.py --synthetic`
uses random ACGT reads (`--read_len 90`). Memory is close to real-data numbers when
the read length matches. Model weights (~470 MB) download from HF the first
time they're used.

**Grid:** reads_per_sample ∈ {64,128,256,512,1024} × batch_size (samples) ∈ {1,2,4} × AMP
fp16 {off,on} × grad-ckpt {off,on}. A step encodes `batch_size × reads_per_sample`
reads at once, i.e. up to 4096 sequences.

**Each config** runs in a fresh process, which isolates OOM and fragmentation. It
does a warm-up step, which creates the AdamW state, then `reset_peak_memory_stats()`,
then one timed forward+backward+optimizer step. `results.csv` records:
- `peak_alloc_mb` (`torch.cuda.max_memory_allocated`) and `peak_reserved_mb`
- `model_mb`, the weights alone
- `step_time_s`
- `oom`
- `max_tokens`, the padded token length in the batch

## Assumptions

- **Single-end, read 1 only.** `fastq-dump --split-files` and keep `_1` (or the
  only file). Mates are ignored.
- **First K spots, not random.** `fastq-dump -X K` takes the head of each run. The
  head of a run can be biased (flowcell tile/position effects). The R reads per
  sample are random *within* those K spots. Use a larger K, or a full download, for
  anything beyond memory profiling.
- **One run per sample.** This holds for the provided mapping (345 runs ↔ 345
  samples, 174 N / 171 Y).
- **Subject-level split later.** Profiling needs no split. Any real training must
  split by sample/subject so reads from one person never appear in both train and
  test.
- Reads shorter than 50 bp are dropped. If a sample has fewer than R reads, sampling
  falls back to drawing with replacement (a warning is printed).
- Padding is to the longest read in the batch, not to a fixed 128 tokens. Memory
  therefore depends on read length. `max_tokens` is recorded per config.

## Limitations

- **Two sequencing batches in the mapping.** Runs `SRR341581–341725` (145
  samples, IDs like `DLF001`/`NOM029`) and `SRR4135xx–4137xx` (199 samples,
  `CON-`/`T2D-`) look like two cohorts/stages. `SRR1778450` (T2D-095) is an outlier.
  Download selection is label-balanced but mixes batches. For later training, use
  `--run_prefix` to stay within one batch, or stratify by batch to avoid confounding.
- **Single-step timing.** One measured step per config is noisy and includes no
  data loading. Treat times as relative, not as throughput.
- **Peak allocated ≠ what fits.** The CUDA context (~300–500 MB) and allocator
  fragmentation sit on top. Compare `peak_reserved_mb` against the T4's ~15 GB.
- **Plain PyTorch attention** is slower and uses more memory than flash attention.
  These numbers are an upper bound relative to an Ampere+ GPU running the Triton
  kernel. Reads are short (~20–30 tokens), so attention is a small share of
  memory anyway.
- No DataLoader, gradient accumulation, scheduler or evaluation. All were left out on
  purpose for this profile.
- Tested locally on CPU only: data scripts, model load, freezing, checkpoint
  equivalence, the grid driver and report rendering. The GPU numbers come from
  running on Kaggle.
