#!/usr/bin/env bash
# End-to-end memory profile on a Kaggle T4 (Settings: GPU T4, Internet on).
# Usage (from a notebook cell):
#   !git clone https://github.com/sabasiddique1/metagenbert-memprofile.git
#   !cd metagenbert-memprofile && bash kaggle_run.sh
# Synthetic runs come first; the SRA download runs last, so a download
# failure never blocks the main results.
set -euo pipefail
cd "$(dirname "$0")"

# 1. Python deps: keep Kaggle's preinstalled torch/numpy/pandas/matplotlib
pip install -q transformers==4.45.2 einops==0.8.0 tabulate==0.9.0

# 2. Synthetic grid (60 configs) + its report
python profile_step.py --synthetic --read_len 90 --out results_synthetic.csv
python report.py --results results_synthetic.csv --out_dir report/synthetic

# 3. Ceiling run: where each variant runs out of memory (1 sample/step)
python profile_step.py --synthetic --read_len 90 --reads 4096 6144 8192 10240 12288 --batch 1 \
  --out results_ceiling.csv

# 4. Real data last: sra-tools, then one sequencing batch (SRR413*),
#    10 runs x 5000 spots, 512 reads/sample
SRA_VER=3.1.1
if [ ! -d "sratoolkit.${SRA_VER}-ubuntu64" ]; then
  wget -q "https://ftp-trace.ncbi.nlm.nih.gov/sra/sdk/${SRA_VER}/sratoolkit.${SRA_VER}-ubuntu64.tar.gz"
  tar xzf "sratoolkit.${SRA_VER}-ubuntu64.tar.gz"
fi
export PATH="$PWD/sratoolkit.${SRA_VER}-ubuntu64/bin:$PATH"
# Non-interactive config so fastq-dump doesn't ask for vdb-config
mkdir -p ~/.ncbi
[ -f ~/.ncbi/user-settings.mkfg ] || \
  printf '/LIBS/GUID = "%s"\n' "$(cat /proc/sys/kernel/random/uuid)" > ~/.ncbi/user-settings.mkfg

python -m data.download --run_prefix SRR413 --n_runs 10 --spots 5000 --out data/raw
python -m data.sample_reads --manifest data/raw/manifest.csv --reads 512 --out data/reads
python profile_step.py --reads_dir data/reads --reads 64 128 256 512 --out results_real.csv
python report.py --results results_real.csv --out_dir report/real

echo "Done: results_{synthetic,ceiling,real}.csv, report/synthetic, report/real"
