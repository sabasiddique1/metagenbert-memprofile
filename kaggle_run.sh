#!/usr/bin/env bash
# End-to-end memory profile on a Kaggle T4 (Settings: GPU T4, Internet on).
# Usage (from a notebook cell, after cloning the repo):
#   !git clone https://github.com/sabasiddique1/metagenbert-memprofile.git
#   !cd metagenbert-memprofile && bash kaggle_run.sh
set -euo pipefail
cd "$(dirname "$0")"

# 1. sra-tools (fastq-dump), prebuilt binaries
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
fastq-dump --version

# 2. Python deps (keep Kaggle's preinstalled CUDA torch)
grep -v '^torch' requirements.txt > /tmp/req.txt
pip install -q -r /tmp/req.txt

# 3. Synthetic profile: full grid, no SRA data needed
python profile_step.py --synthetic --read_len 90 --out results_synthetic.csv

# 4. Small real-data profile: one sequencing batch (SRR413*), 10 runs x 5000 spots, 512 reads/sample
python -m data.download --run_prefix SRR413 --n_runs 10 --spots 5000 --out data/raw
python -m data.sample_reads --manifest data/raw/manifest.csv --reads 512 --out data/reads
python profile_step.py --reads_dir data/reads --reads 64 128 256 512 --out results_real.csv

# 5. Reports
python report.py --results results_synthetic.csv --out_dir report/synthetic
python report.py --results results_real.csv --out_dir report/real
echo "Done: see report/synthetic and report/real"
