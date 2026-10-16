# Memory profile (Tesla T4)

Cell = peak `torch.cuda.max_memory_allocated` / measured step time. Columns = reads per sample.

|   batch_size | variant         | 64               | 128              | 256              | 512              |
|-------------:|:----------------|:-----------------|:-----------------|:-----------------|:-----------------|
|            1 | fp16 AMP        | 0.79 GB / 0.07 s | 0.93 GB / 0.11 s | 1.20 GB / 0.17 s | 1.77 GB / 0.31 s |
|            1 | fp16 AMP + ckpt | 0.75 GB / 0.08 s | 0.83 GB / 0.12 s | 0.98 GB / 0.18 s | 1.30 GB / 0.33 s |
|            1 | fp32            | 0.85 GB / 0.15 s | 1.09 GB / 0.25 s | 1.55 GB / 0.49 s | 2.49 GB / 1.07 s |
|            1 | fp32 + ckpt     | 0.79 GB / 0.14 s | 0.92 GB / 0.27 s | 1.18 GB / 0.55 s | 1.72 GB / 1.21 s |
|            2 | fp16 AMP        | 0.93 GB / 0.11 s | 1.21 GB / 0.17 s | 1.77 GB / 0.28 s | 2.92 GB / 0.62 s |
|            2 | fp16 AMP + ckpt | 0.83 GB / 0.12 s | 0.99 GB / 0.17 s | 1.30 GB / 0.32 s | 1.98 GB / 0.69 s |
|            2 | fp32            | 1.09 GB / 0.24 s | 1.57 GB / 0.51 s | 2.50 GB / 1.01 s | 4.41 GB / 2.23 s |
|            2 | fp32 + ckpt     | 0.92 GB / 0.27 s | 1.19 GB / 0.58 s | 1.72 GB / 1.13 s | 2.83 GB / 2.56 s |
|            4 | fp16 AMP        | 1.20 GB / 0.17 s | 1.76 GB / 0.28 s | 2.88 GB / 0.58 s | 5.15 GB / 1.29 s |
|            4 | fp16 AMP + ckpt | 0.98 GB / 0.17 s | 1.30 GB / 0.31 s | 1.96 GB / 0.66 s | 3.31 GB / 1.38 s |
|            4 | fp32            | 1.55 GB / 0.46 s | 2.49 GB / 0.95 s | 4.36 GB / 2.02 s | 8.13 GB / 4.48 s |
|            4 | fp32 + ckpt     | 1.18 GB / 0.52 s | 1.71 GB / 1.07 s | 2.80 GB / 2.32 s | 5.00 GB / 5.22 s |
