# Benchmark Results Summary

**Cluster:** Janus (psap-de-h200-cluster), 2× NVIDIA H200, 8 GPUs each
**Model:** zai-org/GLM-5.2-FP8 (dummy weights)
**vLLM:** v0.23.0+
**Date:** 2026-06-21 — 2026-06-22

## 1. Pipeline Parallel (PP=2, TP=8) — Concurrency Sweep

ISL=100K, OSL=10, concurrency 1→32. No MTP (incompatible with PP).

| C | TTFT P50 (s) | TTFT P99 (s) | ITL P50 (ms) | ITL P99 (ms) | Throughput (tok/s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 7.4 | 21.3 | 16 | 220 | 10,989 |
| 2 | 13.9 | 19.5 | 1,015 | 1,152 | 8,539 |
| 4 | 37.5 | 40.7 | 1,018 | 1,154 | 8,658 |
| 8 | 84.2 | 86.3 | 1,021 | 1,152 | 8,608 |
| 16 | 177.1 | 177.8 | 1,022 | 1,152 | 8,602 |
| 32 | 363.8 | 364.1 | 1,022 | 1,152 | 8,575 |

Comparison with GLM-5-FP8 (same topology): GLM-5.2 is 14-17% faster TTFT,
9% lower ITL P50, 28% lower ITL P99, 16% higher throughput.
Improvement attributed to FP8 KV cache (`--kv-cache-dtype fp8`).

## 2. Single-Node Replicas (TP=8, MTP enabled) — Prefix Caching

2× independent replicas, round-robin routing. MTP enabled but 0% acceptance (dummy weights).

### WL1: Unique prompts (ISL=100K, OSL=500, C=8, N=150)

| Metric | Value |
|---|---|
| TTFT P50 | 39,587 ms |
| TTFT P99 | 73,883 ms |
| ITL P50 | 21 ms |
| ITL P99 | 1,161 ms |
| Throughput | 11,704 tok/s |
| Req/s | 0.12 |

### WL2: Shared prefix (80K prefix + 20K unique, OSL=1024, C=8, N=150)

| Metric | Value |
|---|---|
| TTFT P50 | 14,545 ms |
| TTFT P99 | 53,107 ms |
| ITL P50 | 22 ms |
| ITL P99 | 1,075 ms |
| Throughput | 13,091 tok/s |
| Req/s | 0.13 |

**Prefix caching effect:** 63% lower TTFT P50, 12% higher throughput.

## 3. Cross-Node KV Sharing (Mooncake) — Failed

| Config | Throughput | vs Baseline | Status |
|---|---|---|---|
| A: Round-robin + prefix cache | 11,704 tok/s | baseline | completed |
| B: Sticky + CPU offload 20GB | 2,895 tok/s | **-75%** | completed |
| C: Mooncake TCP | 4,688 tok/s | **-60%** | completed |
| C: Mooncake RDMA | ~300s/prompt | **~50× slower** | killed |

Mooncake was tested in embedded mode (wrong choice — standalone-store mode
would reduce contention). See EXPERIMENT_RESULTS.md for full analysis.

## Key Findings

1. **Prefix caching alone delivers 63% TTFT reduction** — no external infra needed
2. **FP8 KV cache gives 16% throughput uplift** (GLM-5.2 vs GLM-5)
3. **MTP + PP is incompatible** — vLLM RFC [#44697](https://github.com/vllm-project/vllm/issues/44697) in progress
4. **CPU offload hurts at high concurrency** — overhead exceeds benefit at C=8, 100K ISL
5. **Mooncake embedded mode is wrong for multi-GPU TP** — use standalone-store mode

## Raw Data Files

| File | Description |
|---|---|
| `pp2-glm5-sweep.txt` | GLM-5 PP=2 concurrency sweep (1→32) |
| `pp2-glm52-sweep.txt` | GLM-5.2 PP=2 concurrency sweep (1→32) |
| `run1-A-wl1.txt` | Config A, WL1 unique prompts |
| `run2-A-wl2.txt` | Config A, WL2 shared prefix |
| `run3-4-B-wl1-wl2.txt` | Config B, WL1 + WL2 (CPU offload) |
| `run6-C-tcp-wl1.txt` | Config C, WL1 (Mooncake TCP) |
