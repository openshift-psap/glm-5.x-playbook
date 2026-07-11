# GLM 5.2 FP8 Single-Node Optimized Config — 8K/1K Workload

A throughput-tuned vLLM configuration for GLM-5.2-FP8 on a single
8×H200 node, for **short-context, high-turnover** workloads:
8,000 input / 1,000 output tokens per request (ISL/OSL 8000/1000).

This is a different animal from the
[practical guide](glm-5.2-single-node-practical-guide.md), which
serves 131K-context agentic sessions. Here the context is bounded
at 10K, so we trade context headroom for scheduler density and
CUDA graph coverage. Same model, same node, different knobs.

## When to Use This Config

- RAG answering, summarization, extraction — prompts ~8K, outputs ~1K
- Latency-sensitive serving at low-to-mid concurrency (1–16)
- Workloads where no request exceeds ~10K total tokens

Do **not** use this for agentic coding sessions — `--max-model-len
10000` will reject any longer request. Use the practical guide.

## Server Configuration

```bash
vllm serve "zai-org/GLM-5.2-FP8" \
  --host 0.0.0.0 \
  --port 30000 \
  --tensor-parallel-size 8 \
  --served-model-name "zai-org/GLM-5.2-FP8" \
  --gpu-memory-utilization 0.80 \
  --max-model-len 10000 \
  --max-num-seqs 48 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --max-cudagraph-capture-size 192 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5,"draft_tensor_parallel_size":8}'
```

Requires vLLM ≥ v0.23.0 (minimum version for GLM-5.2 model support).

### Why Each Knob

| Flag | Value | Rationale |
|------|-------|-----------|
| `--max-model-len` | 10000 | Bounds KV per sequence to 8K in + 1K out + margin. Smaller KV blocks per seq → more concurrent sequences fit, better paging behavior. |
| `--max-num-seqs` | 48 | Caps scheduler batch. At 10K max-len the KV pool supports far more than 48 seqs; 48 keeps ITL flat at C=16 with headroom for MTP draft verification. |
| `--gpu-memory-utilization` | 0.80 | Conservative; leaves HBM for MTP draft KV and CUDA graph pools. At 10K max-len you don't need 0.90+. |
| `--enable-chunked-prefill` + `--max-num-batched-tokens 8192` | | An 8K prompt prefills in one chunk without starving in-flight decodes — TTFT stays sub-second while decode ITL stays single-digit ms. |
| `--max-cudagraph-capture-size` | 192 | Captures CUDA graphs for batch sizes up to 192 — covers target batch (48) plus MTP speculative tokens (48 × (1+5) draft slots). |
| `--speculative-config` | mtp, 5 tokens, draft TP=8 | GLM-5.2's built-in MTP head predicts 5 tokens per forward pass; draft runs at the same TP=8 as the target (valid: vLLM restricts draft TP to 1 or target TP). |

Note the deliberate omissions vs the practical guide: no
`--kv-cache-dtype fp8` (KV pressure is low at 10K max-len; auto dtype
avoids any quantization effect on quality), and no tool-call/reasoning
parsers (this profile targets completion-style serving, not agentic
tool use — add `--tool-call-parser glm47 --reasoning-parser glm45
--enable-auto-tool-choice` if you need them).

## Benchmark: guidellm

Workload matches the internal performance-team standard: ISL/OSL
8000/1000, concurrency sweep 1 → 4 → 16, 300 s per stage.

```bash
guidellm benchmark \
  --target "http://127.0.0.1:30000" \
  --data "prompt_tokens=8000,output_tokens=1000" \
  --rate-type "concurrent" \
  --backend-type "openai_http" \
  --backend-kwargs '{"timeout": 100000}' \
  --model "zai-org/GLM-5.2-FP8" \
  --processor "zai-org/GLM-5.2-FP8" \
  --processor-args '{"trust_remote_code": true}' \
  --rate "1,4,16" \
  --max-seconds "300" \
  --output-path "output_vllm_glm52_spec.json"
```

> **guidellm version pin required.** This CLI form (`--rate-type`,
> `--backend-type`, `--processor`) exists in guidellm ≤ v0.3.x/0.4.x
> and was removed in v0.7+ (replaced by `--profile kind=concurrent`
> style). Install a matching release, e.g. `pip install guidellm==0.3.0`,
> or translate the flags for the new CLI.

## Measured Results

8×H200, GLM-5.2-FP8, TP=8, MTP (5 tokens), 300 s per concurrency
stage, 0 errors at every stage.

| Concurrency | Completed req (300s) | TTFT p50 / p95 (ms) | ITL p50 / p95 (ms) | Req latency p50 (s) | Output tok/s | Total tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1  | 46  | 665 / 671   | 6.0 / 8.2   | 6.7  | 151 | 1,360 |
| 4  | 100 | 692 / 708   | 11.0 / 13.6 | 11.7 | 339 | 3,088 |
| 16 | 189 | 738 / 4,708 | 23.0 / 29.9 | 24.1 | 649 | 6,126 |

### Reading the Numbers

- **TTFT p50 stays ~0.7 s across the whole sweep.** Chunked prefill
  at 8192 batched tokens absorbs an 8K prompt without queueing
  behind decodes. The p95 blowup at C=16 (4.7 s) is prefill queueing
  under saturation — the sign you've hit this config's ceiling.
- **ITL scales sub-linearly** (6 → 23 ms from C=1 → 16): MTP keeps
  accepted-token throughput high while the batch grows.
- **Throughput scales 4.3× from C=1 → C=16** (151 → 649 output
  tok/s). Past C=16, expect ITL to keep climbing; if you need more
  throughput at this ISL/OSL, add a second replica rather than
  raising concurrency.

> **NEEDS VERIFICATION**: results were produced by the internal
> performance team run (June 2026 window). The exact vLLM build and
> driver/firmware of that run are not pinned in this repo — re-run
> the guidellm command above against your own deployment before
> quoting these numbers externally.

## Deploying on Kubernetes

Reuse the StatefulSet from the
[practical guide](glm-5.2-single-node-practical-guide.md#step-1-deploy-with-a-statefulset)
and replace the `vllm serve` args with the command above (keep the
probes, `/dev/shm` mount, and HF token secret as-is; change the
container port to 30000 or keep 8000 and drop `--port 30000`).
