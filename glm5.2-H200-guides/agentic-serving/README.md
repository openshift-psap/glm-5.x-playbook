# GLM 5.2 FP8 on H200 — Deployment Guides

Three validated deployment patterns for
[zai-org/GLM-5.2-FP8](https://huggingface.co/zai-org/GLM-5.2-FP8)
on NVIDIA H200, with vLLM. Each guide is what we actually ran and
measured — configs, benchmark commands, and numbers, including the
things that didn't work.

| # | Pattern | Guide | Workload | Key result | Benchmark tool |
|---|---------|-------|----------|-----------|----------------|
| 1 | Single node, TP=8 + MTP | [Practical guide](glm-5.2-single-node-practical-guide.md) | Long-context agentic (Claude Code, 131K ctx) | 60% prefix-cache hit rate on real Weka traces; 3,256 tok/s synthetic | inference-perf |
| 2 | Single node, throughput-tuned | [8K/1K optimized config](glm-5.2-single-node-8k1k-optimized.md) | Short-context, high-turnover (ISL/OSL 8000/1000, C=1–16) | TTFT p50 ~0.7 s flat across sweep; 649 output tok/s at C=16 | guidellm |
| 3 | 2 nodes, PP=2 TP=8, RoCE | [Multi-node guide](glm-5.2-multi-node-pp2-tp8.md) | Model/KV beyond one node's HBM | ~8.6K tok/s at 100K ISL; +16% vs GLM-5 from FP8 KV | inference-perf |

## Which One?

- Model fits on one node and you serve agents → **1**.
- Bounded short contexts, you want max density → **2**.
- You need more HBM than one node has → **3** (and accept no MTP
  under PP — [vllm#44697](https://github.com/vllm-project/vllm/issues/44697)).

## Requirements (all patterns)

- NVIDIA H200 nodes (8 GPUs/node), Kubernetes or OpenShift with GPU
  operator
- vLLM ≥ v0.23.0 (GLM-5.2 model support starts at v0.23.0)
- HuggingFace token secret `llm-d-hf-token` (key `HF_TOKEN`)
- ~756 GB storage per node for FP8 weights

Manifests live in [`manifests/`](manifests/) (pattern 3) and inline
in the pattern 1 guide. The inference-perf benchmark template is in
[`benchmark-templates/`](benchmark-templates/).
