# GLM-5.2-FP8 Benchmark Playbook

Reproducible benchmark playbook for GLM-5.2-FP8 on NVIDIA H200 with vLLM.
Covers two topologies: pipeline parallel (PP=2, TP=8, 2 nodes) and single-node replicas (TP=8, MTP enabled).

## Prerequisites

- OpenShift/Kubernetes cluster with H200 GPU nodes
- LeaderWorkerSet CRD installed (for PP topology)
- `oc` or `kubectl` CLI configured
- HuggingFace token stored as secret `hf-token` with key `token` in target namespace
- Namespace created (default: `glm-5-pp`)

```bash
export KUBECONFIG=/path/to/kubeconfig.yaml
export NS=glm-5-pp
oc create namespace $NS --dry-run=client -o yaml | oc apply -f -
oc create secret generic hf-token --from-literal=token=$HF_TOKEN -n $NS
```

## Topology 1: Pipeline Parallel (PP=2, TP=8)

Two nodes, one model split across both via pipeline parallelism.
MTP speculative decoding is NOT available with PP ([vllm#44697](https://github.com/vllm-project/vllm/issues/44697)).

### Deploy

```bash
oc apply -f manifests/pp2-tp8-lws.yaml -n $NS
```

### Wait for readiness (~10 min)

```bash
oc logs -f vllm-pp-0 -n $NS
# Ready when: "Application startup complete."
```

### Run benchmark

```bash
oc exec vllm-pp-0 -n $NS -- bash /benchmarks/sweep-concurrency.sh
```

### Cleanup

```bash
oc delete -f manifests/pp2-tp8-lws.yaml -n $NS
```

## Topology 2: Single-Node Replicas (TP=8, MTP enabled)

Independent replicas, one per node. MTP speculative decoding enabled (5 tokens).
Use for: throughput testing, prefix caching evaluation, scaling studies.

### Deploy

```bash
oc apply -f manifests/tp8-replicas.yaml -n $NS
```

### Wait for readiness (~20 min, sequential StatefulSet)

```bash
oc get pods -n $NS -w
# Ready when both vllm-0 and vllm-1 show 1/1 Running
```

### Run benchmarks

```bash
# WL1: Unique prompts (control — no cache benefit)
oc exec vllm-0 -n $NS -- bash /benchmarks/wl1-unique.sh

# WL2: Shared prefix (tests prefix cache reuse)
oc exec vllm-0 -n $NS -- bash /benchmarks/wl2-shared-prefix.sh
```

### Cleanup

```bash
oc delete -f manifests/tp8-replicas.yaml -n $NS
```

## Customizing for Your Workload

Edit the benchmark scripts to match your ISL/OSL:

```bash
# In any benchmark script, change these:
ISL=100000    # → your input sequence length
OSL=500       # → your output sequence length
CONCURRENCY=8 # → your target concurrency
NUM_PROMPTS=150
```

For real prompts instead of random, use `--dataset-name hf` with a HuggingFace dataset,
or `--dataset-name custom` with `--dataset-path /path/to/prompts.jsonl`.

## Cluster Info (Janus Reference)

| Parameter | Value |
|---|---|
| Cluster | psap-de-h200-cluster (IBM rhperfscale) |
| Nodes | 2× NVIDIA H200 (160 CPU, 1.8TB RAM, 8 GPUs each) |
| OpenShift | 4.21.11, Kubernetes 1.34.6 |
| GPU DRA | composite.dra/gpu-nic-pair (GPU + InfiniBand NIC bundle) |
| vLLM | v0.23.0+ (vllm/vllm-openai:latest) |

## Results Summary

See `results/` for raw data and `results/SUMMARY.md` for analysis.
