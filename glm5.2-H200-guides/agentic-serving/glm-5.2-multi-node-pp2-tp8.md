# GLM 5.2 FP8 Multi-Node — PP=2 TP=8 on 2× 8×H200 (RoCE)

Deploy GLM-5.2-FP8 split across **two 8×H200 nodes** with pipeline
parallelism (PP=2) and tensor parallelism within each node (TP=8),
using vLLM's native multi-node launcher (no Ray) and a
LeaderWorkerSet.

Use this when the model plus your KV budget exceeds a single node's
HBM, or when you want more KV headroom for very long contexts than
one node can give. If the model fits on one node, prefer the
[single-node practical guide](glm-5.2-single-node-practical-guide.md)
— it's simpler and gets MTP speculative decoding, which PP does not
(see [Known Limits](#known-limits)).

## Topology

- **PP=2**: model layers split across 2 nodes; cross-node traffic is
  activations at the pipeline boundary (small vs KV transfer — this
  is why PP is the pragmatic cross-node choice on Ethernet/RoCE).
- **TP=8** inside each node over NVLink.
- **LeaderWorkerSet**: leader (node-rank 0) serves the API; the
  worker joins headless via `--nnodes 2 --node-rank 1`.
- vLLM ≥ v0.23.0 (minimum version for GLM-5.2 model support).

## Networking: RoCE

Cross-node NCCL/gloo need working RDMA (or fall back to TCP, slowly).
The manifest ships with RDMA blocks **commented out** because
interface names, HCA lists, and RDMA resource names are
fabric-specific:

- `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` — host interface for
  bootstrap. Set on **both** leader and worker, identically.
- `NCCL_IB_HCA` — HCA prefix or explicit device list.
- `NCCL_IB_GID_INDEX` — RoCEv2 GID index for your fabric.
- RDMA device resource (e.g. `rdma/ib: "1"`) on both roles,
  symmetric. An asymmetric request (leader with RDMA, worker
  without) schedules fine and then hangs at NCCL init — we hit this;
  don't.

Rail-optimized RoCE fabrics work for PP activation transfer. (This is
plain NCCL point-to-point — unlike DeepEP wide-EP, which needs
any-NIC-to-any-NIC reachability.)

Sanity check once pods are up:

```bash
kubectl exec vllm-pp-0 -- nvidia-smi topo -m
kubectl exec vllm-pp-0 -- ibv_devices   # should list your HCAs
```

## Deploy

```bash
export NAMESPACE=glm52-pp2
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
kubectl create secret generic llm-d-hf-token \
  --from-literal="HF_TOKEN=${HF_TOKEN}" -n ${NAMESPACE} \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -n ${NAMESPACE} -f manifests/glm52-pp2-tp8-lws.yaml
```

Both pods must schedule on distinct 8×H200 nodes. Watch the leader:

```bash
kubectl logs -f vllm-pp-0 -n ${NAMESPACE}
# Ready when: "Application startup complete."  (~12 min with cached weights)
```

## Verify

```bash
kubectl exec -it vllm-pp-0 -n ${NAMESPACE} -- \
  curl -s http://localhost:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.2-fp8","prompt":"Hello","max_tokens":10}' | python3 -m json.tool
```

Tool calling and Claude Code connection work exactly as in the
[practical guide](glm-5.2-single-node-practical-guide.md#step-3-verify-tool-calling)
— port-forward `service/vllm-pp-leader 8000:8080` first.

## Measured Results

**Cluster**: 2× 8×H200, OpenShift 4.21.11.
**Weights:** dummy (FP8 layout) — these numbers validate the topology
and scheduler behavior, not end-quality throughput. Re-run with real
weights before quoting externally.

### Concurrency Sweep — ISL=100K, OSL=10

| Concurrency | TTFT P50 (s) | TTFT P99 (s) | ITL P50 (ms) | ITL P99 (ms) | Throughput (tok/s) | Failures |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 7.4 | 21.3 | 16 | 220 | 10,989 | 0 |
| 2 | 13.9 | 19.5 | 1,015 | 1,152 | 8,539 | 0 |
| 4 | 37.5 | 40.7 | 1,018 | 1,154 | 8,658 | 0 |
| 8 | 84.2 | 86.3 | 1,021 | 1,152 | 8,608 | 0 |
| 16 | 177.1 | 177.8 | 1,022 | 1,152 | 8,602 | 0 |
| 32 | 363.8 | 364.1 | 1,022 | 1,152 | 8,575 | 0 |

The ITL step from 16 ms → ~1 s between C=1 and C=2 is the pipeline
bubble: with 100K-token prefills in flight, decode iterations queue
behind prefill chunks crossing the pipeline boundary. Throughput
plateaus at ~8.6K tok/s from C=2 onward — the pipeline is saturated.

### GLM-5.2 vs GLM-5 (same PP=2 topology)

FP8 KV cache (`--kv-cache-dtype fp8`) is the primary differentiator:

| Metric | GLM-5 | GLM-5.2 | Delta |
| --- | --- | --- | --- |
| TTFT P50 (c=8) | 98.2s | 84.2s | **-14%** |
| TTFT P99 (c=8) | 101.0s | 86.3s | **-15%** |
| ITL P50 (c≥2) | ~1,120 ms | ~1,020 ms | **-9%** |
| ITL P99 (c≥2) | ~1,590 ms | ~1,152 ms | **-28%** |
| Throughput (c≥2) | ~7,400 tok/s | ~8,600 tok/s | **+16%** |

## Known Limits

- **No MTP with PP.** GLM-5.2's MTP speculative decoding is not yet
  supported under pipeline parallelism — open RFC
  [vllm#44697](https://github.com/vllm-project/vllm/issues/44697).
  This is the single biggest reason to prefer single-node TP=8 when
  the model fits.
- **Whole-group restarts.** `RecreateGroupOnPodRestart` means any pod
  failure restarts both nodes (~12 min). Budget for it.
- **Weights download twice** (once per node) unless you back the HF
  cache with shared storage.


## Cleanup

```bash
kubectl delete -n ${NAMESPACE} -f manifests/glm52-pp2-tp8-lws.yaml
```
