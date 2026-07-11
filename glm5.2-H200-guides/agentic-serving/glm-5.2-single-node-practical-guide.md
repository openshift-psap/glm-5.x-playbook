# GLM 5.2 FP8 Single-Node Practical Guide

Deploy and benchmark GLM 5.2 FP8 on a single 8×H200 node with vLLM.
No llm-d, no EPP, no P/D disaggregation, no multi-node — just vLLM
serving an OpenAI-compatible API with prefix caching and MTP.

This is what we actually ran and measured.

## What You Get

- vLLM serving `/v1/chat/completions` (OpenAI) and `/v1/messages` (Anthropic)
- TP=8 across 8 H200 GPUs within one node
- FP8 weights + FP8 KV cache (MLA compressed)
- MTP speculative decoding (5 draft tokens)
- Tool calling via `glm47` parser + reasoning via `glm45` parser
- Prefix caching (vLLM native, no external router)
- Claude Code connects via `ANTHROPIC_BASE_URL`

## What You Don't Get

- No llm-d EPP routing (prefix-cache-aware routing across replicas)
- No P/D disaggregation (prefill and decode share the same GPUs)
- No multi-node (no NIXL KV transfer, no RoCE/InfiniBand cross-node)
- No expert parallelism (TP=8, not EP=8)

## Hardware

- 1 node with 8× NVIDIA H200 (141 GB HBM3e each)
- 1,128 GB total GPU memory
- ~743 GB for FP8 weights (actual on-disk: 756 GB including scale factors)
- ~279 GB remaining for KV cache after weights + overhead
- ~6.2M tokens of MLA KV cache capacity (FP8)

## Step 1: Deploy with a StatefulSet

```yaml
# glm52-statefulset.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: vllm
spec:
  serviceName: vllm-headless
  replicas: 1
  selector:
    matchLabels:
      app: vllm-glm52
  template:
    metadata:
      labels:
        app: vllm-glm52
    spec:
      containers:
      - name: vllm
        # GLM-5.2 requires vLLM >= v0.23.0.
        image: vllm/vllm-openai:v0.23.0
        command: ["/bin/bash", "-c"]
        args:
        - |
          vllm serve zai-org/GLM-5.2-FP8 \
            --tensor-parallel-size 8 \
            --kv-cache-dtype fp8 \
            --speculative-config.method mtp \
            --speculative-config.num_speculative_tokens 5 \
            --tool-call-parser glm47 \
            --reasoning-parser glm45 \
            --enable-auto-tool-choice \
            --served-model-name glm-5.2-fp8 \
            --enable-prefix-caching \
            --enable-chunked-prefill \
            --max-model-len 131072 \
            --trust-remote-code \
            --port 8000
        env:
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: llm-d-hf-token
              key: HF_TOKEN
        ports:
        - containerPort: 8000
          name: http
        resources:
          limits:
            nvidia.com/gpu: "8"
            memory: 512Gi
          requests:
            nvidia.com/gpu: "8"
            memory: 512Gi
            cpu: 32
        startupProbe:
          httpGet:
            path: /health
            port: 8000
          periodSeconds: 10
          failureThreshold: 180
        readinessProbe:
          httpGet:
            path: /v1/models
            port: 8000
          periodSeconds: 10
        volumeMounts:
        - name: dshm
          mountPath: /dev/shm
        - name: hf-cache
          mountPath: /root/.cache/huggingface
      volumes:
      - name: dshm
        emptyDir:
          medium: Memory
          sizeLimit: 2Gi
      - name: hf-cache
        # FP8 weights are 756 GB on disk. The node needs that much free
        # ephemeral storage, and every pod reschedule re-downloads.
        # For anything beyond a first run, swap for a PVC or hostPath.
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-headless
spec:
  clusterIP: None
  selector:
    app: vllm-glm52
  ports:
  - port: 8000
    targetPort: 8000
```

```bash
# Create namespace and HF token
kubectl create namespace glm52
kubectl create secret generic llm-d-hf-token \
  --from-literal=HF_TOKEN=$HF_TOKEN -n glm52

# Deploy
kubectl apply -f glm52-statefulset.yaml -n glm52

# Wait (model load takes 15-30 min on first deploy, ~10 min if cached)
kubectl wait --for=condition=Ready pod/vllm-0 -n glm52 --timeout=1800s
```

## Step 2: Verify the Model Is Serving

```bash
kubectl port-forward -n glm52 svc/vllm-headless 8000:8000 &

# Check model is loaded
curl -s http://localhost:8000/v1/models | python3 -c "
import json, sys
d = json.load(sys.stdin)
m = d['data'][0]
print(f'Model: {m[\"id\"]}')
print(f'Max model len: {m[\"max_model_len\"]}')
"
```

Expected: `Model: glm-5.2-fp8`, `Max model len: 131072`

## Step 3: Verify Tool Calling

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-5.2-fp8",
    "messages": [{"role": "user", "content": "Read the file test.txt"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "read_file",
        "description": "Read a file from disk",
        "parameters": {
          "type": "object",
          "properties": {"path": {"type": "string"}},
          "required": ["path"]
        }
      }
    }]
  }' | python3 -c "
import json, sys
r = json.load(sys.stdin)
tc = r['choices'][0]['message'].get('tool_calls')
if tc:
    print(f'Tool: {tc[0][\"function\"][\"name\"]}')
    print(f'Args: {tc[0][\"function\"][\"arguments\"]}')
else:
    print('NO TOOL CALLS - check --tool-call-parser glm47 and --enable-auto-tool-choice')
"
```

Expected: `Tool: read_file`, `Args: {"path": "test.txt"}`

## Step 4: Connect Claude Code

Claude Code only speaks the Anthropic Messages API (`/v1/messages`).
vLLM serves this endpoint natively (requires v0.23.0+ for GLM-5.2).

```bash
# With port-forward already running on localhost:8000
ANTHROPIC_BASE_URL=http://localhost:8000 \
ANTHROPIC_API_KEY=dummy \
ANTHROPIC_MODEL=glm-5.2-fp8 \
ANTHROPIC_SMALL_FAST_MODEL=glm-5.2-fp8 \
CLAUDE_CODE_USE_VERTEX=0 \
claude
```

Test prompts:

| Prompt | Expected behavior |
|--------|------------------|
| `What files are in this directory?` | Calls Bash tool, returns ls output |
| `Create a file called hello.py that prints hello` | Calls Write tool |
| `Now add a main guard to hello.py` | Reads file, calls Edit tool |

## Step 5: Run inference-perf Weka Trace Replay

This replays real Claude Code sessions from the SemiAnalysis Weka
trace corpus against your vLLM endpoint using inference-perf's
graph-based executor.

### 5a. Create the runner pod

```bash
kubectl apply -n glm52 -f - <<'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: inference-perf-runner
spec:
  restartPolicy: Never
  containers:
  - name: runner
    image: python:3.12
    command: ["sleep", "86400"]
    resources:
      requests:
        cpu: "4"
        memory: "8Gi"
    volumeMounts:
    - name: workspace
      mountPath: /workspace
  volumes:
  - name: workspace
    emptyDir: {}
EOF
kubectl wait --for=condition=Ready pod/inference-perf-runner -n glm52 --timeout=300s
```

### 5b. Install inference-perf (needs main branch for weka_trace_replay)

```bash
kubectl exec inference-perf-runner -n glm52 -- bash -c '
export PYTHONUSERBASE=/workspace/.local
cd /workspace
git clone --depth 1 https://github.com/kubernetes-sigs/inference-perf.git
pip install --user pydantic datasets transformers huggingface_hub \
  aiohttp tiktoken aiofiles tokenizers sentencepiece protobuf
cd inference-perf && pip install --user -e .
'
```

### 5c. Write the benchmark config

```bash
ENDPOINT="http://vllm-0.vllm-headless.glm52.svc.cluster.local:8000"

kubectl exec inference-perf-runner -n glm52 -- bash -c "cat > /workspace/config.yaml <<'CFGEOF'
load:
  type: trace_session_replay
  stages:
    - concurrent_sessions: 8
      num_sessions: 50
  num_workers: 4
  worker_max_concurrency: 32
  request_timeout: 120
api:
  type: chat
  streaming: true
server:
  type: vllm
  model_name: glm-5.2-fp8
  base_url: ${ENDPOINT}
tokenizer:
  pretrained_model_name_or_path: Qwen/Qwen2.5-0.5B
data:
  type: weka_trace_replay
  weka_trace_replay:
    hf_dataset_path: semianalysisai/cc-traces-weka-061526
    num_dataset_entries: 50
    use_static_model: true
    static_model_name: glm-5.2-fp8
    default_block_size: 64
    skip_invalid_files: true
    trace_idle_gap_cap_seconds: 1.0
report:
  request_lifecycle:
    summary: true
    per_stage: true
    per_request: true
  goodput:
    constraints:
      ttft: 5.0
      itl: 0.1
    percentile: p90
storage:
  local_storage:
    path: /workspace
CFGEOF"
```

Note on tokenizer: `Qwen/Qwen2.5-0.5B` is used as a lightweight
proxy. inference-perf's weka_trace_replay reconstructs prompts to
match trace token counts. A mismatched tokenizer means the
reconstructed prompts may tokenize to slightly different lengths on
GLM 5.2, reducing prefix cache alignment accuracy. This affects
cache hit rate measurement, not throughput or latency.

### 5d. Run the benchmark

```bash
kubectl exec inference-perf-runner -n glm52 -- bash -c '
export PYTHONUSERBASE=/workspace/.local
export PYTHONPATH=/workspace/.local/lib/python3.12/site-packages:/workspace/inference-perf
export HF_HOME=/workspace/.hf_cache
cd /workspace/inference-perf
python3 -m inference_perf.main -c /workspace/config.yaml
'
```

Expected runtime: 30-90 minutes. 50 sessions, 7,139 events.
Sessions with requests exceeding max-model-len will fail and be
cancelled — this is expected for real agentic traces.

### 5e. Save results locally

```bash
mkdir -p ./results
for f in per_request_lifecycle_metrics.json \
         summary_lifecycle_metrics.json \
         stage_0_lifecycle_metrics.json \
         summary_session_lifecycle_metrics.json; do
  kubectl cp glm52/inference-perf-runner:/workspace/$f ./results/$f 2>/dev/null
done
```

### 5f. Collect Prometheus metrics

```bash
kubectl exec vllm-0 -n glm52 -- curl -s http://localhost:8000/metrics \
  | grep -E "vllm:prefix_cache|vllm:prompt_tokens_total|vllm:generation_tokens_total|vllm:request_success_total"
```

Key metrics:
- `vllm:prefix_cache_hits_total` / `vllm:prefix_cache_queries_total` = cache hit rate
- `vllm:prompt_tokens_total` = total prefill tokens processed
- `vllm:generation_tokens_total` = total output tokens generated

## What We Measured 

Cluster: 1×8×H200, co-located TP=8.
vLLM nightly, GLM-5.2-FP8, MTP 5 tokens, prefix caching on.

### inference-perf Weka Trace Replay (50 sessions, 131K max-model-len)

| Metric | Value |
|--------|-------|
| TTFT median | 5.96s (66K avg input) |
| TTFT P90 | 32.3s |
| ITL P90 | 36.9ms |
| TPOT median | 8.7ms |
| Prefix cache hit rate | 60.1% (476M / 793M tokens) |
| Requests served | 1,219 of 7,139 events |
| Sessions completed | 8 of 50 (42 exceeded 131K context) |
| Prompt tokens processed | 82.8M |

Note: prefix cache hit rate is from vLLM's native prefix caching
on a single replica, not from llm-d EPP routing across replicas.

### Synthetic Load Test (500 requests, 8K fixed input)

| Metric | Value |
|--------|-------|
| Throughput | 3,256 tok/s |
| E2E P50 / P90 / P99 | 4.48s / 5.23s / 7.72s |
| Prefix cache hit rate | 72% |
| Errors | 0 |

## Step 6: Clean Up

```bash
kubectl delete pod inference-perf-runner -n glm52
kubectl delete statefulset vllm -n glm52
kubectl delete svc vllm-headless -n glm52
kubectl delete namespace glm52
```

## Known Issues

| Issue | Impact | Mitigation |
|-------|--------|------------|
| GLM tool-call parsing fails intermittently when backing Claude Code ([#42400](https://github.com/vllm-project/vllm/issues/42400), closed; reported on GLM-5.1) | Intermittent malformed/missed tool calls in long agentic sessions | Update vLLM (fix landed); if it persists, test without MTP |
| MTP streaming can drop first tool-call arguments ([#41967](https://github.com/vllm-project/vllm/issues/41967), closed; reported on Gemma4+MTP) | Truncated tool call JSON in streaming multi-tool responses | Update vLLM; disable MTP (`--speculative-config` flags) if this appears |
| 42/50 Weka sessions exceed 131K | Most real Claude Code sessions need >131K context | Increase `--max-model-len` (trades KV capacity for context length) |
| FP8 weight files are 756 GB, not 744 GB | Calculator underestimates weight memory by ~12 GB | Scale factors and non-quantized layers add ~1.6% overhead |

## What This Guide Does NOT Cover

- **llm-d EPP routing**: requires deploying the llm-d router helm chart and InferencePool CRDs
- **P/D disaggregation**: requires 2 nodes, NIXL KV transfer, routing sidecar on decode
- **Expert parallelism (EP=8)**: requires DeepEP, NVSHMEM, InfiniBand/RoCE — see Elvir's [wide-ep-lws GLM 5.2 configs](https://github.com/llm-d/llm-d/pull/1947)
- **Multi-node PP2 TP8**: see [GLM 5.2 Multi-Node PP=2 TP=8 on H200](glm-5.2-multi-node-pp2-tp8.md)
- **Short-context throughput tuning (8K/1K)**: see [GLM 5.2 Single-Node 8K/1K Optimized Config](glm-5.2-single-node-8k1k-optimized.md)
- **Tiered KV offloading**: requires llm-d with CPU/filesystem KV tiers configured

Each of these adds capability but also complexity. Start here, get
numbers, then add layers.
