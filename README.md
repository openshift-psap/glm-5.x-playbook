# GLM-5 Pipeline Parallel with vLLM + LWS

Multi-node pipeline parallel serving of GLM-5 using LeaderWorkerSet and vLLM's multiprocessing backend.

## Prerequisites

- LWS controller installed on cluster
- Composite DRA webhook installed and enabled
- `oc` CLI with cluster-admin access

## Cluster Setup

```bash
# Create namespace
oc new-project glm-5-pp

# Label namespace for DRA webhook
oc label ns glm-5-pp composite.dra/webhook-enabled=true

# Create HF token secret
oc create secret generic hf-token --from-literal=token=<YOUR_HF_TOKEN> -n glm-5-pp

# Grant SCCs to default service account
oc adm policy add-scc-to-user anyuid -z default -n glm-5-pp
oc adm policy add-scc-to-user privileged -z default -n glm-5-pp
```

## Deploy

### With dummy weights (no model download needed)

```bash
oc apply -f lws-pp-multiproc.yaml -n glm-5-pp
```

### With real weights on NVMe hostPath (poseidon)

```bash
# Download to one node
oc apply -f download-model-job.yaml -n glm-5-pp

# Sync to remaining nodes
oc apply -f sync-model-jobs.yaml -n glm-5-pp

# Or download FP8 to all nodes in parallel
oc apply -f download-glm5-fp8-job.yaml -n glm-5-pp
```

## Manifests

| File | Description |
|------|-------------|
| `lws-pp-multiproc.yaml` | LWS with vLLM multiprocessing backend (PP=2, TP=8, no Ray) |
| `lws-pp-ray.yaml` | LWS with Ray backend variant |
| `download-model-job.yaml` | Download BF16 model to one node's NVMe |
| `download-glm5-fp8-job.yaml` | Parallel FP8 download to all GPU nodes |
| `sync-model-jobs.yaml` | Rsync server + client for cross-node model sync |

## Clean Up

```bash
oc delete lws vllm-pp -n glm-5-pp
oc delete resourceclaimtemplates --all -n glm-5-pp
oc delete resourceclaims --all -n glm-5-pp
```
