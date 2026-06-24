#!/bin/bash
# WL1: Unique prompts — no prefix cache benefit (control workload)
# Run inside a vLLM pod
#
# Customize: ISL=2000 OSL=500 bash wl1-unique.sh

ISL=${ISL:-100000}
OSL=${OSL:-500}
C=${C:-8}
N=${N:-150}
MODEL=${MODEL:-glm-5.2-fp8}
TOKENIZER=${TOKENIZER:-zai-org/GLM-5.2-FP8}
ENDPOINT=${ENDPOINT:-http://vllm-roundrobin:8000}

echo "=== WL1: Unique prompts (control) ==="
echo "ISL=$ISL  OSL=$OSL  C=$C  N=$N"
echo "Endpoint: $ENDPOINT"
echo ""

vllm bench serve \
  --model "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --base-url "$ENDPOINT" \
  --random-input-len "$ISL" \
  --random-output-len "$OSL" \
  --num-prompts "$N" \
  --max-concurrency "$C" \
  --seed 42 \
  --port 8000
