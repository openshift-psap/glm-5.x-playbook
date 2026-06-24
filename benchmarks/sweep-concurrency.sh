#!/bin/bash
# Concurrency sweep benchmark for GLM-5.2-FP8
# Run inside the vLLM leader pod
#
# Customize ISL/OSL for your workload:
#   ISL=2000 OSL=500 bash sweep-concurrency.sh

ISL=${ISL:-100000}
OSL=${OSL:-10}
MODEL=${MODEL:-glm-5.2-fp8}
TOKENIZER=${TOKENIZER:-zai-org/GLM-5.2-FP8}
PORT=${PORT:-8080}

echo "=============================="
echo "  GLM-5.2-FP8 Concurrency Sweep"
echo "  ISL=$ISL  OSL=$OSL"
echo "  $(date -u)"
echo "=============================="

for C in 1 2 4 8 16 32; do
  N=$((C * 10))
  echo ""
  echo "===== Concurrency=$C  Prompts=$N ====="

  vllm bench serve \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --random-input-len "$ISL" \
    --random-output-len "$OSL" \
    --num-prompts "$N" \
    --seed $((ISL + C)) \
    --max-concurrency "$C" \
    --port "$PORT"
done
