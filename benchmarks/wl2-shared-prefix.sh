#!/bin/bash
# WL2: Shared prefix — tests prefix cache reuse
# Run inside a vLLM pod
#
# Customize: PREFIX_LEN=4000 SUFFIX_LEN=1000 OSL=500 bash wl2-shared-prefix.sh

PREFIX_LEN=${PREFIX_LEN:-80000}
SUFFIX_LEN=${SUFFIX_LEN:-20000}
OSL=${OSL:-1024}
NUM_PREFIXES=${NUM_PREFIXES:-10}
C=${C:-8}
N=${N:-150}
MODEL=${MODEL:-glm-5.2-fp8}
TOKENIZER=${TOKENIZER:-zai-org/GLM-5.2-FP8}
ENDPOINT=${ENDPOINT:-http://vllm-roundrobin:8000}

echo "=== WL2: Shared prefix ==="
echo "Prefix=$PREFIX_LEN  Suffix=$SUFFIX_LEN  OSL=$OSL"
echo "Groups=$NUM_PREFIXES  C=$C  N=$N"
echo "Endpoint: $ENDPOINT"
echo ""

vllm bench serve \
  --model "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --base-url "$ENDPOINT" \
  --dataset-name prefix_repetition \
  --prefix-repetition-num-prefixes "$NUM_PREFIXES" \
  --prefix-repetition-prefix-len "$PREFIX_LEN" \
  --prefix-repetition-suffix-len "$SUFFIX_LEN" \
  --prefix-repetition-output-len "$OSL" \
  --num-prompts "$N" \
  --max-concurrency "$C" \
  --seed 123 \
  --port 8000
