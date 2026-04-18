#!/usr/bin/env bash
#
# Run guidellm benchmark with synthetic data across multiple ISL/OSL profiles.
#
# Usage:
#   TARGET=http://host:port MODEL=mymodel bash run_guidellm_synthetic_profiles.sh
#
set -euo pipefail

# --- Required config (no defaults) ---
: "${TARGET:?TARGET must be set (e.g. http://host:8080)}"
: "${MODEL:?MODEL must be set (e.g. openai/gpt-oss-120b)}"
PROCESSOR="${PROCESSOR:-$MODEL}"

# --- Optional config ---
OUTPUT_DIR="${OUTPUT_DIR:-/home/naveen/results}"
MAX_SECONDS="${MAX_SECONDS:-450}"

# Comma-separated concurrency rates (single guidellm run per profile)
RATE="${RATE:-1,2}"

# Output filename prefix
OUTPUT_PREFIX="${OUTPUT_PREFIX:-synthetic}"

# --- Profiles ---
PROFILE_LABELS=(
  "isl128-osl128"
)

PROFILE_DATA=(
  '{"prompt_tokens":128,"output_tokens":128}'
)

echo "=============================================="
echo "GuideLLM synthetic profile benchmarks"
echo "=============================================="
echo "Target:      $TARGET"
echo "Model:       $MODEL"
echo "Processor:   $PROCESSOR"
echo "Output dir:  $OUTPUT_DIR"
echo "Max seconds: $MAX_SECONDS"
echo "Rate:        $RATE"
echo "Profiles:    ${PROFILE_LABELS[*]}"
echo "=============================================="

for idx in "${!PROFILE_LABELS[@]}"; do
  label="${PROFILE_LABELS[$idx]}"
  data="${PROFILE_DATA[$idx]}"
  output_file="${OUTPUT_PREFIX}-${label}.json"

  echo ""
  echo "############################################"
  echo "# Profile: ${label}"
  echo "# Data:    ${data}"
  echo "# Output:  ${output_file}"
  echo "############################################"
  echo ""

   numactl --cpunodebind=0 --membind=0  guidellm benchmark \
    --target "$TARGET" \
    --model "$MODEL" \
    --processor "$PROCESSOR" \
    --data "$data" \
    --rate-type concurrent \
    --rate "$RATE" \
    --backend-kwargs '{"timeout":100000}' \
    --max-seconds "$MAX_SECONDS" \
    --output-dir "$OUTPUT_DIR" \
    --outputs "$output_file"

  echo ""
  echo ">>> Done profile ${label}"
done

echo ""
echo "=============================================="
echo "All runs finished. Results in $OUTPUT_DIR"
echo "=============================================="
