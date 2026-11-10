#!/bin/bash
# Evaluate a trained checkpoint on all 3 tasks and save evaluation/submission.json.
# Run from the handout directory (the one containing evaluation/).
#
# Usage:
#   bash evaluation/run_eval.sh <checkpoint_path> [limit] [temperature] [top_p]
#
# Examples:
#   bash evaluation/run_eval.sh "tinker://3eaa8c2d-...:train:0/sampler_weights/demo"        # 5-sample smoke test
#   bash evaluation/run_eval.sh "tinker://3eaa8c2d-...:train:0/sampler_weights/demo" 0      # full evaluation
#   bash evaluation/run_eval.sh "tinker://3eaa8c2d-...:train:0/sampler_weights/demo" 0 0.3 0.9

set -e

CHECKPOINT_PATH="${1:?Usage: bash evaluation/run_eval.sh <checkpoint_path> [limit] [temperature] [top_p]}"
LIMIT="${2:-5}"
TEMPERATURE="${3:-0.0}"
TOP_P="${4:-1.0}"
BASE_MODEL="Qwen/Qwen3.5-9B-Base"

echo "========================================"
echo "Checkpoint:  $CHECKPOINT_PATH"
echo "Base model:  $BASE_MODEL"
echo "Samples:     $LIMIT (0 = full eval)"
echo "Temperature: $TEMPERATURE"
echo "Top-p:       $TOP_P"
echo "========================================"

LIMIT_ARG=()
if [ "$LIMIT" != "0" ]; then
    LIMIT_ARG=(--limit "$LIMIT")
fi

python -m evaluation.eval_all \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --base_model "$BASE_MODEL" \
    "${LIMIT_ARG[@]}" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P"

echo ""
echo "Done. Submission saved to evaluation/submission.json"
