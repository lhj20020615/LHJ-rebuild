#!/usr/bin/env bash
set -euo pipefail

: "${DATA_DIR:?Set DATA_DIR to YOUR_DATA_DIR, the directory containing train_set.xlsx, val_set.xlsx and test_set.xlsx}"
: "${MODEL_DIR:?Set MODEL_DIR to YOUR_MODEL_DIR, the local model directory}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to YOUR_OUTPUT_DIR, the shared result root}"

python -m src.train_multimodal \
  --data-dir "$DATA_DIR" \
  --model-dir "$MODEL_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --ablation "${ABLATION:-full}" \
  --epochs "${EPOCHS:-15}" \
  --batch-size "${BATCH_SIZE:-8}"

