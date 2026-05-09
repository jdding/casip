#!/usr/bin/env bash
set -euo pipefail

TMALL_INPUT="${TMALL_INPUT:-data/tmall}"
TMALL_OUTPUT="${TMALL_OUTPUT:-results/20260506_tmall_support_audit}"
TMALL_CHUNKSIZE="${TMALL_CHUNKSIZE:-1000000}"
TMALL_WINDOWS="${TMALL_WINDOWS:-pre:501:1001,gap:1001:1101,val:1101:1111,test:1111:1112}"
TMALL_GATE="${TMALL_GATE:-pre_any_gap_silent_val_proxy_test_purchase}"

echo "[run] Tmall no-training support audit"
echo "[run] input=${TMALL_INPUT}"
echo "[run] output=${TMALL_OUTPUT}"
echo "[run] gate=${TMALL_GATE}"

python3 scripts/train_agknet/tmall_no_training_support_audit.py \
  --input-dir "${TMALL_INPUT}" \
  --output-dir "${TMALL_OUTPUT}" \
  --chunksize "${TMALL_CHUNKSIZE}" \
  --windows "${TMALL_WINDOWS}" \
  --gate "${TMALL_GATE}"
