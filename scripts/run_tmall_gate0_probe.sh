#!/usr/bin/env bash
set -euo pipefail

TMALL_INPUT="${TMALL_INPUT:-data/tmall}"
TMALL_OUTPUT="${TMALL_OUTPUT:-results/20260506_tmall_gate0_probe}"
TMALL_WORKERS="${TMALL_WORKERS:-}"
TMALL_CHUNKSIZE="${TMALL_CHUNKSIZE:-1000000}"
TMALL_MAX_ROWS="${TMALL_MAX_ROWS:-0}"
TMALL_WINDOWS="${TMALL_WINDOWS:-pre:501:1001,gap:1001:1101,val:1101:1111,test:1111:1112}"

echo "[run] Tmall Gate-0 metadata probe"
echo "[run] input=${TMALL_INPUT}"
echo "[run] output=${TMALL_OUTPUT}"
echo "[run] chunksize=${TMALL_CHUNKSIZE}"
echo "[run] max_rows=${TMALL_MAX_ROWS}"
echo "[run] windows=${TMALL_WINDOWS}"

python3 scripts/train_agknet/tmall_gate0_fast_probe.py \
  --input-dir "${TMALL_INPUT}" \
  --output-dir "${TMALL_OUTPUT}" \
  --chunksize "${TMALL_CHUNKSIZE}" \
  --max-rows "${TMALL_MAX_ROWS}" \
  --windows "${TMALL_WINDOWS}"
