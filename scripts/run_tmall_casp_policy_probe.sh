#!/usr/bin/env bash
set -euo pipefail

TMALL_INPUT="${TMALL_INPUT:-data/tmall}"
TMALL_OUTPUT="${TMALL_OUTPUT:-results/20260506_tmall_casp_policy_probe}"
TMALL_CHUNKSIZE="${TMALL_CHUNKSIZE:-1000000}"

echo "[run] Tmall CASP policy probe"
echo "[run] input=${TMALL_INPUT}"
echo "[run] output=${TMALL_OUTPUT}"

python3 scripts/train_agknet/tmall_casp_policy_probe.py \
  --input-dir "${TMALL_INPUT}" \
  --output-dir "${TMALL_OUTPUT}" \
  --chunksize "${TMALL_CHUNKSIZE}"
