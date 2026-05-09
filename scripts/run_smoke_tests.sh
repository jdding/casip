#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python3 tools/syntax_check.py
python3 tools/check_results.py

if find . -path './.git' -prune -o \( -name '.DS_Store' -o -name '__MACOSX' -o -name '._*' -o -name '__pycache__' \) -print | grep -q .; then
  echo "Found local archive or cache artifacts." >&2
  exit 1
fi

if find . \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.pkl' -o -name '*.npy' -o -name '*.npz' -o -name '*.pdf' -o -name '*.tex' \) | grep -q .; then
  echo "Found excluded binary/model/paper artifacts." >&2
  exit 1
fi

slash="/"
local_home_pat="${slash}Users${slash}"
remote_root_pat="root""@"
if grep -R -n -E --exclude-dir='.git' --exclude='run_smoke_tests.sh' "${local_home_pat}|${remote_root_pat}" . >/tmp/casp_review_scan.txt; then
  cat /tmp/casp_review_scan.txt >&2
  echo "Found local identity or tool traces." >&2
  exit 1
fi

echo "OK: reviewer smoke tests passed."
