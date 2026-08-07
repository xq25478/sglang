#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_DIR="${ARTIFACT_DIR:?ARTIFACT_DIR must be set}"
mkdir -p "${ARTIFACT_DIR}"

exec python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill-policy round_robin \
  --prefill http://6.200.20.13:8182 18956 \
  --decode http://6.200.20.13:8180 \
  --host 6.200.20.13 \
  --port 8187 \
  --health-check-timeout-secs 25 \
  2>&1 | tee "${ARTIFACT_DIR}/router.log"
