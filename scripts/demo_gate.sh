#!/usr/bin/env bash
#
# RAGGate AI — end-to-end regression gate demo.
#
# Creates two retrieval runs (one good, one regressed), gates them,
# and shows the gate blocking the regression. Exits 0 if the demo
# behaved as expected, 1 otherwise.
#
# Usage:  bash scripts/demo_gate.sh

set -euo pipefail

API="http://127.0.0.1:8000"
THRESHOLDS="config/thresholds.json"
DB="data/raggate.sqlite"

cleanup() {
  if [ -n "${SERVER_PID:-}" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "== RAGGate AI — regression gate demo =="
echo

echo "[1/5] Starting API server..."
rm -f "$DB"
uv run uvicorn raggate.main:app --host 127.0.0.1 --port 8000 >/tmp/raggate-server.log 2>&1 &
SERVER_PID=$!

for i in $(seq 1 30); do
  if curl -s -m 2 "$API/health" >/dev/null 2>&1; then
    echo "      server up (pid $SERVER_PID)"
    break
  fi
  if [ "$i" = "30" ]; then
    echo "      ERROR: server did not become healthy" >&2
    cat /tmp/raggate-server.log >&2
    exit 1
  fi
  sleep 1
done

start_run() {
  local retriever="$1"
  curl -s -X POST "$API/eval/run" \
    -H "Content-Type: application/json" \
    -d "{\"kind\":\"retrieval\",\"retriever\":\"$retriever\",\"k\":5}" \
    | python -c "import sys,json; print(json.load(sys.stdin)['run_id'])"
}

wait_for_run() {
  local id="$1"
  for i in $(seq 1 60); do
    local status
    status=$(curl -s -m 3 "$API/eval/runs/$id" \
      | python -c "import sys,json; print(json.load(sys.stdin)['summary']['status'])" 2>/dev/null || echo "unknown")
    [ "$status" = "succeeded" ] && return 0
    [ "$status" = "failed" ] && echo "      ERROR: run $id failed" >&2 && return 1
    sleep 2
  done
  echo "      ERROR: run $id did not finish in time" >&2
  return 1
}

echo "[2/5] Running baseline (chroma)..."
BASE=$(start_run "chroma")
wait_for_run "$BASE"
echo "      baseline run: $BASE"

echo "[3/5] Running candidate (keyword — regressed)..."
CAND=$(start_run "keyword")
wait_for_run "$CAND"
echo "      candidate run: $CAND"

echo "[4/5] Gating baseline -> candidate (expect FAIL)..."
set +e
uv run raggate-gate check \
  --baseline "$BASE" \
  --candidate "$CAND" \
  --db "$DB" \
  --thresholds "$THRESHOLDS"
FAIL_CODE=$?
set -e

echo
if [ "$FAIL_CODE" = "1" ]; then
  echo "      [ok] gate correctly blocked the regression (exit=$FAIL_CODE)"
else
  echo "      [FAIL] expected exit=1, got exit=$FAIL_CODE" >&2
  exit 1
fi

echo
echo "[5/5] Gating candidate -> baseline (expect PASS)..."
uv run raggate-gate check \
  --baseline "$CAND" \
  --candidate "$BASE" \
  --db "$DB" \
  --thresholds "$THRESHOLDS"
PASS_CODE=$?

echo
if [ "$PASS_CODE" = "0" ]; then
  echo "      [ok] gate correctly allowed the improvement (exit=$PASS_CODE)"
else
  echo "      [FAIL] expected exit=0, got exit=$PASS_CODE" >&2
  exit 1
fi

echo
echo "== Demo complete =="
