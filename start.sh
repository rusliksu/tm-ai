#!/usr/bin/env bash
# Start both servers in the background with the typical Gemini-LLM configuration.
#
# USAGE
#   ./start.sh                        # start AI (Gemini flash-lite) + TM servers
#   GEMINI_MODEL=gemini-2.5-pro ./start.sh
#   LLM_DEBUG=false ./start.sh
#   TM_AI_DIR=/path TM_DIR=/path ./start.sh
#
# REQUIREMENTS
#   Reads `${TM_AI_DIR}/.env` for `GEMINI_API_KEY` — that env file is the only
#   place the key should live. Aborts immediately if the key is missing.
#
# PORTS
#   8000 — AI server (FastAPI)
#   8080 — TM server (Node)
#   If either port is already bound, the script lists the offending PIDs and
#   asks for confirmation before killing them. Decline → script exits.
#
# OUTPUT
#   PIDs printed at the end AND written to /tmp/tm-ai.pids
#   Logs:  /tmp/ai-server.log  (AI)   /tmp/tm-server.log  (TM)
#
# STOP
#   kill <AI_PID> <TM_PID>          # PIDs from this script's output, or:
#   xargs kill < /tmp/tm-ai.pids

set -euo pipefail

TM_AI_DIR="${TM_AI_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
TM_DIR="${TM_DIR:-${TM_AI_DIR}/../terraforming-mars}"
AI_LOG=/tmp/ai-server.log
TM_LOG=/tmp/tm-server.log
PID_FILE=/tmp/tm-ai.pids

# --- env --------------------------------------------------------------------
if [[ -f "${TM_AI_DIR}/.env" ]]; then
  set -a; source "${TM_AI_DIR}/.env"; set +a
fi
: "${GEMINI_API_KEY:?GEMINI_API_KEY is not set (add it to ${TM_AI_DIR}/.env)}"
export AI_TRAINING_LOG_DIR="${AI_TRAINING_LOG_DIR:-${TM_AI_DIR}/logs/training}"

# --- handle in-use ports ----------------------------------------------------
free_port() {
  local port=$1
  # PIDs listening on this port (skip the header line). The trailing `|| true`
  # is critical: `grep -oE` returns 1 when nothing matches (port is free), and
  # under `pipefail` that would propagate into the $(...) assignment and abort
  # the script via `set -e` — silently, after killing on the first port.
  local pids
  pids=$(ss -ltnp "sport = :${port}" 2>/dev/null \
           | awk 'NR>1' \
           | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u | tr '\n' ' ' || true)
  [[ -z "${pids// }" ]] && return 0  # port is free

  echo "⚠ port ${port} is already in use by PID(s):${pids%% }"
  ps -o pid=,user=,cmd= -p ${pids} 2>/dev/null | sed 's/^/    /'
  if [[ -t 0 ]]; then
    read -r -p "  kill these process(es)? [y/N] " answer
  else
    answer=""
  fi
  if [[ "${answer,,}" == "y" || "${answer,,}" == "yes" ]]; then
    # shellcheck disable=SC2086
    kill ${pids} 2>/dev/null || true
    # Give them up to 5s to exit gracefully, then escalate to SIGKILL
    for _ in {1..10}; do
      if ! ss -ltn "sport = :${port}" 2>/dev/null | grep -q LISTEN; then
        echo "  ✔ port ${port} freed"
        return 0
      fi
      sleep 0.5
    done
    # shellcheck disable=SC2086
    kill -9 ${pids} 2>/dev/null || true
    sleep 0.5
    if ss -ltn "sport = :${port}" 2>/dev/null | grep -q LISTEN; then
      echo "❌ could not free port ${port}" >&2
      exit 1
    fi
    echo "  ✔ port ${port} freed (SIGKILL)"
  else
    echo "❌ aborting — port ${port} still in use" >&2
    exit 1
  fi
}

for port in 8000 8080; do
  free_port "${port}"
done

: > "${PID_FILE}"

# --- AI server (Gemini LLM, debug on) ---------------------------------------
echo "▶ starting AI server (Gemini, log: ${AI_LOG})"
(
  cd "${TM_AI_DIR}/tm-ai-server"
  USE_LLM=true LLM_PROVIDER=gemini \
  GEMINI_MODEL="${GEMINI_MODEL:-gemini-2.5-flash-lite}" \
  GEMINI_API_KEY="${GEMINI_API_KEY}" \
  LLM_DEBUG="${LLM_DEBUG:-true}" \
  exec uv run uvicorn tm_ai_server.main:app --host 0.0.0.0 --port 8000
) >> "${AI_LOG}" 2>&1 &
AI_PID=$!
echo "${AI_PID}" >> "${PID_FILE}"
echo "  PID=${AI_PID}"

# --- wait for AI /health ----------------------------------------------------
for i in {1..30}; do
  if curl -fsS http://localhost:8000/health -o /dev/null 2>/dev/null; then
    echo "  ✔ AI /health OK"
    break
  fi
  if ! kill -0 "${AI_PID}" 2>/dev/null; then
    echo "❌ AI server died during startup — see ${AI_LOG}" >&2
    tail -20 "${AI_LOG}" >&2
    exit 1
  fi
  sleep 1
  [[ $i -eq 30 ]] && { echo "❌ AI /health did not respond in 30s — see ${AI_LOG}" >&2; exit 1; }
done

# --- TM server --------------------------------------------------------------
echo "▶ starting TM server (log: ${TM_LOG})"
if [[ ! -f "${TM_DIR}/build/src/server/server.js" ]]; then
  echo "  building TM server first..."
  (cd "${TM_DIR}" && npm run build:server) >> "${TM_LOG}" 2>&1
fi
(
  cd "${TM_DIR}"
  exec node build/src/server/server.js
) >> "${TM_LOG}" 2>&1 &
TM_PID=$!
echo "${TM_PID}" >> "${PID_FILE}"
echo "  PID=${TM_PID}"

# --- wait for TM root -------------------------------------------------------
for i in {1..30}; do
  if curl -fsS http://localhost:8080/ -o /dev/null 2>/dev/null; then
    echo "  ✔ TM server responding on :8080"
    break
  fi
  if ! kill -0 "${TM_PID}" 2>/dev/null; then
    echo "❌ TM server died during startup — see ${TM_LOG}" >&2
    tail -20 "${TM_LOG}" >&2
    exit 1
  fi
  sleep 1
  [[ $i -eq 30 ]] && { echo "❌ TM server did not respond in 30s — see ${TM_LOG}" >&2; exit 1; }
done

# --- done -------------------------------------------------------------------
echo
echo "✅ both servers up"
echo "   AI server :8000  (PID ${AI_PID})  ${AI_LOG}"
echo "   TM server :8080  (PID ${TM_PID})  ${TM_LOG}"
echo "   open: http://localhost:8080"
echo
echo "stop with:  kill ${AI_PID} ${TM_PID}     # or:  xargs kill < ${PID_FILE}"
