#!/usr/bin/env bash
# Start the AI + TM servers in the background.
#
# USAGE
#   ./start.sh                                      # default: deepseek/deepseek-v4-flash
#   OPENROUTER_MODEL=anthropic/claude-sonnet-4-6 ./start.sh
#   OPENROUTER_THINKING=off ./start.sh              # disable model reasoning (faster); auto|on|off
#   OPENROUTER_PROVIDER=Cloudflare ./start.sh       # pin upstream provider (comma-sep order ok)
#   LLM_DEBUG=false ./start.sh
#   TM_AI_DIR=/path TM_DIR=/path ./start.sh
#
# For multi-LLM death-match runs, use:
#   ./start-deathmatch.sh
#
# REQUIREMENTS
#   OPENROUTER_API_KEY must be set in ${TM_AI_DIR}/.env. The AI server derives
#   the provider from the model name: "vendor/model" → OpenRouter, "bare:tag"
#   → local Ollama.
#
# PORTS
#   8000 — AI server (FastAPI)
#   8080 — TM server (Node)
#   If either port is already bound the script lists the offending PIDs and
#   asks for confirmation before killing them. Decline → script exits.
#
# OUTPUT
#   Server PIDs   /tmp/tm-ai.pids
#   Session dir   ./logs/llm-test/<timestamp>/    (created at start)
#       ai-server.log    (AI server)
#       tm-server.log    (TM server)
#   The latest session is also symlinked at ./logs/llm-test/_latest for
#   convenience (`tail -f logs/llm-test/_latest/ai-server.log`).
#
# STOP
#   ./stop.sh

set -euo pipefail

TM_AI_DIR="${TM_AI_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
TM_DIR="${TM_DIR:-${TM_AI_DIR}/../terraforming-mars}"

# --- session log directory ----------------------------------------------------
SESSION_ID="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
SESSION_DIR="${TM_AI_DIR}/logs/llm-test/${SESSION_ID}"
mkdir -p "${SESSION_DIR}"
ln -sfn "${SESSION_ID}" "${TM_AI_DIR}/logs/llm-test/_latest"

AI_LOG="${SESSION_DIR}/ai-server.log"
TM_LOG="${SESSION_DIR}/tm-server.log"
PID_FILE=/tmp/tm-ai.pids

# --- env ----------------------------------------------------------------------
if [[ -f "${TM_AI_DIR}/.env" ]]; then
  set -a; source "${TM_AI_DIR}/.env"; set +a
fi

: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY is not set (add it to ${TM_AI_DIR}/.env)}"
export AI_TRAINING_LOG_DIR="${AI_TRAINING_LOG_DIR:-${TM_AI_DIR}/logs/training}"

# --- handle in-use ports ------------------------------------------------------
free_port() {
  local port=$1
  local pids
  pids=$(ss -ltnp "sport = :${port}" 2>/dev/null \
           | awk 'NR>1' \
           | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u | tr '\n' ' ' || true)
  [[ -z "${pids// }" ]] && return 0

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

# Per-model provider pins (e.g. deepseek → GMICloud/Baidu) live in openrouter.py's
# _MODEL_PROVIDER table; an explicit OPENROUTER_PROVIDER here still overrides them globally.
EFFECTIVE_MODEL="${OPENROUTER_MODEL:-deepseek/deepseek-v4-flash}"

# --- AI server ----------------------------------------------------------------
echo "▶ starting AI server (OpenRouter, model: ${EFFECTIVE_MODEL}, provider: ${OPENROUTER_PROVIDER:-auto}, log: ${AI_LOG})"
(
  cd "${TM_AI_DIR}/tm-ai-server"
  # OPENROUTER_API_KEY is already exported via `set -a; source .env` above.
  OPENROUTER_MODEL="${EFFECTIVE_MODEL}" \
  OPENROUTER_PROVIDER="${OPENROUTER_PROVIDER:-}" \
  OPENROUTER_THINKING="${OPENROUTER_THINKING:-off}" \
  OPENROUTER_MAX_OUTPUT_TOKENS="${OPENROUTER_MAX_OUTPUT_TOKENS:-4096}" \
  LLM_DEBUG="${LLM_DEBUG:-true}" \
  exec uv run uvicorn tm_llm.app:app --host "${HOST:-127.0.0.1}" --port 8000
) >> "${AI_LOG}" 2>&1 &
AI_PID=$!
echo "${AI_PID}" >> "${PID_FILE}"
echo "  PID=${AI_PID}"

# --- wait for AI /health ------------------------------------------------------
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

# --- TM server ----------------------------------------------------------------
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

# --- wait for TM root ---------------------------------------------------------
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

# --- done ---------------------------------------------------------------------
echo
echo "✅ servers up"
echo "   AI server :8000  (PID ${AI_PID})  ${AI_LOG}"
echo "   TM server :8080  (PID ${TM_PID})  ${TM_LOG}"
echo "   session dir      ${SESSION_DIR}"
echo "                    symlinked at logs/llm-test/_latest"
echo
echo "stop with:  ./stop.sh"
