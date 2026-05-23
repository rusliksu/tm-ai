#!/usr/bin/env bash
# Start both servers (and optionally a death-match experiment) in the background.
#
# USAGE
#   ./start.sh                        # start AI (OpenRouter, single model) + TM servers
#   ./start.sh --death-match          # start AI (OpenRouter) + TM + 4-LLM game
#   OPENROUTER_MODEL=anthropic/claude-sonnet-4-6 ./start.sh
#   DEATH_MATCH_MODELS="anthropic/claude-sonnet-4-6,openai/gpt-4o-mini,google/gemini-2.5-flash,deepseek/deepseek-chat" ./start.sh --death-match
#   LLM_DEBUG=false ./start.sh
#   TM_AI_DIR=/path TM_DIR=/path ./start.sh
#
# REQUIREMENTS
#   Both modes require OPENROUTER_API_KEY in ${TM_AI_DIR}/.env
#   (the AI server derives the provider from the model name: "vendor/model" → OpenRouter,
#    "bare:tag" → local Ollama; there is no LLM_PROVIDER / GEMINI_API_KEY any more)
#
# PORTS
#   8000 — AI server (FastAPI)
#   8080 — TM server (Node)
#   If either port is already bound, the script lists the offending PIDs and
#   asks for confirmation before killing them. Decline → script exits.
#
# OUTPUT
#   Server PIDs written to /tmp/tm-ai.pids
#   Game PID written to   /tmp/death-match.pids  (--death-match only)
#   Logs:  /tmp/ai-server.log    (AI server)
#          /tmp/tm-server.log    (TM server)
#          /tmp/death-match.log  (game loop; --death-match only)
#
# STOP
#   ./stop.sh           # kills everything tracked in PID files

set -euo pipefail

# --- parse args ---------------------------------------------------------------
DEATH_MATCH=false
while [[ $# -gt 0 ]]; do
  case $1 in
    --death-match) DEATH_MATCH=true; shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

TM_AI_DIR="${TM_AI_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
TM_DIR="${TM_DIR:-${TM_AI_DIR}/../terraforming-mars}"
AI_LOG=/tmp/ai-server.log
TM_LOG=/tmp/tm-server.log
DM_LOG=/tmp/death-match.log
PID_FILE=/tmp/tm-ai.pids
DM_PID_FILE=/tmp/death-match.pids

# Default 4-LLM lineup for --death-match (override via DEATH_MATCH_MODELS env var)
DEATH_MATCH_MODELS="${DEATH_MATCH_MODELS:-anthropic/claude-sonnet-4-6,openai/gpt-4o-mini,google/gemini-2.5-flash,deepseek/deepseek-chat}"

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

# --- AI server ----------------------------------------------------------------
if [[ "${DEATH_MATCH}" == "true" ]]; then
  echo "▶ starting AI server (OpenRouter / multi-model, log: ${AI_LOG})"
  (
    cd "${TM_AI_DIR}/tm-ai-server"
    # API keys are already exported via `set -a; source .env` above — no need to
    # pass them explicitly here, which would risk exposing them in process listings
    # or debug traces.
    USE_LLM=true \
    LLM_DEBUG="${LLM_DEBUG:-true}" \
    exec uv run uvicorn tm_ai_server.main:app --host 0.0.0.0 --port 8000
  ) >> "${AI_LOG}" 2>&1 &
else
  echo "▶ starting AI server (OpenRouter, single model, log: ${AI_LOG})"
  (
    cd "${TM_AI_DIR}/tm-ai-server"
    # OPENROUTER_API_KEY is already exported via `set -a; source .env` above.
    USE_LLM=true \
    OPENROUTER_MODEL="${OPENROUTER_MODEL:-google/gemini-2.5-flash-lite}" \
    LLM_DEBUG="${LLM_DEBUG:-true}" \
    exec uv run uvicorn tm_ai_server.main:app --host 0.0.0.0 --port 8000
  ) >> "${AI_LOG}" 2>&1 &
fi
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

# --- death-match game loop ----------------------------------------------------
if [[ "${DEATH_MATCH}" == "true" ]]; then
  : > "${DM_LOG}"
  rm -f /tmp/current-game.url  # clear stale URL from any previous run
  echo "▶ launching death-match (log: ${DM_LOG})"
  echo "  models: ${DEATH_MATCH_MODELS}"
  (
    cd "${TM_AI_DIR}"
    exec uv run python scripts/play_game.py \
      --players 4 \
      --models "${DEATH_MATCH_MODELS}"
  ) >> "${DM_LOG}" 2>&1 &
  DM_PID=$!
  echo "${DM_PID}" > "${DM_PID_FILE}"
  echo "  PID=${DM_PID}"

  # Wait for play_game.py to create the game and write the spectator URL
  echo -n "  waiting for game ID"
  for i in {1..20}; do
    if [[ -f /tmp/current-game.url ]]; then
      game_url=$(< /tmp/current-game.url)
      echo ""
      echo "  ✔ game ready: ${game_url}"
      # Open in Chrome (falls back through available launchers)
      if command -v google-chrome &>/dev/null; then
        google-chrome "${game_url}" &>/dev/null &
      elif command -v chromium-browser &>/dev/null; then
        chromium-browser "${game_url}" &>/dev/null &
      elif command -v xdg-open &>/dev/null; then
        xdg-open "${game_url}" &>/dev/null &
      fi
      break
    fi
    if ! kill -0 "${DM_PID}" 2>/dev/null; then
      echo ""
      echo "❌ game loop exited before creating a game — see ${DM_LOG}" >&2
      tail -10 "${DM_LOG}" >&2
      break
    fi
    echo -n "."
    sleep 1
    [[ $i -eq 20 ]] && echo "" && echo "  ⚠ timed out waiting for game URL — check ${DM_LOG}"
  done
fi

# --- done ---------------------------------------------------------------------
echo
echo "✅ servers up"
echo "   AI server :8000  (PID ${AI_PID})  ${AI_LOG}"
echo "   TM server :8080  (PID ${TM_PID})  ${TM_LOG}"
if [[ "${DEATH_MATCH}" == "true" ]]; then
  echo "   death-match      (PID ${DM_PID})  ${DM_LOG}"
  echo
  echo "   follow game:  tail -f ${DM_LOG}"
  echo "   follow AI:    tail -f ${AI_LOG}"
fi
echo
echo "stop with:  ./stop.sh"
