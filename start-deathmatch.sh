#!/usr/bin/env bash
# Start AI + TM servers and run a 4-LLM death-match game.
#
# USAGE
#   ./start-deathmatch.sh
#   DEATH_MATCH_MODELS="deepseek/deepseek-v4-flash,google/gemini-3.1-flash-lite,openai/gpt-5-nano,qwen/qwen3.5-flash-02-23,anthropic/claude-haiku-4.5" ./start-deathmatch.sh
#   LLM_DEBUG=false ./start-deathmatch.sh
#   TM_AI_DIR=/path TM_DIR=/path ./start-deathmatch.sh
#
# REQUIREMENTS
#   OPENROUTER_API_KEY must be set in ${TM_AI_DIR}/.env.
#
# PORTS
#   8000 — AI server (FastAPI)
#   8080 — TM server (Node)
#
# OUTPUT
#   Server PIDs   /tmp/tm-ai.pids
#   Game PID      /tmp/death-match.pids
#   Session dir   ./logs/llm-test/<session-id>/    (used at startup)
#       ai-server.log
#       tm-server.log
#       death-match.log
#   Once the game has been created (game_id known), the session directory is
#   renamed to ./logs/llm-test/<game-id>/ and ./logs/llm-test/_latest points
#   at it. Running processes keep writing to the same inodes after the rename.
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
DM_LOG="${SESSION_DIR}/death-match.log"
PID_FILE=/tmp/tm-ai.pids
DM_PID_FILE=/tmp/death-match.pids
GAME_ID_FILE=/tmp/current-game.id

# Default 5-LLM lineup — four cheap/fast models (~$0.05-0.25 in / $0.20-1.50 out
# per 1M) plus claude-haiku-4.5 ($1/$5) as Anthropic's current-gen representative.
# Override via DEATH_MATCH_MODELS (and adjust --players below to match).
DEATH_MATCH_MODELS="${DEATH_MATCH_MODELS:-deepseek/deepseek-v4-flash,google/gemini-3.1-flash-lite,openai/gpt-5-nano,qwen/qwen3.5-flash-02-23,anthropic/claude-haiku-4.5}"

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
rm -f "${GAME_ID_FILE}" /tmp/current-game.url

# --- AI server ----------------------------------------------------------------
echo "▶ starting AI server (OpenRouter / multi-model, log: ${AI_LOG})"
(
  cd "${TM_AI_DIR}/tm-ai-server"
  # API keys are already exported via `set -a; source .env` above — no need to
  # pass them explicitly here, which would risk exposing them in process listings.
  LLM_DEBUG="${LLM_DEBUG:-true}" \
  exec uv run uvicorn tm_llm.app:app --host 0.0.0.0 --port 8000
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

# --- death-match game loop ----------------------------------------------------
echo "▶ launching death-match (log: ${DM_LOG})"
echo "  models: ${DEATH_MATCH_MODELS}"
(
  cd "${TM_AI_DIR}"
  exec uv run python scripts/play_game.py \
    --players 5 \
    --models "${DEATH_MATCH_MODELS}"
) >> "${DM_LOG}" 2>&1 &
DM_PID=$!
echo "${DM_PID}" > "${DM_PID_FILE}"
echo "  PID=${DM_PID}"

# --- wait for game_id, then rename session dir to game-id ---------------------
echo -n "  waiting for game ID"
GAME_ID=""
for i in {1..20}; do
  if [[ -f "${GAME_ID_FILE}" ]]; then
    GAME_ID="$(< "${GAME_ID_FILE}")"
    echo ""
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
  [[ $i -eq 20 ]] && echo "" && echo "  ⚠ timed out waiting for game ID — check ${DM_LOG}"
done

if [[ -n "${GAME_ID}" ]]; then
  GAME_DIR="${TM_AI_DIR}/logs/llm-test/${GAME_ID}"
  if [[ -e "${GAME_DIR}" ]]; then
    echo "  ⚠ ${GAME_DIR} already exists — leaving session at ${SESSION_DIR}"
  else
    mv "${SESSION_DIR}" "${GAME_DIR}"
    ln -sfn "${GAME_ID}" "${TM_AI_DIR}/logs/llm-test/_latest"
    AI_LOG="${GAME_DIR}/ai-server.log"
    TM_LOG="${GAME_DIR}/tm-server.log"
    DM_LOG="${GAME_DIR}/death-match.log"
    SESSION_DIR="${GAME_DIR}"
    echo "  ✔ logs renamed → ${GAME_DIR}"
  fi
fi

# Open spectator URL in browser
if [[ -f /tmp/current-game.url ]]; then
  game_url=$(< /tmp/current-game.url)
  echo "  ✔ game ready: ${game_url}"
  if command -v google-chrome &>/dev/null; then
    google-chrome "${game_url}" &>/dev/null &
  elif command -v chromium-browser &>/dev/null; then
    chromium-browser "${game_url}" &>/dev/null &
  elif command -v xdg-open &>/dev/null; then
    xdg-open "${game_url}" &>/dev/null &
  fi
fi

# --- done ---------------------------------------------------------------------
echo
echo "✅ servers up + death-match running"
echo "   AI server :8000  (PID ${AI_PID})  ${AI_LOG}"
echo "   TM server :8080  (PID ${TM_PID})  ${TM_LOG}"
echo "   death-match      (PID ${DM_PID})  ${DM_LOG}"
echo "   session dir      ${SESSION_DIR}"
echo "                    symlinked at logs/llm-test/_latest"
echo
echo "   follow game:  tail -f ${DM_LOG}"
echo "   follow AI:    tail -f ${AI_LOG}"
echo
echo "stop with:  ./stop.sh"
