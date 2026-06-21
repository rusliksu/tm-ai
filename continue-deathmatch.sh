#!/usr/bin/env bash
# Start AI + TM servers and CONTINUE an existing (e.g. crashed) death-match game.
#
# Unlike start-deathmatch.sh this does NOT create a new game or register players: it resumes
# the persisted game via /api/ai/peek, and the AI server restores each player's memory
# (strategy/tactical/token usage) from logs/llm-state/ on its first /move.
#
# USAGE
#   ./continue-deathmatch.sh                 # resume the latest game (logs/llm-test/_latest)
#   ./continue-deathmatch.sh g29e2e3c6738f   # resume a specific game id
#   LLM_DEBUG=false ./continue-deathmatch.sh
#   TM_AI_DIR=/path TM_DIR=/path ./continue-deathmatch.sh
#
# REQUIREMENTS
#   OPENROUTER_API_KEY must be set in ${TM_AI_DIR}/.env.
#   The game must still be in the TM DB — do NOT run `./stop.sh --clean-db` before resuming.
#
# PORTS
#   8000 — AI server (FastAPI)
#   8080 — TM server (Node)
#
# OUTPUT
#   Server PIDs   /tmp/tm-ai.pids
#   Game PID      /tmp/death-match.pids
#   Logs append to the game's existing dir ./logs/llm-test/<game-id>/ (symlinked at _latest).
#
# STOP
#   ./stop.sh

set -euo pipefail

TM_AI_DIR="${TM_AI_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
TM_DIR="${TM_DIR:-${TM_AI_DIR}/../terraforming-mars}"

# --- resolve which game to resume ---------------------------------------------
# Explicit arg wins; else the target of the _latest symlink; else /tmp/current-game.id.
GAME_ID="${1:-}"
if [[ -z "${GAME_ID}" && -L "${TM_AI_DIR}/logs/llm-test/_latest" ]]; then
  GAME_ID="$(basename "$(readlink "${TM_AI_DIR}/logs/llm-test/_latest")")"
fi
if [[ -z "${GAME_ID}" && -f /tmp/current-game.id ]]; then
  GAME_ID="$(< /tmp/current-game.id)"
fi
if [[ -z "${GAME_ID}" ]]; then
  echo "❌ no game id to resume (pass one as \$1, or ensure logs/llm-test/_latest exists)" >&2
  exit 1
fi
echo "▶ resuming game ${GAME_ID}"

# --- log directory: append to the game's existing dir -------------------------
SESSION_DIR="${TM_AI_DIR}/logs/llm-test/${GAME_ID}"
mkdir -p "${SESSION_DIR}"
ln -sfn "${GAME_ID}" "${TM_AI_DIR}/logs/llm-test/_latest"

AI_LOG="${SESSION_DIR}/ai-server.log"
TM_LOG="${SESSION_DIR}/tm-server.log"
DM_LOG="${SESSION_DIR}/death-match.log"
PID_FILE=/tmp/tm-ai.pids
DM_PID_FILE=/tmp/death-match.pids

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
echo "▶ starting AI server (OpenRouter / multi-model, log: ${AI_LOG})"
(
  cd "${TM_AI_DIR}/tm-ai-server"
  # API keys are already exported via `set -a; source .env` above — no need to
  # pass them explicitly here, which would risk exposing them in process listings.
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

# --- resume the game ----------------------------------------------------------
echo "▶ resuming death-match game ${GAME_ID} (log: ${DM_LOG})"
(
  cd "${TM_AI_DIR}"
  exec uv run python scripts/play_game.py --resume "${GAME_ID}"
) >> "${DM_LOG}" 2>&1 &
DM_PID=$!
echo "${DM_PID}" > "${DM_PID_FILE}"
echo "  PID=${DM_PID}"

# Give the driver a moment; fail fast if it can't pick up the game (e.g. not in DB).
sleep 3
if ! kill -0 "${DM_PID}" 2>/dev/null; then
  echo "❌ resume driver exited immediately — see ${DM_LOG}" >&2
  tail -15 "${DM_LOG}" >&2
  exit 1
fi

# --- done ---------------------------------------------------------------------
echo
echo "✅ servers up + death-match resuming"
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
