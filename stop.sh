#!/usr/bin/env bash
# Stop the running tm-ai experiment: AI server, TM server, and game loop.
#
# USAGE
#   ./stop.sh              # stop all processes tracked in PID files
#   ./stop.sh --clean-db   # also remove current game(s) from the TM SQLite DB
#
# PID files consumed:
#   /tmp/tm-ai.pids         — AI server + TM server (written by start.sh)
#   /tmp/death-match.pids   — game loop process     (written by start.sh --death-match)

set -euo pipefail

CLEAN_DB=false
while [[ $# -gt 0 ]]; do
  case $1 in
    --clean-db) CLEAN_DB=true; shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

TM_AI_DIR="${TM_AI_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
TM_DIR="${TM_DIR:-${TM_AI_DIR}/../terraforming-mars}"
PID_FILE=/tmp/tm-ai.pids
DM_PID_FILE=/tmp/death-match.pids
DB_PATH="${TM_DIR}/db/game.db"

killed_any=false

kill_pids_from_file() {
  local file=$1
  local label=$2
  [[ -f "${file}" ]] || return 0
  while IFS= read -r pid; do
    [[ -z "${pid}" ]] && continue
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null && echo "  killed ${label} PID ${pid}" || true
      killed_any=true
    else
      echo "  ${label} PID ${pid} already gone"
    fi
  done < "${file}"
  rm -f "${file}"
}

echo "▶ stopping death-match game loop..."
kill_pids_from_file "${DM_PID_FILE}" "game-loop"

echo "▶ stopping AI + TM servers..."
kill_pids_from_file "${PID_FILE}" "server"

# Give processes a moment to exit, then verify ports are free
sleep 1
for port in 8000 8080; do
  if ss -ltn "sport = :${port}" 2>/dev/null | grep -q LISTEN; then
    pids=$(ss -ltnp "sport = :${port}" 2>/dev/null \
             | awk 'NR>1' \
             | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u | tr '\n' ' ' || true)
    echo "  ⚠ port ${port} still in use by PID(s) ${pids% } — sending SIGKILL"
    # shellcheck disable=SC2086
    kill -9 ${pids} 2>/dev/null || true
    killed_any=true
  fi
done

if [[ "${killed_any}" == "false" ]]; then
  echo "  nothing was running"
fi

# --- optional DB cleanup ------------------------------------------------------
if [[ "${CLEAN_DB}" == "true" ]]; then
  if [[ ! -f "${DB_PATH}" ]]; then
    echo "⚠ DB not found at ${DB_PATH} — skipping cleanup"
  else
    # Remove all self-play / AI games (isSelfPlay=1) — these are the experiment games.
    # Self-play games are not shown in the human UI, so dropping them is safe.
    removed=$(python3 - <<EOF
import sqlite3, json
conn = sqlite3.connect("${DB_PATH}")
cur = conn.execute("SELECT game_id, game FROM games")
to_delete = []
for row in cur.fetchall():
    try:
        g = json.loads(row[1])
        if g.get("isSelfPlay") or all(p.get("isAI") for p in g.get("players", [{}])):
            to_delete.append(row[0])
    except Exception:
        pass
for gid in to_delete:
    conn.execute("DELETE FROM games WHERE game_id = ?", (gid,))
conn.commit()
conn.close()
print(f"removed {len(to_delete)} game(s): {', '.join(to_delete) or 'none'}")
EOF
    )
    echo "  DB cleanup: ${removed}"
  fi
fi

echo
echo "✅ done"
