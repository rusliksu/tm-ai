"""
Drive a full TM game via the self-play API, with the AI server making every move.

Usage (run from the tm-ai repo root, or any directory):
    python scripts/play_game.py [options]
    # or from tm-ai-server/:
    uv run python ../scripts/play_game.py [options]

Options:
    --tm-url      TM server base URL      (default: http://localhost:8080)
    --ai-url      AI server base URL      (default: http://localhost:8000)
    --board       Board name              (default: random)
    --players     Number of players 2-4   (default: 2)
    --verbose     Print full state JSON each turn

The script drives both players through the self-play API (/api/ai/new-game +
/api/ai/step), calling the AI server /move for each decision. With USE_LLM=true
on the AI server both players use the LLM; without it they use the neural net.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any


def post_json(url: str, payload: dict, timeout: int = 300) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:400]
        raise RuntimeError(f"HTTP {e.code} from {url}: {body}") from e


def build_move_request(game_id: str, player_id: str, state: dict, waiting_for: dict) -> dict:
    return {
        "game_id": game_id,
        "player_id": player_id,
        "state": {**state, "waitingFor": waiting_for},
        "legal_actions": [
            {
                "action_id": "provide_input",
                "type": "or",
                "title": "Choose",
                "payload": {"input": waiting_for},
            }
        ],
        "metadata": {"schema_version": 1},
    }


def describe_decision(waiting_for: dict) -> str:
    wf_type = waiting_for.get("type", "?")
    title = waiting_for.get("title")
    if isinstance(title, dict):
        title = title.get("message", "")
    return f"{wf_type}" + (f": {title}" if title else "")


def play_game(tm_url: str, ai_url: str, board: str | None, player_count: int, verbose: bool) -> None:
    # --- Create game ---
    new_game_payload: dict[str, Any] = {"playerCount": player_count}
    if board:
        new_game_payload["boardName"] = board

    print(f"Creating game (board={board or 'random'}, players={player_count}) ...", flush=True)
    data = post_json(f"{tm_url}/api/ai/new-game", new_game_payload, timeout=30)

    game_id   = data["game_id"]
    player_id = data["player_id"]
    state     = data["state"]
    wf        = data["waitingFor"]

    player_names: dict[str, str] = {}
    for p in ([state.get("player", {})] + (state.get("opponents") or [])):
        pid = p.get("id")
        if pid:
            player_names[pid] = p.get("name", pid[:8])

    print(f"Game {game_id}  players: {list(player_names.values())}\n", flush=True)

    turn = 0
    last_gen = 0

    while True:
        turn += 1
        gen = state.get("game", {}).get("generation", 1)
        if gen != last_gen:
            if last_gen:
                print()
            print(f"=== Generation {gen} ===", flush=True)
            last_gen = gen

        pname    = player_names.get(player_id, player_id[:8])
        decision = describe_decision(wf)
        print(f"  [{turn:3d}] {pname:<12} {decision}", end="", flush=True)

        if verbose:
            print(f"\n  state: {json.dumps(state, indent=2)}")

        # --- Ask AI server for a move ---
        move_req = build_move_request(game_id, player_id, state, wf)
        t0 = time.time()
        try:
            move_data = post_json(f"{ai_url}/move", move_req, timeout=300)
        except RuntimeError as e:
            print(f"\n  ✗ AI server error: {e}")
            sys.exit(1)
        elapsed = time.time() - t0
        input_response = move_data["input_response"]
        print(f"  → {json.dumps(input_response)[:80]}  ({elapsed:.1f}s)", flush=True)

        # --- Submit move to TM server ---
        step_payload = {"game_id": game_id, "player_id": player_id, "input_response": input_response}
        try:
            step_data = post_json(f"{tm_url}/api/ai/step", step_payload, timeout=30)
        except RuntimeError as e:
            print(f"  ✗ TM server rejected move: {e}")
            sys.exit(1)

        if step_data.get("done"):
            print(f"\n=== Game over (gen {gen}) ===\n")
            result = step_data.get("result") or {}
            for entry in sorted(result.get("playerResults", []), key=lambda r: r.get("rank", 99)):
                rank = entry.get("rank", "?")
                name = entry.get("name", "?")
                vp   = entry.get("vp_total", "?")
                tr   = entry.get("tr", "?")
                print(f"  #{rank}  {name:<12}  {vp} VP  (TR {tr})")
            print()
            return

        state     = step_data["state"]
        wf        = step_data["waitingFor"]
        player_id = step_data["player_id"]

        # Update name map with any new player info
        for p in ([state.get("player", {})] + (state.get("opponents") or [])):
            pid = p.get("id")
            if pid and pid not in player_names:
                player_names[pid] = p.get("name", pid[:8])


def main() -> None:
    parser = argparse.ArgumentParser(description="Drive a TM game via the self-play API")
    parser.add_argument("--tm-url",  default="http://localhost:8080")
    parser.add_argument("--ai-url",  default="http://localhost:8000")
    parser.add_argument("--board",   default=None, help="tharsis | hellas | elysium (default: random)")
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    play_game(args.tm_url, args.ai_url, args.board, args.players, args.verbose)


if __name__ == "__main__":
    main()
