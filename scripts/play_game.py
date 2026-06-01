"""
Drive a full TM game via the self-play API, with the AI server making every move.

Usage (run from the tm-ai repo root, or any directory):
    python scripts/play_game.py [options]
    # or from tm-ai-server/:
    uv run python ../scripts/play_game.py [options]

Options:
    --tm-url      TM server base URL        (default: http://localhost:8080)
    --ai-url      AI server base URL        (default: http://localhost:8000)
    --board       Board name                (default: random)
    --players     Number of players 2-5     (default: 2)
    --models      Comma-separated model IDs, one per player (default: AI server default)
                  Fewer models than players → last model is reused for remaining players.
                  Examples:
                    --models anthropic/claude-opus-4.7,openai/gpt-5.5-pro
                    --models "anthropic/claude-opus-4.7,openai/gpt-5.5-pro,google/gemini-pro-latest,deepseek/deepseek-v4-pro,x-ai/grok-4.3"
    --verbose     Print full state JSON each turn

Models are registered with POST /player/register before the game starts, so each
player uses a different LLM. Without --models all players use OPENROUTER_MODEL (or
OLLAMA_MODEL if no API key is set).

Token/cost summary is logged at game end via POST /game-done.
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


def _extract_server_error(raw: str) -> str:
    """Pull the error string out of 'HTTP 400 from <url>: {"error":"..."}'."""
    try:
        idx = raw.find(": {")
        if idx >= 0:
            body = json.loads(raw[idx + 2:])
            return body.get("error", raw)
    except Exception:
        pass
    return raw


_MAX_STEP_RETRIES = 2


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


def register_players(
    ai_url: str,
    game_id: str,
    player_ids: list[str],
    models: list[str],
) -> dict[str, str]:
    """Register each player_id with its model. Returns {player_id: model}."""
    assigned: dict[str, str] = {}
    for i, pid in enumerate(player_ids):
        model = models[min(i, len(models) - 1)] if models else None
        payload: dict[str, Any] = {"player_id": pid, "game_id": game_id}
        if model:
            payload["model"] = model
        try:
            resp = post_json(f"{ai_url}/player/register", payload, timeout=15)
            assigned[pid] = resp.get("model", model or "?")
            print(f"  Registered player {pid[:8]}… → {assigned[pid]}", flush=True)
        except RuntimeError as e:
            print(f"  ⚠ Could not register player {pid[:8]}…: {e}", flush=True)
            assigned[pid] = model or "default"
    return assigned


def model_short_name(model: str) -> str:
    """Derive a short display name from an OpenRouter model ID."""
    # "anthropic/claude-sonnet-4-6" → "Claude-Sonnet-4-6"
    # "openai/gpt-4o-mini"          → "GPT-4o-mini"
    # "google/gemini-flash-latest"  → "Gemini-Flash"
    # "deepseek/deepseek-v4-pro"    → "DeepSeek-V4-Pro"
    # "qwen3:4b"                    → "Qwen3-4b"
    name = model.split("/")[-1] if "/" in model else model.replace(":", "-")
    # Capitalise each dash-separated word for readability
    parts = name.replace("_", "-").split("-")
    caps = "-".join(p.capitalize() if not p[0].isupper() else p for p in parts if p)
    # Drop redundant provider prefix in the name (e.g. "Deepseek-V4-Pro" from "deepseek-v4-pro")
    provider = model.split("/")[0] if "/" in model else ""
    if provider and caps.lower().startswith(provider.lower().replace("-", "")):
        stripped = caps[len(provider):].strip("-")
        # Only drop the prefix if what remains contains a version/digit (e.g. "V4-Pro", "R1")
        # Keep it for generic names like "Chat" that need context to be identifiable
        if stripped and ("-" in stripped or any(c.isdigit() for c in stripped)):
            caps = stripped
    return caps.strip("-") or name


def play_game(
    tm_url: str,
    ai_url: str,
    board: str | None,
    player_count: int,
    models: list[str],
    verbose: bool,
) -> None:
    # Derive player names from model IDs
    player_name_list = [model_short_name(models[min(i, len(models) - 1)]) if models else f"AI-{i+1}"
                        for i in range(player_count)]

    # --- Create game ---
    new_game_payload: dict[str, Any] = {"playerCount": player_count, "playerNames": player_name_list}
    if board:
        new_game_payload["boardName"] = board

    print(f"Creating game (board={board or 'random'}, players={player_count}) ...", flush=True)
    data = post_json(f"{tm_url}/api/ai/new-game", new_game_payload, timeout=30)

    game_id      = data["game_id"]
    player_id    = data["player_id"]
    spectator_id = data.get("spectator_id")
    state        = data["state"]
    wf           = data["waitingFor"]

    # Write spectator URL + game_id to temp files so start scripts can pick them up:
    #   /tmp/current-game.url — opened in browser (spectator view)
    #   /tmp/current-game.id  — used by start-deathmatch.sh to rename the session
    #                           log directory to ./logs/llm-test/<game-id>/
    spectator_url = f"{tm_url}/spectator?id={spectator_id}" if spectator_id else f"{tm_url}"
    try:
        with open("/tmp/current-game.url", "w") as _f:
            _f.write(spectator_url + "\n")
    except OSError:
        pass
    try:
        with open("/tmp/current-game.id", "w") as _f:
            _f.write(game_id)
    except OSError:
        pass

    # Collect all player IDs in seat order: active player first, then opponents
    all_players: list[dict] = [state.get("player", {})] + (state.get("opponents") or [])
    player_ids = [p["id"] for p in all_players if p.get("id")]
    player_names: dict[str, str] = {p["id"]: p.get("name", p["id"][:8])
                                    for p in all_players if p.get("id")}

    # --- Register players with their models ---
    print(f"\nGame {game_id}  spectator: {spectator_url}", flush=True)
    assigned_models = register_players(ai_url, game_id, player_ids, models)

    # Show lineup
    print("\nLineup:")
    for pid in player_ids:
        print(f"  {player_names.get(pid, pid[:8]):<14} {assigned_models.get(pid, '?')}")
    print(flush=True)

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
        print(f"  [{turn:3d}] {pname:<14} {decision}", end="", flush=True)

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

        # --- Submit move to TM server (with retry-on-rejection) ---
        step_data: dict = {}
        step_last_error: str | None = None
        for step_attempt in range(_MAX_STEP_RETRIES + 1):
            if step_attempt > 0:
                # Re-ask AI server with the rejection error
                move_req["last_error"] = step_last_error
                try:
                    move_data = post_json(f"{ai_url}/move", move_req, timeout=300)
                except RuntimeError as e:
                    print(f"\n  ✗ AI server error on retry {step_attempt}: {e}")
                    sys.exit(1)
                input_response = move_data["input_response"]
                print(f"  ↩ retry {step_attempt}/{_MAX_STEP_RETRIES}: {json.dumps(input_response)[:70]}",
                      flush=True)

            step_payload = {"game_id": game_id, "player_id": player_id, "input_response": input_response}
            try:
                step_data = post_json(f"{tm_url}/api/ai/step", step_payload, timeout=30)
                break
            except RuntimeError as e:
                step_last_error = _extract_server_error(str(e))
                if step_attempt < _MAX_STEP_RETRIES:
                    print(f"\n  ⚠ TM rejected (attempt {step_attempt + 1}): {step_last_error[:100]}",
                          flush=True)
                else:
                    print(f"\n  ✗ TM rejected after {_MAX_STEP_RETRIES} retries: {step_last_error}")
                    sys.exit(1)

        if step_data.get("done"):
            print(f"\n=== Game over (gen {gen}) ===\n")
            result = step_data.get("result") or {}
            for entry in sorted(result.get("playerResults", []), key=lambda r: r.get("rank", 99)):
                rank  = entry.get("rank", "?")
                name  = entry.get("name", "?")
                vp    = entry.get("vp_total", "?")
                tr    = entry.get("tr", "?")
                model = assigned_models.get(entry.get("playerId", ""), "?")
                print(f"  #{rank}  {name:<14}  {vp} VP  (TR {tr})  [{model}]")
            print()

            # Flush per-player token/cost summary
            try:
                post_json(f"{ai_url}/game-done", {"game_id": game_id}, timeout=10)
            except RuntimeError:
                pass
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
    parser.add_argument(
        "--models", default="",
        help="Comma-separated OpenRouter model IDs, one per player seat (left to right). "
             "Fewer than --players → last model repeated.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]

    play_game(args.tm_url, args.ai_url, args.board, args.players, models, args.verbose)


if __name__ == "__main__":
    main()
