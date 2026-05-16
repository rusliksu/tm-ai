"""
Interactive TM game: Claude Code vs Gemini LLM.

State persisted to /tmp/tm_game.json between calls.

Usage (from tm-ai repo root, run inside tm-ai-server uv env):
  uv run --project tm-ai-server python scripts/play_interactive.py --new [--board X]
  uv run --project tm-ai-server python scripts/play_interactive.py          # show state
  uv run --project tm-ai-server python scripts/play_interactive.py --choice N
  uv run --project tm-ai-server python scripts/play_interactive.py --initial-response '{...}'

Claude = player 0 (blue).  Gemini = player 1 (red).
Gemini turns advance automatically.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any

# Use the real encoding module so response construction is always correct.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../tm-ai-server/src"))
from tm_ai_server.encoding import flatten_options, index_to_response
from tm_ai_server.game_knowledge import CARD_DB

STATE_FILE   = "/tmp/tm_game.json"
TM_DEFAULT   = "http://localhost:8080"
AI_DEFAULT   = "http://localhost:8000"


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def post_json(url: str, payload: dict, timeout: int = 300) -> dict:
    body = json.dumps(payload).encode()
    req  = urllib.request.Request(url, data=body,
                                  headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        txt = e.read().decode(errors="replace")[:600]
        raise RuntimeError(f"HTTP {e.code} from {url}:\n{txt}") from e


# ---------------------------------------------------------------------------
# State file
# ---------------------------------------------------------------------------

def load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        sys.exit("No game in progress. Run with --new first.")
    with open(STATE_FILE) as f:
        return json.load(f)

def save_state(gs: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(gs, f, indent=2)


# ---------------------------------------------------------------------------
# Card lookup helper
# ---------------------------------------------------------------------------

def card_info(name: str) -> str:
    entry = CARD_DB.get(name) or {}
    cost  = entry.get("cost")
    desc  = entry.get("description", "")
    tags  = entry.get("tags") or []
    parts = []
    if cost is not None:
        parts.append(f"{cost}MC")
    if tags:
        parts.append("/".join(tags))
    if desc:
        parts.append(desc[:80])
    return "  " + "  ".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# State rendering
# ---------------------------------------------------------------------------

def render_state(gs: dict) -> str:
    state = gs["state"]
    wf    = gs["waiting_for"]
    g     = state.get("game", {})
    p     = state.get("player", {})
    prod  = {k: v for k, v in p.get("production", {}).items() if v}
    opps  = state.get("opponents") or []
    my_id = p.get("id", "")

    lines = [f"\n{'='*64}"]
    lines.append(f"Gen {g.get('generation',1)} | Temp {g.get('temperature',-30)}°C | "
                 f"O₂ {g.get('oxygen',0)}% | Oceans {g.get('oceanCount',0)}/9  |  Turn {gs['turn']}")
    if g.get("temperature", -30) >= 8:
        lines.append("⚠  Temperature MAXED — Convert Heat wastes an action")
    if g.get("oxygen", 0) >= 14:
        lines.append("⚠  O₂ MAXED — Greenery tiles give no terraforming benefit")

    vp   = p.get("victoryPoints")
    corp = ", ".join(p.get("corporations") or [])
    lines.append(f"\nCLAUDE  ({corp})")
    lines.append(f"  TR:{p.get('terraformRating',20)}  VP:{vp}  "
                 f"MC:{p.get('megacredits',0)}  St:{p.get('steel',0)}  "
                 f"Ti:{p.get('titanium',0)}  Pl:{p.get('plants',0)}  "
                 f"En:{p.get('energy',0)}  He:{p.get('heat',0)}")
    if prod:
        lines.append(f"  prod: {prod}")
    tags = {k: v for k, v in p.get("tags", {}).items() if v}
    if tags:
        lines.append(f"  tags: {tags}")
    played = p.get("playedCards") or []
    if played:
        lines.append(f"  played ({len(played)}): {', '.join(played)}")

    hand = p.get("cardsInHand") or []
    if hand:
        lines.append(f"\n  Hand ({len(hand)} cards):")
        for c in hand:
            lines.append(f"    {c}{card_info(c)}")

    ms_raw = state.get("milestones") or []
    aw_raw = state.get("awards") or []
    ms_claimed = [f"{m['name']}({'me' if m.get('playerId')==my_id else 'opp'})"
                  for m in ms_raw if m.get("playerId")]
    aw_funded  = [f"{a['name']}({'me' if a.get('playerId')==my_id else 'opp'})"
                  for a in aw_raw if a.get("playerId")]
    if ms_claimed:
        lines.append(f"  milestones: {ms_claimed}")
    if aw_funded:
        lines.append(f"  awards:     {aw_funded}")

    avail_ms = g.get("availableMilestones") or []
    avail_aw = g.get("availableAwards") or []
    if avail_ms:
        lines.append(f"  avail milestones: {[m['name'] for m in avail_ms]}")
    if avail_aw:
        lines.append(f"  avail awards: {[a['name'] for a in avail_aw]}")

    for opp in opps:
        ovp   = opp.get("victoryPoints")
        ocorp = ", ".join(opp.get("corporations") or [])
        oprod = {k: v for k, v in opp.get("production", {}).items() if v}
        lines.append(f"\nGEMINI  ({ocorp})")
        lines.append(f"  TR:{opp.get('terraformRating',20)}  VP:{ovp}  "
                     f"MC:{opp.get('megacredits',0)}  St:{opp.get('steel',0)}  "
                     f"Ti:{opp.get('titanium',0)}  Pl:{opp.get('plants',0)}  "
                     f"En:{opp.get('energy',0)}  He:{opp.get('heat',0)}")
        if oprod:
            lines.append(f"  prod: {oprod}")
        oplayed = opp.get("playedCards") or []
        if oplayed:
            lines.append(f"  played ({len(oplayed)}): {', '.join(oplayed)}")

    recent = g.get("recentLog") or []
    if recent:
        lines.append(f"\n  Recent Gemini actions:")
        for e in recent[-8:]:
            lines.append(f"    {e}")

    # Decision
    lines.append(f"\n{'─'*64}")
    wf_type = wf.get("type", "?")
    title   = wf.get("title", "")
    if isinstance(title, dict):
        title = title.get("message", "")
    lines.append(f"DECISION ({wf_type}): {title}")

    if wf_type == "initialCards":
        _render_initial_cards(wf, lines)
    else:
        opts = flatten_options(wf)
        for i, opt in enumerate(opts, 1):
            extra = card_info(opt["title"]) if wf_type in ("projectCard", "card") else ""
            lines.append(f"  {i:3d}. {opt['title']}{extra}")
        lines.append(f"\n--choice N  (1–{len(opts)})")

    lines.append("=" * 64)
    return "\n".join(lines)


def _render_initial_cards(wf: dict, lines: list[str]) -> None:
    """Render corporation + starting card options for initialCards phase."""
    for sub_i, sub in enumerate(wf.get("options") or []):
        sub_type  = sub.get("type", "")
        sub_title = sub.get("title", sub_type)
        if isinstance(sub_title, dict):
            sub_title = sub_title.get("message", sub_type)
        lines.append(f"\n  [Sub-decision {sub_i}] {sub_title} (type={sub_type})")
        cards = sub.get("cards") or []
        if cards:
            for j, c in enumerate(cards):
                name = c.get("name", "?") if isinstance(c, dict) else str(c)
                entry = CARD_DB.get(name) or {}
                cost  = c.get("cost") if isinstance(c, dict) else entry.get("cost")
                desc  = entry.get("description", "")[:80]
                tags  = entry.get("tags") or (c.get("tags") if isinstance(c, dict) else []) or []
                lines.append(f"    {j}: {name}  {cost}MC  [{'/'.join(tags)}]  {desc}")
    lines.append("\nUse --initial-response '{JSON}' to submit.")
    lines.append("Example: --initial-response '{\"type\":\"initialCards\","
                 "\"responses\":[{\"type\":\"card\",\"cards\":[\"HelionCorp\"]},"
                 "{\"type\":\"card\",\"cards\":[\"ImportedHydrogen\"]}]}'")


# ---------------------------------------------------------------------------
# Gemini move
# ---------------------------------------------------------------------------

def gemini_move(gs: dict) -> dict:
    return gemini_move_with_error(gs, None)


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

def tm_step(gs: dict, ir: dict) -> dict:
    return post_json(f"{gs['tm_url']}/api/ai/step", {
        "game_id":        gs["game_id"],
        "player_id":      gs["current_player_id"],
        "input_response": ir,
    }, timeout=30)


def apply_step(gs: dict, step: dict) -> None:
    gs["turn"]              += 1
    gs["current_player_id"]  = step.get("player_id")
    gs["state"]              = step.get("state") or gs["state"]
    gs["waiting_for"]        = step.get("waitingFor") or {}
    gs["done"]               = step.get("done", False)
    gs["result"]             = step.get("result")


def advance_gemini(gs: dict) -> None:
    while not gs.get("done") and gs["current_player_id"] == gs["gemini_player_id"]:
        wf      = gs["waiting_for"]
        wf_type = wf.get("type", "?")
        title   = wf.get("title", "")
        if isinstance(title, dict):
            title = title.get("message", "")
        gen = gs["state"].get("game", {}).get("generation", 1)
        print(f"  [Gemini G{gen} turn {gs['turn']+1}] {wf_type}"
              + (f": {str(title)[:50]}" if title else ""), end="", flush=True)
        t0 = time.time()
        last_error: str | None = None
        for attempt in range(3):
            ir = gemini_move_with_error(gs, last_error)
            elapsed = time.time() - t0
            try:
                step = tm_step(gs, ir)
                break
            except RuntimeError as e:
                last_error = str(e).split("\n", 1)[0]
                print(f"\n    ⚠ Gemini move rejected (attempt {attempt+1}/3): {last_error[:80]}", flush=True)
                if attempt == 2:
                    print("    ✗ All Gemini retries failed — falling back to Pass", flush=True)
                    ir = _fallback_pass(gs["waiting_for"])
                    step = tm_step(gs, ir)
                    break
        print(f"  → {json.dumps(ir)[:70]}  ({elapsed:.1f}s)", flush=True)
        apply_step(gs, step)


def gemini_move_with_error(gs: dict, last_error: str | None = None) -> dict:
    state = gs["state"]
    wf    = gs["waiting_for"]
    payload: dict = {
        "game_id":   gs["game_id"],
        "player_id": gs["current_player_id"],
        "state":     {**state, "waitingFor": wf},
        "legal_actions": [{"action_id": "provide_input", "type": "or",
                           "title": "Choose", "payload": {"input": wf}}],
        "metadata": {"schema_version": 1},
    }
    if last_error:
        payload["last_error"] = last_error
    data = post_json(f"{gs['ai_url']}/move", payload, timeout=300)
    return data["input_response"]


def _fallback_pass(wf: dict) -> dict:
    """Return a pass/end-turn fallback when all retries are exhausted."""
    if wf.get("type") == "or":
        opts = wf.get("options") or []
        # 1. Look for an option whose type is explicitly "pass" or "end"
        for i, opt in enumerate(opts):
            t = opt.get("type", "") if isinstance(opt, dict) else ""
            if t in ("pass", "end"):
                return {"type": "or", "index": i, "response": {"type": t}}
        # 2. Look for an option whose title contains "pass" (case-insensitive)
        for i, opt in enumerate(opts):
            title = opt.get("title", "") if isinstance(opt, dict) else ""
            if isinstance(title, dict):
                title = title.get("message", "")
            if "pass" in title.lower():
                return {"type": "or", "index": i, "response": {"type": "option"}}
        # 3. Look for "end" in title as a last resort
        for i, opt in enumerate(opts):
            title = opt.get("title", "") if isinstance(opt, dict) else ""
            if isinstance(title, dict):
                title = title.get("message", "")
            if "end" in title.lower():
                return {"type": "or", "index": i, "response": {"type": "option"}}
    return {"type": "option"}


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

def print_result(gs: dict) -> None:
    result = gs.get("result") or {}
    print(f"\n{'='*64}")
    print("GAME OVER")
    print(f"{'='*64}")
    for e in sorted(result.get("playerResults", []), key=lambda r: r.get("rank", 99)):
        who = " ← CLAUDE" if e.get("playerId") == gs["claude_player_id"] else " ← GEMINI"
        print(f"  #{e.get('rank','?')}  {e.get('name','?'):<14}  "
              f"{e.get('vp_total','?')} VP  (TR {e.get('tr','?')}){who}")
    print()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_new(args: argparse.Namespace) -> None:
    payload: dict[str, Any] = {"playerCount": 2}
    if args.board:
        payload["boardName"] = args.board

    print(f"Creating game (board={args.board or 'random'}) ...", flush=True)
    data = post_json(f"{args.tm_url}/api/ai/new-game", payload, timeout=30)

    game_id   = data["game_id"]
    state     = data["state"]
    wf        = data["waitingFor"]
    first_pid = data["player_id"]

    opp_ids = [o["id"] for o in (state.get("opponents") or []) if o.get("id")]
    gemini_pid = opp_ids[0] if opp_ids else None

    pnames: dict[str, str] = {}
    for pp in [state.get("player", {})] + (state.get("opponents") or []):
        if pp.get("id"):
            pnames[pp["id"]] = pp.get("name", pp["id"][:8])

    gs: dict[str, Any] = {
        "game_id":           game_id,
        "claude_player_id":  first_pid,
        "gemini_player_id":  gemini_pid,
        "current_player_id": first_pid,
        "state":             state,
        "waiting_for":       wf,
        "turn":              0,
        "done":              False,
        "result":            None,
        "tm_url":            args.tm_url,
        "ai_url":            args.ai_url,
    }
    save_state(gs)

    print(f"Game {game_id}")
    print(f"  Claude = {pnames.get(first_pid, first_pid[:8])}")
    print(f"  Gemini = {pnames.get(gemini_pid, str(gemini_pid)[:8])}\n")

    advance_gemini(gs)
    save_state(gs)
    if gs.get("done"):
        print_result(gs)
    else:
        print(render_state(gs))


def cmd_show(_args: argparse.Namespace) -> None:
    gs = load_state()
    if gs.get("done"):
        print_result(gs)
        return
    advance_gemini(gs)
    save_state(gs)
    if gs.get("done"):
        print_result(gs)
    else:
        print(render_state(gs))


def cmd_choice(args: argparse.Namespace) -> None:
    gs = load_state()
    if gs.get("done"):
        print_result(gs)
        return
    if gs["current_player_id"] != gs["claude_player_id"]:
        print("Not Claude's turn — advancing Gemini ...")
        advance_gemini(gs)
        save_state(gs)
        print(render_state(gs))
        return

    wf   = gs["waiting_for"]
    opts = flatten_options(wf)
    n    = args.choice
    if n < 1 or n > len(opts):
        sys.exit(f"Choice must be 1–{len(opts)}")

    chosen = opts[n - 1]
    ir     = index_to_response(wf, chosen["index"])
    gen    = gs["state"].get("game", {}).get("generation", 1)
    print(f"  [Claude G{gen} turn {gs['turn']+1}] → {chosen['title'][:60]}  ({json.dumps(ir)[:70]})",
          flush=True)

    step = tm_step(gs, ir)
    apply_step(gs, step)
    if gs.get("done"):
        save_state(gs)
        print_result(gs)
        return

    advance_gemini(gs)
    save_state(gs)
    if gs.get("done"):
        print_result(gs)
    else:
        print(render_state(gs))


def cmd_initial_response(args: argparse.Namespace) -> None:
    gs = load_state()
    if gs.get("done"):
        print_result(gs)
        return
    ir = json.loads(args.initial_response)
    gen = gs["state"].get("game", {}).get("generation", 1)
    print(f"  [Claude G{gen} turn {gs['turn']+1}] initialCards → {json.dumps(ir)[:120]}", flush=True)
    step = tm_step(gs, ir)
    apply_step(gs, step)
    if gs.get("done"):
        save_state(gs)
        print_result(gs)
        return
    advance_gemini(gs)
    save_state(gs)
    if gs.get("done"):
        print_result(gs)
    else:
        print(render_state(gs))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Claude Code vs Gemini — Terraforming Mars")
    p.add_argument("--new",              action="store_true")
    p.add_argument("--board",            default=None)
    p.add_argument("--choice",           type=int, default=None, metavar="N")
    p.add_argument("--initial-response", type=str, default=None, metavar="JSON")
    p.add_argument("--tm-url",           default=TM_DEFAULT)
    p.add_argument("--ai-url",           default=AI_DEFAULT)
    args = p.parse_args()

    if args.new:
        cmd_new(args)
    elif args.initial_response:
        cmd_initial_response(args)
    elif args.choice is not None:
        cmd_choice(args)
    else:
        cmd_show(args)


if __name__ == "__main__":
    main()
