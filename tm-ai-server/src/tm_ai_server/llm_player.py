"""
LLM-based action selection — supports Ollama (local) and Gemini (cloud).

Session-per-game architecture:
- Setup (initialCards): full system prompt (TM rules + board context) sent ONCE to
  initialise the session. The model writes an opening strategy document.
- All subsequent decisions (prelude / action): continue the same session. Only the
  current game state and numbered options are sent — rules/board context are in
  session memory, so they are NOT repeated each turn.
- Strategy: requested at setup, updated on demand via STRATEGY_UPDATE in action prompts.
  Stored in _game_strategies for logging and context-trim recovery.
- Context trimming: if an Ollama session grows beyond MAX_SESSION_MESSAGES, it is
  rebuilt as: original system + strategy reminder + last 40 messages.

Providers:
  Ollama  — messages[] array; Ollama server reuses KV cache for unchanged prefix.
  Gemini  — Chat API (client.chats.create + chat.send_message); history cached server-side.
            Setup uses generate_content with think=True, then chat is initialised with
            that exchange as history.

Env vars:
  USE_LLM=true              Enable this module (checked in inference.py)
  LLM_PROVIDER              'ollama' (default) or 'gemini'
  LLM_DEBUG=true            Log full prompts and raw responses

  Ollama (LLM_PROVIDER=ollama):
    OLLAMA_URL              Base URL  (default: http://localhost:11434)
    OLLAMA_MODEL            Model tag (default: qwen3:4b)
    OLLAMA_TIMEOUT          Request timeout seconds (default: 600)

  Gemini (LLM_PROVIDER=gemini):
    GEMINI_API_KEY          Google AI Studio API key (required)
    GEMINI_MODEL            Model name (default: gemini-2.5-flash)
"""

from __future__ import annotations
import logging
import os
import re

import requests

from .encoding import flatten_options, index_to_response, _default_response
from .game_knowledge import CARD_DB, format_card_context, format_config_context

logger = logging.getLogger(__name__)

_LLM_PROVIDER   = os.getenv("LLM_PROVIDER", "ollama").lower()
_LLM_DEBUG      = os.getenv("LLM_DEBUG", "false").lower() == "true"

# Ollama settings
_OLLAMA_URL     = os.getenv("OLLAMA_URL",   "http://localhost:11434")
_OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL", "qwen3:4b")
_OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "600"))

# Gemini settings
_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
_GEMINI_MODEL   = os.getenv("GEMINI_MODEL",  "gemini-2.5-flash")

_gemini_client = None  # lazy-initialised

SETUP_TYPES = {"initialCards", "prelude"}

# Per-game strategy documents: game_id → strategy text (for logging and trim recovery)
_game_strategies: dict[str, str] = {}

# Per-game session history: game_id → messages list (Ollama) or Chat object (Gemini)
_game_sessions: dict[str, list[dict]] = {}
_game_chat_sessions: dict[str, object] = {}
_session_base_system: dict[str, str] = {}  # game_id → original system (for trim)

# Trim Ollama sessions when they grow beyond this many messages (system + user/assistant pairs)
_MAX_SESSION_MESSAGES = 62  # ~30 game turns before trim

# ---------------------------------------------------------------------------
# Terraforming Mars rules reference — injected into every system prompt
# ---------------------------------------------------------------------------

TM_RULES = """
=== TERRAFORMING MARS — RULES REFERENCE ===

OBJECTIVE: Most Victory Points (VPs) wins.
VP sources:
  • Terraform Rating (TR): 1 VP per TR (start at 20, rises with terraforming)
  • Greenery tiles: 1 VP each
  • City tiles: 1 VP per adjacent greenery tile (any owner)
  • Milestones: 5 VP each (max 3 claimed per whole game, pay 8 MC to claim)
  • Awards: 5 VP for 1st place / 2 VP for 2nd (max 3 funded, pay 8/14/20 MC)
  • Cards: VPs printed on individual cards

GLOBAL PARAMETERS (game ends when ALL three are maxed):
  • Temperature: -30°C → +8°C  (38 steps). Raise: 8 heat, Asteroid SP, or cards.
  • Oxygen:       0%   → 14%   (14 steps). Raise: place greenery tile or cards.
  • Oceans:       0   → 9 tiles.           Place: Aquifer SP or cards.
  Each step raised = +1 TR (=+1 income and +1 VP).

RESOURCES (produced every generation):
  • MegaCredits (MC): currency. INCOME = TR + MC-production each generation.
  • Steel:     pays for BUILDING-tag cards at 2 MC/cube.
  • Titanium:  pays for SPACE-tag cards at 3 MC/cube.
  • Plants:    8 plants → greenery tile (+1 oxygen, +1 TR).
  • Energy:    unused energy converts to heat at end of generation.
  • Heat:      8 heat → raise temperature +1°C (+1 TR).
  MC production can be negative (minimum −5).

CARD TYPES:
  • GREEN  (automated): one-time effects; tags always count thereafter.
  • BLUE   (active):    ongoing effects OR once-per-generation actions (red arrow).
  • RED    (event):     one-time effects; tags count ONLY when played (then face-down).

TAGS: Building, Space, Science, Power, Earth, Jovian, Venus, Plant, Microbe,
      Animal, City, Event, Wild.
  Important: Steel discounts building-tag cards; titanium discounts space-tag cards.

TURN STRUCTURE each generation:
  1. Player order shifts clockwise; generation marker advances.
  2. Research phase: draw 4 cards, buy any for 3 MC each, discard rest.
  3. Action phase: take turns doing 1 or 2 actions until all players pass.
  4. Production phase: all energy → heat; collect resources per production tracks.

AVAILABLE ACTIONS (choose 1 or 2 per turn):
  A. Play a card from hand (pay its cost; use steel for building, titanium for space).
  B. Use a standard project.
  C. Claim a milestone (8 MC + meet its requirement).
  D. Fund an award (8 MC 1st / 14 MC 2nd / 20 MC 3rd funded).
  E. Use the action on a blue card (once per generation per card; pay cost if any).
  F. Convert 8 plants into a greenery tile (+1 oxygen, +1 TR, place next to own tile).
  G. Convert 8 heat into +1 temperature (+1 TR).

STANDARD PROJECTS (always available to any player):
  1. Sell patents:  discard N cards → gain N MC.
  2. Power plant:   11 MC → +1 energy production.
  3. Asteroid:      14 MC → +1 temperature (+1 TR).
  4. Aquifer:       18 MC → place ocean tile (+1 TR, +placement bonus).
  5. Greenery:      23 MC → place greenery (+1 oxygen, +1 TR).
  6. City:          25 MC → place city tile + 1 MC production.

MILESTONES (5 VP, costs 8 MC; only 3 total can be claimed in entire game):
  1. Terraformer: TR ≥ 35.
  2. Mayor:       own ≥ 3 city tiles.
  3. Gardener:    own ≥ 3 greenery tiles.
  4. Builder:     ≥ 8 building tags in play.
  5. Planner:     ≥ 16 cards in hand when claimed.

AWARDS (5 VP 1st / 2 VP 2nd; costs 8/14/20 MC; only 3 total funded per game):
  1. Landlord:   most tiles on the board.
  2. Banker:     highest MC production.
  3. Scientist:  most science tags.
  4. Thermalist: most heat resource cubes.
  5. Miner:      most steel + titanium resource cubes.

TILE PLACEMENT:
  • Ocean:   only on reserved blue spaces; other players placing next to it get +2 MC.
  • Greenery: must place next to own tile if possible; otherwise any free space.
  • City:    cannot be adjacent to another city (exception: Noctis City).
  • Scoring: greenery = 1 VP; city = 1 VP per adjacent greenery (end of game).

INITIAL SETUP (generation 1 — no research phase):
  • Choose exactly 1 corporation from the 2 dealt to you.
  • From the 10 dealt project cards, buy any number at 3 MC each (to add to hand).
    (The cost shown on a card is its PLAY cost during the game, not the buy cost.)
  • You CANNOT buy more cards than: floor(starting_MC / 3).
  • Unselected cards are discarded. Cards in hand are played during future action phases.

PAYMENT:
  • Pay card play cost in MC; optionally substitute steel (building) or titanium (space).
  • Steel = 2 MC value, titanium = 3 MC value toward their card types.
  • You cannot overpay in MC; overpaying with steel/titanium is allowed (surplus lost).

STRATEGIC TIPS:
  • Prioritise increasing MC production — it compounds every generation.
  • Match your corporation's ability to the cards you buy initially.
  • Plan milestone/award strategy early; opponents can block you.
  • Steel and titanium production turbocharge expensive building/space cards.
  • Science tags matter for the Scientist award and many card requirements.
  • Greenery placement near your cities multiplies your end-game VP.
  • Passing early saves MC but gives opponents tempo; balance carefully.
  • Calculate MC cost per VP — 15 MC per VP is the rough benchmark.
  • Slow the game deliberately if your per-generation card VP > opponent's.
  • Accelerate terraforming if you have high TR or need to end before opponents
    can catch up.

ADVANCED STRATEGIES:
  Science/Jupiter engine:
    – Science tags reduce costs and unlock many cards — chain them.
    – Jupiter tags are rare; the few cards that score them are extremely valuable.
      Prioritise all Jupiter-tagged cards and pair with Titan production
      (most Jupiter cards use titanium for payment).
    – Physics Complex: +2 VP/generation once energy production ≥ 6.

  Titanium engine:
    – High titanium production makes expensive space-tag cards nearly free.
    – Asteroid events also damage opponents' plants, slowing their greenery.

  Card-VP vs board-VP (depends on player count):
    – Many players → focus on card-based VPs (animals, science, Jupiter).
      Board VP (city × greenery) is harder when space is contested.
    – Fewer players → board VP is viable; you have more turns to build.

  Pace control:
    – If your engine generates more points per generation, SLOW terraforming:
      avoid raising global parameters unless the card benefit outweighs the
      tempo gift to opponents.
    – If you are behind in VP but ahead in TR, ACCELERATE — end the game
      before opponents' engines overtake you.

OPPONENT ANALYSIS — read opponents constantly:
  • Their played cards reveal their engine (energy → heat, plant engine, etc.).
  • Funded awards signal what they are optimising for — don't help them win it.
  • Claimed milestones tell you what to block or race for next.
  • High hand size + few played cards → they are building toward a big combo.
  • Low MC + many played cards → they over-extended; they may pass soon.

DRAFTING (when research phase offers card selection):
  • Do not pass a card that strongly benefits an opponent's visible engine
    UNLESS you have a card that is strictly better for YOUR own engine.
  • Deny opponent synergy cards (e.g. an opponent building Jupiter engine —
    withhold Jupiter-tag cards even at personal cost).
  • In late game, pass weak cards freely; in early game, card denial matters more.

=== END RULES REFERENCE ===
"""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def select_action_llm(state: dict, waiting_for: dict) -> tuple[dict, dict]:
    """Return (input_response, debug) using the configured LLM provider."""
    game_id = state.get("game", {}).get("id", "unknown")
    wf_type  = waiting_for.get("type", "")

    try:
        if wf_type in SETUP_TYPES:
            return _select_setup(state, waiting_for, game_id)
        else:
            return _select_action(state, waiting_for, game_id)
    except Exception as exc:
        logger.error("LLM selection failed (game=%s type=%s): %s — using default",
                     game_id, wf_type, exc, exc_info=True)
        return _default_response(waiting_for), {"llm_error": str(exc)}


# ---------------------------------------------------------------------------
# LLM provider helpers — session-per-game
# ---------------------------------------------------------------------------

def _ensure_gemini_client() -> None:
    global _gemini_client
    if not _GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(api_key=_GEMINI_API_KEY)


def _call_llm_init(game_id: str, system: str, user: str, think: bool = False) -> str:
    """Start a new session for game_id and return the first response.

    Sends the full system prompt (TM rules + board context) once. All subsequent
    calls via _call_llm_continue omit the system and rely on session memory.
    """
    _session_base_system[game_id] = system
    if _LLM_DEBUG:
        logger.info("=== LLM INIT (provider=%s game=%s) system ===\n%s",
                    _LLM_PROVIDER, game_id, system)
        logger.info("=== LLM INIT user ===\n%s", user)

    if _LLM_PROVIDER == "gemini":
        text = _init_gemini_session(game_id, system, user, think)
    else:
        text = _init_ollama_session(game_id, system, user, think)

    if _LLM_DEBUG:
        logger.info("=== LLM INIT response ===\n%s", text)
    return text


def _call_llm_continue(game_id: str, user: str) -> str:
    """Continue the existing session for game_id (no system re-sent).

    Falls back to a stateless call with rules + strategy if the session was lost
    (e.g. server restart mid-game).
    """
    if _LLM_DEBUG:
        logger.info("=== LLM CONTINUE (provider=%s game=%s) user ===\n%s",
                    _LLM_PROVIDER, game_id, user)

    if _LLM_PROVIDER == "gemini":
        chat = _game_chat_sessions.get(game_id)
        if chat is None:
            logger.warning("No Gemini session for game %s — falling back to stateless", game_id)
            text = _stateless_fallback(game_id, user)
        else:
            text = _continue_gemini_session(game_id, user, chat)
    else:
        session = _game_sessions.get(game_id)
        if session is None:
            logger.warning("No Ollama session for game %s — falling back to stateless", game_id)
            text = _stateless_fallback(game_id, user)
        else:
            text = _continue_ollama_session(game_id, user, session)

    if _LLM_DEBUG:
        logger.info("=== LLM CONTINUE response ===\n%s", text)
    return text


def _stateless_fallback(game_id: str, user: str) -> str:
    """Single stateless call when no session exists (server restart recovery)."""
    strategy = _game_strategies.get(game_id, "Play a balanced game — maximise TR and card synergies.")
    system = (
        TM_RULES + "\n\n"
        "You are a Terraforming Mars player. Your current strategy:\n"
        f"{strategy}\n\n"
        "Pick the single best action. Respond ONLY:\n"
        "CHOICE: <number>\n"
        "STRATEGY_UPDATE: <full updated strategy or 'no change'>"
    )
    if _LLM_PROVIDER == "gemini":
        _ensure_gemini_client()
        from google.genai import types
        response = _gemini_client.models.generate_content(  # type: ignore[union-attr]
            model=_GEMINI_MODEL,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return response.text
    else:
        payload: dict = {
            "model": _OLLAMA_MODEL, "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        }
        r = requests.post(f"{_OLLAMA_URL}/api/chat", json=payload, timeout=_OLLAMA_TIMEOUT)
        r.raise_for_status()
        return r.json()["message"]["content"]


# ---------------------------------------------------------------------------
# Ollama session management
# ---------------------------------------------------------------------------

def _init_ollama_session(game_id: str, system: str, user: str, think: bool) -> str:
    session: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    payload: dict = {"model": _OLLAMA_MODEL, "stream": False, "messages": session}
    if _OLLAMA_MODEL.startswith("qwen3"):
        payload["think"] = think
    r = requests.post(f"{_OLLAMA_URL}/api/chat", json=payload, timeout=_OLLAMA_TIMEOUT)
    r.raise_for_status()
    text: str = r.json()["message"]["content"]
    session.append({"role": "assistant", "content": text})
    _game_sessions[game_id] = session
    return text


def _continue_ollama_session(game_id: str, user: str, session: list[dict]) -> str:
    session.append({"role": "user", "content": user})
    _trim_ollama_session_if_needed(game_id, session)
    payload: dict = {"model": _OLLAMA_MODEL, "stream": False, "messages": session}
    # think=False for action phase
    if _OLLAMA_MODEL.startswith("qwen3"):
        payload["think"] = False
    r = requests.post(f"{_OLLAMA_URL}/api/chat", json=payload, timeout=_OLLAMA_TIMEOUT)
    r.raise_for_status()
    text: str = r.json()["message"]["content"]
    session.append({"role": "assistant", "content": text})
    return text


def _trim_ollama_session_if_needed(game_id: str, session: list[dict]) -> None:
    """If the session is too long, rebuild it: base system+strategy + last 40 messages."""
    if len(session) <= _MAX_SESSION_MESSAGES:
        return
    strategy = _game_strategies.get(game_id, "")
    base = _session_base_system.get(game_id, session[0]["content"])
    system_with_reminder = base
    if strategy:
        system_with_reminder = (
            base + f"\n\n[CONTEXT TRIM — current strategy to continue with:\n{strategy}]"
        )
    tail = session[-40:]  # keep last 20 exchanges
    session.clear()
    session.append({"role": "system", "content": system_with_reminder})
    session.extend(tail)
    logger.info("Ollama session trimmed for game %s — kept last 40 messages + strategy reminder",
                game_id)


# ---------------------------------------------------------------------------
# Gemini session management
# ---------------------------------------------------------------------------

def _init_gemini_session(game_id: str, system: str, user: str, think: bool) -> str:
    """Setup via generate_content (supports think=True), then create Chat with history."""
    _ensure_gemini_client()
    from google.genai import types

    thinking_budget = 1024 if think else 0
    response = _gemini_client.models.generate_content(  # type: ignore[union-attr]
        model=_GEMINI_MODEL,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
        ),
    )
    text: str = response.text

    # Create chat with the setup exchange as initial history.
    # Action calls use thinking_budget=0 (fast responses).
    history = [
        types.Content(role="user",  parts=[types.Part.from_text(text=user)]),
        types.Content(role="model", parts=[types.Part.from_text(text=text)]),
    ]
    chat = _gemini_client.chats.create(  # type: ignore[union-attr]
        model=_GEMINI_MODEL,
        config=types.GenerateContentConfig(
            system_instruction=system,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
        history=history,
    )
    _game_chat_sessions[game_id] = chat
    return text


def _continue_gemini_session(game_id: str, user: str, chat: object) -> str:
    response = chat.send_message(user)  # type: ignore[attr-defined]
    return response.text


# ---------------------------------------------------------------------------
# Setup phase (initialCards / prelude)
# ---------------------------------------------------------------------------

def _select_setup(state: dict, waiting_for: dict, game_id: str) -> tuple[dict, dict]:
    g = state.get("game", {})
    wf_type = waiting_for.get("type", "")
    has_session = game_id in (
        _game_chat_sessions if _LLM_PROVIDER == "gemini" else _game_sessions
    )

    logger.info("LLM setup call (game=%s type=%s board=%s exps=%s session=%s)",
                game_id, wf_type, g.get("boardName", "?"), g.get("expansions", []), has_session)

    user = _build_setup_prompt(state, waiting_for, game_id)

    if wf_type == "initialCards" or not has_session:
        # First call of the game: send full system prompt and start a new session.
        # format_config_context includes board, expansions, variants, AND the actual
        # milestones/awards for this specific game (even if randomised).
        game_ctx = format_config_context(g)
        system = (
            TM_RULES + "\n\n" + game_ctx + "\n\n"
            "You are an expert Terraforming Mars strategist making the opening decisions. "
            "Think step by step about card synergies, engine building, the specific milestones "
            "and awards listed above, and any active game variants. "
            "Remember everything in this session — you will continue playing this game in "
            "subsequent messages without receiving these rules again. "
            "Follow the EXACT output format requested — no extra text before or after."
        )
        text = _call_llm_init(game_id, system, user, think=True)
    else:
        # Prelude comes after initialCards in the same game — continue the session.
        text = _call_llm_continue(game_id, user)

    logger.info("Setup LLM response (game=%s):\n%s", game_id, text[:1000])

    input_response, strategy = _parse_setup_response(text, waiting_for, game_id)
    _game_strategies[game_id] = strategy
    logger.info("Game %s strategy stored:\n%s", game_id, strategy)
    return input_response, {"llm_phase": "setup", "strategy": strategy[:300]}


def _build_setup_prompt(state: dict, waiting_for: dict, game_id: str) -> str:
    wf_type = waiting_for.get("type", "")
    options = waiting_for.get("options", [])
    lines: list[str] = []

    if wf_type == "initialCards":
        # Identify sub-options by title
        corp_opt    = _find_option(options, ("corporation",))
        prelude_opt = _find_option(options, ("prelude",))
        ceo_opt     = _find_option(options, ("ceo",))
        project_opt = _find_option(options, ("project", "initial", "cards to buy"))

        corps   = corp_opt.get("cards", []) if corp_opt else []
        buyable = project_opt.get("cards", []) if project_opt else []

        lines += [
            "# Terraforming Mars — Opening Decisions",
            "",
            "## Corporation Choices (choose exactly 1)",
        ]
        for i, c in enumerate(corps, 1):
            name = c.get("name", f"Corp {i}")
            entry = CARD_DB.get(name)
            desc = f" — {entry['description']}" if entry and entry.get("description") else ""
            lines.append(f"  {i}. {name}{desc}")

        if buyable:
            lines += [
                "",
                "## Project Cards Available to Add to Hand",
                "  Each costs 3 MC to buy now. Play cost (during game) shown in [brackets].",
                "  You cannot buy more cards than: floor(chosen_corporation_starting_MC / 3).",
            ]
            for i, c in enumerate(buyable, 1):
                name = c.get("name", f"Card {i}")
                play_cost = c.get("calculatedCost", "?")
                entry = CARD_DB.get(name)
                tags = entry.get("tags") or [] if entry else []
                tag_str = f" [{', '.join(tags)}]" if tags else ""
                desc = f" — {entry['description']}" if entry and entry.get("description") else ""
                lines.append(f"  {i}. {name}{tag_str}  [play cost: {play_cost} MC]{desc}")

        if prelude_opt:
            preludes = prelude_opt.get("cards", [])
            lines += ["", "## Prelude Cards (choose 2 from these)"]
            for i, c in enumerate(preludes, 1):
                name = c.get("name", f"Prelude {i}")
                entry = CARD_DB.get(name)
                desc = f" — {entry['description']}" if entry and entry.get("description") else ""
                lines.append(f"  {i}. {name}{desc}")

        if ceo_opt:
            ceos = ceo_opt.get("cards", [])
            lines += ["", "## CEO Card (choose 1)"]
            for i, c in enumerate(ceos, 1):
                name = c.get("name", f"CEO {i}")
                entry = CARD_DB.get(name)
                desc = f" — {entry['description']}" if entry and entry.get("description") else ""
                lines.append(f"  {i}. {name}{desc}")

        lines += [
            "",
            "Analyse the corporations and project cards. Choose the corporation that best",
            "synergises with the available project cards and write a clear strategic plan.",
            "This strategy is your memory for the whole game — you will update it each turn.",
            "After this, each game turn you will receive game state + numbered options and",
            "must reply CHOICE: <number> and STRATEGY_UPDATE: <full strategy or 'no change'>.",
            "",
            "Respond in EXACTLY this format (copy card/corporation names exactly as listed):",
            "CORPORATION: <exact name from list above>",
            "BUY_CARDS: <exact comma-separated names from project card list, or 'none'>",
        ]
        if prelude_opt:
            lines.append("PRELUDE_CARDS: <exact comma-separated names of 2 prelude cards>")
        if ceo_opt:
            lines.append("CEO_CARD: <exact name of 1 CEO card>")
        lines += [
            "STRATEGY:",
            "<150-250 words: engine type, priority tags, milestone/award targets, key card",
            " synergies, pace plan (accelerate or slow terraforming), opponent watch-outs>",
        ]

    elif wf_type == "prelude":
        p    = state.get("player", {})
        prod = {k: v for k, v in p.get("production", {}).items() if v}
        lines += [
            "# Terraforming Mars — Prelude Selection",
            f"State: MC={p.get('megacredits',0)} Production={prod}",
            "",
            "## Prelude Options",
        ]
        for i, opt in enumerate(options, 1):
            title = _node_title(opt, i)
            if opt.get("type") == "card":
                names = ", ".join(c.get("name","") for c in opt.get("cards", []))
                lines.append(f"  {i}. {names}")
            else:
                lines.append(f"  {i}. {title}")
        lines += [
            "",
            "Choose the prelude that best advances your strategy.",
            "",
            "Respond in EXACTLY this format:",
            "CHOICE: <option number>",
            "STRATEGY_UPDATE: <your full updated strategy — repeat every element you want to",
            " keep plus any changes; this REPLACES your memory entirely. Write 'no change'",
            " only if truly nothing has changed.>",
        ]

    return "\n".join(lines)


def _find_option(options: list, keywords: tuple) -> dict | None:
    """Return first option whose title contains any of the keywords (case-insensitive)."""
    for opt in options:
        t = _node_title(opt, 0).lower()
        if any(k in t for k in keywords):
            return opt
    return None


def _parse_setup_response(
    text: str, waiting_for: dict, game_id: str
) -> tuple[dict, str]:
    wf_type = waiting_for.get("type", "")
    options = waiting_for.get("options", [])

    if wf_type == "initialCards":
        corp_opt    = _find_option(options, ("corporation",))
        prelude_opt = _find_option(options, ("prelude",))
        ceo_opt     = _find_option(options, ("ceo",))
        project_opt = _find_option(options, ("project", "initial", "cards to buy"))

        corps   = [c.get("name","") for c in (corp_opt or {}).get("cards", [])]
        buyable = [c.get("name","") for c in (project_opt or {}).get("cards", [])]

        # --- Parse CORPORATION ---
        chosen_corp = corps[0] if corps else ""
        m = re.search(r"CORPORATION:\s*(.+)", text)
        if m:
            raw = m.group(1).strip().rstrip(".")
            for c in corps:
                if c.lower() in raw.lower() or raw.lower() in c.lower():
                    chosen_corp = c
                    break

        # --- Parse BUY_CARDS ---
        bought: list[str] = []
        m = re.search(r"BUY_CARDS:\s*(.+?)(?:\nPRELUDE|\nCEO|\nSTRATEGY|\Z)",
                      text, re.DOTALL | re.IGNORECASE)
        if m:
            raw_list = m.group(1).strip()
            if raw_list.lower() not in ("none", "none.", ""):
                for raw_name in re.split(r",\s*|\n", raw_list):
                    raw_name = raw_name.strip().lstrip("-•").strip().rstrip(".")
                    if not raw_name:
                        continue
                    for b in buyable:
                        if b.lower() in raw_name.lower() or raw_name.lower() in b.lower():
                            if b not in bought:
                                bought.append(b)
                            break

        # --- Parse PRELUDE_CARDS ---
        prelude_chosen: list[str] = []
        if prelude_opt:
            preludes = [c.get("name","") for c in prelude_opt.get("cards", [])]
            m2 = re.search(r"PRELUDE_CARDS:\s*(.+?)(?:\nCEO|\nSTRATEGY|\Z)",
                           text, re.DOTALL | re.IGNORECASE)
            if m2:
                for raw_name in re.split(r",\s*|\n", m2.group(1).strip()):
                    raw_name = raw_name.strip().rstrip(".")
                    for p in preludes:
                        if p.lower() in raw_name.lower() or raw_name.lower() in p.lower():
                            if p not in prelude_chosen:
                                prelude_chosen.append(p)
                            break
            # Fallback: first 2
            if len(prelude_chosen) < 2:
                prelude_chosen = preludes[:2]

        # --- Parse CEO_CARD ---
        ceo_chosen: str = ""
        if ceo_opt:
            ceos = [c.get("name","") for c in ceo_opt.get("cards", [])]
            m3 = re.search(r"CEO_CARD:\s*(.+?)(?:\nSTRATEGY|\Z)",
                           text, re.DOTALL | re.IGNORECASE)
            if m3 and ceos:
                raw_ceo = m3.group(1).strip().rstrip(".")
                for c in ceos:
                    if c.lower() in raw_ceo.lower() or raw_ceo.lower() in c.lower():
                        ceo_chosen = c
                        break
            if not ceo_chosen and ceos:
                ceo_chosen = ceos[0]

        # --- Parse STRATEGY ---
        m4 = re.search(r"STRATEGY:\s*(.*)", text, re.DOTALL)
        strategy = m4.group(1).strip() if m4 else text.strip()

        logger.info("Setup parsed: corp=%r buy=%r prelude=%r ceo=%r",
                    chosen_corp, bought, prelude_chosen, ceo_chosen)

        # Build exactly len(options) responses, one per sub-option
        responses = []
        for opt in options:
            t = _node_title(opt, 0).lower()
            if "corporation" in t:
                responses.append({"type": "card", "cards": [chosen_corp] if chosen_corp else corps[:1]})
            elif "prelude" in t:
                responses.append({"type": "card", "cards": prelude_chosen})
            elif "ceo" in t:
                responses.append({"type": "card", "cards": [ceo_chosen] if ceo_chosen else []})
            elif any(k in t for k in ("project", "initial", "cards to buy")):
                responses.append({"type": "card", "cards": bought})
            else:
                responses.append(_default_response(opt))

        return {"type": "initialCards", "responses": responses}, strategy

    elif wf_type == "prelude":
        m = re.search(r"CHOICE:\s*(\d+)", text)
        idx = int(m.group(1)) - 1 if m else 0
        idx = max(0, min(idx, len(options) - 1))
        m2 = re.search(r"STRATEGY_UPDATE:\s*(.*)", text, re.DOTALL)
        old = _game_strategies.get(game_id, "Play balanced.")
        strategy = old
        if m2:
            upd = m2.group(1).strip()
            if upd and not upd.lower().startswith("no change"):
                strategy = upd
        return index_to_response(waiting_for, idx), strategy

    return _default_response(waiting_for), "Play a balanced game."


# ---------------------------------------------------------------------------
# Action phase
# ---------------------------------------------------------------------------

def _select_action(state: dict, waiting_for: dict, game_id: str) -> tuple[dict, dict]:
    options = flatten_options(waiting_for)
    if not options:
        return _default_response(waiting_for), {}

    user = _build_action_prompt(state, waiting_for, options)
    logger.debug("LLM action (game=%s type=%s options=%d)",
                 game_id, waiting_for.get("type"), len(options))
    text = _call_llm_continue(game_id, user)
    logger.debug("Action response (game=%s): %s", game_id, text[:300])

    return _parse_action_response(text, options, waiting_for, game_id)


def _build_action_prompt(state: dict, waiting_for: dict, options: list[dict]) -> str:
    g    = state.get("game", {})
    p    = state.get("player", {})
    prod = {k: v for k, v in p.get("production", {}).items() if v}
    tags = {k: v for k, v in p.get("tags", {}).items() if v}
    played = p.get("playedCards", [])  # list of strings from stateMapping
    my_id = p.get("id", "")
    ms_raw = state.get("milestones", [])
    aw_raw = state.get("awards", [])
    ms = [f"{m.get('name','?')} ({'you' if m.get('playerId')==my_id else 'opponent'})"
          for m in ms_raw]
    aw = [f"{a.get('name','?')} ({'you' if a.get('playerId')==my_id else 'opponent'})"
          for a in aw_raw]

    lines = [
        f"Gen {g.get('generation',1)} | Temp {g.get('temperature',-30)}°C | "
        f"O₂ {g.get('oxygen',0)}% | Oceans {g.get('oceanCount',0)}/9",
        f"TR:{p.get('terraformRating',20)}  MC:{p.get('megacredits',0)}  "
        f"St:{p.get('steel',0)}  Ti:{p.get('titanium',0)}  "
        f"Pl:{p.get('plants',0)}  En:{p.get('energy',0)}  He:{p.get('heat',0)}",
    ]
    if prod:
        lines.append(f"Production: {prod}")
    if tags:
        lines.append(f"Tags: {tags}")
    if played:
        lines.append(f"Played ({len(played)}): {', '.join(played[:14])}"
                     f"{'…' if len(played) > 14 else ''}")
    lines.append(f"Hand: {p.get('handSize', 0)} cards")
    opponents = state.get("opponents") or []
    for i, opp in enumerate(opponents, 1):
        opp_prod  = {k: v for k, v in opp.get("production", {}).items() if v}
        opp_tags  = {k: v for k, v in opp.get("tags", {}).items() if v}
        opp_cards = opp.get("playedCards", [])  # list of strings
        label = f"Opponent{'' if len(opponents)==1 else i}"
        lines.append(
            f"{label}: TR:{opp.get('terraformRating',20)}  MC:{opp.get('megacredits',0)}  "
            f"prod:{opp_prod}  tags:{opp_tags}"
        )
        if opp_cards:
            lines.append(f"  {label} played: {', '.join(opp_cards[:16])}"
                         f"{'…' if len(opp_cards) > 16 else ''}")
    if ms:
        lines.append(f"Milestones claimed: {ms}")
    if aw:
        lines.append(f"Awards funded: {aw}")

    title = (waiting_for.get("title") or "").strip()
    if title:
        lines += ["", f"Decision: {title}"]

    # Inject card descriptions if this decision involves selecting specific cards
    card_names_in_decision = _extract_card_names(waiting_for)
    if card_names_in_decision:
        ctx = format_card_context(card_names_in_decision, header="Card descriptions:", max_cards=20)
        if ctx:
            lines += ["", ctx]

    lines.append("\nChoose from:")
    for i, opt in enumerate(options, 1):
        lines.append(f"  {i}. {opt['title']}")

    lines += [
        "",
        "CHOICE: <number>",
        "STRATEGY_UPDATE: <your full updated strategy — repeat everything you want to keep plus "
        "any changes; this REPLACES your memory so omit nothing. Write 'no change' only if "
        "truly nothing has changed.>",
    ]
    return "\n".join(lines)


def _parse_action_response(
    text: str, options: list[dict], waiting_for: dict, game_id: str
) -> tuple[dict, dict]:
    m = re.search(r"CHOICE:\s*(\d+)", text)
    chosen = int(m.group(1)) - 1 if m else 0
    chosen = max(0, min(chosen, len(options) - 1))

    m2 = re.search(r"STRATEGY_UPDATE:\s*(.*)", text, re.DOTALL)
    if m2:
        upd = m2.group(1).strip()
        if upd and not upd.lower().startswith("no change"):
            _game_strategies[game_id] = upd
            logger.info("Strategy updated for game %s", game_id)

    option = options[chosen]
    logger.info("Action choice game=%s: %d. %s", game_id, chosen + 1, option["title"])
    return index_to_response(waiting_for, option["index"]), {
        "llm_choice": chosen + 1,
        "llm_option": option["title"],
    }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _node_title(node: dict, fallback: int) -> str:
    title = node.get("title", "")
    if isinstance(title, str) and title:
        return title
    if isinstance(title, dict):
        return str(title.get("message", f"Option {fallback}"))
    return f"Option {fallback}"


def _extract_card_names(waiting_for: dict, max_depth: int = 3) -> list[str]:
    """Recursively collect card names from a waitingFor node (for card-selection decisions)."""
    if max_depth <= 0:
        return []
    names: list[str] = []
    wf_type = waiting_for.get("type", "")
    if wf_type == "card":
        for c in waiting_for.get("cards", []):
            name = c.get("name", "") if isinstance(c, dict) else str(c)
            if name and name not in names:
                names.append(name)
    elif wf_type in ("or", "and"):
        for opt in waiting_for.get("options", []):
            for n in _extract_card_names(opt, max_depth - 1):
                if n not in names:
                    names.append(n)
    return names
