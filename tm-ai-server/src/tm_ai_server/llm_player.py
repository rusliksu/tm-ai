"""
LLM-based action selection — supports Ollama (local) and Gemini (cloud).

Session-per-game architecture:
- Setup (initialCards): full system prompt (TM rules + board context) sent ONCE to
  initialise the session. The model writes an opening strategy document.
- All subsequent decisions (prelude / action): continue the same session. Only the
  current game state and numbered options are sent — rules/board context are in
  session memory, so they are NOT repeated each turn.
- Thinking is enabled on EVERY turn (setup, prelude, actions, per-gen reflection).
  Default budget controllable via GEMINI_THINKING_BUDGET (default 1024 tokens).
- Per-generation strategy update: at the first action of each new generation, the LLM
  is asked to restate its strategy (standing, engine, milestone/award targets, next-gen
  priority). The response becomes natural chat history (no destructive rebuild) and is
  stored in _game_strategies for debugging + session recovery.
- Context trimming: both providers. Ollama: rebuilt after MAX_SESSION_MESSAGES (~30 turns).
  Gemini: trimmed after GEMINI_MAX_TURNS (default 80) by keeping last 40 message pairs +
  strategy summary. At ~1938 tokens/turn of accumulated context, 80 turns ≈ 155K tokens
  per request — safely under the 1M/min paid-tier quota for two concurrent players.

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
    GEMINI_MODEL            Model name (default: gemini-2.5-flash-lite)
                            Supported: gemini-3-pro, gemini-3-flash, gemini-3-flash-lite,
                            gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite
                            (any Google AI Studio model name is accepted verbatim)
    GEMINI_THINKING_BUDGET  Thinking tokens per turn (default: 1024)
"""

from __future__ import annotations
import hashlib
import logging
import os
import re
import time

import requests

from .encoding import flatten_options, index_to_response, _default_response
from .game_knowledge import CARD_DB, format_card_context, format_config_context, format_board_layout

logger = logging.getLogger(__name__)

_LLM_PROVIDER   = os.getenv("LLM_PROVIDER", "ollama").lower()
_LLM_DEBUG      = os.getenv("LLM_DEBUG", "false").lower() == "true"

# Ollama settings
_OLLAMA_URL     = os.getenv("OLLAMA_URL",   "http://localhost:11434")
_OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL", "qwen3:4b")
_OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "600"))

# Gemini settings
_GEMINI_API_KEY        = os.getenv("GEMINI_API_KEY", "")
_GEMINI_MODEL          = os.getenv("GEMINI_MODEL",  "gemini-2.5-flash-lite")
_GEMINI_THINKING_BUDGET = int(os.getenv("GEMINI_THINKING_BUDGET", "512"))   # setup/prelude always use 1024

_gemini_client = None  # lazy-initialised

SETUP_TYPES = {"initialCards", "prelude"}

# Per-game strategy documents: game_id → strategy text (for logging and trim recovery)
_game_strategies: dict[str, str] = {}

# Per-game session history: game_id → messages list (Ollama) or Chat object (Gemini)
_game_sessions: dict[str, list[dict]] = {}
_game_chat_sessions: dict[str, object] = {}

# Gemini context caching: game_id → {name, created_at, system}
_game_cache_info: dict[str, dict] = {}
_CACHE_REFRESH_AFTER = 50 * 60  # seconds — refresh TTL before cache expires at 3600s
_session_base_system: dict[str, str] = {}  # game_id → original system (for trim)

# Per-generation Gemini strategy update: game_id → last seen generation number
_gemini_last_generation: dict[str, int] = {}

# Hand-description elision: game_id → generation number on which the hand was last shown
# with full descriptions. On subsequent action turns in the same generation, card names
# are shown only with "(desc. shown earlier this gen)" to save ~400–600 tokens per turn.
_hand_shown_generation: dict[str, int] = {}

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
  • Temperature: -30°C → +8°C  (each step = +2°C, 20 steps). Raise: 8 heat, Asteroid SP, or cards.
  • Oxygen:       0%   → 14%   (14 steps).                    Raise: place greenery tile or cards.
  • Oceans:       0   → 9 tiles.                              Place: Aquifer SP or cards.
  • Venus:        0%   → 30%   (each step = +2%, 15 steps).   Raise: Venus expansion cards/SPs.
  Each step raised = +1 TR (=+1 income and +1 VP).

GLOBAL-PARAMETER MILESTONE BONUSES (one-time, awarded to the player who triggers the threshold):
  • Temperature reaches -24°C : +1 HEAT PRODUCTION to the raising player.
  • Temperature reaches -20°C : +1 HEAT PRODUCTION to the raising player.
  • Temperature reaches   0°C : the raising player places 1 OCEAN tile.
  • Oxygen reaches 8%         : temperature automatically rises +1 step (free TR + further bonuses).
  • Venus reaches 8%          : the raising player DRAWS 1 CARD.
  • Venus reaches 16%         : the raising player gains +1 TR.
  These are big — timing your terraforming step to hit a threshold yourself is worth ~10 MC of value.
  If an opponent is about to hit a threshold, consider racing them to it.

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
  • Ocean:   only on reserved blue spaces; any tile placed next to it gets its player +2 MC.
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
  • Don't fund awards in early phase as this gives opponents a clear target to contest.
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

def select_action_llm(state: dict, waiting_for: dict, last_error: str | None = None) -> tuple[dict, dict]:
    """Return (input_response, debug) using the configured LLM provider."""
    game_id = state.get("game", {}).get("id", "unknown")
    wf_type  = waiting_for.get("type", "")

    try:
        if wf_type in SETUP_TYPES:
            return _select_setup(state, waiting_for, game_id)
        else:
            return _select_action(state, waiting_for, game_id, last_error=last_error)
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


def _log_prompt(header: str, text: str) -> None:
    prefixed = '\n'.join(f'> {line}' for line in text.splitlines())
    logger.info("%s\n%s", header, prefixed)


def _log_response(header: str, text: str | None) -> None:
    if text is None:
        logger.info("%s\n< (empty/None response)", header)
        return
    prefixed = '\n'.join(f'< {line}' for line in text.splitlines())
    logger.info("%s\n%s", header, prefixed)


def _call_llm_init(game_id: str, system: str, user: str, think: bool = False) -> str:
    """Start a new session for game_id and return the first response.

    Sends the full system prompt (TM rules + board context) once. All subsequent
    calls via _call_llm_continue omit the system and rely on session memory.
    """
    _session_base_system[game_id] = system
    if _LLM_DEBUG:
        _log_prompt(f"=== INIT system (provider={_LLM_PROVIDER} game={game_id}) ===", system)
        _log_prompt(f"=== INIT user (game={game_id}) ===", user)

    if _LLM_PROVIDER == "gemini":
        text = _init_gemini_session(game_id, system, user, think)
    else:
        text = _init_ollama_session(game_id, system, user, think)

    if _LLM_DEBUG:
        _log_response(f"=== INIT response (game={game_id}) ===", text)
    return text


def _call_llm_continue(game_id: str, user: str, max_output_tokens: int | None = None) -> str:
    """Continue the existing session for game_id (no system re-sent).

    Falls back to a session-recovery call with rules + strategy if the session
    was lost (e.g. server restart mid-game or 503 exhausted on setup).
    max_output_tokens caps Gemini response length (Ollama: ignored).
    """
    if _LLM_DEBUG:
        _log_prompt(f"=== CONTINUE user (provider={_LLM_PROVIDER} game={game_id}) ===", user)

    if _LLM_PROVIDER == "gemini":
        chat = _game_chat_sessions.get(game_id)
        if chat is None:
            logger.warning("No Gemini session for game %s — recovering session", game_id)
            text = _session_recovery(game_id, user)
        else:
            text = _continue_gemini_session(game_id, user, chat, max_output_tokens=max_output_tokens)
    else:
        session = _game_sessions.get(game_id)
        if session is None:
            logger.warning("No Ollama session for game %s — recovering session", game_id)
            text = _session_recovery(game_id, user)
        else:
            text = _continue_ollama_session(game_id, user, session)

    if _LLM_DEBUG:
        _log_response(f"=== CONTINUE response (game={game_id}) ===", text)
    return text


def _session_recovery(game_id: str, user: str) -> str:
    """Re-initialise a session when none exists (setup 503 exhausted or server restart).

    Creates a proper chat session as a side-effect so all subsequent turns continue it.
    """
    strategy = _game_strategies.get(game_id, "Play a balanced game — maximise TR and card synergies.")
    system = (
        TM_RULES + "\n\n"
        "You are a Terraforming Mars player. Your current strategy:\n"
        f"{strategy}\n\n"
        "Pick the single best action. Respond ONLY:\n"
        "CHOICE: <number>"
    )
    logger.info("Session recovery for game %s (strategy: %.80s…)", game_id, strategy)
    # Initialise a proper session — future turns will use _continue_*
    if _LLM_PROVIDER == "gemini":
        return _init_gemini_session(game_id, system, user, think=False)
    else:
        return _init_ollama_session(game_id, system, user, think=False)


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
    if len(session) > _MAX_SESSION_MESSAGES:
        _capture_strategy_then_trim(game_id, session)
    payload: dict = {"model": _OLLAMA_MODEL, "stream": False, "messages": session}
    if _OLLAMA_MODEL.startswith("qwen3"):
        payload["think"] = True  # think on every turn (was False — caused shallow choices)
    r = requests.post(f"{_OLLAMA_URL}/api/chat", json=payload, timeout=_OLLAMA_TIMEOUT)
    r.raise_for_status()
    text: str = r.json()["message"]["content"]
    session.append({"role": "assistant", "content": text})
    return text


def _capture_strategy_then_trim(game_id: str, session: list[dict]) -> None:
    """Ask the model for its current strategy, save it, then rebuild the session."""
    # Temporarily swap in the strategy-capture request (keep the pending user msg aside)
    pending_user = session.pop()
    session.append({
        "role": "user",
        "content": (
            "Before we continue: write out your current strategy in 150-200 words — "
            "engine type, priority tags, milestone/award targets, pace plan, key watch-outs. "
            "Include a TABLEAU section listing every card you have played and its key ongoing effect."
        ),
    })
    payload: dict = {"model": _OLLAMA_MODEL, "stream": False, "messages": session}
    if _OLLAMA_MODEL.startswith("qwen3"):
        payload["think"] = False
    try:
        r = requests.post(f"{_OLLAMA_URL}/api/chat", json=payload, timeout=_OLLAMA_TIMEOUT)
        r.raise_for_status()
        strategy = r.json()["message"]["content"].strip()
        _game_strategies[game_id] = strategy
        logger.info("Captured strategy before Ollama trim (game=%s): %.100s…", game_id, strategy)
    except Exception as exc:
        logger.warning("Strategy capture before trim failed (game=%s): %s", game_id, exc)

    # Restore the pending user message and rebuild the trimmed session
    session.pop()  # remove strategy_request
    session.append(pending_user)

    strategy = _game_strategies.get(game_id, "")
    base = _session_base_system.get(game_id, session[0]["content"])
    system_with_reminder = (
        base + f"\n\n[CONTEXT TRIM — current strategy:\n{strategy}]" if strategy else base
    )
    tail = session[-40:]
    session.clear()
    session.append({"role": "system", "content": system_with_reminder})
    session.extend(tail)
    logger.info("Ollama session trimmed for game %s — kept last 40 messages + strategy", game_id)


# ---------------------------------------------------------------------------
# Gemini session management
# ---------------------------------------------------------------------------

_GEMINI_RETRY_ATTEMPTS = 3
_GEMINI_RETRY_DELAY    = 5  # seconds for 503 back-off (doubles: 5, 10)

# Token cap for action-turn responses — keeps context history from growing unboundedly.
# Strategy updates and setup phases are left uncapped (they need full reasoning).
_GEMINI_ACTION_MAX_OUTPUT_TOKENS: int = int(os.getenv("GEMINI_ACTION_MAX_OUTPUT_TOKENS", "350"))

# Trim Gemini chat history when the session exceeds this many turns (user+model pairs).
# At ~1938 tokens/turn of context growth, 80 turns ≈ 155K tokens of history per call —
# a safe headroom under the 1M/min quota with two concurrent players.
_MAX_GEMINI_TURNS: int = int(os.getenv("GEMINI_MAX_TURNS", "80"))

# Per-game turn counter for Gemini sessions (for history trimming)
_gemini_turn_count: dict[str, int] = {}


def _is_transient_gemini_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("503", "unavailable", "429", "rate limit", "overloaded", "resource exhausted"))


def _parse_api_retry_delay(exc: Exception) -> float | None:
    """Extract retryDelay seconds from a Google API error response, or return None."""
    m = re.search(r"retryDelay.*?(\d+\.?\d*)s", str(exc))
    return float(m.group(1)) if m else None


def _gemini_with_retry(fn):
    """Call fn() with back-off on transient Gemini errors.

    429 quota errors: waits the exact retryDelay from the API response (+ 2s buffer) so
    the minute-window quota resets before retrying. This avoids the old 5s/10s pattern
    which retried inside the same quota window and tripled token consumption per failure.
    503/overloaded: standard exponential back-off (5s, 10s).
    """
    last_exc: Exception | None = None
    for attempt in range(_GEMINI_RETRY_ATTEMPTS):
        try:
            return fn()
        except Exception as exc:
            if attempt < _GEMINI_RETRY_ATTEMPTS - 1 and _is_transient_gemini_error(exc):
                msg = str(exc)
                if "429" in msg or "resource exhausted" in msg.lower():
                    api_delay = _parse_api_retry_delay(exc)
                    wait = (api_delay + 2.0) if api_delay else 62.0  # default: full minute + buffer
                    logger.warning(
                        "Gemini 429 quota (attempt %d/%d) — waiting %.0fs (API retryDelay=%.0fs)",
                        attempt + 1, _GEMINI_RETRY_ATTEMPTS, wait, api_delay or 0,
                    )
                else:
                    wait = _GEMINI_RETRY_DELAY * (2 ** attempt)
                    logger.warning(
                        "Gemini transient error (attempt %d/%d): %s — retrying in %ds",
                        attempt + 1, _GEMINI_RETRY_ATTEMPTS, exc, int(wait),
                    )
                time.sleep(wait)
                last_exc = exc
            else:
                raise
    raise last_exc  # type: ignore[misc]


def _create_gemini_cache(system: str) -> str | None:
    """Create a Gemini context cache for `system`. Returns cache name or None on failure."""
    from google.genai import types
    try:
        cache = _gemini_client.caches.create(  # type: ignore[union-attr]
            model=_GEMINI_MODEL,
            config=types.CreateCachedContentConfig(
                system_instruction=system,
                ttl="3600s",
            ),
        )
        logger.info("Created Gemini context cache: %s", cache.name)
        return cache.name
    except Exception as exc:
        logger.warning("Gemini cache creation failed (falling back to inline system): %s", exc)
        return None


def _maybe_refresh_gemini_cache(game_id: str) -> None:
    """Refresh the cache TTL after 50 min; rebuild chat without cache if refresh fails."""
    info = _game_cache_info.get(game_id)
    if not info or time.time() - info["created_at"] < _CACHE_REFRESH_AFTER:
        return
    from google.genai import types
    try:
        _gemini_client.caches.update(  # type: ignore[union-attr]
            name=info["name"],
            config=types.UpdateCachedContentConfig(ttl="3600s"),
        )
        info["created_at"] = time.time()
        logger.info("Refreshed Gemini cache TTL for game %s", game_id)
    except Exception as exc:
        logger.warning("Gemini cache TTL refresh failed (game=%s): %s — rebuilding chat without cache", game_id, exc)
        _rebuild_gemini_chat_without_cache(game_id, info["system"])
        del _game_cache_info[game_id]


def _rebuild_gemini_chat_without_cache(game_id: str, system: str) -> None:
    """Reconstruct the chat session using inline system_instruction after cache expiry."""
    from google.genai import types
    old_chat = _game_chat_sessions.get(game_id)
    history = old_chat.get_history() if old_chat else []  # type: ignore[attr-defined]
    chat = _gemini_client.chats.create(  # type: ignore[union-attr]
        model=_GEMINI_MODEL,
        config=types.GenerateContentConfig(
            system_instruction=system,
            thinking_config=types.ThinkingConfig(thinking_budget=_GEMINI_THINKING_BUDGET),
        ),
        history=history,
    )
    _game_chat_sessions[game_id] = chat
    logger.info("Rebuilt Gemini chat without cache for game %s (cache expired)", game_id)


def _init_gemini_session(game_id: str, system: str, user: str, think: bool) -> str:
    """Setup via generate_content (supports think=True), then create Chat with history."""
    _ensure_gemini_client()
    from google.genai import types

    thinking_budget = 1024 if think else 0

    # Attempt to create a context cache for the system prompt (saves per-turn token cost).
    # Include a hash of the system in the display name so stale caches are detectable.
    sys_hash = hashlib.sha256(system.encode()).hexdigest()[:12]
    cache_name = _create_gemini_cache(system)
    if cache_name:
        _game_cache_info[game_id] = {"name": cache_name, "created_at": time.time(), "system": system}
        logger.debug("Using cached content %s for game %s (hash=%s)", cache_name, game_id, sys_hash)
        init_config = types.GenerateContentConfig(
            cached_content=cache_name,
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
        )
        chat_config = types.GenerateContentConfig(
            cached_content=cache_name,
            thinking_config=types.ThinkingConfig(thinking_budget=_GEMINI_THINKING_BUDGET),
        )
    else:
        init_config = types.GenerateContentConfig(
            system_instruction=system,
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
        )
        chat_config = types.GenerateContentConfig(
            system_instruction=system,
            thinking_config=types.ThinkingConfig(thinking_budget=_GEMINI_THINKING_BUDGET),
        )

    def _call_init():
        return _gemini_client.models.generate_content(  # type: ignore[union-attr]
            model=_GEMINI_MODEL,
            contents=user,
            config=init_config,
        )

    response = _gemini_with_retry(_call_init)
    text: str = response.text

    # Create chat with the setup exchange as initial history.
    # Action calls use _GEMINI_THINKING_BUDGET (default 1024) — think on every turn.
    history = [
        types.Content(role="user",  parts=[types.Part.from_text(text=user)]),
        types.Content(role="model", parts=[types.Part.from_text(text=text)]),
    ]
    chat = _gemini_client.chats.create(  # type: ignore[union-attr]
        model=_GEMINI_MODEL,
        config=chat_config,
        history=history,
    )
    _game_chat_sessions[game_id] = chat
    return text


_PER_GEN_STRATEGY_PROMPT = (
    "=== End of Generation {prev_gen}, start of Generation {gen} ===\n"
    "{global_status}"
    "{income_note}"
    "Before your next action, restate your strategy in ~120-180 words:\n"
    "1. STANDING: your VP/TR vs each opponent. Ahead, level, or behind?\n"
    "2. ENGINE: primary path to victory — engine type, key cards in play, how you score each gen.\n"
    "3. MILESTONE TARGET: which of the 5 milestones will you claim (5 VP, max 3 in whole game, 8 MC)?\n"
    "   List the requirement and your current progress. IF YOU ALREADY MEET ONE, "
    "claim it next action — don't let opponents block you!\n"
    "4. AWARD TARGET: which award will you fund and place 1st in (5 VP for 1st, 2 VP for 2nd, "
    "max 3 funded at 8/14/20 MC)? Don't fund awards you can't win.\n"
    "5. NEXT-GEN PRIORITY: concrete plan for this generation's actions. "
    "Reminder: DO NOT use Convert Heat if temperature is already 8°C, and DO NOT place "
    "greenery tiles if O₂ is already 14% (both are wasted actions at max)."
)


def _per_generation_strategy_update(game_id: str, chat: object, generation: int, state: dict) -> None:
    """Ask the LLM to restate its strategy at each generation boundary.

    The response becomes natural chat history (no destructive rebuild) and is stored in
    _game_strategies for debugging + session recovery. Replaces the old _trim_gemini_session,
    which re-injected stale strategy via a fake user/model pair at chat[0] and caused the
    model to paraphrase that stale anchor every subsequent generation.
    """
    g = state.get("game", {})
    p = state.get("player", {})

    # Warn about any already-maxed global parameters so the AI doesn't waste actions.
    temp = g.get("temperature", -30)
    oxygen = g.get("oxygen", 0)
    oceans = g.get("oceanCount", 0)
    maxed_warnings: list[str] = []
    if temp >= 8:
        maxed_warnings.append("temperature is at maximum (8°C) — DO NOT use Convert Heat")
    if oxygen >= 14:
        maxed_warnings.append("O₂ is at maximum (14%) — DO NOT place greenery tiles")
    if oceans >= 9:
        maxed_warnings.append("all 9 oceans are placed")
    global_status = ("⚠ Global parameters: " + "; ".join(maxed_warnings) + "\n") if maxed_warnings else ""

    # Inform the AI of its production income for this generation so it can plan accurately.
    prod = p.get("production", {})
    mc_prod = prod.get("megacredits", 0)
    tr = p.get("terraformRating", 20)
    mc_income = tr + mc_prod
    income_parts = [f"MC:{mc_income} (TR:{tr} + prod:{mc_prod:+d})"]
    for res_label, key in [("steel", "steel"), ("titanium", "titanium"),
                           ("plants", "plants"), ("energy", "energy"), ("heat", "heat")]:
        v = prod.get(key, 0)
        if v:
            income_parts.append(f"{res_label}:{v}")
    income_note = "Your production income this generation: " + ", ".join(income_parts) + "\n"

    prompt = _PER_GEN_STRATEGY_PROMPT.format(
        prev_gen=generation - 1,
        gen=generation,
        global_status=global_status,
        income_note=income_note,
    )
    if _LLM_DEBUG:
        _log_prompt(f"=== PER-GEN STRATEGY UPDATE (game={game_id} gen={generation}) ===", prompt)
    try:
        response = _gemini_with_retry(lambda: chat.send_message(prompt))  # type: ignore[attr-defined]
        strategy = response.text.strip()
        _game_strategies[game_id] = strategy
        logger.info("Per-gen strategy update (game=%s gen=%d):\n%s", game_id, generation, strategy)
        if _LLM_DEBUG:
            _log_response(f"=== PER-GEN STRATEGY UPDATE response (game={game_id} gen={generation}) ===", strategy)
    except Exception as exc:
        logger.warning("Per-gen strategy update failed (game=%s gen=%d): %s", game_id, generation, exc)


def _maybe_per_generation_update(game_id: str, generation: int, state: dict) -> None:
    """When generation number increases, run a per-generation strategy update.

    Fires at the first action call of generation N+1. The Gemini chat retains full history
    (1M-token context window — no trim needed for a typical 15-gen game).
    """
    last_gen = _gemini_last_generation.get(game_id)
    if last_gen is not None and generation > last_gen:
        chat = _game_chat_sessions.get(game_id)
        if chat is not None:
            logger.info("Generation bump %d→%d for game %s — running strategy update",
                        last_gen, generation, game_id)
            _per_generation_strategy_update(game_id, chat, generation, state)
    _gemini_last_generation[game_id] = generation


def _trim_gemini_session(game_id: str) -> None:
    """Trim old Gemini chat history to prevent unbounded context growth.

    Keeps the strategy (from _game_strategies) plus the last 40 message pairs
    (20 user + 20 model) so the per-request token count stays bounded regardless
    of game length. The per-gen strategy updates ensure recent intent is preserved.
    """
    from google.genai import types
    chat = _game_chat_sessions.get(game_id)
    if chat is None:
        return
    history: list = chat.get_history()  # type: ignore[attr-defined]
    keep = 40  # user+model messages to retain
    if len(history) <= keep:
        return

    strategy = _game_strategies.get(game_id, "Play a balanced game — maximise TR and card synergies.")
    cache_info = _game_cache_info.get(game_id)
    system = cache_info["system"] if cache_info else _session_base_system.get(game_id, TM_RULES)

    # Inject strategy as the first kept message pair (synthetic, not sent to API)
    strategy_summary = f"[Session trimmed to last {keep//2} turns. Your current strategy: {strategy}]"
    trimmed_history = [
        types.Content(role="user",  parts=[types.Part(text=strategy_summary)]),
        types.Content(role="model", parts=[types.Part(text="Understood. Continuing.")]),
    ] + history[-keep:]

    if cache_info:
        new_chat = _gemini_client.chats.create(  # type: ignore[union-attr]
            model=_GEMINI_MODEL,
            config=types.GenerateContentConfig(
                cached_content=cache_info["name"],
                thinking_config=types.ThinkingConfig(thinking_budget=_GEMINI_THINKING_BUDGET),
            ),
            history=trimmed_history,
        )
    else:
        new_chat = _gemini_client.chats.create(  # type: ignore[union-attr]
            model=_GEMINI_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=system,
                thinking_config=types.ThinkingConfig(thinking_budget=_GEMINI_THINKING_BUDGET),
            ),
            history=trimmed_history,
        )
    _game_chat_sessions[game_id] = new_chat
    _gemini_turn_count[game_id] = 0
    logger.info("Trimmed Gemini session for game %s — kept last %d messages", game_id, keep)


def _continue_gemini_session(game_id: str, user: str, chat: object,
                              max_output_tokens: int | None = None) -> str:
    _maybe_refresh_gemini_cache(game_id)
    # Re-fetch chat: _maybe_refresh_gemini_cache may have rebuilt it into _game_chat_sessions.
    # Using the stale reference causes a 403 because the old chat object still references
    # the expired cache.
    chat = _game_chat_sessions.get(game_id) or chat

    # Trim history if session has grown too long (prevents unbounded input token growth).
    turn = _gemini_turn_count.get(game_id, 0)
    if turn >= _MAX_GEMINI_TURNS:
        _trim_gemini_session(game_id)
        chat = _game_chat_sessions.get(game_id) or chat
    _gemini_turn_count[game_id] = turn + 1

    if max_output_tokens:
        from google.genai import types
        cfg = types.GenerateContentConfig(max_output_tokens=max_output_tokens)
        response = _gemini_with_retry(lambda: chat.send_message(user, config=cfg))  # type: ignore[attr-defined]
    else:
        response = _gemini_with_retry(lambda: chat.send_message(user))  # type: ignore[attr-defined]
    text: str | None = response.text
    if not text:
        logger.warning("Gemini returned empty/None text for game %s", game_id)
        return ""
    return text


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
        board_spaces = state.get("boardSpaces") or []
        board_layout = format_board_layout(board_spaces)
        system = (
            TM_RULES + "\n\n" + game_ctx + "\n\n"
            + (board_layout + "\n\n" if board_layout else "")
            + "You are an expert Terraforming Mars strategist making the opening decisions. "
            "Think step by step about card synergies, engine building, the specific milestones "
            "and awards listed above, and any active game variants. "
            "Remember everything in this session — you will continue playing this game in "
            "subsequent messages without receiving these rules again. "
            "IMPORTANT: Memorize every card you play and its ongoing effects. "
            "Subsequent prompts will NOT list your played cards — that is your session memory. "
            "Your strategy should always include a TABLEAU section listing what you have in play. "
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
            "This strategy is your memory for the whole game — keep it in mind as you play.",
            "After this, each game turn you will receive game state + numbered options and",
            "must reply with only: CHOICE: <number>",
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

def _select_action(state: dict, waiting_for: dict, game_id: str, last_error: str | None = None) -> tuple[dict, dict]:
    options = flatten_options(waiting_for)
    if not options:
        return _default_response(waiting_for), {}

    # At each generation boundary, ask the Gemini session to restate its strategy.
    if _LLM_PROVIDER == "gemini":
        generation = state.get("game", {}).get("generation", 1)
        _maybe_per_generation_update(game_id, generation, state)

    user = _build_action_prompt(state, waiting_for, options, last_error=last_error, game_id=game_id)
    # Append a brevity instruction so the model doesn't produce multi-paragraph reasoning
    # that accumulates in the chat history and inflates future input-token counts.
    user += "\n\nBrief reasoning (1-2 sentences), then CHOICE: N on its own line. No text after CHOICE."
    logger.debug("LLM action (game=%s type=%s options=%d)",
                 game_id, waiting_for.get("type"), len(options))
    text = _call_llm_continue(game_id, user,
                              max_output_tokens=_GEMINI_ACTION_MAX_OUTPUT_TOKENS if _LLM_PROVIDER == "gemini" else None)
    logger.debug("Action response (game=%s): %s", game_id, text[:300])

    return _parse_action_response(text, options, waiting_for, game_id, player=state.get("player"))


# ---------------------------------------------------------------------------
# AI Trainer advice endpoint
# ---------------------------------------------------------------------------

_TRAINER_SYSTEM_SUFFIX = (
    "\n\nYou are a strategy coach helping a human player. Your job is to advise, not to play.\n"
    "Output rules (STRICT — failure to follow breaks the UI):\n"
    "  • Plain text ONLY. No markdown. No bullet points. No asterisks, no headers, no backticks.\n"
    "  • Structure EVERY coaching response in exactly two parts:\n"
    "      1. Strategy (2-3 sentences): describe the overall strategic direction this player "
    "should pursue given their engine, score, and the game situation.\n"
    "      2. Action (1-2 sentences): state clearly and concretely what to do RIGHT NOW with "
    "this specific decision, and why it fits the strategy.\n"
    "  • End EVERY response with a machine-readable block:\n"
    "      <recommendation>\n"
    "      CHOICE: <number>\n"
    "      [PAYMENT: MC=<n>[, STEEL=<n>] ...   # only if payment is required]\n"
    "      </recommendation>\n"
    "  • For setup decisions, follow the requested CORPORATION / BUY_CARDS / PRELUDE_CARDS / "
    "CEO_CARD / STRATEGY format inside the recommendation block instead.\n"
    "  • Production floor: Steel/Titanium/Plants/Energy/Heat production CANNOT go below 0. "
    "Only MC production can be negative (down to -5). Never recommend playing a card that "
    "would reduce any non-MC production below its current level if that level is already 0."
)


def _build_trainer_system(state: dict) -> str:
    """Build the trainer system prompt: TM rules + game config + board + coaching persona."""
    from .game_knowledge import format_config_context, format_board_layout
    g = state.get("game", {})
    game_ctx = format_config_context(g)
    board_layout = format_board_layout(state.get("boardSpaces") or [])
    system = (
        TM_RULES + "\n\n" + game_ctx + "\n\n"
        + (board_layout + "\n\n" if board_layout else "")
    )
    return system + _TRAINER_SYSTEM_SUFFIX


def _select_setup_advise(
    state: dict,
    waiting_for: dict,
    trainer_game_id: str,
    user_question: str | None,
) -> tuple[str, dict]:
    """Coach a human through initialCards / prelude. Reuses the setup prompt builder."""
    game_id_part = trainer_game_id.split(":", 2)[1] if trainer_game_id.startswith("trainer:") else trainer_game_id
    setup_prompt = _build_setup_prompt(state, waiting_for, game_id_part)

    prompt_lines = [
        setup_prompt,
        "",
        "You are coaching a human, not playing. Provide 1-3 short plain-text sentences explaining",
        "the best opening choice, then output the setup decision inside a <recommendation> block",
        "using the SAME CORPORATION / BUY_CARDS / PRELUDE_CARDS / CEO_CARD / STRATEGY format from",
        "the request above.",
    ]
    if user_question:
        prompt_lines.append(f'\nUser\'s question: "{user_question}"')
    user = "\n".join(prompt_lines)

    has_session = trainer_game_id in (
        _game_chat_sessions if _LLM_PROVIDER == "gemini" else _game_sessions
    )
    if not has_session:
        system = _build_trainer_system(state)
        text = _call_llm_init(trainer_game_id, system, user, think=True)
    else:
        text = _call_llm_continue(trainer_game_id, user)

    rec_match = re.search(r"<recommendation>(.*?)</recommendation>", text, re.DOTALL | re.IGNORECASE)
    if rec_match:
        rec_text = rec_match.group(1).strip()
        advice_text = (text[:rec_match.start()] + text[rec_match.end():]).strip()
    else:
        rec_text = text
        advice_text = text

    recommendation, _ = _parse_setup_response(rec_text, waiting_for, game_id_part)
    return advice_text or "(no advice text)", recommendation


def select_action_advise(
    state: dict,
    waiting_for: dict,
    game_id: str,
    player_id: str,
    user_question: str | None = None,
) -> tuple[str, dict]:
    """Return (advice_text, recommendation_input_response) for the AI Trainer feature.

    Each player gets an isolated session via the namespace 'trainer:<game_id>:<player_id>'
    so the LLM never confuses the two players' tableaux or strategies.

    Supports both setup phases (initialCards / prelude) and action turns.
    """
    wf_type = waiting_for.get("type", "")
    trainer_game_id = f"trainer:{game_id}:{player_id}"

    # ------------------------------------------------------------------
    # Setup-phase coaching (initialCards / prelude) — the action-options
    # flattener returns nothing for these, so they had no advice path before.
    # ------------------------------------------------------------------
    if wf_type in SETUP_TYPES:
        return _select_setup_advise(state, waiting_for, trainer_game_id, user_question)

    options = flatten_options(waiting_for)
    if not options:
        return ("No actions available.", _default_response(waiting_for))

    if _LLM_PROVIDER == "gemini":
        generation = state.get("game", {}).get("generation", 1)
        _maybe_per_generation_update(trainer_game_id, generation, state)

    prompt_lines: list[str] = []
    prompt_lines.append(_build_action_prompt(state, waiting_for, options))
    if user_question:
        prompt_lines.append(f"\nUser's question: \"{user_question}\"")
    prompt_lines += [
        "",
        "Coach the human in 1-3 short sentences (plain text, no markdown), then end with:",
        "<recommendation>",
        "CHOICE: <number>",
        "[PAYMENT: MC=<n>[, STEEL=<n>] ...  # only when payment is required]",
        "</recommendation>",
    ]
    user = "\n".join(prompt_lines)

    # Use or initialise a per-player trainer session
    if trainer_game_id not in (_game_chat_sessions if _LLM_PROVIDER == "gemini" else _game_sessions):
        system = _build_trainer_system(state)
        text = _call_llm_init(trainer_game_id, system, user, think=True)
    else:
        text = _call_llm_continue(trainer_game_id, user)

    # Extract <recommendation>...</recommendation> block
    rec_match = re.search(r"<recommendation>(.*?)</recommendation>", text, re.DOTALL | re.IGNORECASE)
    if rec_match:
        rec_text = rec_match.group(1).strip()
        advice_text = text[:rec_match.start()].strip()
        if not advice_text:
            advice_text = text[rec_match.end():].strip()
    else:
        rec_text = text
        advice_text = text

    # Parse the recommendation using the same logic as action responses
    m = re.search(r"CHOICE:\s*(\d+)", rec_text)
    chosen = int(m.group(1)) - 1 if m else 0
    chosen = max(0, min(chosen, len(options) - 1))
    recommendation = index_to_response(waiting_for, options[chosen]["index"])

    wf_type = waiting_for.get("type", "")
    if wf_type in ("projectCard", "payment"):
        payment = _parse_payment_line(rec_text)
        if payment:
            payment = _correct_payment(payment, waiting_for, state.get("player", {}))
            recommendation = {**recommendation, "payment": payment}

    return advice_text, recommendation


# Payment resource values (MC equivalent per unit)
_PAYMENT_VALUES = {
    "steel": 2, "titanium": 3, "heat": 1, "plants": 3,
    "microbes": 2, "floaters": 3, "seeds": 5, "graphene": 4,
    "lunaArchivesScience": 1, "kuiperAsteroids": 1, "auroraiData": 3, "spireScience": 2,
}

# Mapping from prompt keyword to Payment field name
_PAYMENT_KEYS = {
    "MC": "megacredits", "MEGACREDITS": "megacredits",
    "STEEL": "steel", "TITANIUM": "titanium", "HEAT": "heat", "PLANTS": "plants",
    "MICROBES": "microbes", "FLOATERS": "floaters", "SEEDS": "seeds",
    "GRAPHENE": "graphene", "LUNA": "lunaArchivesScience",
    "LUNAARCHIVESSCIENCE": "lunaArchivesScience", "KUIPER": "kuiperAsteroids",
    "KUIPERASTEROIDS": "kuiperAsteroids", "AURORA": "auroraiData",
    "AURORAIDATA": "auroraiData", "SPIRE": "spireScience", "SPIRESCIENCE": "spireScience",
}

def _empty_payment() -> dict:
    return {
        "megacredits": 0, "steel": 0, "titanium": 0, "heat": 0, "plants": 0,
        "microbes": 0, "floaters": 0, "lunaArchivesScience": 0, "spireScience": 0,
        "seeds": 0, "auroraiData": 0, "graphene": 0, "kuiperAsteroids": 0,
    }


def _parse_payment_line(text: str) -> dict | None:
    """Parse PAYMENT: MC=5, STEEL=2, TITANIUM=3 into a full Payment dict, or None if absent."""
    m = re.search(r"PAYMENT:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if not m:
        return None
    parts = re.findall(r"([A-Z_]+)\s*=\s*(\d+)", m.group(1), re.IGNORECASE)
    if not parts:
        return None
    payment = _empty_payment()
    for k, v in parts:
        field = _PAYMENT_KEYS.get(k.upper())
        if field:
            payment[field] = int(v)
    return payment


def _correct_payment(payment: dict, waiting_for: dict, player: dict) -> dict:
    """Clamp payment to available resources and ensure it covers the required cost.

    Prevents "You do not have that many resources to spend" rejections by validating
    the AI-specified payment before submitting it. If the payment exceeds available
    resources, each component is clamped to what the player actually has, and MC is
    topped up to cover any resulting shortfall.

    Returns the corrected payment dict. Logs a warning if a correction was needed.
    """
    wf_type = waiting_for.get("type", "")
    po = waiting_for.get("paymentOptions") or {}

    # Required cost
    if wf_type == "projectCard":
        cost = waiting_for.get("card", {}).get("calculatedCost", 0) if isinstance(waiting_for.get("card"), dict) else 0
        # Fallback: look inside the options list for calculatedCost
        if not cost:
            for opt in (waiting_for.get("options") or []):
                c = opt.get("calculatedCost") or opt.get("card", {}).get("calculatedCost", 0)
                if c:
                    cost = c
                    break
    elif wf_type == "payment":
        cost = waiting_for.get("amount", 0)
    else:
        return payment

    # Available player resources
    avail: dict[str, int] = {
        "megacredits": player.get("megacredits", 0),
        "steel":       player.get("steel", 0)       if wf_type == "projectCard" else 0,
        "titanium":    player.get("titanium", 0)     if wf_type == "projectCard" else 0,
        "heat":        player.get("heat", 0)         if po.get("heat") else 0,
        "plants":      player.get("plants", 0)       if po.get("plants") else 0,
    }
    for k in ("microbes", "floaters", "seeds", "graphene",
              "lunaArchivesScience", "kuiperAsteroids", "auroraiData", "spireScience"):
        avail[k] = waiting_for.get(k) or 0  # special resources tracked in waitingFor

    # Clamp each component to available
    corrected = dict(payment)
    changed = False
    for field, cap in avail.items():
        if corrected.get(field, 0) > cap:
            corrected[field] = cap
            changed = True

    # Zero out disallowed resources (steel/titanium only valid for projectCard)
    if wf_type != "projectCard":
        for field in ("steel", "titanium"):
            if corrected.get(field, 0):
                corrected[field] = 0
                changed = True

    # Compute total value after clamping
    total_value = corrected.get("megacredits", 0)
    for field, rate in _PAYMENT_VALUES.items():
        total_value += corrected.get(field, 0) * rate

    # If we still can't cover the cost (e.g. insufficient resources overall), top up MC
    shortfall = cost - total_value
    if shortfall > 0:
        extra_mc = min(shortfall, avail["megacredits"] - corrected.get("megacredits", 0))
        if extra_mc > 0:
            corrected["megacredits"] = corrected.get("megacredits", 0) + extra_mc
            changed = True

    if changed:
        logger.warning(
            "Payment corrected: %s → %s (cost=%d avail=%s)",
            {k: v for k, v in payment.items() if v},
            {k: v for k, v in corrected.items() if v},
            cost,
            {k: v for k, v in avail.items() if v},
        )

    return corrected


def _format_payment_section(waiting_for: dict, player: dict) -> str:
    """Build the payment options block shown in the action prompt."""
    wf_type = waiting_for.get("type", "")
    po = waiting_for.get("paymentOptions") or {}

    mc    = player.get("megacredits", 0)
    steel = player.get("steel", 0)
    ti    = player.get("titanium", 0)
    heat  = player.get("heat", 0)

    lines = ["", "Payment resources available:"]
    lines.append(f"  MC: {mc} (always, 1:1)")

    if wf_type == "projectCard":
        if steel > 0:
            lines.append(f"  STEEL: {steel} cubes @ 2 MC each  — only for cards with [building] tag")
        if ti > 0:
            if po.get("lunaTradeFederationTitanium"):
                lines.append(f"  TITANIUM: {ti} cubes @ 3 MC each  — any card (Luna Trade Federation)")
            else:
                lines.append(f"  TITANIUM: {ti} cubes @ 3 MC each  — only for cards with [space] tag")

    if po.get("heat") and heat > 0:
        lines.append(f"  HEAT: {heat} cubes @ 1 MC each  — (corp special ability)")

    # Show special resources that are available (non-zero in waitingFor model)
    for wf_key, label, rate in [
        ("microbes",            "MICROBES",      2),
        ("floaters",            "FLOATERS",      3),
        ("seeds",               "SEEDS",         5),
        ("graphene",            "GRAPHENE",      4),
        ("lunaArchivesScience", "LUNA_SCIENCE",  1),
        ("kuiperAsteroids",     "KUIPER",        1),
        ("auroraiData",         "AURORA_DATA",   3),
        ("spireScience",        "SPIRE_SCIENCE", 2),
    ]:
        amt = waiting_for.get(wf_key) or 0
        if amt > 0:
            lines.append(f"  {label}: {amt} @ {rate} MC each")

    if wf_type == "projectCard":
        # Show the exact cost so Gemini doesn't have to infer it from the card list
        cost = 0
        card = waiting_for.get("card")
        if isinstance(card, dict):
            cost = card.get("calculatedCost", 0)
        cost_str = f" (must cover {cost} MC)" if cost else ""
        lines += [
            "",
            f"After CHOICE, specify how you pay{cost_str}:",
            "  PAYMENT: MC=<n>[, STEEL=<n>][, TITANIUM=<n>][, HEAT=<n>][, ...]",
            "  RULE: MC ≤ your current MC. STEEL cubes × 2 + TITANIUM cubes × 3 count toward cost.",
            "  You can overpay with non-MC resources (surplus discarded). You cannot overpay in MC.",
        ]
    else:  # payment type
        amount = waiting_for.get("amount", 0)
        lines[1] = f"Payment resources available (need {amount} MC total):"
        lines += [
            "",
            f"Specify payment: PAYMENT: MC=<n>[, HEAT=<n>][, ...]",
            f"  MC ≤ {mc}. Total value must equal (or exceed) {amount}.",
        ]
    return "\n".join(lines)


def _get_card_desc_for_option(opt: dict) -> str:
    """Return a short description if the option title references a played card action."""
    title = opt.get("title", "")
    if not isinstance(title, str):
        return ""
    m = re.match(r"Use (.+?)(?:'s)? action\b", title, re.IGNORECASE)
    if m:
        card_name = m.group(1).strip()
        entry = CARD_DB.get(card_name)
        if entry and entry.get("description"):
            desc = entry["description"]
            return desc[:100] if len(desc) > 100 else desc
    return ""


_BONUS_ABBREV = {
    "steel": "St", "titanium": "Ti", "plant": "Pl", "card": "Cd",
    "heat": "He", "MC": "MC", "ocean": "Oc", "animal": "An",
    "microbe": "Mi", "energy": "En", "data": "Da", "science": "Sc",
    "energy production": "EP", "temperature": "Tp",
}


def _build_action_prompt(state: dict, waiting_for: dict, options: list[dict], last_error: str | None = None,
                          game_id: str = "") -> str:
    g      = state.get("game", {})
    p      = state.get("player", {})
    prod   = {k: v for k, v in p.get("production", {}).items() if v}
    tags   = {k: v for k, v in p.get("tags", {}).items() if v}
    my_id  = p.get("id", "")
    ms_raw = state.get("milestones", [])
    aw_raw = state.get("awards", [])
    wf_type = waiting_for.get("type", "")

    # Build a spaceId → space_info lookup for annotating tile placement options
    _space_index: dict[str, dict] = {}
    for s in (state.get("boardSpaces") or []):
        sid = s.get("id")
        if sid:
            _space_index[sid] = s
    ms = [f"{m.get('name','?')} ({'you' if m.get('playerId')==my_id else 'opponent'})"
          for m in ms_raw]
    aw = [f"{a.get('name','?')} ({'you' if a.get('playerId')==my_id else 'opponent'})"
          for a in aw_raw]

    lines: list[str] = []
    if last_error:
        lines += [
            f"⚠ Your previous response was rejected: \"{last_error}\"",
            "Please choose a different option or correct your payment/selection.",
            "",
        ]

    temp = g.get("temperature", -30)
    oxygen = g.get("oxygen", 0)
    oceans = g.get("oceanCount", 0)

    my_vp = p.get("victoryPoints")
    vp_str = f"  VP:{my_vp}" if my_vp is not None else ""
    lines += [
        f"Gen {g.get('generation',1)} | Temp {temp}°C | O₂ {oxygen}% | Oceans {oceans}/9",
        f"TR:{p.get('terraformRating',20)}{vp_str}  MC:{p.get('megacredits',0)}  "
        f"St:{p.get('steel',0)}  Ti:{p.get('titanium',0)}  "
        f"Pl:{p.get('plants',0)}  En:{p.get('energy',0)}  He:{p.get('heat',0)}",
    ]

    # Warn when global params are maxed so the AI doesn't waste actions.
    if temp >= 8:
        lines.append("⚠ Temperature is at maximum (8°C). DO NOT use Convert Heat — it is a wasted action.")
    if oxygen >= 14:
        lines.append("⚠ O₂ is at maximum (14%). DO NOT place greenery tiles — they give no benefit.")
    if oceans >= 9:
        lines.append("⚠ All 9 oceans are placed. No more ocean tiles can be placed.")
    if prod:
        lines.append(f"Production: {prod}")
    if tags:
        lines.append(f"Tags: {tags}")
    # Played cards are in session memory — not repeated each turn.
    # Production floor rule: Steel/Titanium/Plants/Energy/Heat production cannot go below 0.
    # Only MC production can be negative (minimum -5). Don't play cards that would reduce
    # non-MC production below 0 — the game will reject those actions.
    lines.append("Note: Only MC production can go negative (min -5). Steel/Ti/Plants/Energy/Heat production CANNOT go below 0.")

    # Cards in hand with descriptions — skip for tile/payment/numeric decisions
    # where the hand plays no role (saves ~100–300 tokens per such turn).
    # Within a generation, show full descriptions only on the FIRST action turn;
    # subsequent turns in the same gen show names only to avoid re-sending ~500 tokens
    # of card descriptions that are already in the model's session context.
    hand_cards = p.get("cardsInHand") or []
    generation = g.get("generation", 1)
    _hand_irrelevant = wf_type in ("space", "payment", "amount")
    if hand_cards and not _hand_irrelevant:
        last_shown_gen = _hand_shown_generation.get(game_id, -1) if game_id else -1
        if last_shown_gen == generation:
            # Already shown with descriptions this gen — name-only to save tokens
            name_list = ", ".join(hand_cards[:30])
            lines += ["", f"Your hand ({len(hand_cards)} cards): {name_list}",
                      "  (Full descriptions shown earlier this generation — rely on your session memory.)"]
        else:
            ctx = format_card_context(hand_cards, header=f"Your hand ({len(hand_cards)} cards):", max_cards=30)
            lines += ["", ctx]
            if game_id:
                _hand_shown_generation[game_id] = generation

    # Opponents
    opponents = state.get("opponents") or []
    for i, opp in enumerate(opponents, 1):
        opp_prod = {k: v for k, v in opp.get("production", {}).items() if v}
        opp_tags = {k: v for k, v in opp.get("tags", {}).items() if v}
        opp_name = opp.get("name", f"Opponent{'' if len(opponents) == 1 else i}")
        opp_vp = opp.get("victoryPoints")
        opp_vp_str = f" VP:{opp_vp}" if opp_vp is not None else ""
        lines.append(
            f"{opp_name}: TR:{opp.get('terraformRating',20)}{opp_vp_str}  MC:{opp.get('megacredits',0)}  "
            f"prod:{opp_prod}  tags:{opp_tags}"
        )
    if ms:
        lines.append(f"Milestones claimed: {ms}")
    if aw:
        lines.append(f"Awards funded: {aw}")

    # Recent game events (current generation log — OPPONENT moves and system messages only;
    # your own moves are already in your session memory above).
    recent_log = g.get("recentLog") or []
    if recent_log:
        lines += ["", f"Recent opponent actions / events ({len(recent_log)}):"]
        for entry in recent_log:
            lines.append(f"  {entry}")

    # Decision title
    title_raw = waiting_for.get("title")
    if isinstance(title_raw, str):
        title = title_raw.strip()
    elif isinstance(title_raw, dict):
        title = title_raw.get("message", "")
    else:
        title = ""
    if title:
        lines += ["", f"Decision: {title}"]

    # Tile placement tips: greenery adjacency to cities; city separation rule.
    if wf_type == "space":
        title_lower = title.lower()
        if "greenery" in title_lower:
            own_cities = sum(
                1 for s in (state.get("boardSpaces") or [])
                if s.get("tileType") == "city" and s.get("playerColor") == p.get("color")
            )
            if own_cities:
                lines.append(
                    f"Placement tip: place this greenery ADJACENT to one of your {own_cities} "
                    f"city tile(s) — each adjacent greenery scores +1 VP for the city at game end."
                )
            else:
                lines.append(
                    "Placement tip: you have no cities yet. Consider placing this greenery where "
                    "a future city can sit next to it, or near the center for flexibility."
                )
        elif "city" in title_lower:
            lines.append(
                "Placement tip: place this city where greenery tiles can later surround it — "
                "each adjacent greenery scores +1 VP for this city at game end. "
                "Cities cannot be adjacent to other cities."
            )

    # Card descriptions for explicit card-selection decisions (research, discard, etc.)
    if wf_type == "card":
        card_names_in_decision = _extract_card_names(waiting_for)
        if card_names_in_decision:
            if _is_card_decision_about_hand(card_names_in_decision, hand_cards):
                lines += ["", "Cards to choose from: (see hand above)"]
            else:
                ctx = format_card_context(card_names_in_decision, header="Cards to choose from:", max_cards=20)
                if ctx:
                    lines += ["", ctx]

    # Payment section
    if wf_type in ("projectCard", "payment"):
        lines.append(_format_payment_section(waiting_for, p))

    # Options list — annotate card-action and space options with inline info
    lines.append("\nChoose from:")
    for i, opt in enumerate(options, 1):
        title = opt["title"]
        card_desc = _get_card_desc_for_option(opt)
        # Annotate space options with placement bonuses and type
        if wf_type == "space" and _space_index:
            space_info = _space_index.get(title, {})
            bonuses: list[str] = space_info.get("b") or []
            stype: str = space_info.get("t", "land")
            x, y = space_info.get("x", "?"), space_info.get("y", "?")
            bonus_str = "+".join(_BONUS_ABBREV.get(b, b) for b in bonuses) if bonuses else "no bonus"
            volcanic_tag = " [volcanic]" if space_info.get("v") else ""
            lines.append(f"  {i}. hex-{title} ({x},{y}) [{stype}]{volcanic_tag}  placement bonus: {bonus_str}")
        elif card_desc:
            lines.append(f"  {i}. {title}  — {card_desc}")
        else:
            lines.append(f"  {i}. {title}")

    if wf_type in ("projectCard", "payment"):
        lines += ["", "CHOICE: <number>", "PAYMENT: MC=<n>[, STEEL=<n>][, TITANIUM=<n>][, HEAT=<n>]..."]
    else:
        lines += ["", "CHOICE: <number>"]
    return "\n".join(lines)


def _parse_action_response(
    text: str, options: list[dict], waiting_for: dict, game_id: str,
    player: dict | None = None,
) -> tuple[dict, dict]:
    m = re.search(r"CHOICE:\s*(\d+)", text)
    chosen = int(m.group(1)) - 1 if m else 0
    chosen = max(0, min(chosen, len(options) - 1))
    option = options[chosen]
    logger.info("Action choice game=%s: %d. %s", game_id, chosen + 1, option["title"])
    response = index_to_response(waiting_for, option["index"])

    # Override payment if AI provided PAYMENT: line; validate and clamp to available resources.
    wf_type = waiting_for.get("type", "")
    if wf_type in ("projectCard", "payment"):
        payment = _parse_payment_line(text)
        if payment:
            payment = _correct_payment(payment, waiting_for, player or {})
            response = {**response, "payment": payment}

    return response, {
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


def _is_card_decision_about_hand(card_names: list[str], cards_in_hand: list) -> bool:
    """Return True when all cards in the decision are already described in the hand block."""
    if not card_names or not cards_in_hand:
        return False
    hand_set = {c if isinstance(c, str) else c.get("name", "") for c in cards_in_hand}
    return all(name in hand_set for name in card_names)


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
