"""
LLM-based action selection — supports OpenRouter (cloud, any model) and Ollama (local).

Architecture:
- One LLMPlayer instance per AI player, stored in _player_registry keyed by player_id.
- Session key is always the unique player_id from the TM server (never game_id alone).
- Provider is derived from the model name: "model/name" → OpenRouter, "bare:tag" → Ollama.
- Model capabilities (caching, thinking) are probed once per model and cached in memory.
  Known models are pre-seeded to skip probing. Unknown models are probed on first use.
  If a feature fails mid-session the error is caught and the model is marked as incapable
  so subsequent calls skip it without retrying.
- Token usage and cost are tracked per-player; log_game_token_summary aggregates by game.

Providers:
  OpenRouter — single openai.OpenAI client pointed at https://openrouter.ai/api/v1.
               Uses standard messages[] format. Supports caching (Anthropic models) and
               thinking (Anthropic + some others) via extra_headers / extra_body.
  Ollama     — direct HTTP to localhost:11434 via requests. Uses same messages[] format.
               Thinking via "think": True in payload (qwen3 and capable models).

Env vars:
  USE_LLM                     Enable LLM player (default: false)
  OPENROUTER_API_KEY          Required for OpenRouter models
  OPENROUTER_MODEL            Default model (default: anthropic/claude-opus-4-7)
  OPENROUTER_THINKING_BUDGET  Thinking tokens for capable models (default: 512;
                              setup/prelude always use 1024)
  OPENROUTER_MAX_OUTPUT_TOKENS Cap on action response length (default: 350)
  OPENROUTER_MAX_TURNS        Trim session after this many turns (default: 80)
  OLLAMA_URL                  Ollama base URL (default: http://localhost:11434)
  OLLAMA_MODEL                Default Ollama model (default: qwen3:4b)
  OLLAMA_TIMEOUT              Ollama request timeout seconds (default: 600)
  LLM_DEBUG                   Log full prompts and responses (default: false)
"""

from __future__ import annotations
import logging
import os
import re
import time

import requests

from .encoding import flatten_options, index_to_response, _default_response
from .game_knowledge import CARD_DB, format_card_context, format_config_context, format_board_layout

logger = logging.getLogger(__name__)

_LLM_DEBUG = os.getenv("LLM_DEBUG", "false").lower() == "true"

# OpenRouter settings
_OPENROUTER_API_KEY         = os.getenv("OPENROUTER_API_KEY", "")
_OPENROUTER_MODEL           = os.getenv("OPENROUTER_MODEL", "anthropic/claude-opus-4-7")
_OPENROUTER_THINKING_BUDGET = int(os.getenv("OPENROUTER_THINKING_BUDGET", "512"))
_OPENROUTER_MAX_OUTPUT_TOKENS = int(os.getenv("OPENROUTER_MAX_OUTPUT_TOKENS", "350"))
_OPENROUTER_MAX_TURNS       = int(os.getenv("OPENROUTER_MAX_TURNS", "80"))

# Ollama settings
_OLLAMA_URL     = os.getenv("OLLAMA_URL",   "http://localhost:11434")
_OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL", "qwen3:4b")
_OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "600"))

SETUP_TYPES = {"initialCards", "prelude"}

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

# Registry: player_id → LLMPlayer instance
_player_registry: dict[str, LLMPlayer] = {}  # type: ignore[name-defined]

# Game → player mapping for summary logging
_game_players: dict[str, list[str]] = {}

# Capability cache: model → {caching: bool, thinking: bool}
_model_capabilities: dict[str, dict] = {}

# Pricing cache: model → (input_usd_per_token, output_usd_per_token)
_pricing_cache: dict[str, tuple[float, float]] = {}
_pricing_fetched = False
_pricing_lock = __import__("threading").Lock()

# Lazy-init OpenRouter client
_openrouter_client = None

# Games whose summary has been logged (avoid duplicate logging)
_game_summary_logged: set[str] = set()

# Maximum session length before trimming (message count including system)
_MAX_SESSION_MESSAGES = 62  # ~30 user+assistant pairs + system


# ---------------------------------------------------------------------------
# Known model capabilities — skip probing for these
# ---------------------------------------------------------------------------

_KNOWN_CAPABILITIES: dict[str, dict] = {
    # Anthropic via OpenRouter — caching and thinking both available
    # OpenRouter uses dot notation (4.7) as well as dash (4-7) — both are seeded
    "anthropic/claude-opus-4.7":        {"caching": True,  "thinking": True},
    "anthropic/claude-opus-4.7-fast":   {"caching": True,  "thinking": True},
    "anthropic/claude-opus-4.6-fast":   {"caching": True,  "thinking": True},
    "anthropic/claude-opus-4-7":        {"caching": True,  "thinking": True},
    "anthropic/claude-opus-4-5":        {"caching": True,  "thinking": True},
    "anthropic/claude-sonnet-4-6":      {"caching": True,  "thinking": True},
    "anthropic/claude-sonnet-4-5":      {"caching": True,  "thinking": True},
    "anthropic/claude-haiku-4-5":       {"caching": True,  "thinking": False},
    "anthropic/claude-3-7-sonnet":      {"caching": True,  "thinking": True},
    "anthropic/claude-3-5-sonnet":      {"caching": True,  "thinking": False},
    "anthropic/claude-3-5-haiku":       {"caching": True,  "thinking": False},
    # OpenAI via OpenRouter — automatic caching (no config needed), no explicit thinking
    "openai/gpt-5.5-pro":              {"caching": False, "thinking": False},
    "openai/gpt-5.5":                  {"caching": False, "thinking": False},
    "openai/gpt-5.4-pro":              {"caching": False, "thinking": False},
    "openai/gpt-5.4":                  {"caching": False, "thinking": False},
    "openai/gpt-4o":                    {"caching": False, "thinking": False},
    "openai/gpt-4o-mini":               {"caching": False, "thinking": False},
    "openai/o3":                        {"caching": False, "thinking": False},
    "openai/o4-mini":                   {"caching": False, "thinking": False},
    "openai/o3-mini":                   {"caching": False, "thinking": False},
    # xAI Grok via OpenRouter — 4.x series supports explicit caching and reasoning
    "x-ai/grok-4.3":                    {"caching": True,  "thinking": True},
    "x-ai/grok-4.20":                   {"caching": True,  "thinking": True},
    "x-ai/grok-4.20-multi-agent":       {"caching": True,  "thinking": True},
    "x-ai/grok-3":                      {"caching": False, "thinking": False},
    "x-ai/grok-3-mini":                 {"caching": False, "thinking": True},
    "x-ai/grok-2-1212":                 {"caching": False, "thinking": False},
    # Google via OpenRouter
    "google/gemini-2.5-pro":            {"caching": False, "thinking": True},
    "google/gemini-2.5-pro-preview":    {"caching": False, "thinking": True},
    "google/gemini-2.5-flash":          {"caching": False, "thinking": True},
    "google/gemini-2.5-flash-lite":     {"caching": False, "thinking": False},
    "google/gemini-3.1-flash-lite":     {"caching": False, "thinking": True},
    "google/gemini-3.1-flash-lite-preview": {"caching": False, "thinking": True},
    "google/gemini-3.1-pro-preview":    {"caching": False, "thinking": True},
    "google/gemini-2.0-flash-001":      {"caching": False, "thinking": False},
    "google/gemini-2.0-flash-lite-001": {"caching": False, "thinking": False},
    # DeepSeek via OpenRouter — auto-caching (like OpenAI), built-in reasoning
    "deepseek/deepseek-v4-pro":         {"caching": False, "thinking": True},
    "deepseek/deepseek-v4-flash":       {"caching": False, "thinking": True},
    "deepseek/deepseek-r1":             {"caching": False, "thinking": True},
    "deepseek/deepseek-chat":           {"caching": False, "thinking": False},
    "deepseek/deepseek-chat-v3-0324":   {"caching": False, "thinking": False},
    "deepseek/deepseek-v3.2":           {"caching": False, "thinking": False},
    # Ollama local
    "qwen3:4b":                         {"caching": False, "thinking": True},
    "qwen3:8b":                         {"caching": False, "thinking": True},
    "qwen3:14b":                        {"caching": False, "thinking": True},
    "qwen3:32b":                        {"caching": False, "thinking": True},
    "llama3.2:3b":                      {"caching": False, "thinking": False},
    "llama3.3:70b":                     {"caching": False, "thinking": False},
}


# ---------------------------------------------------------------------------
# Provider resolution
# ---------------------------------------------------------------------------

def _provider_for(model: str) -> str:
    """Return 'openrouter' if model contains '/', else 'ollama'."""
    return "openrouter" if "/" in model else "ollama"


def _ensure_openrouter_client() -> None:
    global _openrouter_client
    if _openrouter_client is None:
        import openai
        _openrouter_client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=_OPENROUTER_API_KEY,
        )
        _prefetch_pricing()


# ---------------------------------------------------------------------------
# Capability detection (probe on first use, cache result)
# ---------------------------------------------------------------------------

def _get_model_capabilities(model: str) -> dict:
    """Return {caching: bool, thinking: bool} for model. Probe if unknown."""
    if model in _model_capabilities:
        return _model_capabilities[model]
    # Check pre-seeded table first (exact match, then prefix match)
    if model in _KNOWN_CAPABILITIES:
        _model_capabilities[model] = dict(_KNOWN_CAPABILITIES[model])
        return _model_capabilities[model]
    for key, caps in _KNOWN_CAPABILITIES.items():
        if model.startswith(key) or key.startswith(model.split(":")[0]):
            _model_capabilities[model] = dict(caps)
            logger.info("Capability inferred for %s from known entry %s: %s", model, key, caps)
            return _model_capabilities[model]
    # Unknown model — probe both features (one tiny call each)
    if _provider_for(model) == "openrouter":
        _ensure_openrouter_client()
        caching  = _probe_caching(model)
        thinking = _probe_thinking(model)
    else:
        # Ollama: check if model advertises thinking by testing with think=True
        caching  = False
        thinking = _probe_ollama_thinking(model)
    caps = {"caching": caching, "thinking": thinking}
    _model_capabilities[model] = caps
    logger.info("Probed capabilities for %s: %s", model, caps)
    return caps


def _probe_caching(model: str) -> bool:
    try:
        _openrouter_client.chat.completions.create(  # type: ignore[union-attr]
            model=model,
            messages=[{
                "role": "system",
                "content": [{"type": "text", "text": "test",
                             "cache_control": {"type": "ephemeral"}}],
            }, {"role": "user", "content": "ping"}],
            max_tokens=1,
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
        )
        return True
    except Exception as e:
        logger.debug("Caching probe for %s failed: %s", model, e)
        return False


def _probe_thinking(model: str) -> bool:
    try:
        if model.startswith("anthropic/"):
            extra_body = {"thinking": {"type": "enabled", "budget_tokens": 256}}
            extra_headers = {"anthropic-beta": "interleaved-thinking-2025-05-14"}
        else:
            extra_body = {"reasoning": {"max_tokens": 256}}
            extra_headers = {}
        kwargs: dict = dict(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=10,
            extra_body=extra_body,
        )
        if extra_headers:
            kwargs["extra_headers"] = extra_headers
        _openrouter_client.chat.completions.create(**kwargs)  # type: ignore[union-attr]
        return True
    except Exception as e:
        logger.debug("Thinking probe for %s failed: %s", model, e)
        return False


def _probe_ollama_thinking(model: str) -> bool:
    try:
        r = requests.post(
            f"{_OLLAMA_URL}/api/chat",
            json={"model": model, "stream": False, "messages": [{"role": "user", "content": "ping"}], "think": True},
            timeout=30,
        )
        return r.ok
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Pricing (OpenRouter models API)
# ---------------------------------------------------------------------------

def _prefetch_pricing() -> None:
    """Fetch all OpenRouter model pricing once and cache it. Thread-safe."""
    global _pricing_fetched
    if _pricing_fetched:
        return
    with _pricing_lock:
        if _pricing_fetched:
            return
        try:
            r = requests.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {_OPENROUTER_API_KEY}"},
                timeout=10,
            )
            r.raise_for_status()
            for entry in r.json().get("data", []):
                mid     = entry.get("id", "")
                pricing = entry.get("pricing") or {}
                raw_in  = pricing.get("prompt")      or pricing.get("input")  or 0
                raw_out = pricing.get("completion")  or pricing.get("output") or 0
                try:
                    in_p  = float(raw_in)
                    out_p = float(raw_out)
                except (TypeError, ValueError):
                    in_p, out_p = 0.0, 0.0
                _pricing_cache[mid] = (in_p, out_p)
            _pricing_fetched = True
            logger.info("Fetched OpenRouter model pricing (%d models)", len(_pricing_cache))
        except Exception as e:
            logger.warning("OpenRouter pricing fetch failed: %s", e)


def _get_openrouter_pricing(model: str) -> tuple[float, float]:
    """Return (input_usd_per_token, output_usd_per_token) from the pre-fetched cache."""
    if not _pricing_fetched:
        _prefetch_pricing()
    return _pricing_cache.get(model, (0.0, 0.0))


# ---------------------------------------------------------------------------
# Retry logic (generic)
# ---------------------------------------------------------------------------

_RETRY_ATTEMPTS = 3
_RETRY_BASE_DELAY = 5.0  # seconds, doubles on each attempt


def _is_transient_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("503", "unavailable", "429", "rate limit",
                                   "overloaded", "resource exhausted", "timeout"))


def _parse_retry_delay(exc: Exception) -> float | None:
    m = re.search(r"retry.?after[:\s]+(\d+\.?\d*)", str(exc), re.IGNORECASE)
    if not m:
        m = re.search(r"retryDelay.*?(\d+\.?\d*)s", str(exc))
    return float(m.group(1)) if m else None


def _with_retry(fn, attempts: int = _RETRY_ATTEMPTS, base_delay: float = _RETRY_BASE_DELAY):
    """Call fn() with exponential back-off on transient errors (429/503)."""
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if attempt < attempts - 1 and _is_transient_error(exc):
                msg = str(exc)
                if "429" in msg or "resource exhausted" in msg.lower():
                    api_delay = _parse_retry_delay(exc)
                    wait = (api_delay + 2.0) if api_delay else 62.0
                    logger.warning("Rate-limited (attempt %d/%d) — waiting %.0fs",
                                   attempt + 1, attempts, wait)
                else:
                    wait = base_delay * (2 ** attempt)
                    logger.warning("Transient error (attempt %d/%d): %s — retrying in %.0fs",
                                   attempt + 1, attempts, exc, wait)
                time.sleep(wait)
                last_exc = exc
            else:
                raise
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# LLMPlayer class
# ---------------------------------------------------------------------------

class LLMPlayer:
    """Stateful LLM session for one AI player in a TM game."""

    def __init__(self, player_id: str, game_id: str, model: str) -> None:
        self.player_id = player_id
        self.game_id   = game_id
        self.model     = model
        self.provider  = _provider_for(model)

        # Session state
        self.session:               list[dict] = []
        self.base_system:           str = ""
        self.strategy:              str = ""
        self.last_generation:       int = -1
        self.hand_shown_generation: int = -1

        # Token tracking (per-player)
        self.token_usage: dict = {
            "calls": 0, "input": 0, "output": 0,
            "cache_read": 0, "cache_write": 0, "thinking": 0,
        }
        self.summary_logged: bool = False

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def init_session(self, system: str, user: str, think: bool = True) -> str:
        """Start a new session: send system + first user message, store response."""
        self.base_system = system
        if _LLM_DEBUG:
            _log_prompt(f"=== INIT system (player={self.player_id} model={self.model}) ===", system)
            _log_prompt(f"=== INIT user (player={self.player_id}) ===", user)

        if self.provider == "openrouter":
            text = self._call_openrouter(system, user, think=think)
        else:
            text = self._call_ollama(system, user, think=think)

        if _LLM_DEBUG:
            _log_response(f"=== INIT response (player={self.player_id}) ===", text)
        return text

    def continue_session(self, user: str, max_output_tokens: int | None = None) -> str:
        """Continue the session with a new user message. Recovers if session is lost."""
        if not self.session:
            logger.warning("No session for player %s — recovering", self.player_id)
            return self.recover_session(user)

        if _LLM_DEBUG:
            _log_prompt(f"=== CONTINUE user (player={self.player_id}) ===", user)

        if self.provider == "openrouter":
            text = self._continue_openrouter(user, max_output_tokens=max_output_tokens)
        else:
            text = self._continue_ollama(user)

        if _LLM_DEBUG:
            _log_response(f"=== CONTINUE response (player={self.player_id}) ===", text)
        return text

    def recover_session(self, user: str) -> str:
        """Re-initialise from strategy when session is lost (server restart / exhaustion)."""
        strategy = self.strategy or "Play a balanced game — maximise TR and card synergies."
        system = (
            TM_RULES + "\n\n"
            "You are a Terraforming Mars player. Your current strategy:\n"
            f"{strategy}\n\n"
            "Pick the single best action. Respond ONLY:\n"
            "CHOICE: <number>"
        )
        logger.info("Session recovery for player %s (strategy: %.80s…)", self.player_id, strategy)
        return self.init_session(system, user, think=False)

    def trim_session(self) -> None:
        """Trim old session history to prevent unbounded context growth."""
        if len(self.session) <= _MAX_SESSION_MESSAGES:
            return
        strategy = self.strategy or ""
        system_msg = self.session[0] if self.session else {"role": "system", "content": self.base_system}
        base = system_msg.get("content", self.base_system)
        if isinstance(base, list):
            # Extract text from cache_control content blocks
            base = " ".join(b.get("text", "") for b in base if isinstance(b, dict))
        system_with_reminder = (
            base + f"\n\n[CONTEXT TRIM — current strategy:\n{strategy}]" if strategy else base
        )
        tail = self.session[-40:]  # keep last 40 messages (user+assistant pairs)
        self.session.clear()
        if self.provider == "openrouter":
            caps = _get_model_capabilities(self.model)
            sys_content = (
                [{"type": "text", "text": system_with_reminder,
                  "cache_control": {"type": "ephemeral"}}]
                if caps.get("caching") else system_with_reminder
            )
            self.session.append({"role": "system", "content": sys_content})
        else:
            self.session.append({"role": "system", "content": system_with_reminder})
        self.session.extend(tail)
        logger.info("Session trimmed for player %s — kept last 40 messages + strategy", self.player_id)

    # ------------------------------------------------------------------
    # Internal call helpers
    # ------------------------------------------------------------------

    def _build_system_message(self, system: str) -> dict:
        """Wrap system text in cache_control block if the model supports caching."""
        caps = _get_model_capabilities(self.model)
        if caps.get("caching"):
            return {
                "role": "system",
                "content": [{"type": "text", "text": system,
                             "cache_control": {"type": "ephemeral"}}],
            }
        return {"role": "system", "content": system}

    def _call_openrouter(self, system: str, user: str, think: bool) -> str:
        """First call: build messages[], store in self.session, return response text."""
        _ensure_openrouter_client()
        sys_msg  = self._build_system_message(system)
        messages = [sys_msg, {"role": "user", "content": user}]
        text = self._do_openrouter_call(messages, think=think, max_output_tokens=None)
        self.session = messages + [{"role": "assistant", "content": text}]
        return text

    def _continue_openrouter(self, user: str, max_output_tokens: int | None) -> str:
        """Continue call: append user msg, call, append response."""
        self.trim_session()
        self.session.append({"role": "user", "content": user})
        text = self._do_openrouter_call(self.session, think=True, max_output_tokens=max_output_tokens)
        self.session.append({"role": "assistant", "content": text})
        return text

    def _do_openrouter_call(
        self,
        messages: list[dict],
        think: bool,
        max_output_tokens: int | None,
    ) -> str:
        caps = _get_model_capabilities(self.model)
        kwargs: dict = {"model": self.model, "messages": messages}
        if max_output_tokens:
            kwargs["max_tokens"] = max_output_tokens

        extra_headers: dict = {}
        extra_body: dict    = {}

        if think and caps.get("thinking"):
            budget = _OPENROUTER_THINKING_BUDGET
            if self.model.startswith("anthropic/"):
                extra_body["thinking"] = {"type": "enabled", "budget_tokens": budget}
                extra_headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"
            else:
                # OpenRouter unified reasoning parameter for non-Anthropic models
                extra_body["reasoning"] = {"max_tokens": budget}

        if caps.get("caching"):
            # Ensure the anthropic-beta header covers caching too
            existing = extra_headers.get("anthropic-beta", "")
            if "prompt-caching" not in existing:
                extra_headers["anthropic-beta"] = (
                    existing + ",prompt-caching-2024-07-31" if existing
                    else "prompt-caching-2024-07-31"
                )

        if extra_headers:
            kwargs["extra_headers"] = extra_headers
        if extra_body:
            kwargs["extra_body"] = extra_body

        # Capability-fallback loop: retry without the failing feature on first error
        for _attempt in range(3):
            try:
                response = _with_retry(lambda: _openrouter_client.chat.completions.create(**kwargs))  # type: ignore[union-attr]
                break
            except Exception as e:
                if _is_transient_error(e):
                    raise  # transient (429/503) — don't permanently disable capabilities
                err = str(e).lower()
                if "cache" in err and caps.get("caching"):
                    caps["caching"] = False
                    _model_capabilities[self.model]["caching"] = False
                    # Rebuild messages without cache_control blocks
                    messages = _strip_cache_control(messages)
                    kwargs["messages"] = messages
                    extra_headers.pop("anthropic-beta", None)
                    kwargs.pop("extra_headers", None)
                    logger.info("Disabled caching for %s after error: %s", self.model, e)
                    continue
                if ("thinking" in err or "budget" in err or "reasoning" in err) and caps.get("thinking"):
                    caps["thinking"] = False
                    _model_capabilities[self.model]["thinking"] = False
                    extra_body.pop("thinking", None)
                    extra_body.pop("reasoning", None)
                    if extra_body:
                        kwargs["extra_body"] = extra_body
                    else:
                        kwargs.pop("extra_body", None)
                    logger.info("Disabled thinking for %s after error: %s", self.model, e)
                    continue
                raise

        text = _extract_text(response)
        self._accum_openrouter(response)
        return text

    def _call_ollama(self, system: str, user: str, think: bool) -> str:
        """First Ollama call: build messages[], return response."""
        caps = _get_model_capabilities(self.model)
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        payload: dict = {"model": self.model, "stream": False, "messages": messages}
        if caps.get("thinking"):
            payload["think"] = think
        r = _with_retry(lambda: requests.post(f"{_OLLAMA_URL}/api/chat",
                                              json=payload, timeout=_OLLAMA_TIMEOUT))
        r.raise_for_status()
        data = r.json()
        text: str = data["message"]["content"]
        messages.append({"role": "assistant", "content": text})
        self.session = messages
        self._accum_ollama(data, "init")
        return text

    def _continue_ollama(self, user: str) -> str:
        """Continue Ollama call: trim if needed, append user + response."""
        if len(self.session) > _MAX_SESSION_MESSAGES:
            self._capture_ollama_strategy()
            self.trim_session()
        caps = _get_model_capabilities(self.model)
        self.session.append({"role": "user", "content": user})
        payload: dict = {"model": self.model, "stream": False, "messages": self.session}
        if caps.get("thinking"):
            payload["think"] = True
        r = _with_retry(lambda: requests.post(f"{_OLLAMA_URL}/api/chat",
                                              json=payload, timeout=_OLLAMA_TIMEOUT))
        r.raise_for_status()
        data = r.json()
        text: str = data["message"]["content"]
        self.session.append({"role": "assistant", "content": text})
        self._accum_ollama(data, "continue")
        return text

    def _capture_ollama_strategy(self) -> None:
        """Ask the model for its strategy before trimming the Ollama session."""
        pending = self.session.pop()  # temporarily remove pending user msg
        self.session.append({
            "role": "user",
            "content": (
                "Before we continue: write out your current strategy in 150-200 words — "
                "engine type, priority tags, milestone/award targets, pace plan, key watch-outs. "
                "Include a TABLEAU section listing every card you have played and its key ongoing effect."
            ),
        })
        payload: dict = {"model": self.model, "stream": False,
                         "messages": self.session, "think": False}
        try:
            r = requests.post(f"{_OLLAMA_URL}/api/chat", json=payload, timeout=_OLLAMA_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            self.strategy = data["message"]["content"].strip()
            self._accum_ollama(data, "strategy_capture")
            logger.info("Captured Ollama strategy for player %s: %.100s…", self.player_id, self.strategy)
        except Exception as exc:
            logger.warning("Strategy capture failed for player %s: %s", self.player_id, exc)
        self.session.pop()           # remove strategy request
        self.session.append(pending) # restore pending user msg

    # ------------------------------------------------------------------
    # Token accounting
    # ------------------------------------------------------------------

    def _accum_openrouter(self, response) -> None:
        usage = getattr(response, "usage", None)
        in_tok  = int(getattr(usage, "prompt_tokens",     0) or 0)
        out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
        # Anthropic-style cache fields (OpenRouter passes them through)
        details = getattr(usage, "prompt_tokens_details", None)
        cache_read  = int(getattr(details, "cached_tokens", 0) or 0) if details else 0
        if not cache_read:
            cache_read  = int(getattr(usage, "cache_read_input_tokens",     0) or 0)
        cache_write = int(getattr(usage, "cache_creation_input_tokens",     0) or 0)

        self.token_usage["calls"]      += 1
        self.token_usage["input"]      += in_tok
        self.token_usage["output"]     += out_tok
        self.token_usage["cache_read"] += cache_read
        self.token_usage["cache_write"]+= cache_write

        in_p, out_p = _get_openrouter_pricing(self.model)
        billed_in = in_tok - cache_read
        cost = (billed_in * in_p + cache_read * in_p * 0.1 + out_tok * out_p)
        running_cost = self._running_cost()
        logger.info(
            "tokens player=%s model=%s  in=%d out=%d cache_read=%d cache_write=%d"
            "  |  total calls=%d in=%d out=%d est_cost=$%.4f",
            self.player_id, self.model, in_tok, out_tok, cache_read, cache_write,
            self.token_usage["calls"], self.token_usage["input"],
            self.token_usage["output"], running_cost,
        )

    def _accum_ollama(self, data: dict, call_type: str) -> None:
        in_tok  = data.get("prompt_eval_count", 0) or 0
        out_tok = data.get("eval_count",         0) or 0
        self.token_usage["calls"]  += 1
        self.token_usage["input"]  += in_tok
        self.token_usage["output"] += out_tok
        logger.info(
            "tokens[%s] player=%s model=%s  in=%d out=%d  |  total calls=%d in=%d out=%d",
            call_type, self.player_id, self.model, in_tok, out_tok,
            self.token_usage["calls"], self.token_usage["input"], self.token_usage["output"],
        )

    def _running_cost(self) -> float:
        in_p, out_p = _get_openrouter_pricing(self.model)
        u = self.token_usage
        billed_in = u["input"] - u["cache_read"]
        return billed_in * in_p + u["cache_read"] * in_p * 0.1 + u["output"] * out_p

    def log_token_summary(self) -> None:
        if self.summary_logged:
            return
        self.summary_logged = True
        u = self.token_usage
        if self.provider == "openrouter":
            in_p, out_p = _get_openrouter_pricing(self.model)
            billed_in   = u["input"] - u["cache_read"]
            cost_in     = billed_in            * in_p
            cost_cached = u["cache_read"]      * in_p * 0.1
            cost_out    = u["output"]          * out_p
            total_cost  = cost_in + cost_cached + cost_out
            logger.info(
                "TOKEN SUMMARY player=%s | game=%s | model=%s | calls=%d"
                " | input=%d (billed=%d cached_read=%d cached_write=%d) | output=%d"
                " | cost: in=$%.4f cached=$%.4f out=$%.4f | TOTAL=$%.4f",
                self.player_id, self.game_id, self.model, u["calls"],
                u["input"], billed_in, u["cache_read"], u["cache_write"],
                u["output"], cost_in, cost_cached, cost_out, total_cost,
            )
        else:
            logger.info(
                "TOKEN SUMMARY player=%s | game=%s | model=%s | calls=%d | input=%d | output=%d",
                self.player_id, self.game_id, self.model,
                u["calls"], u["input"], u["output"],
            )


# ---------------------------------------------------------------------------
# Text extraction (strips thinking blocks from Anthropic extended-thinking responses)
# ---------------------------------------------------------------------------

def _extract_text(response) -> str:
    choice = response.choices[0]
    msg = choice.message
    if isinstance(msg.content, str):
        return msg.content or ""
    if isinstance(msg.content, list):
        parts = []
        for block in msg.content:
            btype = getattr(block, "type", None)
            if btype == "text" and hasattr(block, "text"):
                parts.append(block.text)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(msg.content or "")


def _strip_cache_control(messages: list[dict]) -> list[dict]:
    """Return messages with cache_control removed from system content blocks."""
    result = []
    for msg in messages:
        if msg.get("role") == "system" and isinstance(msg.get("content"), list):
            plain = " ".join(
                b.get("text", "") for b in msg["content"]
                if isinstance(b, dict)
            )
            result.append({"role": "system", "content": plain})
        else:
            result.append(msg)
    return result


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _log_prompt(header: str, text: str) -> None:
    prefixed = "\n".join(f"> {line}" for line in text.splitlines())
    logger.info("%s\n%s", header, prefixed)


def _log_response(header: str, text: str | None) -> None:
    if text is None:
        logger.info("%s\n< (empty/None response)", header)
        return
    prefixed = "\n".join(f"< {line}" for line in text.splitlines())
    logger.info("%s\n%s", header, prefixed)


# ---------------------------------------------------------------------------
# Player registry
# ---------------------------------------------------------------------------

def register_player(player_id: str, game_id: str, model: str | None = None) -> "LLMPlayer":
    """Register an AI player with a specific model. Overwrites any existing registration."""
    if model is None:
        model = _OPENROUTER_MODEL if _OPENROUTER_API_KEY else _OLLAMA_MODEL
    player = LLMPlayer(player_id=player_id, game_id=game_id, model=model)
    _player_registry[player_id] = player
    _game_players.setdefault(game_id, [])
    if player_id not in _game_players[game_id]:
        _game_players[game_id].append(player_id)
    logger.info("Registered player %s (game=%s model=%s provider=%s)",
                player_id, game_id, model, player.provider)
    return player


def get_or_create_player(player_id: str, game_id: str) -> "LLMPlayer":
    """Return existing player or auto-create with default model (warn if auto-created)."""
    if player_id not in _player_registry:
        logger.warning("Player %s not pre-registered — auto-creating with default model", player_id)
        register_player(player_id, game_id)
    return _player_registry[player_id]


def log_game_token_summary(game_id: str) -> None:
    """Log per-player token summary for all AI players in a game."""
    if game_id in _game_summary_logged:
        return
    _game_summary_logged.add(game_id)
    player_ids = _game_players.get(game_id, [])
    if not player_ids:
        logger.info("TOKEN SUMMARY game=%s  (no registered players)", game_id)
        return
    totals: dict = {"calls": 0, "input": 0, "output": 0}
    for pid in player_ids:
        player = _player_registry.get(pid)
        if player:
            player.log_token_summary()
            totals["calls"]  += player.token_usage["calls"]
            totals["input"]  += player.token_usage["input"]
            totals["output"] += player.token_usage["output"]
    logger.info(
        "TOKEN SUMMARY game=%s | players=%d | total calls=%d in=%d out=%d",
        game_id, len(player_ids), totals["calls"], totals["input"], totals["output"],
    )


# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------

def validate_llm_config() -> None:
    """Called at server startup. Warns if LLM is enabled but credentials are missing."""
    if os.getenv("USE_LLM", "false").lower() != "true":
        return
    if _OPENROUTER_API_KEY:
        _ensure_openrouter_client()
        logger.info("OpenRouter client ready (default model: %s)", _OPENROUTER_MODEL)
    else:
        # Check Ollama reachability
        try:
            r = requests.get(f"{_OLLAMA_URL}/api/tags", timeout=5)
            r.raise_for_status()
            names = [m["name"] for m in r.json().get("models", [])]
            if _OLLAMA_MODEL not in names:
                logger.warning("OLLAMA_MODEL=%r not found in Ollama. Available: %s",
                               _OLLAMA_MODEL, names)
            else:
                logger.info("Ollama model validated: %s", _OLLAMA_MODEL)
        except Exception as exc:
            logger.warning("Cannot reach Ollama at %s: %s", _OLLAMA_URL, exc)


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
       WEAK — only useful if energy production < 3/gen or you need a card threshold.
       Once you can already convert heat regularly, more energy gives almost nothing.
       NEVER choose Power Plant when City, Asteroid, or Aquifer is also affordable.
  3. Asteroid:      14 MC → +1 temperature (+1 TR).  [GOOD VALUE: 14 MC per TR]
  4. Aquifer:       18 MC → place ocean tile (+1 TR, +placement bonus).  [GOOD]
  5. Greenery:      23 MC → place greenery (+1 oxygen, +1 TR).
  6. City:          25 MC → place city tile + 1 MC production.
       HIGH VALUE — gives TR + permanent +1 MC income + board VP potential +
       progress toward Mayor milestone. Almost always better than Power Plant.

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
  • BUY AS MANY CARDS AS POSSIBLE — 3 MC per card is cheap relative to in-game value.
    A thin opening hand starves your engine for the entire game. Unless a card has
    zero synergy with your corporation, buy it. Aim to buy the maximum you can afford.

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
  • TR is income AND VP: each +1 TR permanently raises your MC income by 1/generation
    AND scores +1 VP at game end. Raising a global parameter is doubly valuable —
    treat terraforming as your primary objective, not an afterthought.
  • Convert heat/plants before passing: ≥8 heat with temperature < 8°C = a FREE +1 TR
    sitting unused (action G). ≥8 plants with O₂ < 14% = free greenery + TR (action F).
    These zero-MC actions are the highest-value moves in the game. NEVER end your turn
    or pass a generation while you have enough heat or plants to convert.
  • Match cards to your resources in the research phase: if you have stockpiled titanium,
    BUY space-tag cards — titanium pays for them at 3 MC/cube. If you have steel, BUY
    building-tag cards — steel pays at 2 MC/cube. Resources sitting in your stockpile
    with no cards to spend them on are pure wasted production.
  • Energy production has diminishing returns: once you convert heat to temperature every
    generation, extra energy-to-heat adds nothing. Power Plant SP (11 MC) is almost never
    the right Standard Project — City SP (25 MC) gives TR + income + board VP + milestone
    progress and is almost always better value. Only build Power Plants when your energy
    production is below ~3/gen or a specific card requires it.

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

  City-greenery VP engine:
    – Place your first city EARLY — it anchors a region for your greeneries and scores
      1 VP per greenery that ends up adjacent to it. Each greenery you place near a city
      is worth 2 VP (1 greenery VP + 1 city-adjacency VP) instead of just 1 VP.
    – Choose the city location where 4-5 adjacent hexes are free land (not ocean-reserved)
      so you have room to surround it with greeneries later.
    – A greenery adjacent to 2 cities scores +2 city VP (1 per city) plus its own 1 VP
      = 3 VP total. A greenery adjacent to 3 cities = 4 VP total.
    – Ideal 3-city triangle layout: place 3 cities so they share common adjacent hexes.
      Then fill those shared hexes with greeneries:
        (a) One central greenery adjacent to all 3 cities: 3 city VPs + 1 greenery VP = 4 VP
        (b) One greenery between each pair of cities (adjacent to 2 cities): 2+1 = 3 VP each
      Result: 3 cities + 4 greeneries = 13+ VP from board tiles alone.
    – Cities cannot be adjacent to each other, but two cities CAN both be adjacent to the
      same hex — that shared hex is where your high-value greenery goes.
    – Deny opponents: avoid placing greeneries next to their cities unless you have no choice.

OPPONENT ANALYSIS — read opponents constantly:
  • Their played cards reveal their engine (energy → heat, plant engine, etc.).
  • Funded awards signal what they are optimising for — don't help them win it.
  • Claimed milestones tell you what to block or race for next.
  • High hand size + few played cards → they are building toward a big combo.
  • Low MC + many played cards → they over-extended; they may pass soon.
  • Hand size gap: if they have 8+ cards and you have 2, they have 4× more engine
    options per turn. A persistent hand-size deficit means your engine will be weaker
    every generation — prioritise buying more cards in the research phase.

DRAFTING (when research phase offers card selection):
  • Do not pass a card that strongly benefits an opponent's visible engine
    UNLESS you have a card that is strictly better for YOUR own engine.
  • Deny opponent synergy cards (e.g. an opponent building Jupiter engine —
    withhold Jupiter-tag cards even at personal cost).
  • In late game, pass weak cards freely; in early game, card denial matters more.

=== END RULES REFERENCE ===
"""


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def select_action_llm(
    state: dict,
    waiting_for: dict,
    game_id: str,
    player_id: str,
    last_error: str | None = None,
) -> tuple[dict, dict]:
    """Return (input_response, debug) using the registered LLM player."""
    player = get_or_create_player(player_id, game_id)
    wf_type = waiting_for.get("type", "")

    try:
        if wf_type in SETUP_TYPES:
            return _select_setup(state, waiting_for, player)
        else:
            return _select_action(state, waiting_for, player, last_error=last_error)
    except Exception as exc:
        logger.error("LLM selection failed (player=%s type=%s): %s — using default",
                     player_id, wf_type, exc, exc_info=True)
        return _default_response(waiting_for), {"llm_error": str(exc)}


# ---------------------------------------------------------------------------
# Per-generation strategy update
# ---------------------------------------------------------------------------

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
    "IMPORTANT: if you currently have >=8 heat and temperature < 8°C, 'Convert 8 heat' MUST be "
    "one of your first actions this generation (free +1 TR = +1 VP + +1 MC income every gen). "
    "If you have >=8 plants and O₂ < 14%, 'Convert 8 plants' MUST be a first action (free +1 TR). "
    "These are your highest-value zero-cost actions — never skip them. "
    "Also identify which cards in your hand you plan to play and why they fit your strategy. "
    "Reminder: DO NOT use Convert Heat if temperature is already 8°C, and DO NOT place "
    "greenery tiles if O₂ is already 14% (both are wasted actions at max).\n\n"
    "!! MANDATORY DEFERRAL LOOP CHECK !!\n"
    "Scroll back through your last 2-3 strategy updates visible in this conversation. "
    "For every goal you listed in previous NEXT-GEN PRIORITY sections: has the relevant "
    "game state actually changed, or did you write the same priority again without acting?\n"
    "Examples of deferral loops:\n"
    "  • 'Build 3rd city for Mayor' stated last gen → still have same city count → LOOP\n"
    "  • 'Claim milestone X' stated last gen → milestone still unclaimed → LOOP\n"
    "  • 'Play card Y' stated last gen → card still in hand → LOOP\n"
    "  • 'Fund award Z' stated last gen → award still unfunded → LOOP\n"
    "If you detect a deferral loop:\n"
    "  1. Name it explicitly: 'I have deferred [action] for N generations.'\n"
    "  2. Execute it as your ABSOLUTE FIRST action this generation — before heat conversion, "
    "before card plays, before any other action. Nothing else comes first.\n"
    "  3. Do NOT write it as a plan for a third time. Repeating the plan without executing "
    "it is a critical failure: you lose the milestone/award/card value AND fall further "
    "behind your opponent who is scoring every generation while you plan."
)


def _per_generation_strategy_update(player: LLMPlayer, generation: int, state: dict) -> None:
    g = state.get("game", {})
    p = state.get("player", {})
    temp   = g.get("temperature", -30)
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

    prod = p.get("production", {})
    mc_prod  = prod.get("megacredits", 0)
    tr       = p.get("terraformRating", 20)
    mc_income = tr + mc_prod
    income_parts = [f"MC:{mc_income} (TR:{tr} + prod:{mc_prod:+d})"]
    for res_label, key in [("steel", "steel"), ("titanium", "titanium"),
                           ("plants", "plants"), ("energy", "energy"), ("heat", "heat")]:
        v = prod.get(key, 0)
        if v:
            income_parts.append(f"{res_label}:{v}")
    income_note = "Your production income this generation: " + ", ".join(income_parts) + "\n"

    current_heat   = p.get("heat", 0)
    current_plants = p.get("plants", 0)
    if current_heat >= 8 and temp < 8:
        income_note += (
            f">> HEAT STOCKPILE: {current_heat} — 'Convert 8 heat' is AVAILABLE NOW (+1 TR, 0 MC cost). "
            f"Plan this as your first action this generation.\n"
        )
    if current_plants >= 8 and oxygen < 14:
        income_note += (
            f">> PLANTS STOCKPILE: {current_plants} — 'Convert 8 plants' is AVAILABLE NOW (+1 TR, 0 MC cost). "
            f"Plan this as your first action this generation.\n"
        )

    prompt = _PER_GEN_STRATEGY_PROMPT.format(
        prev_gen=generation - 1,
        gen=generation,
        global_status=global_status,
        income_note=income_note,
    )
    if _LLM_DEBUG:
        _log_prompt(f"=== PER-GEN STRATEGY UPDATE (player={player.player_id} gen={generation}) ===", prompt)
    try:
        strategy = player.continue_session(prompt)
        player.strategy = strategy.strip()
        logger.info("Per-gen strategy update (player=%s gen=%d):\n%s",
                    player.player_id, generation, player.strategy)
        if _LLM_DEBUG:
            _log_response(f"=== PER-GEN STRATEGY response (player={player.player_id} gen={generation}) ===",
                          player.strategy)
    except Exception as exc:
        logger.warning("Per-gen strategy update failed (player=%s gen=%d): %s",
                       player.player_id, generation, exc)


def _maybe_per_generation_update(player: LLMPlayer, generation: int, state: dict) -> None:
    if player.last_generation >= 0 and generation > player.last_generation:
        logger.info("Generation bump %d→%d for player %s — running strategy update",
                    player.last_generation, generation, player.player_id)
        _per_generation_strategy_update(player, generation, state)
    player.last_generation = generation


# ---------------------------------------------------------------------------
# Setup phase (initialCards / prelude)
# ---------------------------------------------------------------------------

def _select_setup(state: dict, waiting_for: dict, player: LLMPlayer) -> tuple[dict, dict]:
    g = state.get("game", {})
    wf_type = waiting_for.get("type", "")
    has_session = bool(player.session)

    logger.info("LLM setup call (player=%s type=%s board=%s exps=%s session=%s)",
                player.player_id, wf_type, g.get("boardName", "?"), g.get("expansions", []), has_session)

    user = _build_setup_prompt(state, waiting_for)

    if wf_type == "initialCards" or not has_session:
        game_ctx  = format_config_context(g)
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
        text = player.init_session(system, user, think=True)
    else:
        text = player.continue_session(user)

    logger.info("Setup LLM response (player=%s):\n%s", player.player_id, text[:1000])

    input_response, strategy = _parse_setup_response(text, waiting_for, player)
    player.strategy = strategy
    logger.info("Player %s strategy stored:\n%s", player.player_id, strategy)
    return input_response, {"llm_phase": "setup", "strategy": strategy[:300]}


def _build_setup_prompt(state: dict, waiting_for: dict) -> str:
    wf_type = waiting_for.get("type", "")
    options = waiting_for.get("options", [])
    lines: list[str] = []

    if wf_type == "initialCards":
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
                "  BUYING GUIDANCE: buy as many cards as you can afford that have any synergy",
                "  with your corporation. 3 MC is cheap — a larger opening hand gives you more",
                "  engine options every draft round. A thin hand starves your engine all game.",
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
    text: str, waiting_for: dict, player: LLMPlayer
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

        chosen_corp = corps[0] if corps else ""
        m = re.search(r"CORPORATION:\s*(.+)", text)
        if m:
            raw = m.group(1).strip().rstrip(".")
            for c in corps:
                if c.lower() in raw.lower() or raw.lower() in c.lower():
                    chosen_corp = c
                    break

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
            if len(prelude_chosen) < 2:
                prelude_chosen = preludes[:2]

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

        m4 = re.search(r"STRATEGY:\s*(.*)", text, re.DOTALL)
        strategy = m4.group(1).strip() if m4 else text.strip()

        logger.info("Setup parsed: corp=%r buy=%r prelude=%r ceo=%r",
                    chosen_corp, bought, prelude_chosen, ceo_chosen)

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
        old = player.strategy or "Play balanced."
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

def _select_action(
    state: dict, waiting_for: dict, player: LLMPlayer, last_error: str | None = None
) -> tuple[dict, dict]:
    options = flatten_options(waiting_for)
    if not options:
        return _default_response(waiting_for), {}

    generation = state.get("game", {}).get("generation", 1)
    _maybe_per_generation_update(player, generation, state)

    user = _build_action_prompt(state, waiting_for, options, last_error=last_error, player=player)
    user += (
        "\n\nBefore choosing, check: does this action advance my engine, milestone/award targets, and pace plan?"
        " 1-2 sentences explaining how your choice fits your strategy, then CHOICE: N on its own line."
        " If playing a project card, add PAYMENT: MC=<n>[, STEEL=<n>][, TITANIUM=<n>] on the next line."
    )
    logger.debug("LLM action (player=%s type=%s options=%d)",
                 player.player_id, waiting_for.get("type"), len(options))
    text = player.continue_session(user, max_output_tokens=_OPENROUTER_MAX_OUTPUT_TOKENS
                                   if player.provider == "openrouter" else None)
    logger.debug("Action response (player=%s): %s", player.player_id, text[:300])

    result = _parse_action_response(text, options, waiting_for, player.player_id,
                                    player=state.get("player"))

    g = state.get("game", {})
    if (g.get("temperature", -30) >= 8
            and g.get("oxygen", 0) >= 14
            and g.get("oceanCount", 0) >= 9):
        log_game_token_summary(player.game_id)

    return result


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
    g = state.get("game", {})
    game_ctx    = format_config_context(g)
    board_layout = format_board_layout(state.get("boardSpaces") or [])
    system = (
        TM_RULES + "\n\n" + game_ctx + "\n\n"
        + (board_layout + "\n\n" if board_layout else "")
    )
    return system + _TRAINER_SYSTEM_SUFFIX


def _get_trainer_player(game_id: str, player_id: str, state: dict) -> LLMPlayer:
    """Return (or create) the trainer session for a given player."""
    trainer_id = f"trainer:{player_id}"
    if trainer_id not in _player_registry:
        base = _player_registry.get(player_id)
        model = base.model if base else (_OPENROUTER_MODEL if _OPENROUTER_API_KEY else _OLLAMA_MODEL)
        register_player(trainer_id, game_id, model)
    return _player_registry[trainer_id]


def _select_setup_advise(
    state: dict,
    waiting_for: dict,
    trainer: LLMPlayer,
    user_question: str | None,
) -> tuple[str, dict]:
    setup_prompt = _build_setup_prompt(state, waiting_for)
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

    if not trainer.session:
        system = _build_trainer_system(state)
        text = trainer.init_session(system, user, think=True)
    else:
        text = trainer.continue_session(user)

    rec_match = re.search(r"<recommendation>(.*?)</recommendation>", text, re.DOTALL | re.IGNORECASE)
    if rec_match:
        rec_text   = rec_match.group(1).strip()
        advice_text = (text[:rec_match.start()] + text[rec_match.end():]).strip()
    else:
        rec_text   = text
        advice_text = text

    # Use a dummy player for the strategy store in _parse_setup_response
    recommendation, _ = _parse_setup_response(rec_text, waiting_for, trainer)
    return advice_text or "(no advice text)", recommendation


def select_action_advise(
    state: dict,
    waiting_for: dict,
    game_id: str,
    player_id: str,
    user_question: str | None = None,
) -> tuple[str, dict]:
    """Return (advice_text, recommendation_input_response) for the AI Trainer feature."""
    wf_type = waiting_for.get("type", "")
    trainer = _get_trainer_player(game_id, player_id, state)

    if wf_type in SETUP_TYPES:
        return _select_setup_advise(state, waiting_for, trainer, user_question)

    options = flatten_options(waiting_for)
    if not options:
        return ("No actions available.", _default_response(waiting_for))

    generation = state.get("game", {}).get("generation", 1)
    _maybe_per_generation_update(trainer, generation, state)

    prompt_lines: list[str] = [_build_action_prompt(state, waiting_for, options)]
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

    if not trainer.session:
        system = _build_trainer_system(state)
        text = trainer.init_session(system, user, think=True)
    else:
        text = trainer.continue_session(user)

    rec_match = re.search(r"<recommendation>(.*?)</recommendation>", text, re.DOTALL | re.IGNORECASE)
    if rec_match:
        rec_text   = rec_match.group(1).strip()
        advice_text = text[:rec_match.start()].strip()
        if not advice_text:
            advice_text = text[rec_match.end():].strip()
    else:
        rec_text   = text
        advice_text = text

    m = re.search(r"CHOICE:\s*(\d+)", rec_text)
    chosen = int(m.group(1)) - 1 if m else 0
    chosen = max(0, min(chosen, len(options) - 1))
    recommendation = index_to_response(waiting_for, options[chosen]["index"])

    wf_type2 = waiting_for.get("type", "")
    if wf_type2 in ("projectCard", "payment"):
        payment = _parse_payment_line(rec_text)
        if payment:
            payment = _correct_payment(payment, waiting_for, state.get("player", {}))
            recommendation = {**recommendation, "payment": payment}

    return advice_text, recommendation


# ---------------------------------------------------------------------------
# Payment helpers
# ---------------------------------------------------------------------------

_PAYMENT_VALUES = {
    "steel": 2, "titanium": 3, "heat": 1, "plants": 3,
    "microbes": 2, "floaters": 3, "seeds": 5, "graphene": 4,
    "lunaArchivesScience": 1, "kuiperAsteroids": 1, "auroraiData": 3, "spireScience": 2,
}

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
    m = re.search(r"PAYMENT:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if not m:
        return None
    parts = re.findall(r"([A-Z_]+)\s*=\s*(\d+)", m.group(1), re.IGNORECASE)
    if not parts:
        return None
    payment = _empty_payment()
    for key, val in parts:
        field = _PAYMENT_KEYS.get(key.upper())
        if field and field in payment:
            payment[field] = int(val)
    return payment


def _correct_payment(payment: dict, waiting_for: dict, player: dict) -> dict:
    """Clamp payment fields to available resources and ensure the total covers the cost."""
    wf_type = waiting_for.get("type", "")
    payment = dict(payment)

    mc_avail = player.get("megacredits", 0)
    st_avail = player.get("steel",      0)
    ti_avail = player.get("titanium",   0)
    ht_avail = player.get("heat",       0)
    pl_avail = player.get("plants",     0)

    # Only allow steel/titanium for project card payments
    is_project = wf_type == "projectCard"
    if not is_project:
        payment["steel"]    = 0
        payment["titanium"] = 0

    # Clamp all non-MC resources to available
    for field, available in [
        ("steel",    st_avail), ("titanium", ti_avail),
        ("heat",     ht_avail), ("plants",   pl_avail),
    ]:
        if payment.get(field, 0) > available:
            payment[field] = available

    # Clamp MC
    if payment.get("megacredits", 0) > mc_avail:
        payment["megacredits"] = mc_avail

    # Determine cost and compute what's already covered by non-MC resources
    if is_project:
        card_node = waiting_for.get("card", {})
        cost = card_node.get("calculatedCost", 0) if card_node else waiting_for.get("amount", 0)
    else:
        cost = waiting_for.get("amount", 0)

    covered = sum(
        payment.get(field, 0) * _PAYMENT_VALUES.get(field, 0)
        for field in _PAYMENT_VALUES
    )
    needed_mc = max(0, cost - covered)
    if payment.get("megacredits", 0) < needed_mc:
        payment["megacredits"] = min(needed_mc, mc_avail)
        if mc_avail < needed_mc:
            logger.warning(
                "Payment underfunded: cost=%d covered_by_resources=%d need_mc=%d have_mc=%d — "
                "player cannot fully cover this card",
                cost, covered, needed_mc, mc_avail,
            )

    return payment


# ---------------------------------------------------------------------------
# Action prompt builder
# ---------------------------------------------------------------------------

# Track which generation the full hand was shown for each player (to elide repeats)
_hand_shown_generation: dict[str, int] = {}


def _get_card_desc_for_option(opt: dict) -> str:
    """Return 'Use <CardName> action — <description>' for blue-card actions."""
    card = opt.get("card") or {}
    name = card.get("name", "")
    if not name:
        return ""
    info = CARD_DB.get(name, {})
    desc = info.get("description", "")
    return f"Use {name} action — {desc}" if desc else f"Use {name} action"


def _build_action_prompt(
    state: dict,
    waiting_for: dict,
    options: list[dict],
    last_error: str | None = None,
    player: LLMPlayer | None = None,
) -> str:
    g  = state.get("game",   {})
    p  = state.get("player", {})
    ms = state.get("milestones", [])
    aw = state.get("awards",     [])

    game_id   = player.player_id if player else "unknown"
    temp   = g.get("temperature", -30)
    oxygen = g.get("oxygen", 0)
    oceans = g.get("oceanCount", 0)

    prod = {k: v for k, v in p.get("production", {}).items() if v}
    tags = {k: v for k, v in p.get("tags", {}).items() if v}

    tr     = p.get("terraformRating", 20)
    mc     = p.get("megacredits", 0)
    mc_prod = p.get("production", {}).get("megacredits", 0)
    mc_income = tr + mc_prod

    lines: list[str] = []

    if last_error:
        lines.append(f'⚠ Your previous response was rejected: "{last_error}"')
        lines.append("Please correct your choice based on the error above.")
        lines.append("")

    lines += [
        f"Gen {g.get('generation',1)} | Temp:{temp}°C O₂:{oxygen}% Oceans:{oceans}/9",
        f"You: TR:{tr} VP:{p.get('victoryPoints','?')}  MC:{mc}(income:{mc_income})  "
        f"Steel:{p.get('steel',0)} Ti:{p.get('titanium',0)}  "
        f"Plants:{p.get('plants',0)} Energy:{p.get('energy',0)} Heat:{p.get('heat',0)}",
    ]

    if temp >= 8:
        lines.append("⚠ Temperature is at maximum (8°C). DO NOT use Convert Heat — it is a wasted action.")
    if oxygen >= 14:
        lines.append("⚠ O₂ is at maximum (14%). DO NOT place greenery tiles — they give no benefit.")
    if oceans >= 9:
        lines.append("⚠ All 9 oceans are placed. No more ocean tiles can be placed.")

    heat_now   = p.get("heat", 0)
    plants_now = p.get("plants", 0)
    if heat_now >= 8 and temp < 8:
        lines.append(
            f">> ACTION AVAILABLE: you have {heat_now} heat (>=8) and temperature is not maxed."
            f" 'Convert 8 heat' raises temperature +1 TR (+1 VP, +1 MC income every remaining gen)."
            f" Do this BEFORE passing."
        )
    if plants_now >= 8 and oxygen < 14:
        lines.append(
            f">> ACTION AVAILABLE: you have {plants_now} plants (>=8) and O2 is not maxed."
            f" 'Convert 8 plants' places a greenery tile (+1 O2, +1 TR, +1 VP). Do this BEFORE passing."
        )
    if prod:
        lines.append(f"Production: {prod}")
    if tags:
        lines.append(f"Tags: {tags}")
    lines.append("Note: Only MC production can go negative (min -5). Steel/Ti/Plants/Energy/Heat production CANNOT go below 0.")

    # Cards in hand
    hand_cards = p.get("cardsInHand") or []
    generation = g.get("generation", 1)
    wf_type    = waiting_for.get("type", "")
    _hand_irrelevant = wf_type in ("space", "payment", "amount")
    if hand_cards and not _hand_irrelevant:
        last_shown_gen = player.hand_shown_generation if player else -1
        if last_shown_gen == generation:
            name_list = ", ".join(hand_cards[:30])
            lines += ["", f"Your hand ({len(hand_cards)} cards): {name_list}",
                      "  (Full descriptions shown earlier this generation — rely on your session memory.)"]
        else:
            ctx = format_card_context(hand_cards, header=f"Your hand ({len(hand_cards)} cards):", max_cards=30)
            lines += ["", ctx]
            if player:
                player.hand_shown_generation = generation
            # Effective cost annotations for steel/titanium discounts
            steel_now = p.get("steel", 0)
            ti_now    = p.get("titanium", 0)
            eff_notes: list[str] = []
            for card_name in hand_cards[:30]:
                card_name_str = card_name if isinstance(card_name, str) else card_name.get("name", "")
                info = CARD_DB.get(card_name_str, {})
                cost = info.get("cost", 0) or 0
                card_tags = info.get("tags") or []
                if "building" in card_tags and steel_now >= 2:
                    discount = min(steel_now, cost // 2) * 2
                    eff = max(0, cost - discount)
                    if discount > 0:
                        eff_notes.append(f"  {card_name_str}: {cost}MC → {eff}MC effective (use {discount//2} steel)")
                elif "space" in card_tags and ti_now >= 3:
                    discount = min(ti_now, cost // 3) * 3
                    eff = max(0, cost - discount)
                    if discount > 0:
                        eff_notes.append(f"  {card_name_str}: {cost}MC → {eff}MC effective (use {discount//3} titanium)")
            if eff_notes:
                lines += ["Effective cost with your resources (steel/titanium):"] + eff_notes

    # Opponents
    opponents = state.get("opponents") or []
    for i, opp in enumerate(opponents, 1):
        opp_prod = {k: v for k, v in opp.get("production", {}).items() if v}
        opp_tags = {k: v for k, v in opp.get("tags", {}).items() if v}
        opp_name = opp.get("name", f"Opponent{'' if len(opponents) == 1 else i}")
        opp_vp   = opp.get("victoryPoints")
        opp_vp_str = f" VP:{opp_vp}" if opp_vp is not None else ""
        opp_hs   = opp.get("handSize")
        opp_hs_str = f"  hand:{opp_hs}cards" if opp_hs is not None else ""
        lines.append(
            f"{opp_name}: TR:{opp.get('terraformRating',20)}{opp_vp_str}  MC:{opp.get('megacredits',0)}"
            f"{opp_hs_str}  prod:{opp_prod}  tags:{opp_tags}"
        )
    if ms:
        lines.append(f"Milestones claimed: {ms}")
    if aw:
        lines.append(f"Awards funded: {aw}")

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
        title = str(title_raw.get("message", "Select action"))
    else:
        title = "Select action"

    lines += ["", f"Decision: {title}", "Options:"]

    # Resource hint for card-selection decisions
    if wf_type == "card":
        steel_now = p.get("steel", 0)
        ti_now    = p.get("titanium", 0)
        resource_hints = []
        if ti_now >= 3:
            resource_hints.append(f"you have {ti_now} titanium → favor SPACE-tag cards (titanium pays at 3 MC/cube)")
        if steel_now >= 3:
            resource_hints.append(f"you have {steel_now} steel → favor BUILDING-tag cards (steel pays at 2 MC/cube)")
        if resource_hints:
            lines.append("Resource tip: " + "; ".join(resource_hints) + ".")

    # Tile placement tips
    if wf_type == "space":
        board = state.get("board") or []
        own_color = p.get("color", "")
        own_cities = [t for t in board if t.get("tileType") == "city" and t.get("playerColor") == own_color]
        opp_cities = [t for t in board if t.get("tileType") == "city" and t.get("playerColor") != own_color]

        tile_title = title.lower()
        if "greenery" in tile_title:
            lines += [
                "PLACEMENT TIP — Greenery VP math:",
                "  • Adjacent to 1 of your cities: +1 city VP + 1 greenery VP = 2 VP total",
                "  • Adjacent to 2 of your cities: +2 city VP + 1 greenery VP = 3 VP total",
                "  • Adjacent to 3 of your cities: +3 city VP + 1 greenery VP = 4 VP total",
                "  • Maximize: pick the hex adjacent to the most of YOUR cities.",
                "  ⚠ Never place adjacent to an opponent's city — you give them +1 VP for free.",
            ]
            if opp_cities:
                lines.append(f"  (Opponent has {len(opp_cities)} cities — avoid their adjacency.)")

        elif "city" in tile_title:
            if not own_cities:
                lines += [
                    "PLACEMENT TIP — First city (critical decision):",
                    "  • Choose a hex with 4-5 adjacent free LAND hexes (not ocean-reserved).",
                    "  • This gives you room to place greeneries around it for 2 VP each.",
                    "  • Aim for a spot where you can eventually build 2-3 cities and surround",
                    "    them with shared greeneries (each shared greenery = 3-4 VP).",
                    "  • Ideal: a position that lets you form a triangle of 3 cities later,",
                    "    sharing central hexes worth 4 VP each.",
                ]
            else:
                lines += [
                    "PLACEMENT TIP — Subsequent city:",
                    f"  • You already have {len(own_cities)} city/cities. Place close to your existing city",
                    "    so you can place greeneries adjacent to BOTH cities (3 VP per greenery).",
                    "  • A greenery adjacent to 2 cities = 3 VP; adjacent to 3 cities = 4 VP.",
                    "  • Forming a triangle with your 3rd city sets up the highest-value greenery layout.",
                ]

    # Options list
    card_names_in_decision = _extract_card_names(waiting_for)
    is_hand_decision = _is_card_decision_about_hand(card_names_in_decision, hand_cards)

    for opt in options:
        idx    = opt["index"] + 1
        title2 = opt["title"]
        desc   = _get_card_desc_for_option(opt)

        if wf_type == "card" and is_hand_decision and card_names_in_decision:
            title2_clean = str(title2)[:70]
            lines.append(f"  {idx}. {title2_clean} (see hand above)")
        elif desc:
            lines.append(f"  {idx}. {desc}")
        else:
            title2_clean = str(title2)[:70]
            lines.append(f"  {idx}. {title2_clean}")

    # Payment section
    if wf_type in ("projectCard", "payment"):
        lines += _format_payment_section(waiting_for, p)

    return "\n".join(lines)


def _format_payment_section(waiting_for: dict, player: dict) -> list[str]:
    lines: list[str] = ["", "Payment:"]
    wf_type = waiting_for.get("type", "")
    if wf_type == "projectCard":
        card = waiting_for.get("card", {})
        cost = card.get("calculatedCost", "?")
        name = card.get("name", "?")
        lines.append(f"  Card: {name}  Cost: {cost} MC")
        st = player.get("steel",    0)
        ti = player.get("titanium", 0)
        mc = player.get("megacredits", 0)
        card_tags = []
        info = CARD_DB.get(name, {})
        if info:
            card_tags = info.get("tags") or []
        can_use_steel = "building" in card_tags
        can_use_ti    = "space"    in card_tags
        lines.append(f"  Available: MC={mc}" +
                     (f" Steel={st}(worth {st*2}MC)" if can_use_steel and st else "") +
                     (f" Titanium={ti}(worth {ti*3}MC)" if can_use_ti and ti else ""))
        lines += [
            "  Reply with: PAYMENT: MC=<n>[, STEEL=<n>][, TITANIUM=<n>]...",
            "  Rules: steel only for building-tag, titanium only for space-tag.",
            "  You cannot overpay in MC; overpaying in steel/titanium is OK (surplus lost).",
        ]
    else:
        amount = waiting_for.get("amount", 0)
        mc = player.get("megacredits", 0)
        lines.append(f"  Amount: {amount} MC  Available MC: {mc}")
        lines += [
            "  Reply with: PAYMENT: MC=<n>[, HEAT=<n>]...",
            "  (No steel/titanium allowed for standard project payments.)",
        ]
    return lines


# ---------------------------------------------------------------------------
# Action response parser
# ---------------------------------------------------------------------------

def _auto_payment_for_card(card_name: str, sub_node: dict, player: dict) -> dict:
    """Generate optimal steel/titanium payment for a project card using CARD_DB tags."""
    cards = sub_node.get("cards", []) if isinstance(sub_node, dict) else []
    card_info = next((c for c in cards if c.get("name") == card_name), {})
    cost = card_info.get("calculatedCost", 0)
    entry = CARD_DB.get(card_name, {})
    tags = entry.get("tags", []) if entry else []

    ti_avail = player.get("titanium", 0) if "space" in tags else 0
    st_avail = player.get("steel",    0) if "building" in tags else 0

    ti_used = min(ti_avail, (cost + 2) // 3)
    remaining = max(0, cost - ti_used * 3)
    st_used = min(st_avail, (remaining + 1) // 2)
    remaining = max(0, remaining - st_used * 2)
    mc_used = min(remaining, player.get("megacredits", 0))

    payment = _empty_payment()
    payment["megacredits"] = mc_used
    payment["steel"]       = st_used
    payment["titanium"]    = ti_used
    return payment


def _parse_action_response(
    text: str, options: list[dict], waiting_for: dict, player_id: str,
    player: dict | None = None,
) -> tuple[dict, dict]:
    m = re.search(r"CHOICE:\s*(\d+)", text)
    if not m:
        logger.warning(
            "No CHOICE line in response (player=%s) — defaulting to option 1. Response: %.300s",
            player_id, text,
        )
    chosen = int(m.group(1)) - 1 if m else 0
    chosen = max(0, min(chosen, len(options) - 1))
    option = options[chosen]
    logger.info("Action choice player=%s: %d. %s", player_id, chosen + 1, option["title"])
    response = index_to_response(waiting_for, option["index"])

    wf_type = waiting_for.get("type", "")
    p = player or {}

    if wf_type in ("projectCard", "payment"):
        payment = _parse_payment_line(text)
        if payment:
            payment = _correct_payment(payment, waiting_for, p)
            response = {**response, "payment": payment}
        else:
            logger.warning(
                "No PAYMENT line for %s decision (player=%s) — using default payment. Response: %.200s",
                wf_type, player_id, text,
            )
    elif response.get("type") == "or":
        # or-option that resolves to a project card play — extract card + payment
        inner = response.get("response", {})
        if inner.get("type") == "projectCard":
            sub_node = option.get("node", {})
            available_cards = sub_node.get("cards", []) if isinstance(sub_node, dict) else []
            card_name = inner.get("card", "")
            payment = _parse_payment_line(text)
            if payment:
                # AI wrote an explicit PAYMENT line — find which card it named in the text
                # (only when PAYMENT present; avoids false matches on incidental words)
                text_lower = text.lower()
                for c in available_cards:
                    cname = c.get("name", "")
                    if cname and cname.lower() in text_lower:
                        card_name = cname
                        break
            if not card_name:
                if available_cards:
                    card_name = available_cards[0].get("name", "")
                    logger.warning(
                        "or→projectCard: no card name found in response (player=%s) — using first: %r. "
                        "Response: %.200s", player_id, card_name, text,
                    )
                else:
                    logger.error(
                        "or→projectCard: no card name and no available cards (player=%s). "
                        "Response: %.200s", player_id, text,
                    )
            if payment:
                card_info = next((c for c in available_cards if c.get("name") == card_name), {})
                stub_wf = {"type": "projectCard", "amount": card_info.get("calculatedCost", 0)}
                payment = _correct_payment(payment, stub_wf, p)
            else:
                payment = _auto_payment_for_card(card_name, sub_node, p)
                logger.info(
                    "Auto-payment for card %r (player=%s, no PAYMENT line): %s",
                    card_name, player_id, payment,
                )
            inner = {"type": "projectCard", "card": card_name, "payment": payment}
            response = {**response, "response": inner}

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
    if not card_names or not cards_in_hand:
        return False
    hand_set = {c if isinstance(c, str) else c.get("name", "") for c in cards_in_hand}
    return all(name in hand_set for name in card_names)


def _extract_card_names(waiting_for: dict, max_depth: int = 3) -> list[str]:
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
