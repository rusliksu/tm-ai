"""OpenRouter client — the single LLM backend.

One `openai.OpenAI` client pointed at OpenRouter. Provides:
  • single_shot(model, system, user, ...) -> (text, usage)  — one stateless completion
  • capability table (caching/thinking) with safe defaults, no live probing
  • pricing prefetch + cost() helper for token accounting
  • transient-error retry (network blips / 429 / 5xx retried; everything else raised)
  • truncation handling for models that blow max_tokens on hidden reasoning
  • per-call capability fallback (disable caching/thinking if a model rejects it)
"""
from __future__ import annotations
import logging
import re
import threading
import time

import requests

from . import config

logger = logging.getLogger(__name__)

_client = None  # lazy openai.OpenAI

# ---------------------------------------------------------------------------
# Capabilities — static table, safe defaults for unknown models (no probing)
# ---------------------------------------------------------------------------

# {caching, thinking}. Caching = explicit Anthropic ephemeral cache_control. Other
# providers (OpenAI/DeepSeek/Gemini) auto-cache server-side; we don't set cache_control.
_KNOWN_CAPABILITIES: dict[str, dict] = {
    "anthropic/claude-opus-4.7":   {"caching": True,  "thinking": True},
    "anthropic/claude-opus-4-7":   {"caching": True,  "thinking": True},
    "anthropic/claude-opus-4-5":   {"caching": True,  "thinking": True},
    "anthropic/claude-sonnet-4-6": {"caching": True,  "thinking": True},
    "anthropic/claude-sonnet-4-5": {"caching": True,  "thinking": True},
    "anthropic/claude-haiku-4-5":  {"caching": True,  "thinking": False},
    "openai/gpt-5.5":              {"caching": False, "thinking": False},
    "openai/gpt-4o":               {"caching": False, "thinking": False},
    "openai/gpt-4o-mini":          {"caching": False, "thinking": False},
    "openai/o3":                   {"caching": False, "thinking": False},
    "openai/o4-mini":              {"caching": False, "thinking": False},
    "x-ai/grok-4.3":               {"caching": True,  "thinking": True},
    "google/gemini-2.5-pro":       {"caching": False, "thinking": True},
    "google/gemini-2.5-flash":     {"caching": False, "thinking": True},
    "google/gemini-2.5-flash-lite": {"caching": False, "thinking": False},
    "deepseek/deepseek-v4-pro":    {"caching": False, "thinking": True},
    "deepseek/deepseek-v4-flash":  {"caching": False, "thinking": True},
    "deepseek/deepseek-r1":        {"caching": False, "thinking": True},
    "deepseek/deepseek-chat":      {"caching": False, "thinking": False},
}

_capabilities: dict[str, dict] = {}


def _base_capabilities(model: str) -> dict:
    """Raw {caching, thinking} from the static table. Unknown models default to caching
    for anthropic/ models and thinking enabled (disabled later if the provider rejects it)."""
    if model in _KNOWN_CAPABILITIES:
        return dict(_KNOWN_CAPABILITIES[model])
    for key, caps in _KNOWN_CAPABILITIES.items():
        if model.startswith(key) or key.startswith(model):
            return dict(caps)
    caps = {"caching": model.startswith("anthropic/"), "thinking": True}
    logger.info("Unknown model %s — using default capabilities %s", model, caps)
    return caps


def get_capabilities(model: str) -> dict:
    """Return {caching, thinking} for a model, applying the global OPENROUTER_THINKING
    override ("on"/"off") once at cache-build time. The cached dict is also what the
    on-error path mutates to disable a capability the provider rejected, so subsequent
    calls keep that disable."""
    if model in _capabilities:
        return _capabilities[model]
    caps = _base_capabilities(model)
    if config.OPENROUTER_THINKING == "off":
        caps["thinking"] = False
    elif config.OPENROUTER_THINKING == "on":
        caps["thinking"] = True
    _capabilities[model] = caps
    return caps


def ensure_client() -> None:
    global _client
    if _client is None:
        import openai
        _client = openai.OpenAI(base_url="https://openrouter.ai/api/v1",
                                api_key=config.OPENROUTER_API_KEY)
        _prefetch_pricing()


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

_pricing_cache: dict[str, tuple[float, float]] = {}
_pricing_fetched = False
_pricing_lock = threading.Lock()

_KNOWN_PRICING: dict[str, tuple[float, float]] = {
    "anthropic/claude-opus-4-7":   (15e-6, 75e-6),
    "anthropic/claude-sonnet-4-6": (3e-6, 15e-6),
    "anthropic/claude-haiku-4-5":  (0.8e-6, 4e-6),
    "openai/gpt-4o":               (2.5e-6, 10e-6),
    "openai/gpt-4o-mini":          (0.15e-6, 0.6e-6),
    "google/gemini-2.5-flash":     (0.3e-6, 1.2e-6),
    "google/gemini-2.5-pro":       (1.25e-6, 10e-6),
    "deepseek/deepseek-chat":      (0.27e-6, 1.1e-6),
    "deepseek/deepseek-r1":        (0.55e-6, 2.19e-6),
}


def _prefetch_pricing() -> None:
    global _pricing_fetched
    if _pricing_fetched:
        return
    with _pricing_lock:
        if _pricing_fetched:
            return
        try:
            r = requests.get("https://openrouter.ai/api/v1/models",
                             headers={"Authorization": f"Bearer {config.OPENROUTER_API_KEY}"},
                             timeout=10)
            r.raise_for_status()
            for entry in r.json().get("data", []):
                mid = entry.get("id", "")
                pricing = entry.get("pricing") or {}
                raw_in = pricing.get("prompt") or pricing.get("input") or 0
                raw_out = pricing.get("completion") or pricing.get("output") or 0
                try:
                    _pricing_cache[mid] = (float(raw_in), float(raw_out))
                except (TypeError, ValueError):
                    _pricing_cache[mid] = (0.0, 0.0)
            _pricing_fetched = True
            logger.info("Fetched OpenRouter model pricing (%d models)", len(_pricing_cache))
        except Exception as e:
            logger.warning("OpenRouter pricing fetch failed: %s", e)


def get_pricing(model: str) -> tuple[float, float]:
    if not _pricing_fetched:
        _prefetch_pricing()
    result = _pricing_cache.get(model)
    if result and result != (0.0, 0.0):
        return result
    for known_id, known_price in _KNOWN_PRICING.items():
        if model.startswith(known_id) or known_id.startswith(model):
            return known_price
    return (0.0, 0.0)


def cost(model: str, usage: dict) -> float:
    """Estimated USD cost for accumulated usage (cached reads billed at 10%)."""
    in_p, out_p = get_pricing(model)
    billed_in = usage.get("input", 0) - usage.get("cache_read", 0)
    return billed_in * in_p + usage.get("cache_read", 0) * in_p * 0.1 + usage.get("output", 0) * out_p


# ---------------------------------------------------------------------------
# Retry on transient errors
# ---------------------------------------------------------------------------

_RETRY_SCHEDULE = [1.0, 5.0, 10.0]
_RETRY_FOREVER_DELAY = 30.0


def is_transient_error(exc: Exception) -> bool:
    try:
        import openai
        if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError)):
            return True
        if isinstance(exc, openai.APIStatusError) and 500 <= getattr(exc, "status_code", 0) < 600:
            return True
    except ImportError:
        pass
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    msg = str(exc).lower()
    return any(k in msg for k in (
        "503", "unavailable", "429", "rate limit", "overloaded", "resource exhausted",
        "timeout", "connection error", "connection refused", "connection reset",
        "remote end closed connection", "broken pipe", "network is unreachable",
        "no route to host", "name or service not known",
        "temporary failure in name resolution", "ssl", "tls",
    ))


def _parse_retry_delay(exc: Exception) -> float | None:
    m = re.search(r"retry.?after[:\s]+(\d+\.?\d*)", str(exc), re.IGNORECASE)
    if not m:
        m = re.search(r"retryDelay.*?(\d+\.?\d*)s", str(exc))
    return float(m.group(1)) if m else None


def with_retry(fn, max_attempts: int | None = None):
    """Call fn() retrying transient errors on a 1s/5s/10s/30s… schedule (forever by default)."""
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:
            if not is_transient_error(exc):
                raise
            if max_attempts is not None and attempt + 1 >= max_attempts:
                raise
            api_delay = None
            if "429" in str(exc) or "resource exhausted" in str(exc).lower():
                api_delay = _parse_retry_delay(exc)
            scheduled = _RETRY_SCHEDULE[attempt] if attempt < len(_RETRY_SCHEDULE) else _RETRY_FOREVER_DELAY
            wait = max(api_delay + 2.0, scheduled) if api_delay else scheduled
            logger.warning("Transient error (attempt %d): %s — retrying in %.0fs", attempt + 1, exc, wait)
            time.sleep(wait)
            attempt += 1


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------

def _extract_text(response) -> str:
    choice = response.choices[0]
    msg = choice.message
    if isinstance(msg.content, str):
        return msg.content or ""
    if isinstance(msg.content, list):
        parts = []
        for block in msg.content:
            if getattr(block, "type", None) == "text" and hasattr(block, "text"):
                parts.append(block.text)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(msg.content or "")


def _is_truncated(response, text: str) -> bool:
    """True iff cut off at max_tokens with no usable answer (no CHOICE line / empty)."""
    try:
        choice = response.choices[0]
    except Exception:
        return False
    if (getattr(choice, "finish_reason", None) or "").lower() != "length":
        return False
    if not text or not text.strip():
        return True
    return not re.search(r"CHOICE:\s*\d", text)


def _strip_cache_control(messages: list[dict]) -> list[dict]:
    result = []
    for msg in messages:
        if msg.get("role") == "system" and isinstance(msg.get("content"), list):
            plain = " ".join(b.get("text", "") for b in msg["content"] if isinstance(b, dict))
            result.append({"role": "system", "content": plain})
        else:
            result.append(msg)
    return result


def _system_message(model: str, system: str) -> dict:
    if get_capabilities(model).get("caching"):
        return {"role": "system",
                "content": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]}
    return {"role": "system", "content": system}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _provider_routing(model: str) -> dict | None:
    """OpenRouter `provider` routing block for a request, or None to let OpenRouter decide.

    An explicit OPENROUTER_PROVIDER pins one provider with NO fallback so prompt caching stays
    warm and a slow/changing provider can't be chosen — OpenRouter was otherwise floating deepseek
    between SiliconFlow/Alibaba, resetting the cache on each switch. Without a pin, open models are
    sorted by throughput (Anthropic/OpenAI already resolve to a single host)."""
    if config.OPENROUTER_PROVIDER:
        order = [pp.strip() for pp in config.OPENROUTER_PROVIDER.split(",") if pp.strip()]
        if order:
            return {"order": order, "allow_fallbacks": False, "require_parameters": True}
    if not model.startswith(("anthropic/", "openai/")):
        return {"sort": "throughput", "require_parameters": True}
    return None


def single_shot(model: str, system: str, user: str, *, think: bool = True,
                thinking_budget: int | None = None,
                max_output_tokens: int | None = None) -> tuple[str, dict]:
    """One stateless [system, user] completion. Returns (text, usage_delta).

    `usage_delta` has keys: input, output, cache_read, cache_write. Keeping `system`
    byte-identical across turns lets prompt-caching reuse the rules prefix.
    """
    ensure_client()
    caps = get_capabilities(model)
    messages = [_system_message(model, system), {"role": "user", "content": user}]

    kwargs: dict = {"model": model, "messages": messages}
    if max_output_tokens:
        kwargs["max_tokens"] = max_output_tokens

    extra_headers: dict = {}
    extra_body: dict = {}

    if think and caps.get("thinking"):
        budget = thinking_budget or config.OPENROUTER_THINKING_BUDGET
        if model.startswith("anthropic/"):
            extra_body["thinking"] = {"type": "enabled", "budget_tokens": budget}
            extra_headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"
        else:
            extra_body["reasoning"] = {"max_tokens": budget}

    if caps.get("caching"):
        existing = extra_headers.get("anthropic-beta", "")
        extra_headers["anthropic-beta"] = (
            existing + ",prompt-caching-2024-07-31" if existing else "prompt-caching-2024-07-31"
        )

    routing = _provider_routing(model)
    if routing:
        extra_body["provider"] = routing

    if extra_headers:
        kwargs["extra_headers"] = extra_headers
    if extra_body:
        kwargs["extra_body"] = extra_body

    response = None
    for _trunc in range(config.OPENROUTER_TRUNCATION_RETRIES + 1):
        response = None
        for _attempt in range(3):
            try:
                response = with_retry(lambda: _client.chat.completions.create(**kwargs))
                break
            except Exception as e:
                if is_transient_error(e):
                    raise
                err = str(e).lower()
                if "cache" in err and caps.get("caching"):
                    caps["caching"] = False
                    _capabilities[model]["caching"] = False
                    messages = _strip_cache_control(messages)
                    kwargs["messages"] = messages
                    extra_headers.pop("anthropic-beta", None)
                    kwargs.pop("extra_headers", None)
                    logger.info("Disabled caching for %s after error: %s", model, e)
                    continue
                if ("thinking" in err or "budget" in err or "reasoning" in err) and caps.get("thinking"):
                    caps["thinking"] = False
                    _capabilities[model]["thinking"] = False
                    extra_body.pop("thinking", None)
                    extra_body.pop("reasoning", None)
                    if extra_body:
                        kwargs["extra_body"] = extra_body
                    else:
                        kwargs.pop("extra_body", None)
                    logger.info("Disabled thinking for %s after error: %s", model, e)
                    continue
                raise

        text = _extract_text(response)
        if not _is_truncated(response, text):
            break
        if _trunc >= config.OPENROUTER_TRUNCATION_RETRIES:
            logger.warning("Response truncated and retries exhausted for %s — best-effort (%d chars)",
                           model, len(text or ""))
            break
        cur_max = int(kwargs.get("max_tokens") or config.OPENROUTER_MAX_OUTPUT_TOKENS)
        new_max = min(config.OPENROUTER_TRUNCATION_MAX, cur_max * 2)
        if new_max <= cur_max:
            break
        kwargs["max_tokens"] = new_max
        for key in ("thinking", "reasoning"):
            if key in extra_body:
                budget_key = "budget_tokens" if key == "thinking" else "max_tokens"
                old = int(extra_body[key].get(budget_key, 1024))
                extra_body[key][budget_key] = max(256, old // 2)
                kwargs["extra_body"] = extra_body
        logger.warning("Response truncated for %s — retry with max_tokens=%d (was %d)",
                       model, new_max, cur_max)

    usage = _usage_delta(response)
    return text, usage


def _usage_delta(response) -> dict:
    usage = getattr(response, "usage", None)
    in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
    out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
    details = getattr(usage, "prompt_tokens_details", None)
    cache_read = int(getattr(details, "cached_tokens", 0) or 0) if details else 0
    if not cache_read:
        cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    cache_write = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    return {"input": in_tok, "output": out_tok, "cache_read": cache_read, "cache_write": cache_write}
