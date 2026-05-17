# Plan: OpenRouter Multi-Model LLM Player

**Goal**: Replace the Gemini-specific monolithic LLM player with a class-based architecture
where each AI player gets its own `LLMPlayer` instance, backed by OpenRouter (or Ollama for
local dev). OpenAI, Anthropic, Gemini, and Grok can play against each other in one game.

---

## Architecture Overview

### Current state (problems)
- Global dicts (`_game_sessions`, `_game_strategies`, etc.) keyed by `game_id`
- One game = one AI player = fine; two AI players in one game = session collision
- Hard-wired Gemini client with Google-specific cache lifecycle
- Token usage aggregated per game, not per player
- `select_action_llm` ignores `player_id` even though it's in `MoveRequest`

### Target state
```
AI server process
└── _player_registry: dict[player_id → LLMPlayer]
    ├── player "p1abc" → LLMPlayer(model="anthropic/claude-opus-4-7")
    ├── player "p2def" → LLMPlayer(model="openai/gpt-4o")
    ├── player "p3ghi" → LLMPlayer(model="x-ai/grok-3")
    └── player "p4jkl" → LLMPlayer(model="google/gemini-2.5-pro")
```

Each `LLMPlayer`:
- Holds its own messages[] session history
- Knows which provider to use (derived from model string)
- Probes and caches model capabilities (caching, thinking)
- Tracks its own token usage and cost

---

## Provider Resolution (model string determines everything)

No `LLM_PROVIDER` global enum. Provider is derived from the model name:

| Model string | Provider | Example |
|---|---|---|
| Contains `/` | OpenRouter | `anthropic/claude-opus-4-7`, `openai/gpt-4o`, `x-ai/grok-3`, `google/gemini-2.5-pro` |
| No `/` | Ollama (localhost) | `qwen3:4b`, `llama3.2:3b` |

Default model: `OPENROUTER_MODEL` env var if `OPENROUTER_API_KEY` is set, else `OLLAMA_MODEL`.

---

## Model Capability Detection

For each model, on first use, the server probes whether caching and thinking can be specified.
Results are cached in memory (`_model_capabilities: dict[str, dict]`) so probing happens once
per model per server lifetime. Known capabilities for common models are pre-seeded to skip the
probe entirely (with a fallback to probing for unknown models).

### Caching probe
Send a minimal request with `cache_control: {type: ephemeral}` on the system message
(+ `anthropic-beta: prompt-caching-2024-07-31` header). If the API accepts it → caching
supported. If error → caching unsupported, mark and never try again.

### Thinking probe
Send a minimal request with `thinking: {type: enabled, budget_tokens: 256}` (+ appropriate
beta header for Anthropic). If accepted → thinking supported.

### Graceful fallback loop
```
attempt_call(model, messages, config=full):
    try:
        return _do_call(model, messages, use_cache=True, use_thinking=True)
    except UnsupportedFeatureError as e:
        if "cache" in e → mark model caps[caching]=False, retry without cache
        if "thinking" in e → mark model caps[thinking]=False, retry without thinking
    → final attempt with just defaults
```

### Pre-seeded capability table (skip probing for known models)
```python
_KNOWN_CAPABILITIES = {
    # Anthropic via OpenRouter
    "anthropic/claude-opus-4-7":        {"caching": True,  "thinking": True},
    "anthropic/claude-sonnet-4-6":      {"caching": True,  "thinking": True},
    "anthropic/claude-haiku-4-5":       {"caching": True,  "thinking": False},
    # OpenAI via OpenRouter (automatic caching, no thinking config)
    "openai/gpt-4o":                    {"caching": False, "thinking": False},
    "openai/o3":                        {"caching": False, "thinking": False},
    "openai/o4-mini":                   {"caching": False, "thinking": False},
    # xAI
    "x-ai/grok-3":                      {"caching": False, "thinking": False},
    "x-ai/grok-3-mini":                 {"caching": False, "thinking": True},
    # Google via OpenRouter
    "google/gemini-2.5-pro":            {"caching": False, "thinking": False},
    "google/gemini-2.5-flash":          {"caching": False, "thinking": False},
    # Ollama (local)
    "qwen3:4b":                         {"caching": False, "thinking": True},
}
```
Unknown models: probe at first use, cache the result.

---

## LLMPlayer Class Design

```python
class LLMPlayer:
    player_id: str
    game_id: str
    model: str              # e.g. "anthropic/claude-opus-4-7" or "qwen3:4b"
    provider: str           # "openrouter" | "ollama"

    session: list[dict]     # full messages[] history (role/content pairs)
    base_system: str        # original system prompt (for session recovery)
    strategy: str           # most recent strategy document (from per-gen update)
    last_generation: int    # last generation for which a strategy update was sent
    hand_shown_generation: int  # last generation where full hand was shown

    token_usage: dict       # {calls, input, output, cache_read, cache_write, thinking}
    summary_logged: bool    # guard: log summary only once

    # Methods
    def init_session(system, user, think) -> str
    def continue_session(user, max_output_tokens) -> str
    def recover_session(user) -> str
    def trim_session() -> None
    def log_token_summary() -> None
```

Registry (module-level):
```python
_player_registry: dict[str, LLMPlayer] = {}
_game_players: dict[str, list[str]] = {}  # game_id → [player_id, ...]

def register_player(player_id, game_id, model) -> LLMPlayer
def get_player(player_id) -> LLMPlayer | None
def get_or_create_player(player_id, game_id, state) -> LLMPlayer
```

---

## Token / Cost Tracking (per-player)

Each `LLMPlayer` accumulates:
- `input`, `output`, `cache_read`, `cache_write`, `thinking` token counts
- Running cost estimate using model pricing fetched from `GET /api/v1/models`

`log_game_token_summary(game_id)` iterates `_game_players[game_id]` and calls
`player.log_token_summary()` for each, then logs a per-game aggregate.

### Pricing fetch
```python
def _get_openrouter_pricing(model: str) -> tuple[float, float]:
    """Returns (input_usd_per_token, output_usd_per_token). Cached per model."""
```
Fetches `GET https://openrouter.ai/api/v1/models`, finds `data[].id == model`,
reads `pricing.prompt` and `pricing.completion`. Cached in `_pricing_cache`.

---

## New `/player/register` Endpoint

Called by the TM server (or play script) at game creation, once per AI player:

```
POST /player/register
{
  "player_id": "p1abc",
  "game_id":   "g123",
  "model":     "anthropic/claude-opus-4-7"
}
→ {"ok": true, "player_id": "p1abc", "model": "anthropic/claude-opus-4-7"}
```

If `player_id` is already registered (e.g., server restart recovery), updates the model.
If model is omitted, uses `OPENROUTER_MODEL` (default) or `OLLAMA_MODEL`.

---

## Environment Variables

Remove all `GEMINI_*` and `LLM_PROVIDER`. New vars:

| Var | Default | Description |
|-----|---------|-------------|
| `USE_LLM` | `false` | Enable LLM player (unchanged) |
| `OPENROUTER_API_KEY` | — | Required for OpenRouter models |
| `OPENROUTER_MODEL` | `anthropic/claude-opus-4-7` | Default model when not registered |
| `OPENROUTER_THINKING_BUDGET` | `512` | Thinking tokens (for models that support it) |
| `OPENROUTER_MAX_OUTPUT_TOKENS` | `350` | Cap on action response length |
| `OPENROUTER_MAX_TURNS` | `80` | Trim session after N message pairs |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama base URL (unchanged) |
| `OLLAMA_MODEL` | `qwen3:4b` | Default Ollama model (unchanged) |
| `OLLAMA_TIMEOUT` | `600` | Ollama timeout seconds (unchanged) |
| `LLM_DEBUG` | `false` | Log prompts and responses (unchanged) |

---

## Files Changed

| File | Change |
|------|--------|
| `tm_ai_server/llm_player.py` | Full rewrite: LLMPlayer class + registry + OpenRouter backend |
| `tm_ai_server/inference.py` | Pass `player_id` and `game_id` to `select_action_llm` |
| `tm_ai_server/main.py` | Add `POST /player/register` endpoint; pass `player_id`/`game_id` |
| `tm_ai_server/schemas.py` | Add `PlayerRegisterRequest`, `PlayerRegisterResponse` |
| `tm_ai_server/config.py` | Remove `GEMINI_*` constants, add `OPENROUTER_*` constants |
| `pyproject.toml` | `uv add openai`; remove `google-genai` |
| `tests/test_llm_player.py` | Update for class-based API; add capability probe tests |
| `CLAUDE.md` | Update env vars table, architecture section |
| `scripts/play_game.py` | Call `/player/register` before starting game |
| `scripts/play_4way.py` | New: launch 4-player game with 4 different models |

---

## Implementation Steps

---

### Step 1 — Git commit current changes

Commit all `llm_player.py` changes made in this session (suggestions 1–6 from game analysis):
deferral loop check, standard project annotations, heat/plant conversion warnings,
initial card buying guidance, opponent hand size, effective card cost display.

```bash
cd /home/pmunk/workspace/tm-ai/tm-ai-server
git add src/tm_ai_server/llm_player.py
git commit -m "feat: improve LLM action prompt — deferral loop check, SP annotations, resource hints"
```

---

### Step 2 — Add `openai` dependency, remove `google-genai`

```bash
cd /home/pmunk/workspace/tm-ai/tm-ai-server
uv add openai
uv remove google-genai
```

Verify `pyproject.toml` and `uv.lock` updated.

---

### Step 3 — Add config constants (`config.py`)

Remove `GEMINI_*` env reads. Add:
```python
OPENROUTER_API_KEY          = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL            = os.getenv("OPENROUTER_MODEL", "anthropic/claude-opus-4-7")
OPENROUTER_THINKING_BUDGET  = int(os.getenv("OPENROUTER_THINKING_BUDGET", "512"))
OPENROUTER_MAX_OUTPUT_TOKENS= int(os.getenv("OPENROUTER_MAX_OUTPUT_TOKENS", "350"))
OPENROUTER_MAX_TURNS        = int(os.getenv("OPENROUTER_MAX_TURNS", "80"))
```

---

### Step 4 — Update `schemas.py`

Add:
```python
class PlayerRegisterRequest(BaseModel):
    player_id: str
    game_id: str
    model: Optional[str] = None  # None → use OPENROUTER_MODEL default

class PlayerRegisterResponse(BaseModel):
    ok: bool
    player_id: str
    model: str
```

---

### Step 5 — Update `main.py`

**a)** Pass `player_id` and `game_id` through to `select_action`:
```python
@app.post("/move", response_model=MoveResponse)
async def move(request: MoveRequest):
    ...
    input_response, debug_info = select_action(
        state, waiting_for, game_spec,
        game_id=request.game_id,
        player_id=request.player_id,
        last_error=request.last_error,
    )
```

**b)** Add `/player/register` endpoint:
```python
@app.post("/player/register", response_model=PlayerRegisterResponse)
async def player_register(body: PlayerRegisterRequest):
    from .llm_player import register_player
    player = register_player(body.player_id, body.game_id, body.model)
    return PlayerRegisterResponse(ok=True, player_id=player.player_id, model=player.model)
```

**c)** Update `/game-done` to iterate all registered players for the game:
```python
@app.post("/game-done")
async def game_done(body: dict):
    game_id = body.get("game_id", "")
    from .llm_player import log_game_token_summary
    log_game_token_summary(game_id)
    return {"ok": True, "game_id": game_id}
```
(Internally `log_game_token_summary` will now call each player's own summary logger.)

---

### Step 6 — Update `inference.py`

Add `game_id` and `player_id` params to `select_action` and forward them to `select_action_llm`:

```python
def select_action(
    state, waiting_for, game_spec=None, model=None, last_error=None,
    game_id=None, player_id=None,   # ← new
) -> tuple[dict, dict]:
    if os.getenv("USE_LLM", "false").lower() == "true":
        from .llm_player import select_action_llm
        return select_action_llm(
            state, waiting_for,
            game_id=game_id, player_id=player_id,
            last_error=last_error,
        )
    ...
```

`select_advice` already receives `game_id` and `player_id` — no change needed there.

---

### Step 7 — Rewrite `llm_player.py` (main work)

The file is restructured into these sections (order reflects file layout):

#### 7a. Module-level state (slim)

```python
_player_registry: dict[str, LLMPlayer] = {}
_game_players: dict[str, list[str]] = {}       # game_id → [player_id]
_model_capabilities: dict[str, dict] = {}       # model → {caching, thinking}
_pricing_cache: dict[str, tuple[float, float]] = {}   # model → (in_price, out_price)
_openrouter_client = None                       # lazy-init openai.OpenAI instance
```

#### 7b. Provider helpers

```python
def _provider_for(model: str) -> str:
    """'openrouter' if model contains '/', else 'ollama'."""
    return "openrouter" if "/" in model else "ollama"

def _ensure_openrouter_client() -> None:
    """Lazy-init the openai.OpenAI client pointed at OpenRouter."""
    global _openrouter_client
    if _openrouter_client is None:
        import openai
        _openrouter_client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
```

#### 7c. Capability detection

```python
# Pre-seeded known capabilities (see table above)
_KNOWN_CAPABILITIES: dict[str, dict] = { ... }

def _get_model_capabilities(model: str) -> dict[str, bool]:
    """Returns {caching: bool, thinking: bool}. Cached after first probe."""
    if model in _model_capabilities:
        return _model_capabilities[model]
    if model in _KNOWN_CAPABILITIES:
        _model_capabilities[model] = _KNOWN_CAPABILITIES[model].copy()
        return _model_capabilities[model]
    # Unknown model: probe both features
    caps = {"caching": _probe_caching(model), "thinking": _probe_thinking(model)}
    _model_capabilities[model] = caps
    logger.info("Probed capabilities for %s: %s", model, caps)
    return caps

def _probe_caching(model: str) -> bool:
    """Probe whether model accepts cache_control. Returns True if supported."""
    _ensure_openrouter_client()
    try:
        _openrouter_client.chat.completions.create(
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
    """Probe whether model accepts thinking config."""
    _ensure_openrouter_client()
    try:
        _openrouter_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            extra_body={"thinking": {"type": "enabled", "budget_tokens": 256}},
            extra_headers={"anthropic-beta": "interleaved-thinking-2025-05-14"},
        )
        return True
    except Exception as e:
        logger.debug("Thinking probe for %s failed: %s", model, e)
        return False
```

#### 7d. Pricing

```python
def _get_openrouter_pricing(model: str) -> tuple[float, float]:
    """(input_usd_per_token, output_usd_per_token). Fetched once, cached."""
    if model in _pricing_cache:
        return _pricing_cache[model]
    try:
        import httpx
        r = httpx.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            timeout=10,
        )
        r.raise_for_status()
        for entry in r.json().get("data", []):
            mid = entry.get("id", "")
            pricing = entry.get("pricing", {})
            in_p  = float(pricing.get("prompt",     0)) if pricing.get("prompt")     else 0.0
            out_p = float(pricing.get("completion", 0)) if pricing.get("completion") else 0.0
            _pricing_cache[mid] = (in_p, out_p)
        return _pricing_cache.get(model, (0.0, 0.0))
    except Exception as e:
        logger.warning("OpenRouter pricing fetch failed: %s", e)
        return (0.0, 0.0)
```

Note: `httpx` is likely already available (FastAPI dependency). If not, use `requests`.

#### 7e. Retry logic (generic)

```python
def _is_transient_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("503", "unavailable", "429", "rate limit",
                                   "overloaded", "resource exhausted", "timeout"))

def _parse_retry_delay(exc: Exception) -> float | None:
    m = re.search(r"retry.?after[:\s]+(\d+\.?\d*)", str(exc), re.IGNORECASE)
    return float(m.group(1)) if m else None

def _with_retry(fn, attempts=3, base_delay=5.0):
    """Generic exponential back-off retry for transient errors."""
    last_exc = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if attempt < attempts - 1 and _is_transient_error(exc):
                msg = str(exc)
                if "429" in msg or "resource exhausted" in msg.lower():
                    api_delay = _parse_retry_delay(exc)
                    wait = (api_delay + 2.0) if api_delay else 62.0
                else:
                    wait = base_delay * (2 ** attempt)
                logger.warning("Transient error (attempt %d/%d): %s — retrying in %.0fs",
                               attempt + 1, attempts, exc, wait)
                time.sleep(wait)
                last_exc = exc
            else:
                raise
    raise last_exc
```

#### 7f. LLMPlayer class

```python
class LLMPlayer:
    def __init__(self, player_id: str, game_id: str, model: str):
        self.player_id  = player_id
        self.game_id    = game_id
        self.model      = model
        self.provider   = _provider_for(model)

        # Session
        self.session: list[dict] = []
        self.base_system: str = ""
        self.strategy: str = ""
        self.last_generation: int = -1
        self.hand_shown_generation: int = -1

        # Token usage
        self.token_usage: dict = {
            "calls": 0, "input": 0, "output": 0,
            "cache_read": 0, "cache_write": 0, "thinking": 0,
        }
        self.summary_logged: bool = False
```

**`init_session(system, user, think=True) -> str`**
1. Store `self.base_system = system`
2. Resolve capabilities: `caps = _get_model_capabilities(self.model)`
3. Build system message:
   - If `caps["caching"]`: wrap as `[{"type":"text","text":system,"cache_control":{"type":"ephemeral"}}]`
   - Else: plain string
4. Build messages: `[{role:system, content:sys_msg}, {role:user, content:user}]`
5. Call `_call_openrouter(self, messages, think=think)` or `_call_ollama(self, messages, think=think)`
6. Append `{role:assistant, content:response}` to `self.session`
7. Return response text

**`continue_session(user, max_output_tokens=None) -> str`**
1. If `self.session` is empty → `return self.recover_session(user)`
2. Append `{role:user, content:user}` to `self.session`
3. Call `_call_openrouter(self, self.session, max_output_tokens=max_output_tokens)`
   or `_call_ollama(self, self.session)`
4. Append assistant response
5. Return response text

**`recover_session(user) -> str`**
- Reconstruct minimal session: system + strategy summary + user message
- Same logic as current `_session_recovery` but operating on `self`

**`trim_session() -> None`**
- If `len(self.session) > OPENROUTER_MAX_TURNS * 2`:
  1. Keep `self.session[0]` (system message)
  2. Keep last 40 message pairs
  3. Insert a strategy-summary fake pair at position 1 (as current code does)

**`_accum(call_type, in_tok, out_tok, cache_read=0, cache_write=0, thinking=0)`**
- Add to `self.token_usage`
- Compute running cost estimate and log

**`log_token_summary()`**
- Log per-player summary: player_id, model, call counts, tokens, cost

#### 7g. OpenRouter call function

```python
def _call_openrouter(
    player: LLMPlayer,
    messages: list[dict],
    think: bool = False,
    max_output_tokens: int | None = None,
) -> str:
    _ensure_openrouter_client()
    caps = _get_model_capabilities(player.model)

    kwargs = {
        "model": player.model,
        "messages": messages,
    }
    if max_output_tokens:
        kwargs["max_tokens"] = max_output_tokens

    extra_headers = {}
    extra_body = {}

    if think and caps["thinking"]:
        budget = OPENROUTER_THINKING_BUDGET
        extra_body["thinking"] = {"type": "enabled", "budget_tokens": budget}
        extra_headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"

    if caps["caching"]:
        extra_headers["anthropic-beta"] = extra_headers.get("anthropic-beta",
                                          "prompt-caching-2024-07-31")

    if extra_headers:
        kwargs["extra_headers"] = extra_headers
    if extra_body:
        kwargs["extra_body"] = extra_body

    def _do():
        return _openrouter_client.chat.completions.create(**kwargs)

    # Graceful capability fallback loop
    for attempt in range(3):
        try:
            response = _with_retry(_do)
            break
        except Exception as e:
            err = str(e).lower()
            if "cache" in err and caps.get("caching"):
                caps["caching"] = False
                _model_capabilities[player.model]["caching"] = False
                # Rebuild system message without cache_control
                messages = _strip_cache_control(messages)
                kwargs["messages"] = messages
                logger.info("Disabled caching for %s after error", player.model)
                continue
            if "thinking" in err and caps.get("thinking"):
                caps["thinking"] = False
                _model_capabilities[player.model]["thinking"] = False
                extra_body.pop("thinking", None)
                kwargs["extra_body"] = extra_body
                logger.info("Disabled thinking for %s after error", player.model)
                continue
            raise

    # Extract text (strip thinking blocks for Anthropic models)
    text = _extract_text(response)

    # Token accounting
    usage = response.usage
    in_tok  = getattr(usage, "prompt_tokens",     0) or 0
    out_tok = getattr(usage, "completion_tokens", 0) or 0
    cache_read  = getattr(usage, "prompt_tokens_details", None)
    cache_write = 0
    if cache_read is not None:
        cache_read  = getattr(cache_read, "cached_tokens", 0) or 0
    else:
        # Try Anthropic-style fields
        cache_read  = getattr(usage, "cache_read_input_tokens",     0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    thinking_tok = 0  # thinking tokens not separately reported by OpenRouter yet
    player._accum("call", in_tok, out_tok, cache_read, cache_write, thinking_tok)

    return text
```

#### 7h. Text extraction (handle thinking blocks)

```python
def _extract_text(response) -> str:
    """Extract text content from completion, stripping thinking blocks."""
    choice = response.choices[0]
    msg = choice.message
    # Standard: message.content is a string
    if isinstance(msg.content, str):
        return msg.content
    # Anthropic extended thinking: content is a list of blocks
    if isinstance(msg.content, list):
        parts = [b.text for b in msg.content
                 if hasattr(b, "type") and b.type == "text" and hasattr(b, "text")]
        return "\n".join(parts)
    return str(msg.content or "")
```

#### 7i. Ollama call (minimal changes from current)

```python
def _call_ollama(
    player: LLMPlayer,
    messages: list[dict],
    think: bool = True,
    max_output_tokens: int | None = None,
) -> str:
    payload = {
        "model": player.model,
        "stream": False,
        "messages": messages,
    }
    if player.model.startswith("qwen3") or _get_model_capabilities(player.model).get("thinking"):
        payload["think"] = think
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    text: str = data["message"]["content"]
    player._accum("call", data.get("prompt_eval_count", 0), data.get("eval_count", 0))
    return text
```

#### 7j. Registry functions

```python
def register_player(player_id: str, game_id: str, model: str | None = None) -> LLMPlayer:
    if model is None:
        model = OPENROUTER_API_KEY and OPENROUTER_MODEL or OLLAMA_MODEL
    player = LLMPlayer(player_id=player_id, game_id=game_id, model=model)
    _player_registry[player_id] = player
    _game_players.setdefault(game_id, [])
    if player_id not in _game_players[game_id]:
        _game_players[game_id].append(player_id)
    logger.info("Registered player %s (game=%s model=%s)", player_id, game_id, model)
    return player

def get_or_create_player(player_id: str, game_id: str, state: dict) -> LLMPlayer:
    if player_id not in _player_registry:
        logger.warning("Player %s not pre-registered — creating with default model", player_id)
        register_player(player_id, game_id)
    return _player_registry[player_id]

def log_game_token_summary(game_id: str) -> None:
    player_ids = _game_players.get(game_id, [])
    totals = {"calls": 0, "input": 0, "output": 0, "cost": 0.0}
    for pid in player_ids:
        player = _player_registry.get(pid)
        if player:
            player.log_token_summary()
            totals["calls"] += player.token_usage["calls"]
            totals["input"] += player.token_usage["input"]
            totals["output"] += player.token_usage["output"]
    logger.info(
        "TOKEN SUMMARY game=%s | players=%d | total calls=%d in=%d out=%d",
        game_id, len(player_ids), totals["calls"], totals["input"], totals["output"],
    )
```

#### 7k. Entry points (minimal changes)

```python
def select_action_llm(
    state: dict, waiting_for: dict,
    game_id: str, player_id: str,
    last_error: str | None = None,
) -> tuple[dict, dict]:
    player = get_or_create_player(player_id, game_id, state)
    wf_type = waiting_for.get("type")
    if wf_type in SETUP_TYPES:
        return _select_setup(state, waiting_for, player)
    return _select_action(state, waiting_for, player, last_error=last_error)
```

All internal prompt functions (`_select_setup`, `_select_action`, `_build_action_prompt`,
`_build_setup_prompt`, `_per_generation_strategy_update`, etc.) are updated to accept
`player: LLMPlayer` instead of `game_id: str`, and call `player.init_session` /
`player.continue_session` instead of `_call_llm_init` / `_call_llm_continue`.

All other logic (TM_RULES, option parsing, response parsing, payment correction, tile
placement tips, deferral loop prompt) is **unchanged**.

#### 7l. Trainer (advise) sessions

Trainer sessions use a separate `LLMPlayer` instance with a different model or the same model.
Session key: `f"trainer:{player_id}"`. The trainer player is registered at first advise call
if not already present:

```python
def select_action_advise(state, waiting_for, game_id, player_id, user_question=None):
    trainer_key = f"trainer:{player_id}"
    if trainer_key not in _player_registry:
        base_player = _player_registry.get(player_id)
        model = base_player.model if base_player else OPENROUTER_MODEL
        register_player(trainer_key, game_id, model)
    trainer = _player_registry[trainer_key]
    ...
```

---

### Step 8 — Update tests (`tests/test_llm_player.py`)

Changes required:
- Replace any use of `_game_sessions`, `_game_chat_sessions` with `_player_registry`
- Replace `select_action_llm(state, wf)` calls with `select_action_llm(state, wf, game_id="g1", player_id="p1")`
- Add test: `register_player("p1", "g1", "anthropic/claude-opus-4-7")` creates LLMPlayer with correct provider
- Add test: capability probe result is cached (probe not called twice for same model)
- Add test: caching failure falls back and marks model capabilities correctly
- Add test: `log_game_token_summary("g1")` calls `log_token_summary` on each registered player

Run: `uv run pytest tests/ -v` — all 34 tests must pass plus new tests.

---

### Step 9 — Update `scripts/play_game.py` and add `scripts/play_4way.py`

**`play_game.py`**: after creating the game and getting player IDs from the TM server,
call `POST /player/register` for each AI player before starting the game loop.

**`scripts/play_4way.py`** (new script):
```python
# Creates a 4-player game where each player uses a different model.
MODELS = [
    "anthropic/claude-opus-4-7",   # player 1 (red)
    "openai/gpt-4o",               # player 2 (yellow)
    "x-ai/grok-3",                 # player 3 (green)
    "google/gemini-2.5-pro",       # player 4 (blue)
]
# 1. POST /api/game/new with 4 AI players
# 2. For each player_id returned, POST /player/register with corresponding model
# 3. Run game loop (poll /api/ai/step or let server drive via requestAiMove)
```

---

### Step 10 — Update `CLAUDE.md`

- Update env vars table (remove `GEMINI_*`, add `OPENROUTER_*`)
- Update LLM Player section (class-based, registry, provider resolution)
- Update Running the Stack examples

---

## Open Items (verify before/during implementation)

1. **`httpx` availability**: FastAPI pulls in `httpx` transitively (via `httpx` as a starlette dep).
   If not available, use `requests` (already in the project) for the pricing fetch.

2. **Thinking block extraction**: OpenRouter's response format for thinking blocks may differ
   slightly between Anthropic and xAI Grok-mini. Verify the response structure against the
   OpenRouter docs and adjust `_extract_text()` accordingly.

3. **`anthropic-beta` header values**: The exact beta header strings for prompt caching
   (`prompt-caching-2024-07-31`) and thinking (`interleaved-thinking-2025-05-14`) should be
   confirmed against current OpenRouter/Anthropic docs at implementation time — they change.

4. **TM server: 4-player all-AI**: verify `ApiCreateGame.ts` allows all 4 player slots to have
   `isAI=true`. If restricted, a small server-side change is needed.

5. **Trainer model**: decide whether trainer (advise endpoint) always uses the same model as the
   game player, or gets its own `OPENROUTER_TRAINER_MODEL` env var. Lean toward same model.

---

## Estimated Effort

| Step | Est. |
|------|------|
| 1 Git commit | 5 min |
| 2–4 Deps, config, schemas | 20 min |
| 5–6 main.py + inference.py | 30 min |
| 7a–7e Module globals, providers, capabilities, pricing, retry | 2h |
| 7f–7h LLMPlayer class, OpenRouter call, text extraction | 2h |
| 7i–7k Ollama, registry, entry points | 1h |
| 7l Trainer sessions | 30 min |
| 7m Prompt functions: swap game_id → player param | 1h |
| 8 Tests | 1h |
| 9 Scripts | 45 min |
| 10 CLAUDE.md | 20 min |
| **Total** | **~9h** |
