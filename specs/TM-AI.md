## TM AI Server Specification

### Goal

A FastAPI server that picks Terraforming Mars moves for 1–n LLM players. The TM server fork
calls `POST /move` with the full game state and a decision tree; the AI server returns a
valid `InputResponse`. It is **LLM-only** and **OpenRouter-only** — no neural net, no
training, no Ollama, no in-game coach.

## Tech Stack

- Python 3.11+, managed with `uv`.
- FastAPI + uvicorn (HTTP), Pydantic (schemas).
- `openai` SDK pointed at OpenRouter (`https://openrouter.ai/api/v1`).
- `requests` (pricing prefetch).
- No torch / numpy / RL libraries.

## Project Structure

```
tm-ai-server/
  main.py                 # shim: re-exports tm_llm.app for uvicorn
  src/tm_llm/
    app.py                # FastAPI app + lifespan (prune on start, persist on stop)
    config.py             # env-var defaults
    schemas.py            # Pydantic request/response models
    options.py            # flatten_options / index_to_response / _default_response
    payment.py            # PAYMENT parse / correct / validate / auto-generate
    knowledge.py          # CARD_DB loader + config/board/expansion/variant context
    board.py              # live board rendering + hex adjacency (matches TM)
    openrouter.py         # client, capabilities, pricing, cost, retry, truncation
    prompts.py            # TM_RULES + STRATEGY_PRIMER + prompt builders + parsers
    player.py             # LLMPlayer: memory, token accounting, persistence
    registry.py           # registry, get_or_create, token summaries, state persistence
    engine.py             # orchestration: setup / action (retry) / per-gen reflection
  tests/                  # test_options, test_prompts, test_board (19 tests)
```

## API Contract

### `POST /move`

Request (`schemas.MoveRequest`):
```json
{
  "game_id": "g...", "player_id": "p...",
  "state": {
    "game": {"id","phase","generation","oxygen","temperature","oceanCount",
             "boardName","expansions","availableMilestones","availableAwards",
             "gameVariants","recentLog"},
    "player": {"...resources/production/tags...","cardsInHand","playedCards",
               "cardResources","corporations","boardTiles","victoryPoints"},
    "opponents": [{"...","handSize"}],
    "board": [{"id","x","y","tileType","playerColor"}],
    "boardSpaces": [{"id","x","y","t","b","v?","tile?","pc?"}],
    "milestones": [{"name","playerId"}], "awards": [{"name","playerId"}],
    "waitingFor": { /* PlayerInputModel decision tree */ }
  },
  "legal_actions": [...], "metadata": {"schema_version": 1},
  "last_error": "optional — set when the previous response was rejected"
}
```
Response: `{"input_response": { /* InputResponse */ }, "debug": {...}}`.
`input_response` is passed straight to `player.process()`.

**InputResponse wire format** (see `options.py`): `OrOptions` is
`{type:"or", index:N, response:<InputResponse>}` — not `responses:[{index:N}]`.
Other types: `option`, `card{cards:[...]}`, `projectCard{card,payment}`, `space{spaceId}`,
`amount{amount}`, `player{player}`, `colony{colonyName}`, `delegate{player}`,
`party{partyName}`, `resource{resourceType}`, `and{responses:[...]}`,
`initialCards{responses:[...]}`.

### `POST /player/register`
`{player_id, game_id, model?}` → `{ok, player_id, model}`. Assigns a model to a seat before
the game (one call per AI player). Without `model`, the default `OPENROUTER_MODEL` is used.

### `POST /game-done`
`{game_id}` → logs the per-player token/cost summary and deletes the game's state files.

### `GET /health` → `{status:"ok"}`.   `GET /version` → `{model_version, git_commit, config}`.

## Pydantic Schemas (`schemas.py`)

`MoveRequest`, `MoveRequestState`, `GameContext`, `PlayerContext`, `PlayerProduction`,
`LegalAction`, `Metadata`, `MoveResponse`, `PlayerRegisterRequest/Response`,
`HealthResponse`, `VersionResponse`. Field names mirror the TM server's camelCase output.

## LLM Player

### One `LLMPlayer` per AI player (`player.py`, keyed by `player_id` in `registry.py`)

State: `model`, `strategy` (coarse inter-generation memory + backup/switch), `tactical`
(intra-generation next steps), `action_system` (cached per-game system prompt, not
persisted), `last_generation`, `token_usage`.

### Stateless action turns

The action phase does not keep a chat session. Each `/move` is one `single_shot` of
`[system, user]` (`engine._select_action` → `player.single_shot` → `openrouter.single_shot`):

- **system** (`prompts.build_action_system`) = `TM_RULES` (rules) + `STRATEGY_PRIMER`
  (condensed strategy) + game config (`knowledge.format_config_context`) + static board
  layout (`knowledge.format_board_layout`). Byte-identical every turn → prompt-cached.
- **user** (`prompts.build_action_prompt`) = state header, conditional one-line advisories
  (heat/plant conversion, income gap, milestone-claim advisory), live board or candidate-hex
  adjacency (`board.py`), tableau, compact hand (`knowledge.format_card_context`, 1 line/card),
  opponents, milestone & award status, recent log, the player's `STRATEGY` + `TACTICAL`,
  numbered options, payment section.

### Two-part memory

- `tactical` — rewritten each turn via a `TACTICAL:` line (`prompts.capture_tactical`).
- `strategy` — refreshed only at a generation bump by `engine._maybe_per_generation_update`
  (one extra `single_shot` fed the prior strategy via `prompts.build_pergen_prompt`).

### Setup phase (`initialCards` / `prelude`)

`engine._select_setup` does one `single_shot` with `prompts.build_setup_system`;
`prompts.parse_setup_response` extracts corporation / buys / preludes / CEO and the strategy
doc (which becomes `player.strategy`).

### Validation + retry (`engine._select_action`)

Up to `MAX_ACTION_RETRIES` (2) within a single `/move`. Before submitting, the server checks
the response has a `CHOICE:` line and an affordable `PAYMENT:` (`payment.check_payment_valid`).
On failure it resends the full prompt with an error banner. On exhaustion it falls back to
Pass when the only problem is a missing CHOICE. A TM-server rejection (`last_error`) is
prepended as a banner on the next call.

### Payment (`payment.py`)

`parse_payment_line` reads `PAYMENT: MC=n[, STEEL=n][, TITANIUM=n][, HEAT=n]...`;
`correct_payment` clamps to available resources and the card's tag rules (steel only for
building tags, titanium only for space tags) and tops up MC to cover the cost;
`auto_payment_for_card` generates an optimal steel/titanium payment when no PAYMENT line is
present (e.g. nested `or→projectCard`).

### Board awareness (`board.py`)

`render_live_board` lists placed tiles grouped by owner (normal turns). `render_space_choices`
describes each offered candidate hex on a `space` decision — placement bonus + adjacent own
cities / own greeneries / opponent cities / oceans — using adjacency computed exactly the way
TM's `Board.computeAdjacentSpaces` does (pointy-top offset grid keyed on the middle row), so
the model never infers hex geometry from raw (x,y).

### OpenRouter client (`openrouter.py`)

`single_shot(model, system, user, *, think, thinking_budget, max_output_tokens) → (text, usage)`.
Capabilities (`caching`, `thinking`) come from a static table with safe defaults for unknown
models (caching for `anthropic/`, thinking on, disabled at runtime if rejected). Anthropic
uses `extra_body["thinking"]` + caching `cache_control`; others use `extra_body["reasoning"]`
and (for non-OpenAI) a pinned highest-throughput provider so sticky routing enables their
automatic caching. Handles transient-error retry (1s/5s/10s/30s…), truncation retry (widen
`max_tokens`, halve reasoning budget), and per-call capability fallback. Pricing is prefetched
once; `cost()` tracks per-player spend.

### State persistence (`registry.py`, see `LLM-state-persistence.md`)

On graceful shutdown `save_all_active_players` writes each in-flight player's
`{model, strategy, tactical, last_generation, token_usage}` to
`logs/llm-state/<player_id>.json`. On the next `/move`, `get_or_create_player` lazily restores
it (validating `game_id`; stale files deleted). `prune_stale_state` runs at startup;
`/game-done` clears a finished game's files. `action_system` is not persisted (regenerated).

### Env vars (`config.py`)

| Var | Default | Description |
|-----|---------|-------------|
| `OPENROUTER_API_KEY` | — | Required for all model calls |
| `OPENROUTER_MODEL` | `anthropic/claude-opus-4-7` | Default model (per-player override via `/player/register`) |
| `OPENROUTER_THINKING_BUDGET` | `1024` | Thinking tokens for setup / per-gen reflection |
| `OPENROUTER_ACTION_THINKING_BUDGET` | `512` | Thinking tokens for action turns |
| `OPENROUTER_MAX_OUTPUT_TOKENS` | `4096` | Max output per action (must exceed thinking budget) |
| `OPENROUTER_TRUNCATION_MAX` / `_RETRIES` | `16384` / `2` | Truncation-retry ceiling / attempts |
| `LLM_DEBUG` | `false` | Log prompts/responses |
| `LLM_STATE_PERSIST` | `true` | Persist per-player state on graceful shutdown |
| `LLM_STATE_DIR` | `<repo>/logs/llm-state` | Per-player state files |
| `LLM_STATE_MAX_AGE_DAYS` | `7` | Prune state files older than this at startup |

## Running

```bash
source ~/workspace/tm-ai/.env
cd tm-ai-server
OPENROUTER_MODEL=anthropic/claude-sonnet-4-6 \
  uv run uvicorn tm_llm.app:app --host 0.0.0.0 --port 8000

# Multi-model game: register each seat, then drive it
curl -X POST http://localhost:8000/player/register -H 'Content-Type: application/json' \
  -d '{"player_id":"p1","game_id":"g1","model":"anthropic/claude-opus-4-7"}'
```

## Design decisions

- **Stateless turns + persisted notes** beat a growing chat session: the TM server already
  sends complete authoritative state every turn, so the snapshot replaces history and the
  cached system prefix stays byte-identical for prompt caching.
- **Server-side board adjacency**: LLMs are unreliable at offset-hex geometry; the server has
  the data, so it describes neighbours instead.
- **Single-sourced data**: standard-project costs and milestone thresholds live once in
  `prompts.py` and drive both the rules prose and runtime annotations.
