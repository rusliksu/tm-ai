# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Goal

Run **1–n LLM players** in a Terraforming Mars game — both UI-created games with AI players
and headless multi-LLM "death match" games. The AI server is OpenRouter-only and LLM-only.
There is **no** neural-net/PPO training, no supervised pipeline, no Ollama, and no in-game
AI Trainer (all removed in the 0.2 reimplementation).

## Two-Repo Architecture

- **`/home/pmunk/workspace/tm-ai`** (this repo) — Python FastAPI LLM AI server (`tm_llm`).
- **`/home/pmunk/workspace/terraforming-mars`** (sibling, branch `feat/ai-player`) —
  TypeScript/Node.js game server fork. Minimal adaptions from `main`: an `isAI` auto-move
  path, an AI state mapping, and a self-play driver API.

Claude Code permissions for both repos are in `.claude/settings.json`. Canonical specs live
in `specs/` in this repo only.

## Commands

All Python commands run from `tm-ai-server/` using `uv`:

```bash
# Run the AI server (OpenRouter — needs OPENROUTER_API_KEY in .env)
cd tm-ai-server && uv run uvicorn tm_llm.app:app --host 0.0.0.0 --port 8000

# Tests
cd tm-ai-server && uv run pytest tests/ -v
cd tm-ai-server && uv run pytest tests/test_board.py -v

# Add a dependency
cd tm-ai-server && uv add <package>
```

For the TM game server (`/home/pmunk/workspace/terraforming-mars`):

```bash
npm run build:server                                   # build TypeScript
node build/src/server/server.js >> /tmp/tm-server.log 2>&1 &   # start
# Regenerate the card DB consumed by the AI server:
npx tsx src/server/tools/extract_card_db.ts > ../tm-ai/data/card_db.json
```

## Project Structure

```
tm-ai-server/
  pyproject.toml            # uv-managed; deps: fastapi, uvicorn, pydantic, requests, openai
  main.py                   # shim: re-exports tm_llm.app for uvicorn
  src/tm_llm/
    app.py                  # FastAPI app — /health, /version, /move, /player/register, /game-done
    config.py               # env-var defaults (model, thinking budgets, max tokens, state dir)
    schemas.py              # Pydantic models matching TM server camelCase output
    options.py              # flatten_options / index_to_response / _default_response (decision tree)
    payment.py              # PAYMENT parse / correct / validate / auto-generate
    knowledge.py            # CARD_DB loader + board/expansion/variant context formatting
    board.py                # live board rendering + hex adjacency (matches TM's algorithm)
    openrouter.py           # OpenRouter client: single_shot, retry, truncation, pricing, cost
    prompts.py              # TM_RULES + STRATEGY_PRIMER + prompt builders + response parsers
    player.py               # LLMPlayer: strategy/tactical memory, token accounting, persistence
    registry.py             # player registry, get_or_create, token summaries, state persistence
    engine.py               # orchestration: setup / action (retry loop) / per-gen reflection
  tests/
    test_options.py         # decision-tree enumeration + InputResponse construction
    test_prompts.py         # CHOICE/PAYMENT/TACTICAL parsing + payment validation
    test_board.py           # hex adjacency (vs TM algorithm) + live-board rendering

data/card_db.json           # 970-card DB; regenerate via extract_card_db.ts (see Commands)
logs/llm-test/<session>/    # death-match session logs; _latest symlinks the newest
logs/llm-state/<player>.json # per-player durable state (strategy/tactical) across restarts
specs/                      # TM-AI.md, TM-adaption.md, LLM-state-persistence.md
scripts/play_game.py        # death-match driver (uses the self-play API)
start.sh / start-deathmatch.sh / stop.sh
```

## API Contract

The TM game server calls `POST /move` with:
- `state.game` — global state: `generation`, `oxygen`, `temperature`, `oceanCount`,
  `boardName`, `expansions`, `availableMilestones`, `availableAwards`, `gameVariants`,
  `recentLog` (serialized log since the start of the current generation, including the AI's
  **own** moves — the AI is stateless per turn, so this is how it learns what it did).
- `state.player` — active player: resources, production, tags, `cardsInHand`, `playedCards`,
  `cardResources` (per-card), `corporations`, `boardTiles`.
- `state.opponents` — all other players: same fields plus `handSize` (hand is secret).
- `state.board` — placed tiles (id, x, y, tileType, playerColor).
- `state.boardSpaces` — every hex (id, x, y, type, bonuses, current tile + owner).
- `state.milestones` / `state.awards` — claimed/funded.
- `state.waitingFor` — the `PlayerInputModel` decision tree.
- `last_error` — optional; set when the previous response was rejected so the AI can correct.

Response: `{input_response:{...}, debug:{...}}` — `input_response` goes to `player.process()`.

**Critical:** `OrOptions` response is `{type:"or", index:N, response:<InputResponse>}` — not
`responses:[{index:N}]`. See `options.py` docstring for all node types.

## Self-Play (death match) API

Two TM server endpoints drive headless all-LLM games (`isSelfPlay=true` suppresses the
auto-`requestAiMove` and DB-only-persists the game):

- `POST /api/ai/new-game` `{boardName, playerCount, playerNames}` →
  `{game_id, player_id, spectator_id, state, waitingFor, game_spec}`
- `POST /api/ai/step` `{game_id, player_id, input_response}` →
  `{done, player_id, state, waitingFor, result}`

`spectator_id` builds `http://localhost:8080/spectator?id=<spectator_id>` to watch.

## LLM player design (tm_llm)

**OpenRouter only.** Provider routing, capability probing, and Ollama are gone. Model
capabilities (caching/thinking) come from a static table in `openrouter.py` with safe
defaults for unknown models (caching for `anthropic/`, thinking on — disabled at runtime if
the provider rejects it). Pricing is prefetched once; `cost()` tracks per-player spend.

**Stateless action turns + two-part memory.** Each `/move` is one self-contained
`single_shot` of `[system, user]`:
- **system** (`prompts.build_action_system`) = `TM_RULES` (rules only) + `STRATEGY_PRIMER`
  (condensed) + game config + static board layout. Byte-identical every turn so prompt
  caching reuses it; cached on the player as `action_system`.
- **user** (`prompts.build_action_prompt`) = state header, conditional one-line advisories,
  live board / candidate-hex adjacency (from `board.py`), tableau, **compact** hand
  (1 line/card, descriptions capped), opponents, milestone & award status, recent log, the
  player's own memory, numbered options, payment section.

Two-part memory carried between turns (both on `LLMPlayer`):
- **`tactical`** (intra-generation) — the model rewrites it each turn via a `TACTICAL:` line;
  `prompts.capture_tactical` stores it.
- **`strategy`** (inter-generation) — engine plan + milestone/award targets + backup/switch.
  Refreshed only at a generation bump by `engine._maybe_per_generation_update` (one extra
  `single_shot` fed the prior strategy).

**Setup phase** (`initialCards`/`prelude`) — `engine._select_setup` does one `single_shot`
with `build_setup_system`; `parse_setup_response` extracts corp/buys/preludes/CEO + the
strategy doc.

**Validation retry loop** (`engine._select_action`, ≤2 retries): if the response lacks a
`CHOICE:` line or the PAYMENT is underfunded (`payment.check_payment_valid`), resend the full
prompt with an error banner. On exhaustion, fall back to Pass when the only error is a
missing CHOICE. `last_error` from the TM server is prepended as a rejection banner.

**Board awareness** (`board.py`): `render_live_board` shows placed tiles by owner on normal
turns; `render_space_choices` describes each offered candidate hex (placement bonus +
adjacent own/opponent cities, own greeneries, oceans) on `space` decisions, using adjacency
computed exactly the way TM's `Board.computeAdjacentSpaces` does — so the model never infers
hex geometry from raw (x,y).

**State persistence** (`registry.py`, see `specs/LLM-state-persistence.md`): on graceful
shutdown `save_all_active_players` writes each in-flight player's `{model, strategy, tactical,
last_generation, token_usage}` to `logs/llm-state/<player_id>.json`. On the next `/move`,
`get_or_create_player` lazily restores it (validating `game_id`). `/game-done` logs the token
summary and deletes the game's state files. `action_system` is not persisted (regenerated).

**Multi-model death match:** `POST /player/register` assigns a model per seat before the
game; `scripts/play_game.py --models "a/m1,b/m2,..."` registers one model per player and runs
a full game; `POST /game-done` flushes the per-player token/cost summary.

## TM fork integration points

Key files in `/home/pmunk/workspace/terraforming-mars/src/server/`:
- `Player.ts` — `isAI`; `setWaitingFor` triggers `requestAiMove` when `isAI && !isSelfPlay`;
  `requestAiMove(lastError?, retryCount)` retries up to 2× on `process()` failure (sending
  the error as `last_error`) then falls back to `aiFallbackResponse()`; `_aiMoveInProgress`
  guards re-entry.
- `Game.ts` / `IGame.ts` — `isSelfPlay` flag; set before `gotoInitialPhase()`.
- `ai/AiClient.ts` — HTTP client; `requestMove` only.
- `ai/stateMapping.ts` — `buildAiRequestState`: full state incl. `cardsInHand` (self only),
  `recentLog` (own + others, cap 25), `board`, `boardSpaces`, milestones/awards, variants.
- `routes/ApiAiSelfPlay.ts` — `/api/ai/new-game`, `/api/ai/step`.
- `tools/extract_card_db.ts` — regenerates `data/card_db.json`.

## Running the stack

**API keys live in `tm-ai/.env`. Always load via `source`, never read the file directly:**

```bash
source /home/pmunk/workspace/tm-ai/.env

# AI server + TM server (single model, default deepseek/deepseek-v4-flash)
./start.sh
OPENROUTER_MODEL=anthropic/claude-sonnet-4-6 ./start.sh

# Servers + a multi-LLM death match
./start-deathmatch.sh
DEATH_MATCH_MODELS="anthropic/claude-sonnet-4-6,openai/gpt-4o-mini,google/gemini-2.5-flash,deepseek/deepseek-chat" ./start-deathmatch.sh

# Stop everything (PIDs in /tmp/tm-ai.pids, /tmp/death-match.pids)
./stop.sh
./stop.sh --clean-db   # also removes self-play games from the TM SQLite DB

# Manual game (both servers up)
uv run python scripts/play_game.py --players 4 \
  --models "anthropic/claude-sonnet-4-6,openai/gpt-4o-mini,google/gemini-2.5-flash,deepseek/deepseek-chat"
# --board tharsis|hellas|elysium (default random); --verbose
```

Logs go to `./logs/llm-test/<session>/`; `_latest` symlinks the newest. The spectator URL is
written to `/tmp/current-game.url` (the death-match script tries to open it).

## Env vars (tm_llm/config.py)

| Var | Default | Description |
|-----|---------|-------------|
| `OPENROUTER_API_KEY` | — | Required for all model calls |
| `OPENROUTER_MODEL` | `anthropic/claude-opus-4-7` | Default model (per-player override via `/player/register`) |
| `OPENROUTER_THINKING` | `auto` | Global reasoning override: `auto` (per-model table), `off` (force off — faster), `on` (force on) |
| `OPENROUTER_THINKING_BUDGET` | `1024` | Thinking tokens for setup / per-gen reflection |
| `OPENROUTER_ACTION_THINKING_BUDGET` | `512` | Thinking tokens for action turns |
| `OPENROUTER_MAX_OUTPUT_TOKENS` | `4096` | Max output per action (must exceed thinking budget) |
| `LLM_DEBUG` | `false` | Log prompts/responses |
| `LLM_STATE_PERSIST` | `true` | Persist per-player state on graceful shutdown |
| `LLM_STATE_DIR` | `<repo>/logs/llm-state` | Per-player state files |
| `LLM_STATE_MAX_AGE_DAYS` | `7` | Prune state files older than this at startup |
