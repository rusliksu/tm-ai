# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Two-Repo Architecture

This project spans two repositories:

- **`/home/pmunk/workspace/tm-ai`** (this repo) — Python FastAPI AI server
- **`/home/pmunk/workspace/terraforming-mars`** (sibling repo, branch `feat/ai-player`) — TypeScript/Node.js game server fork

Claude Code permissions for both repos are in `.claude/settings.json`. The canonical spec files live in `specs/` in this repo only — `terraforming-mars` no longer has its own copy.

## Commands

All Python commands run from `tm-ai-server/` using `uv`:

```bash
# Run the AI server
cd tm-ai-server && MODEL_PATH=../models/checkpoint_best.pt uv run uvicorn tm_ai_server.main:app --host 0.0.0.0 --port 8000

# Run tests
cd tm-ai-server && uv run pytest tests/ -v

# Run a single test
cd tm-ai-server && uv run pytest tests/test_encoding.py::test_flatten_or_options -v

# Phase 1 supervised training
cd tm-ai-server && uv run python -m tm_ai_server.training.train_supervised \
    --data-dir ../logs/training --output-dir ../models --epochs 50

# Phase 2 PPO self-play training (TM server must be running)
cd tm-ai-server && uv run python -m tm_ai_server.training.train_ppo \
    --checkpoint ../models/checkpoint_best.pt \
    --output-dir ../models \
    --log-dir ../logs/selfplay \
    --total-steps 5_000_000 \
    --checkpoint-interval 100

# Add a dependency
cd tm-ai-server && uv add <package>
```

For the TM game server (`/home/pmunk/workspace/terraforming-mars`):

```bash
# Build TypeScript
npm run build:server

# Start TM server
node build/src/server/server.js >> /tmp/tm-server.log 2>&1 &

# Re-export historical training data
npx tsx src/server/tools/export_training_data.ts /home/pmunk/workspace/tm-ai/logs/training
```

## Project Structure

```
tm-ai-server/
  pyproject.toml              # uv-managed; src layout, setuptools build system
  main.py                     # Shim: adds src/ to path, re-exports for uvicorn
  src/tm_ai_server/
    main.py                   # FastAPI app — /move, /health, /version
    schemas.py                # Pydantic models matching TM server camelCase output
    config.py                 # STATE_DIM=492, 199-card vocab, normalisation caps
    model.py                  # PolicyValueNet (backbone + policy head + value head)
    encoding.py               # encode_state(), flatten_options(), index_to_response(), response_to_index()
    inference.py              # load_model(), select_action(); routes to LLM if USE_LLM=true
    llm_player.py             # LLM player (OpenRouter cloud or Ollama local): setup + action prompts + strategy doc + board/card context injection
    game_knowledge.py         # CARD_DB (970 cards), BOARD_INFO, EXPANSION_INFO; format_card_context/format_game_context
    training/
      dataset.py              # TMDataset: reads per-game JSONL logs from Plan B
      train_supervised.py     # Phase 1: cross-entropy policy + MSE value, checkpointing
      env_tm.py               # Phase 2: Gymnasium env wrapping /api/ai/new-game + /api/ai/step
      train_ppo.py            # Phase 2: MaskablePPO with full per-run logging
  tests/
    test_encoding.py          # 16 tests for encode_state, flatten_options, index_to_response
    test_schemas.py           # 4 tests for Pydantic schema parsing

data/
  card_db.json              # 970-card DB from TM server; regenerate: cd terraforming-mars && npx tsx src/server/tools/extract_card_db.ts > ../tm-ai/data/card_db.json
logs/training/                # Plan B JSONL files — live human/AI games
logs/selfplay/                # Self-play run directories: <run_id>/manifest.json, metrics.jsonl, game JSONLs
models/                       # Supervised checkpoints (checkpoint_best.pt, checkpoint_latest.pt)
models/selfplay/<run_id>/     # PPO checkpoints: checkpoint_<N>.pt, checkpoint_best.pt, checkpoint_latest.pt
specs/
  TM-AI.md                   # AI server spec: schemas, model arch, training pipeline
  TM-adaption.md             # TM server integration spec (canonical — only copy)
```

## API Contract

The TM game server calls `POST /move` with:
- `state.game` — global state: `generation`, `oxygen`, `temperature`, `oceanCount`, `boardName`, `expansions`, `availableMilestones`, `availableAwards`, `gameVariants`, `recentLog` (serialized game log entries since start of current generation — includes **the AI's own moves as well as opponents' and system messages**; the AI is stateless per-turn, so this log is how it learns what it did this generation)
- `state.player` — active player: resources, production, tags, `cardsInHand` (list of card names), `playedCards`, `cardResources` (per-card), `corporations`
- `state.opponents` — all other players: same fields plus `handSize` (count only — hand is secret)
- `state.board` — placed tiles (id, x, y, tileType, playerColor)
- `state.milestones` / `state.awards` — claimed/funded
- `state.waitingFor` — full `PlayerInputModel` decision tree
- `legal_actions[0]` — always `{action_id:"provide_input", payload:{input:<PlayerInputModel>}}`
- `last_error` — optional; set when the previous AI response was rejected by `player.process()`, so the AI can correct its choice

Response: `{input_response:{...}, debug:{...}}` — `input_response` goes directly to `player.process()`.

**Critical:** `OrOptions` response is `{type:"or", index:N, response:<InputResponse>}` — **not** `responses:[{index:N}]`. See `encoding.py` docstring for all types.

## Self-Play API (Phase 2)

Two new TM server endpoints for PPO training:

**`POST /api/ai/new-game`** — creates a self-play game (both AI, `isSelfPlay=true` suppresses auto `requestAiMove()` AND skips JSONL logging — game persists to DB only):
```json
{"boardName":"tharsis", "playerCount":2, "playerNames":["Claude","GPT"]}
→ {"game_id":"g...", "player_id":"p...", "spectator_id":"s...", "state":{...}, "waitingFor":{...}, "game_spec":{...}}
```
`spectator_id` allows constructing `http://localhost:8080/spectator?id=<spectator_id>` to watch the game in a browser.

**`POST /api/ai/step`** — applies one player's decision, returns next state:
```json
{"game_id":"g...", "player_id":"p...", "input_response":{...}}
→ {"done":false, "player_id":"p...", "state":{...}, "waitingFor":{...}, "result":null}
→ {"done":true, "player_id":null, "state":null, "waitingFor":null, "result":{...}}
```

`game.isSelfPlay=true` prevents `setWaitingFor()` from auto-triggering `requestAiMove()`.

## State Encoding

`STATE_DIM = 492`:
- Global (9): generation, temperature, oxygen, oceans, phase one-hot (5 classes)
- Self (230): 7 resources + 6 production + 13 tags + 1 handSize + 199 per-card resources + 1 played_count + 3 board_tiles
- Opponent (230): same as self (handSize visible to all players)
- Milestones/Awards (4): ms_self, ms_total, aw_self, aw_total
- Config (19): player_count + 5 board one-hot + 13 expansion flags

`cardResources` is sent as `{cardName: count}` per-card (199-card vocab in `config.py`).
Config dims are always zeroed (game_spec=None) for train/inference consistency.

## Training Data

**`logs/training/`** — supervised training data (human + AI live games). JSONL format:
- Line 1: `{type:"meta", game_spec:{...}, players:[...]}`
- Middle: `{type:"turn", state:{...}, waitingFor:{...}, input_response:{...}, is_human:bool}`
- Last: `{type:"result", endGeneration:N, playerResults:[...]}`

**`logs/selfplay/<run_id>/`** — PPO run artifacts (`manifest.json`, `metrics.jsonl`, model checkpoints). Per-game JSONLs are **not** written here anymore — self-play games persist to the TM server DB only; use `export_training_data.ts` to extract training data if/when needed.

## TM Server Integration Points

Key files in `/home/pmunk/workspace/terraforming-mars/src/server/`:
- `Player.ts` — isAI, setWaitingFor (Plan B capture + AI trigger), process (logs turn), requestAiMove with fallback
  - `_aiMoveInProgress` flag prevents infinite retry loop
  - `!game.isSelfPlay` guard prevents auto-trigger during self-play
  - `requestAiMove(lastError?, retryCount)` — retries up to 2× on `process()` failure, sending error message as `last_error` in next request; falls back to `aiFallbackResponse()` after max retries
  - After successful `process()`, re-triggers `requestAiMove()` if new `waitingFor` was set
- `Game.ts` — `isSelfPlay: boolean` field; set via `newInstance(..., isSelfPlay=true)` before `gotoInitialPhase()`; writeResult in gotoEndGame
- `IGame.ts` — `isSelfPlay: boolean` in interface
- `ai/AiClient.ts` — HTTP client to AI server; `MoveRequestPayload` includes optional `last_error?: string`
- `ai/stateMapping.ts` — full state; includes `cardsInHand` (self player only), `recentLog` (all moves + system messages since generation start — own moves now included, cap 25), `boardName`, `expansions`, `availableMilestones`, `availableAwards`, `gameVariants`; `cardResources` is per-card `{name: count}`
- `ai/TrainingLogger.ts` — writeMeta/appendTurn/writeResult; **skipped entirely for `game.isSelfPlay` games** in Player.process() and Game.gotoEndGame()
- `routes/ApiAiSelfPlay.ts` — `POST /api/ai/new-game` and `POST /api/ai/step`
- `routes/ApiAiAdvice.ts` — `POST /api/ai/advice` and `POST /api/ai/play-recommendation` (AI Trainer; opt-in per-player via UI toggle, no game-wide flag)
- `routes/ApiCreateGame.ts` — isAI flag, writeMeta at game creation
- `common/app/paths.ts` — `API_AI_NEW_GAME`, `API_AI_STEP`, `API_AI_ADVICE`, `API_AI_PLAY_RECOMMENDATION` path constants
- `tools/extract_card_db.ts` — extracts card DB including prelude/CEO descriptions via renderData traversal
- `client/components/ai/AiTrainerChat.vue` — coaching chat sidebar; auto-fetches advice on each new decision, shows Play Recommendation button
- `client/components/PlayerHome.vue` — per-player 🤖 toggle button + sidebar; navigatePage hotkey handler skips `<input>` / `<textarea>` / contentEditable so chat typing doesn't trigger page jumps

## Running the Stack

**API keys and env vars live in `tm-ai/.env`. Always load them with `source`, never read the file directly:**
```bash
source /home/pmunk/workspace/tm-ai/.env
```

```bash
# Load env vars first
source /home/pmunk/workspace/tm-ai/.env

# 1. Start AI server — neural net mode (default)
cd tm-ai-server && MODEL_PATH=../models/checkpoint_best.pt \
  uv run uvicorn tm_ai_server.main:app --host 0.0.0.0 --port 8000

# The provider is derived from the model name — no LLM_PROVIDER flag.
# "vendor/model" → OpenRouter (needs OPENROUTER_API_KEY); "bare:tag" → local Ollama.

# 1. Start AI server — Ollama/LLM mode (local, requires Ollama running)
cd tm-ai-server && USE_LLM=true OLLAMA_MODEL=qwen3:4b LLM_DEBUG=true \
  uv run uvicorn tm_ai_server.main:app --host 0.0.0.0 --port 8000 >> /tmp/ai-server.log 2>&1 &

# 1. Start AI server — OpenRouter mode (single model, cloud — requires OPENROUTER_API_KEY in .env)
#    Any vendor/model works, e.g. anthropic/claude-sonnet-4-6, google/gemini-2.5-flash-lite.
cd tm-ai-server && USE_LLM=true OPENROUTER_MODEL=google/gemini-2.5-flash-lite LLM_DEBUG=true \
  uv run uvicorn tm_ai_server.main:app --host 0.0.0.0 --port 8000 >> /tmp/ai-server.log 2>&1 &

# 2. Build and start TM server (from terraforming-mars/)
npm run build:server
node build/src/server/server.js >> /tmp/tm-server.log 2>&1 &

# 3. Open http://localhost:8080 and create a game with an AI player
# — or run PPO self-play training (see Commands above)
```

### Quick-start scripts (recommended)

```bash
# Start everything + run a 4-LLM death match (AI server on OpenRouter, TM server, play_game.py)
./start.sh --death-match

# Start AI server (OpenRouter, single model — default google/gemini-2.5-flash-lite) + TM server only
./start.sh

# Override default death-match lineup
DEATH_MATCH_MODELS="anthropic/claude-sonnet-4-6,openai/gpt-4o-mini,google/gemini-2.5-flash,deepseek/deepseek-chat" \
  ./start.sh --death-match

# Stop everything (kills PIDs tracked in /tmp/tm-ai.pids and /tmp/death-match.pids)
./stop.sh
./stop.sh --clean-db   # also removes all self-play games from the TM SQLite DB

# Run a multi-LLM game manually (both servers must be up)
uv run python scripts/play_game.py \
  --players 4 \
  --models "anthropic/claude-sonnet-4-6,openai/gpt-4o-mini,google/gemini-2.5-flash,deepseek/deepseek-chat"
# --board tharsis|hellas|elysium   (default: random)
# --verbose                         (print full state JSON each turn)
```

`start.sh` writes `PID` files to `/tmp/tm-ai.pids` (servers) and `/tmp/death-match.pids` (game loop). The game's spectator URL is written to `/tmp/current-game.url` once the game is created, and the script attempts to open it in Chrome/Chromium automatically.

## LLM Player (llm_player.py)

Supports two providers, selected by the **model name** (no `LLM_PROVIDER` flag): **OpenRouter** (cloud, multi-model — any `vendor/model` id, one player per model) and **Ollama** (local, free — a `bare:tag` model). `_provider_for(model)` does the routing. Env vars:

| Var | Default | Description |
|-----|---------|-------------|
| `USE_LLM` | `false` | Enable LLM player |
| `LLM_DEBUG` | `false` | Log prompts (`>` prefix) and responses (`<` prefix) |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama base URL |
| `OLLAMA_MODEL` | `qwen3:4b` | Ollama model tag (used when no OpenRouter key/model is set) |
| `OLLAMA_TIMEOUT` | `600` | Ollama timeout (s) |
| `OPENROUTER_API_KEY` | — | OpenRouter API key (required for OpenRouter models) |
| `OPENROUTER_MODEL` | `anthropic/claude-opus-4-7` | Default model for OpenRouter (per-player override via `POST /player/register`) |
| `OPENROUTER_THINKING_BUDGET` | `1024` | Thinking tokens for setup/prelude/per-gen reflection |
| `OPENROUTER_ACTION_THINKING_BUDGET` | `512` | Thinking tokens for tactical action turns (smaller = faster for reasoning models) |
| `OPENROUTER_MAX_OUTPUT_TOKENS` | `4096` | Max output tokens per action (must exceed the thinking budget or response is truncated before CHOICE) |
| `OPENROUTER_MAX_TURNS` | `44` | Trim session history at this many messages (keeps last 28 + strategy doc) |
| `LLM_STATE_PERSIST` | `true` | Persist per-player state to disk on graceful shutdown |
| `LLM_STATE_DIR` | `<repo>/logs/llm-state` | Where per-player state JSONs are written (one file per `player_id`) |
| `LLM_STATE_MAX_AGE_DAYS` | `7` | Files older than this are pruned at server startup |

Gemini, GPT, DeepSeek, Grok, etc. are reached **through OpenRouter** by model id (e.g. `google/gemini-2.5-flash-lite`); there is no longer a dedicated Gemini provider or `GEMINI_*` env var.

**Stateless-turn architecture (action phase)**: the action phase does **not** use a growing chat session. Each `/move` is a self-contained call (`LLMPlayer.single_shot`) of `[system, user]`:
- **system** = `TM_RULES` + game config + board layout, built once per game and stored in `player.action_system` (`_ensure_action_system`). It is byte-identical every turn so prompt-caching reuses it.
- **user** = full state snapshot (tableau, hand, all players, log) + the player's own two-part memory.

Because the TM server sends complete authoritative state every turn, no chat history is needed — the snapshot replaces it. The setup phase (`initialCards`/`prelude`) still uses the short multi-step session (`init_session`/`continue_session`); that session is simply not carried into the action phase.

**Two-part memory** (both fields on `LLMPlayer`, fed into every action prompt):
- **Part 1 — `player.tactical`** (intra-generation): a short ordered next-steps list. The model rewrites it inside each action response via a `TACTICAL:` line; `_capture_tactical()` parses and stores it; it's shown back next turn. No extra LLM call.
- **Part 2 — `player.strategy`** (inter-generation, coarse): engine plan, milestone/award targets, standing, **plus a maintained BACKUP plan and a SWITCH DECISION** (keep primary vs pivot). Refreshed only at generation bumps.

**Setup phase** (`initialCards`/`prelude`): `think=True` → chain-of-thought corp + card selection, writes a 150–250 word strategy document ending with a BACKUP PLAN line. Stored in `player.strategy`; `player.tactical` starts empty.

**Action phase prompt** shows:
- Current resources/production/tags
- `Your tableau (N cards in play):` from `playedCards` + per-card resources (authoritative — no longer relies on session memory)
- `Your hand (N cards):` with cost, tags, description for each card (full descriptions every turn — no cross-turn elision)
- Opponent resources/production/tags (incl. VP standing)
- `This generation's events so far:` from serialized game log — **now includes the AI's own moves**
- `=== YOUR MEMORY ===` block: STRATEGY (Part 2) + TACTICAL PLAN (Part 1)
- Numbered options with blue-card-action options annotated: `Use Soletta action — <desc>`
- Payment section for `projectCard`/`payment` decisions: AI specifies `PAYMENT: MC=N[, STEEL=N][, TITANIUM=N]...`
- `⚠ GREENERY OPPORTUNITY` reminder when plants ≥ 8 and O₂ is maxed (tiles still give +1 VP each)

**Internal retry loop** (`_select_action`): up to `_MAX_ACTION_RETRIES=2` retries within a single `/move`. Since turns are stateless, each retry **resends the full prompt** with the error banner prepended (not a session append). Before submitting, the server validates:
1. CHOICE line is present — error fed back if missing
2. Payment is sufficient (`_check_payment_valid()`) — feeds back exact resources available and required total
On exhaustion, falls back to the "Pass" option if the only error is a missing CHOICE line; otherwise sends best-effort.

**Payment validation**: `_check_payment_valid()` checks if the AI's PAYMENT covers `calculatedCost`. `_card_resource_values(card_name)` looks up CARD_DB tags to determine whether steel (building-tagged cards, ×2 MC) or titanium (space-tagged, ×3 MC) apply — inapplicable resources are zeroed by `_correct_payment()`. `_auto_payment_for_card()` generates an optimal steel/titanium payment when no PAYMENT line is present.

**External error feedback** (`last_error`): when the TM server rejects an `input_response` (returned as HTTP 400), `play_game.py` re-calls `/move` with `last_error` set. The AI server prepends `⚠ THE GAME SERVER REJECTED YOUR PREVIOUS MOVE: "..."` plus a server-authority reminder to the next LLM turn — applies to both action and setup phases.

**TM_RULES additions**: `RESPONSE FORMAT` section mandates ending with `CHOICE: N` + `PAYMENT: MC=N[, ...]`; `SERVER AUTHORITY` section instructs the model to read rejection errors and never repeat an invalid move.

**Session recovery**: only relevant to the setup phase (the action phase is stateless and self-contained, so a lost session never matters there). If a setup session is lost, `recover_session` re-initialises from the stored strategy.

**Per-generation strategy update**: at each generation bump, `_maybe_per_generation_update` makes a stateless `single_shot` call that is fed the **prior** `player.strategy` text explicitly (no conversation to scroll back through) and asks for a structured restate — standing, engine, milestone/award targets, next-gen priority, **BACKUP plan, and a SWITCH DECISION** (keep primary vs pivot to backup). The deferral-loop check compares the new priorities against the prior strategy shown in-prompt. Response stored in `player.strategy`.

**Think on every turn**: thinking budget applies to every action and per-gen reflection. `thinking` vs `reasoning` params are branched by provider: Anthropic uses `extra_body["thinking"]`; all other OpenRouter models use `extra_body["reasoning"]["max_tokens"]`.

**Provider routing**: for non-Anthropic/non-OpenAI models (DeepSeek, Gemini, xAI, etc.), every `_do_openrouter_call` includes `extra_body["provider"] = {"sort": "throughput", "require_parameters": True}`. This makes OpenRouter consistently pick the highest-throughput provider, enabling its sticky routing so DeepSeek/Gemini automatic prompt caching can warm up (without this, a different provider is selected each call = zero cache hits). `require_parameters` filters out providers that don't support the `reasoning` parameter.

**Capability fallback**: if a model rejects caching or thinking headers, the capability is permanently disabled for that model within the session and the call is retried. Transient 429/503 errors do NOT disable capabilities — they are re-raised for the retry wrapper.

**Pricing**: `_prefetch_pricing()` fetches all OpenRouter model prices at startup (once, thread-safe). `_KNOWN_PRICING` dict provides fallback rates for 9 common models when the API response ID doesn't match exactly (prevents $0 cost tracking for e.g. `anthropic/claude-sonnet-4-6`).

**Description elision**: for discard/keep/draft decisions (`wf_type == "card"`) where all cards are already in the hand block shown earlier *in the same prompt*, the option descriptions are replaced with `"(see hand above)"`. (This is intra-prompt only — cross-turn elision was removed with the stateless rewrite, since there is no session memory to rely on.)

**AI Trainer** (`select_action_advise`): per-player coaching via `POST /advise`. Opt-in per player from the UI toggle — no game-wide flag. Session namespace `trainer:<game_id>:<player_id>` isolates each player's session. Setup phases (`initialCards`, `prelude`) handled by `_select_setup_advise`. System prompt requires plain-text 1-3 sentence coaching plus a `<recommendation>` block; markdown is forbidden. Requires `USE_LLM=true`. Play Recommendation in `AiTrainerChat.vue` reloads the page on success.

**Multi-LLM death match**: `POST /player/register` assigns a model to a player before game start. `play_game.py --models "a/m1,b/m2,..."` registers one model per seat and runs a full game. `POST /game-done` flushes the per-player token/cost summary.

**State persistence across restarts**: on graceful shutdown (FastAPI lifespan `finally`), `save_all_active_players()` writes each in-flight LLM player's durable state (model, strategy, tactical plan, last_generation, token_usage) to `logs/llm-state/<player_id>.json` — one file per AI player. Trainer players and players whose game has already ended (in `_game_summary_logged`) are skipped. On the next startup, `prune_stale_state()` removes files older than `LLM_STATE_MAX_AGE_DAYS`. When TM server next calls `/move`, `get_or_create_player` lazily restores state from the matching JSON (validating `game_id` matches; stale files are deleted). `/game-done` deletes a game's state files after logging the token summary. Setup-phase chat sessions and `action_system` are **not** persisted — both regenerate naturally. Survives graceful shutdowns only; `kill -9` loses the in-memory state. See `specs/LLM-state-persistence.md` for the full design.

## Remaining Work

- **PPO self-play**: run `train_ppo.py` locally, monitor win rate in `metrics.jsonl`
- **Spatial board encoding**: add x/y grid tile positions to `encode_state()` (data already in request)
- **Phase 3 (cloud)**: Docker images + cloud deployment for GPU PPO training (Phase 7 in TODO.md)
