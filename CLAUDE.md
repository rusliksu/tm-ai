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
    llm_player.py             # LLM player (Ollama local or Gemini cloud): setup + action prompts + strategy doc + board/card context injection
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
- `state.game` — global state: `generation`, `oxygen`, `temperature`, `oceanCount`, `boardName`, `expansions`, `availableMilestones`, `availableAwards`, `gameVariants`, `recentLog` (serialized game log entries since start of current generation)
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

**`POST /api/ai/new-game`** — creates a 2-player self-play game (both AI, `isSelfPlay=true` suppresses auto `requestAiMove()`):
```json
{"boardName":"tharsis", "logDir":"logs/selfplay/<run_id>"}
→ {"game_id":"g...", "player_id":"p...", "state":{...}, "waitingFor":{...}, "game_spec":{...}}
```

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

**`logs/selfplay/<run_id>/`** — self-play game JSONLs (same format, written by TrainingLogger with per-run `logDir`).

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
- `ai/stateMapping.ts` — full state; includes `cardsInHand` (self player only), `recentLog` (serialized game log since generation start), `boardName`, `expansions`, `availableMilestones`, `availableAwards`, `gameVariants`; `cardResources` is per-card `{name: count}`
- `ai/TrainingLogger.ts` — accepts optional `logDir` in constructor; writeMeta/appendTurn/writeResult
- `routes/ApiAiSelfPlay.ts` — `POST /api/ai/new-game` and `POST /api/ai/step`
- `routes/ApiCreateGame.ts` — isAI flag, writeMeta at game creation
- `common/app/paths.ts` — `API_AI_NEW_GAME` and `API_AI_STEP` path constants
- `tools/extract_card_db.ts` — extracts card DB including prelude/CEO descriptions via renderData traversal

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

# 1. Start AI server — Ollama/LLM mode (local, requires Ollama running)
cd tm-ai-server && USE_LLM=true LLM_PROVIDER=ollama OLLAMA_MODEL=qwen3:4b LLM_DEBUG=true \
  uv run uvicorn tm_ai_server.main:app --host 0.0.0.0 --port 8000 >> /tmp/ai-server.log 2>&1 &

# 1. Start AI server — Gemini mode (cloud, fast — get key at aistudio.google.com/apikey)
cd tm-ai-server && USE_LLM=true LLM_PROVIDER=gemini GEMINI_API_KEY=$GEMINI_API_KEY LLM_DEBUG=true \
  uv run uvicorn tm_ai_server.main:app --host 0.0.0.0 --port 8000 >> /tmp/ai-server.log 2>&1 &

# 2. Build and start TM server (from terraforming-mars/)
npm run build:server
node build/src/server/server.js >> /tmp/tm-server.log 2>&1 &

# 3. Open http://localhost:8080 and create a game with an AI player
# — or run PPO self-play training (see Commands above)
```

## LLM Player (llm_player.py)

Supports Ollama (local, free) and Gemini (cloud, fast). Select via `LLM_PROVIDER`. Env vars:

| Var | Default | Description |
|-----|---------|-------------|
| `USE_LLM` | `false` | Enable LLM player |
| `LLM_PROVIDER` | `ollama` | `ollama` or `gemini` |
| `LLM_DEBUG` | `false` | Log prompts (`>` prefix) and responses (`<` prefix) |
| `OLLAMA_MODEL` | `qwen3:4b` | Ollama model tag |
| `OLLAMA_TIMEOUT` | `600` | Ollama timeout (s) |
| `GEMINI_API_KEY` | — | Google AI Studio key (required for Gemini) |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model name |

**Session-per-game architecture**: TM rules + board/expansion context sent once at game start (`_call_llm_init`); all subsequent turns continue the same session (`_call_llm_continue`) — no rules repetition.

**Setup phase** (`initialCards`/`prelude`): `think=True` → chain-of-thought corp + card selection, writes a 100–200 word strategy document including a TABLEAU section (played cards + effects). Strategy stored per `game_id` in `_game_strategies`.

**Action phase**: `think=False` → fast direct answer. Strategy NOT re-requested each turn (saves tokens). Prompt shows:
- Current resources/production/tags
- `Your hand (N cards):` with cost, tags, description for each card
- Opponent resources/production/tags
- `Recent events:` from serialized game log (current generation)
- Numbered options with blue-card-action options annotated: `Use Soletta action — <desc>`
- Payment section for `projectCard`/`payment` decisions: AI specifies `PAYMENT: MC=N[, STEEL=N][, TITANIUM=N]...`

**Error feedback**: when `last_error` is set in the request, it is prepended as `⚠ Your previous response was rejected: "..."` so the AI can correct its choice or payment.

**Session recovery**: if a Gemini session is lost (503 exhausted, server restart), `_session_recovery` re-initialises a chat session using the stored strategy as context. Future turns continue it normally.

**Ollama context trim**: when Ollama session exceeds `MAX_SESSION_MESSAGES` (~30 turns), strategy (including TABLEAU) is captured via an extra API call, then the session is rebuilt as: original system + strategy reminder + last 40 messages.

**Gemini transient errors**: `_gemini_with_retry` retries on 503/429 with exponential backoff (5s, 10s, 3 attempts).

## Remaining Work

- **PPO self-play**: run `train_ppo.py` locally, monitor win rate in `metrics.jsonl`
- **Spatial board encoding**: add x/y grid tile positions to `encode_state()` (data already in request)
- **Phase 3 (cloud)**: Docker images + cloud deployment for GPU PPO training (Phase 7 in TODO.md)
