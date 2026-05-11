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
cd tm-ai-server && uv run uvicorn tm_ai_server.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
cd tm-ai-server && uv run pytest tests/ -v

# Run a single test
cd tm-ai-server && uv run pytest tests/test_encoding.py::test_flatten_or_options -v

# Phase 1 supervised training
cd tm-ai-server && uv run python -m tm_ai_server.training.train_supervised \
    --data-dir logs/training --output-dir models --epochs 50

# Add a dependency
cd tm-ai-server && uv add <package>
```

For the TM game server (`/home/pmunk/workspace/terraforming-mars`):

```bash
# Build TypeScript
npm run build:server

# Re-export display logs from SQLite DB → logs/json/
npx tsx src/server/tools/export_all_logs.ts logs/json

# Copy exported logs to tm-ai
cp /home/pmunk/workspace/terraforming-mars/logs/json/*.json /home/pmunk/workspace/tm-ai/logs/json/
```

## Project Structure

```
tm-ai-server/
  pyproject.toml              # uv-managed; src layout, setuptools build system
  main.py                     # Shim: adds src/ to path, re-exports for uvicorn
  src/tm_ai_server/
    main.py                   # FastAPI app — /move, /health, /version
    schemas.py                # Pydantic models matching TM server camelCase output
    config.py                 # STATE_DIM=55, constants for phases/boards/tags/expansions
    model.py                  # PolicyValueNet (backbone + policy head + value head)
    encoding.py               # encode_state(), flatten_options(), index_to_response(), response_to_index()
    inference.py              # load_model(), select_action(); random fallback if no checkpoint
    training/
      dataset.py              # TMDataset: reads per-game JSONL logs from Plan B
      train_supervised.py     # Phase 1: cross-entropy policy + MSE value, checkpointing
      env_tm.py               # Phase 2: Gymnasium env (skeleton; needs TM server Phase-2 endpoints)
      train_ppo.py            # Phase 2: MaskablePPO (skeleton; needs sb3-contrib + env_tm)
  tests/
    test_encoding.py          # 16 tests for encode_state, flatten_options, index_to_response
    test_schemas.py           # 4 tests for Pydantic schema parsing

logs/json/                    # 82 exported game display logs (gitignored) — NOT training data
logs/training/                # Plan B JSONL files — written by TM server during live games
models/                       # Model checkpoints (checkpoint_best.pt, checkpoint_latest.pt)
specs/
  TM-AI.md                   # AI server spec: schemas, model arch, training pipeline
  TM-adaption.md             # TM server integration spec (canonical — only copy)
```

## API Contract

The TM game server calls `POST /move` with:
- `state.game` — global state (camelCase: `generation`, `oxygen`, `temperature`, `oceanCount`)
- `state.player` — active player resources/production/tags/playedCards
- `state.opponents` — all other players (same fields)
- `state.board` — placed tiles (spaceId, x, y, tileType, playerColor)
- `state.milestones` / `state.awards` — claimed/funded
- `state.waitingFor` — full `PlayerInputModel` decision tree
- `legal_actions[0]` — always `{action_id:"provide_input", payload:{input:<PlayerInputModel>}}`

Response: `{input_response:{...}, debug:{...}}` — `input_response` goes directly to `player.process()`.

**Critical:** `OrOptions` response is `{type:"or", index:N, response:<InputResponse>}` — **not** `responses:[{index:N}]`. See `encoding.py` docstring for all types.

## State Encoding

`STATE_DIM = 55` (essential features):
- Global (9): generation, temperature, oxygen, oceans, phase one-hot (5 classes)
- Player (27): 7 resources + 6 production + 13 tags + 1 handSize
- Config (19): player_count + 5 board one-hot + 13 expansion flags

Note: `state.opponents`, `state.board`, `state.milestones`, `state.awards` are now sent by the TM server but not yet encoded. Add them to `encoding.py` and update `STATE_DIM` when extending features.

## Training Data

**`logs/json/`** — display-message logs only, not usable for training.

**`logs/training/`** — real training data from Plan B (live games). JSONL format:
- Line 1: `{type:"meta", game_spec:{...}, players:[...]}` — written at game creation
- Middle lines: `{type:"turn", state:{...}, waitingFor:{...}, input_response:{...}, is_human:bool}`
- Last line: `{type:"result", endGeneration:N, playerResults:[...]}`

`dataset.py` reads this format and skips games missing meta or result lines.

## TM Server Integration Points

Key files in `/home/pmunk/workspace/terraforming-mars/src/server/`:
- `Player.ts` — isAI, setWaitingFor (captures pendingTrainingState), process (logs turn), requestAiMove with fallback, takeAction (saveBeforeTakingAction fixed)
- `Game.ts` — writeResult in gotoEndGame
- `ai/AiClient.ts` — HTTP client to AI server
- `ai/stateMapping.ts` — full state (player + opponents + board + milestones + awards + playedCards)
- `ai/TrainingLogger.ts` — writeMeta / appendTurn / writeResult; per-game JSONL
- `routes/ApiCreateGame.ts` — isAI flag, writeMeta at game creation

## Remaining Work

- **Plan A** (`export_training_data.ts`): re-run engine on 82 historical DB saves to extract training tuples without live games
- **Extend `encode_state()`**: add opponent features, board tile encoding, milestones/awards (data is already in the request)
- **Phase 2**: implement TM server endpoints `/api/ai/new-game` and `/api/ai/step` for Gymnasium env / PPO training
