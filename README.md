# TM-AI — Terraforming Mars AI Agent

A two-repo project that plays Terraforming Mars: a Python AI server makes the moves, and a custom fork of the open-source TM server (`bafolts/terraforming-mars`) drives the game.

The agent can play in three modes:
1. **Neural net** — a trained policy/value network (supervised pretrain + PPO self-play).
2. **LLM** — any OpenRouter model (cloud, e.g. Claude/GPT/Gemini/DeepSeek) or Ollama (local). Reads the game state and picks moves with chain-of-thought. The provider is derived from the model name: `vendor/model` → OpenRouter, `bare:tag` → Ollama.
3. **AI Trainer** — same LLM, but it *coaches a human* via a chat sidebar instead of playing.

## Repo layout

```
~/workspace/
  tm-ai/                    # this repo (Python, FastAPI, PyTorch)
    tm-ai-server/           # AI server (port 8000); /move, /advise, /health
    specs/                  # TM-AI.md + TM-adaption.md (canonical specs)
    logs/training/          # Plan B per-turn JSONL (human games only)
    logs/llm-test/          # archived LLM-play game logs for analysis
    models/                 # checkpoint_best.pt, checkpoint_latest.pt
    data/card_db.json       # 970-card DB extracted from the TM server
    ai-model-design.md      # Option C design rationale (blog source)

  terraforming-mars/        # sibling repo, branch feat/ai-player
                            # (Node.js TM server fork with AI integration)
```

Both repos must be cloned side-by-side. The TM server expects to call `http://localhost:8000` for moves; the AI server reads card data from `../terraforming-mars` via a regenerable JSON.

## Setup — first time

```bash
# 1. Clone both repos
git clone <tm-ai-fork>          ~/workspace/tm-ai
git clone <tm-fork>             ~/workspace/terraforming-mars
cd ~/workspace/terraforming-mars && git checkout feat/ai-player

# 2. AI server (Python via uv)
cd ~/workspace/tm-ai/tm-ai-server
uv sync                          # installs torch, fastapi, sb3-contrib, openai…
uv run pytest tests/             # 46 tests, ~1s

# 3. TM server (Node 20+)
cd ~/workspace/terraforming-mars
npm install
npm run build:server

# 4. Env vars
cp ~/workspace/tm-ai/.env.example ~/workspace/tm-ai/.env  # then edit
source ~/workspace/tm-ai/.env                              # never read .env directly
```

`.env` is the only place secrets live. The TM server reads `AI_TRAINING_LOG_DIR` (where Plan B JSONLs land) and `AI_SERVER_URL`. The AI server reads `OPENROUTER_API_KEY`, `MODEL_PATH`, etc.

## Running the stack

Three TM-server / AI-server combos, depending on what you want:

### A. Play vs. a trained neural net

```bash
# Terminal 1 — AI server (neural net)
cd ~/workspace/tm-ai/tm-ai-server
MODEL_PATH=../models/checkpoint_best.pt \
  uv run uvicorn tm_ai_server.main:app --host 0.0.0.0 --port 8000

# Terminal 2 — TM server
cd ~/workspace/terraforming-mars
node build/src/server/server.js
```

Open `http://localhost:8080`, create a game, tick "AI player?" for the bot. The TM server calls `POST /move` for every decision.

### D. Run a multi-LLM death match (4 models, OpenRouter)

```bash
source ~/workspace/tm-ai/.env   # must have OPENROUTER_API_KEY

cd ~/workspace/tm-ai
./start.sh --death-match        # starts AI server (OpenRouter) + TM server + 4-player game
# Opens the spectator URL in Chrome; follow the game in real time.
# Logs: /tmp/ai-server.log, /tmp/tm-server.log, /tmp/death-match.log

# Override the default lineup
DEATH_MATCH_MODELS="anthropic/claude-sonnet-4-6,openai/gpt-4o-mini,google/gemini-2.5-flash,deepseek/deepseek-chat" \
  ./start.sh --death-match

# Stop everything when done
./stop.sh
```

Default lineup: Claude Sonnet 4-6, GPT-4o-mini, Gemini 2.5 Flash, DeepSeek Chat. See `logs/llm-test/` for archived game logs and analysis.

### B. Play vs. a single LLM (OpenRouter or Ollama)

The provider is chosen by the model name — no `LLM_PROVIDER` flag.

```bash
source ~/workspace/tm-ai/.env

# OpenRouter (cloud; needs OPENROUTER_API_KEY). Any vendor/model id works,
# e.g. anthropic/claude-sonnet-4-6, openai/gpt-4o-mini, google/gemini-2.5-flash-lite.
cd ~/workspace/tm-ai/tm-ai-server
USE_LLM=true OPENROUTER_MODEL=google/gemini-2.5-flash-lite LLM_DEBUG=true \
  uv run uvicorn tm_ai_server.main:app --host 0.0.0.0 --port 8000

# Ollama (local, free; needs `ollama serve` running with the model pulled)
USE_LLM=true OLLAMA_MODEL=qwen3:4b LLM_DEBUG=true \
  uv run uvicorn tm_ai_server.main:app --host 0.0.0.0 --port 8000
```

### C. Human play with AI Trainer coaching

Start any AI server with `USE_LLM=true` (B above), then open a game in the browser. Click the floating 🤖 button (bottom-right) on a player's page to open the coaching sidebar — auto-fetches advice on each new decision, accepts follow-up questions, has a "Play Recommendation" button. Each player has their own toggle and isolated trainer session (namespace `trainer:<game_id>:<player_id>`).

No game-creation checkbox to set; the toggle is per-player and per-game, persisted in `localStorage`.

## Training pipeline

```bash
# Phase 1 — supervised pretrain from Plan B JSONLs
cd ~/workspace/tm-ai/tm-ai-server
uv run python -m tm_ai_server.training.train_supervised \
  --data-dir ../logs/training --output-dir ../models --epochs 50

# Phase 2 — PPO self-play (needs TM server running)
uv run python -m tm_ai_server.training.train_ppo \
  --checkpoint ../models/checkpoint_best.pt \
  --output-dir ../models --log-dir ../logs/selfplay \
  --total-steps 5_000_000 --checkpoint-interval 100
```

PPO self-play games persist to the TM-server DB only — no JSONL is written. To produce training-data JSONLs from those games later, run `export_training_data.ts` against the DB (see TM-adaption.md). PPO run artifacts (`manifest.json`, `metrics.jsonl`, model checkpoints) live in `logs/selfplay/<run_id>/` / `models/selfplay/<run_id>/`.

For cloud GPU training, see `specs/TM-AI.md` Phase 2 — Vast.ai 3090 spot (~$0.20/h) or GCE Spot T4 (~$0.50/h); 5M-step run is ~5h / ~€2–3 per attempt.

## Re-exporting training data from the TM server DB

```bash
cd ~/workspace/terraforming-mars
npx tsx src/server/tools/export_training_data.ts ~/workspace/tm-ai/logs/training
```

The tool skips any game whose JSONL already exists in the output dir, so re-runs are idempotent and fast. After the May-2026 DB purge, 62 human games / 8235 turns export cleanly.

## Development

```bash
# AI server
cd ~/workspace/tm-ai/tm-ai-server
uv run pytest tests/                                         # 46 tests, ~1s
uv run pytest tests/test_encoding.py::test_flatten_or_options -v

# TM server
cd ~/workspace/terraforming-mars
npm run build:server                                         # tsc + tsc-alias
npm run build:client                                         # webpack (slow)
```

Claude Code permissions for both repos live in `tm-ai/.claude/settings.json` (gitignored — local to your machine, not committed). The canonical specs (`specs/TM-AI.md`, `specs/TM-adaption.md`) live in this repo only — there's no copy under `terraforming-mars/`.

---

# Sibling repo: `terraforming-mars` (TM server fork)

A fork of `bafolts/terraforming-mars` on branch `feat/ai-player`. Adds the AI-integration points listed below; the rest of the codebase is upstream and inherits its GPLv3 license.

## Key integration files (relative to `terraforming-mars/src/`)

| File | What it adds |
|---|---|
| `server/Player.ts` | `isAI` flag, `requestAiMove()` with retry+last_error feedback, `_aiMoveInProgress` guard, per-turn `logTrainingTurn()` (skipped for self-play games) |
| `server/Game.ts` | `isSelfPlay` field, `writeResult()` at game end (skipped for self-play) |
| `server/IGame.ts` | `isSelfPlay: boolean` in the interface |
| `server/ai/stateMapping.ts` | `buildAiRequestState()` — full state payload incl. cardsInHand, recentLog (opponents-only filter), boardSpaces, milestones/awards, gameVariants |
| `server/ai/TrainingLogger.ts` | writeMeta / appendTurn / writeResult; per-game JSONL into `AI_TRAINING_LOG_DIR` |
| `server/ai/AiClient.ts` | HTTP client to AI server (`/move`, `/advise`); `MoveRequestPayload` includes optional `last_error` |
| `server/routes/ApiAiSelfPlay.ts` | `POST /api/ai/new-game` + `/api/ai/step` for PPO self-play; response includes `spectator_id` |
| `server/routes/ApiAiAdvice.ts` | `POST /api/ai/advice` + `/api/ai/play-recommendation` for the AI Trainer; per-player, no game-wide gate |
| `server/tools/export_training_data.ts` | Replays the engine on DB saves to produce Plan-B-format JSONLs; skip-if-exists for idempotent re-runs |
| `server/tools/extract_card_db.ts` | Walks the card renderer to extract all 970 cards into `tm-ai/data/card_db.json` |
| `client/components/PlayerHome.vue` | Per-player 🤖 trainer toggle, hotkey isolation in chat input |
| `client/components/ai/AiTrainerChat.vue` | Coaching sidebar (chat + Play Recommendation) |

## Running just the TM server

```bash
cd ~/workspace/terraforming-mars
npm install
npm run build:server
node build/src/server/server.js >> /tmp/tm-server.log 2>&1 &
```

Default `http://localhost:8080`. SQLite DB at `db/game.db` (set `LOCAL_FS_DB` or `POSTGRES_HOST` to swap backends; see `src/server/database/Database.ts`).

## Environment variables (TM server)

| Variable | Default | Description |
|---|---|---|
| `AI_SERVER_URL` | `http://localhost:8000` | Where to send `/move` and `/advise` |
| `AI_TIMEOUT_MS` | `600000` | HTTP timeout for AI calls (10 min — LLMs can be slow) |
| `AI_TRAINING_LOG_DIR` | `ai_training_logs` | Per-game JSONL output dir (set this to `tm-ai/logs/training`) |
| `POSTGRES_HOST` | — | Use Postgres instead of SQLite if set |
| `LOCAL_FS_DB` | — | Use a flat-file DB if set |

## Regenerating the card database

The AI server depends on `~/workspace/tm-ai/data/card_db.json` for card descriptions, tags, and the vocabulary used by the (planned) embedding table. Regenerate after card content changes:

```bash
cd ~/workspace/terraforming-mars
npx tsx src/server/tools/extract_card_db.ts > ~/workspace/tm-ai/data/card_db.json
```

The tool traverses `CardRenderer.renderData` to pull descriptions for prelude / CEO / event cards that lack a plain `description` string.

## License

The TM server fork inherits GPLv3 from upstream. The AI server in `tm-ai/tm-ai-server/` is the same license unless otherwise noted.

---

## Specifications and design docs

- [`specs/TM-AI.md`](specs/TM-AI.md) — AI server: API contract, state encoding, model architecture (current + planned Option C), training pipeline, LLM provider comparison, cloud cost analysis
- [`specs/TM-adaption.md`](specs/TM-adaption.md) — TM-server integration: state mapping, Plan A/B training data, self-play API, AI Trainer routes
- [`ai-model-design.md`](ai-model-design.md) — Option C design rationale (card embeddings + spatial board + per-option scoring), tensor layouts, rejected alternatives. Blog-post source.
- [`CLAUDE.md`](CLAUDE.md) — operational quick reference for Claude Code sessions
- [`TODO.md`](TODO.md) — checklist by phase
