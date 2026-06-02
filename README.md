# TM-AI — Terraforming Mars LLM Agent

A two-repo project that plays Terraforming Mars with LLMs: a Python AI server makes the
moves, and a custom fork of the open-source TM server (`bafolts/terraforming-mars`) drives
the game.

Any OpenRouter model (Claude, GPT, Gemini, DeepSeek, Grok, …) can play. The agent reads the
full game state each turn and picks a move with chain-of-thought reasoning. It runs in two
contexts:

1. **UI game** — create a game in the browser and tick "AI player?" for a seat; the TM
   server calls the AI server for every decision.
2. **Death match** — a headless driver runs an all-LLM game (one model per seat) and a
   spectator URL lets you watch.

> The earlier neural-net/PPO training stack, supervised pipeline, Ollama provider, and
> in-game AI Trainer were removed in the 0.2 reimplementation. This is now LLM-only and
> OpenRouter-only.

## Repo layout

```
~/workspace/
  tm-ai/                    # this repo (Python, FastAPI)
    tm-ai-server/           # AI server (port 8000); /move, /player/register, /game-done, /health
      src/tm_llm/           # the LLM player package
    specs/                  # TM-AI.md + TM-adaption.md + LLM-state-persistence.md
    data/card_db.json       # 970-card DB extracted from the TM server
    logs/llm-test/          # death-match session logs (_latest → newest)
    logs/llm-state/         # per-player durable state across server restarts
    scripts/play_game.py    # death-match driver
    start.sh / start-deathmatch.sh / stop.sh

  terraforming-mars/        # sibling repo, branch feat/ai-player (Node.js TM server fork)
```

Both repos must be cloned side-by-side. The TM server calls `http://localhost:8000` for
moves; the AI server reads card data from `data/card_db.json` (regenerated from the TM repo).

## Setup — first time

```bash
# 1. Clone both repos side by side
git clone <tm-ai-fork>  ~/workspace/tm-ai
git clone <tm-fork>     ~/workspace/terraforming-mars
cd ~/workspace/terraforming-mars && git checkout feat/ai-player

# 2. AI server (Python via uv)
cd ~/workspace/tm-ai/tm-ai-server
uv sync                # installs fastapi, uvicorn, pydantic, requests, openai
uv run pytest tests/   # 19 tests, <1s

# 3. TM server (Node 20+)
cd ~/workspace/terraforming-mars
npm install
npm run build:server

# 4. Env vars
cp ~/workspace/tm-ai/.env.example ~/workspace/tm-ai/.env  # then edit
source ~/workspace/tm-ai/.env                              # never read .env directly
```

`.env` is the only place secrets live. The AI server needs `OPENROUTER_API_KEY`; the TM
server reads `AI_SERVER_URL` and `AI_TIMEOUT_MS`.

## Running the stack

### A. Play vs. an LLM in the browser

```bash
source ~/workspace/tm-ai/.env

# AI server (any OpenRouter model id)
cd ~/workspace/tm-ai/tm-ai-server
OPENROUTER_MODEL=anthropic/claude-sonnet-4-6 LLM_DEBUG=true \
  uv run uvicorn tm_llm.app:app --host 0.0.0.0 --port 8000

# TM server (separate terminal)
cd ~/workspace/terraforming-mars
node build/src/server/server.js
```

Open `http://localhost:8080`, create a game, tick "AI player?" for the bot. The TM server
calls `POST /move` for every decision.

### B. Multi-LLM death match

```bash
source ~/workspace/tm-ai/.env   # must have OPENROUTER_API_KEY

cd ~/workspace/tm-ai
./start-deathmatch.sh           # AI server + TM server + an all-LLM game; opens spectator URL

# Override the default lineup
DEATH_MATCH_MODELS="anthropic/claude-sonnet-4-6,openai/gpt-4o-mini,google/gemini-2.5-flash,deepseek/deepseek-chat" \
  ./start-deathmatch.sh

./stop.sh                       # stop everything
./stop.sh --clean-db            # also remove self-play games from the TM SQLite DB
```

Or start just the servers and drive a game manually:

```bash
./start.sh                                          # AI + TM servers (single default model)
uv run python scripts/play_game.py --players 4 \
  --models "anthropic/claude-sonnet-4-6,openai/gpt-4o-mini,google/gemini-2.5-flash,deepseek/deepseek-chat"
# --board tharsis|hellas|elysium (default random); --verbose for full state JSON
```

Logs land in `./logs/llm-test/<session>/` (`_latest` symlinks the newest). The spectator URL
is written to `/tmp/current-game.url`. A per-player token/cost summary is logged at game end
via `POST /game-done`.

## How the LLM player works

- **OpenRouter only**, one model per player (assigned via `POST /player/register`). Model
  capabilities (prompt caching, extended thinking) come from a static table with safe
  defaults — no runtime probing.
- **Stateless turns + two-part memory.** Each `/move` is one self-contained completion:
  a cached system prompt (rules + condensed strategy primer + game config + board layout)
  plus a per-turn user prompt (current state, compact hand, live board, candidate-hex
  adjacency on placements, opponents, milestone/award status, log, and the player's own
  `STRATEGY` + `TACTICAL` notes). The player rewrites those notes each turn / each
  generation, so no chat history is kept.
- **Validation + retry.** Before submitting, the server checks the response has a `CHOICE:`
  line and an affordable `PAYMENT:`; if not, it resends with an error banner (≤2 retries).
  TM-server rejections come back as `last_error` and are surfaced to the model.
- **Durable state.** On graceful shutdown, each in-flight player's memory is saved to
  `logs/llm-state/<player_id>.json` and restored on the next `/move` (see
  `specs/LLM-state-persistence.md`).

## Development

```bash
# AI server
cd ~/workspace/tm-ai/tm-ai-server
uv run pytest tests/                                  # 19 tests
uv run pytest tests/test_board.py -v

# TM server
cd ~/workspace/terraforming-mars
npm run build:server                                  # tsc + tsc-alias
npm run build:client                                  # webpack (slow)
```

Claude Code permissions for both repos live in `tm-ai/.claude/settings.json` (gitignored).
The canonical specs (`specs/`) live in this repo only.

## Regenerating the card database

The AI server reads `data/card_db.json` for card descriptions and tags. Regenerate after
card content changes:

```bash
cd ~/workspace/terraforming-mars
npx tsx src/server/tools/extract_card_db.ts > ~/workspace/tm-ai/data/card_db.json
```

---

# Sibling repo: `terraforming-mars` (TM server fork)

A fork of `bafolts/terraforming-mars` on branch `feat/ai-player`. It adds the minimal AI
integration points below; the rest is upstream and inherits its GPLv3 license.

| File (under `terraforming-mars/src/`) | What it adds |
|---|---|
| `server/Player.ts` | `isAI` flag; `requestAiMove()` with retry + `last_error` feedback; `_aiMoveInProgress` guard; auto-trigger on new `waitingFor` |
| `server/Game.ts`, `server/IGame.ts` | `isSelfPlay` field (suppresses auto-move during driver games) |
| `server/ai/stateMapping.ts` | `buildAiRequestState()` — full state incl. `cardsInHand`, `recentLog`, `board`, `boardSpaces`, milestones/awards, variants |
| `server/ai/AiClient.ts` | HTTP client to the AI server (`/move`) |
| `server/routes/ApiAiSelfPlay.ts` | `POST /api/ai/new-game` + `/api/ai/step` (death-match driver); response includes `spectator_id` |
| `server/tools/extract_card_db.ts` | Extracts all 970 cards into `tm-ai/data/card_db.json` |

## Running just the TM server

```bash
cd ~/workspace/terraforming-mars
npm install && npm run build:server
node build/src/server/server.js >> /tmp/tm-server.log 2>&1 &
```

Default `http://localhost:8080`; SQLite DB at `db/game.db`.

## Environment variables (TM server)

| Variable | Default | Description |
|---|---|---|
| `AI_SERVER_URL` | `http://localhost:8000` | Where to send `/move` |
| `AI_TIMEOUT_MS` | `600000` | HTTP timeout for AI calls (10 min — LLMs can be slow) |
| `POSTGRES_HOST` / `LOCAL_FS_DB` | — | Swap the DB backend if set |

## License

The TM server fork inherits GPLv3 from upstream. The AI server in `tm-ai/tm-ai-server/` is
the same license unless otherwise noted.

---

## Specifications

- [`specs/TM-AI.md`](specs/TM-AI.md) — AI server: API contract, prompt design, LLM player.
- [`specs/TM-adaption.md`](specs/TM-adaption.md) — TM-server integration: state mapping, self-play API.
- [`specs/LLM-state-persistence.md`](specs/LLM-state-persistence.md) — durable per-player state.
- [`CLAUDE.md`](CLAUDE.md) — operational quick reference for Claude Code sessions.
