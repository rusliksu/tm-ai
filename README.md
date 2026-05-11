# TM AI — Terraforming Mars AI Agent

A local AI server that plays Terraforming Mars alongside a custom fork of the open-source TM server.

## Architecture

```
terraforming-mars/  (Node.js, TypeScript, branch: feat/ai-player)
    └── AI player calls POST /move on the AI server for every decision
tm-ai/              (this repo — Python, FastAPI, PyTorch)
    └── Responds with a valid InputResponse for the current PlayerInputModel
```

The TM server sends the full decision tree (`PlayerInputModel`) to the AI server at each turn. The AI server returns a raw `InputResponse` that the game processes directly.

## Quick Start

### AI Server

```bash
cd tm-ai-server
uv run uvicorn tm_ai_server.main:app --reload --host 0.0.0.0 --port 8000
```

Without a trained model checkpoint the server uses a random-valid-action policy (last `SelectOption` in `OrOptions`, or first available option otherwise). This is sufficient to play full games.

### TM Server with AI Player

```bash
cd ../terraforming-mars
npm start
```

Create a game at `http://localhost:8080`, enable "AI player?" for one or more players. The TM server calls the AI server at `AI_SERVER_URL` (default `http://localhost:8000`).

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `AI_SERVER_URL` | `http://localhost:8000` | URL the TM server calls for AI moves |
| `AI_TIMEOUT_MS` | `5000` | Timeout for AI server requests |
| `AI_TRAINING_LOG_DIR` | `ai_training_logs` | Where the TM server writes JSONL training logs |
| `MODEL_PATH` | `models/checkpoint_latest.pt` | Path to the trained model checkpoint |
| `PORT` | `8000` | AI server port |

## Training Pipeline

### Phase 1 — Supervised Learning

Play games with human + AI players. The TM server automatically writes per-game JSONL logs (`logs/training/{game_id}.jsonl`) with every decision by every player.

Train the policy/value network:

```bash
cd tm-ai-server
uv run python -m tm_ai_server.training.train_supervised \
    --data-dir logs/training \
    --output-dir models \
    --epochs 50
```

Output: `models/checkpoint_best.pt` and `models/checkpoint_latest.pt`.

The server loads the checkpoint automatically on startup (set `MODEL_PATH` env var to override the path).

### Phase 2 — PPO Self-Play (cloud GPU)

Requires implementing the `/api/ai/new-game` and `/api/ai/step` endpoints on the TM server (see `specs/TM-adaption.md`).

```bash
cd tm-ai-server
uv add sb3-contrib
uv run python -m tm_ai_server.training.train_ppo \
    --checkpoint models/checkpoint_best.pt \
    --output-dir models \
    --total-steps 10_000_000
```

## Development

```bash
cd tm-ai-server
uv run pytest tests/ -v          # run tests
uv run pytest tests/test_encoding.py -v   # run one module
```

See `CLAUDE.md` for full architecture details, command reference, and remaining work.

## Specifications

- `specs/TM-AI.md` — AI server: schemas, model architecture, training pipeline
- `specs/TM-adaption.md` — TM server integration: API contract, Plan B logging, Plan A remaining work
