## TM AI Server Specification

### Goal

A local AI server that:
- Receives HTTP requests from the TM server when an AI player must decide
- Returns a valid `input_response` for the given `PlayerInput` decision tree
- Always returns a legal move (baseline behaviour, no trained model required)
- Trains from human game logs via supervised learning (Phase 1)
- Improves via self-play PPO on cloud GPUs (Phase 2)

---

## Tech Stack

| Layer | Choice |
|---|---|
| Language | Python 3.11+ |
| Web API | FastAPI + uvicorn |
| Deep Learning | PyTorch |
| Reinforcement Learning | Stable-Baselines3 (PPO) |
| Package manager | `uv` |
| Deployment | Docker — CPU (local), GPU (cloud) |

---

## Project Structure

```
tm-ai-server/
  pyproject.toml
  uv.lock
  src/
    tm_ai_server/
      __init__.py
      main.py           # FastAPI app and endpoints
      schemas.py        # Pydantic models for request/response
      encoding.py       # State JSON → feature tensors
      model.py          # PolicyValueNet (PyTorch)
      inference.py      # Model loading and action selection; routes to LLM if USE_LLM=true
    llm_player.py     # Ollama LLM player (setup + action phases, strategy document)
      config.py         # Hyperparameters and env var config
      training/
        __init__.py
        dataset.py      # Load and parse training logs
        train_supervised.py
        env_tm.py       # Gymnasium environment wrapping TM server HTTP
        train_ppo.py
  tests/
```

Run locally:
```bash
uv run uvicorn tm_ai_server.main:app --reload --host 0.0.0.0 --port 8000
```

---

## API Contract

Base URL: `http://localhost:8000` (via `AI_SERVER_URL` env var)
Timeout: 5000 ms (via `AI_TIMEOUT_MS` env var)
Content-Type: `application/json`

### `POST /move`

The TM server calls this when an AI player must make a decision. The full `PlayerInputModel` decision tree is provided; the AI server must return a valid `input_response` for it.

**Request body:**
```json
{
  "game_id": "g123...",
  "player_id": "p456...",
  "state": {
    "game": {
      "id": "g123...",
      "phase": "action",
      "generation": 7,
      "oxygen": 8,
      "temperature": -12,
      "oceanCount": 5
    },
    "player": {
      "id": "p456...",
      "name": "Alice",
      "color": "blue",
      "terraformRating": 42,
      "megacredits": 25,
      "steel": 3,
      "titanium": 1,
      "plants": 5,
      "energy": 2,
      "heat": 6,
      "handSize": 4,
      "production": {
        "megacredits": 4,
        "steel": 1,
        "titanium": 0,
        "plants": 2,
        "heat": 0,
        "energy": 1
      },
      "tags": {"science": 2, "building": 3, "space": 1},
      "isAI": true
    },
    "waitingFor": {"type": "or", "title": "Take action", "options": ["..."]}
  },
  "legal_actions": [
    {
      "action_id": "provide_input",
      "type": "or",
      "title": "Take action",
      "payload": {
        "input": {"type": "or", "title": "Take action", "options": ["..."]}
      }
    }
  ],
  "metadata": {"schema_version": 1}
}
```

`legal_actions` always has exactly one entry with `action_id: "provide_input"`. The `PlayerInputModel` decision tree is in `state.waitingFor` and also in `legal_actions[0].payload.input`.

**Response body:**
```json
{
  "input_response": {"type": "or", "index": 2, "response": {"type": "option"}},
  "debug": {
    "policy_logits": [1.2, -0.3, 0.8],
    "value_estimate": 0.35
  }
}
```

`input_response` is passed directly to `player.process()` on the TM server. `debug` is optional and ignored by the game.

**InputResponse wire format** (from `src/common/inputs/InputResponse.ts`):

| Type | Format |
|---|---|
| `OrOptions` | `{type:"or", index:N, response:<InputResponse>}` |
| `AndOptions` | `{type:"and", responses:[<InputResponse>, ...]}` |
| `SelectOption` | `{type:"option"}` |
| `SelectCard` | `{type:"card", cards:[<CardName>, ...]}` |
| `SelectProjectCardToPlay` | `{type:"projectCard", card:<CardName>, payment:{...}}` |
| `SelectSpace` | `{type:"space", spaceId:<SpaceId>}` |
| `SelectAmount` | `{type:"amount", amount:N}` |
| `SelectPlayer` | `{type:"player", player:<Color>}` |
| `SelectColony` | `{type:"colony", colonyName:<ColonyName>}` |
| `SelectDelegate` | `{type:"delegate", player:<Color>}` |
| `SelectParty` | `{type:"party", partyName:<PartyName>}` |

### `GET /health`
```json
{"status": "ok"}
```

### `GET /version`
```json
{
  "model_version": "0.1.0",
  "git_commit": "abc123def",
  "config": {"state_dim": 512, "hidden_sizes": [512, 512, 512]}
}
```

---

## Pydantic Schemas (`schemas.py`)

The current `schemas.py` in `tm-ai-server/` uses the wrong format (old snake_case structure with a nested `global_` field). It must be replaced with the following, which matches the actual TM server output:

```python
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class GameContext(BaseModel):
    id: str
    phase: str
    generation: int
    oxygen: int
    temperature: int
    oceanCount: int


class PlayerProduction(BaseModel):
    megacredits: int
    steel: int
    titanium: int
    plants: int
    heat: int
    energy: int


class PlayerContext(BaseModel):
    id: str
    name: str
    color: str
    terraformRating: int
    megacredits: int
    steel: int
    titanium: int
    plants: int
    energy: int
    heat: int
    handSize: int
    production: PlayerProduction
    tags: Dict[str, int] = {}
    isAI: bool = False


class MoveRequestState(BaseModel):
    game: GameContext
    player: PlayerContext
    waitingFor: Optional[Dict[str, Any]] = None


class LegalAction(BaseModel):
    action_id: str
    type: str
    title: str
    payload: Optional[Dict[str, Any]] = None


class Metadata(BaseModel):
    schema_version: int = 1


class MoveRequest(BaseModel):
    game_id: str
    player_id: str
    state: MoveRequestState
    legal_actions: List[LegalAction]
    metadata: Metadata


class MoveDebug(BaseModel):
    policy_logits: Optional[List[float]] = None
    value_estimate: Optional[float] = None


class MoveResponse(BaseModel):
    input_response: Dict[str, Any]
    debug: Optional[MoveDebug] = None


class HealthResponse(BaseModel):
    status: str = "ok"


class VersionResponse(BaseModel):
    model_version: str
    git_commit: str
    config: Dict[str, Any]
```

---

## State Encoding (`encoding.py`)

Converts the JSON state from the TM server into a fixed-size float32 feature vector.

### Essential features (~150–200 dims, implement first)

- **Global** (5): generation, temperature, oxygen, oceans, phase (one-hot over 5 phases)
- **Current player** (28): TR, megacredits, steel, titanium, plants, energy, heat (7 resources), production for all 6 resources, tags for all 13 tag types, handSize
- **Game config** (20+): player count, board one-hot (tharsis/hellas/elysium/...), enabled expansion flags (corpEra, venus, colonies, prelude, prelude2, turmoil, moon, pathfinders, ...)

### Extended features (add after baseline converges)

- Per opponent (×N players): TR, resources, production, tags, played card count
- Board state: ocean/city/greenery tile counts
- Milestones: which are claimed, by whom
- Awards: which are funded, by whom

### Action encoding

The `waitingFor` `PlayerInputModel` is a recursive tree. Flatten it to a canonical list of leaf options:
- Walk the tree depth-first, collect all leaf choices (e.g. each `OrOptions` option, each selectable card)
- Pad to a fixed maximum (128 options) with -∞ masking for invalid slots
- The AI outputs 128 logits; apply mask before softmax

This avoids a fixed global action space and handles the variable 2–100+ actions per decision.

---

## Model Architecture (`model.py`)

**`PolicyValueNet` (MLP):**

```python
import torch
import torch.nn as nn
from typing import Tuple


class PolicyValueNet(nn.Module):
    def __init__(self, state_dim: int, hidden_sizes: list[int], action_space_size: int):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(0.1)]
            in_dim = h
        self.backbone = nn.Sequential(*layers)
        self.policy_head = nn.Linear(in_dim, action_space_size)
        self.value_head = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        logits = self.policy_head(features)
        logits = logits.masked_fill(~mask, float('-inf'))
        value = self.value_head(features).squeeze(-1)
        return logits, value
```

Default config: `hidden_sizes=[512, 512, 512]`, `action_space_size=128`.

---

## Training Pipeline

### Phase 1 — Supervised learning from human games

- **Input**: `(state_vector, action_mask, chosen_action_index)` tuples from human decision logs
- **Policy loss**: cross-entropy between policy head logits and chosen action index
- **Value loss**: MSE between value head output and normalized final rank reward
- **Reward normalization**: `rank_reward = (num_players - rank) / (num_players - 1)` → range [0, 1]
- **Target**: 10–50 epochs, batch size 64–256, CPU sufficient

### Phase 2 — PPO self-play (cloud GPU)

The `env_tm.py` Gymnasium environment wraps the TM server over HTTP:
- `reset()`: POST to TM server to start a new game, return initial state
- `step(action)`: send `input_response`, receive next state + reward
- **Reward**: relative scoring = `(player_vp - mean_opponent_vp) / reference_vp`
  - Gives learning signal to all players, not just the winner
  - Switch to rank-based reward once agent stabilises

Stable-Baselines3 PPO hyperparameters (starting values):

| Parameter | Value |
|---|---|
| `learning_rate` | 3e-4 |
| `gamma` | 0.99 |
| `gae_lambda` | 0.95 |
| `clip_range` | 0.2 |
| `n_steps` | collect several thousand steps per update |

Initialise PPO from the supervised model checkpoint.

---

## Training Data

### Existing data

The 80 exported JSON files in `logs/json/` **contain only display log messages** (the game event log), not game states or actions. They are not usable for training.

The SQLite database (`db/game.db` in the TM server repo) contains the full `SerializedGame` JSON at every save point: 82 games, 11,192 saves. This is the source for training data.

**Export strategy — two parallel tracks:**

1. **Plan A (historical)**: Re-run the game engine on each DB save to recover the `waitingFor` decision tree. See `TM-adaption.md` for full analysis, risks, and the `export_training_data.ts` implementation plan.

2. **Plan B (ongoing)**: Hook `Player.setWaitingFor()` + `Player.process()` in the TM server to log all decisions (human and AI) in real time. Produces clean training tuples without any inference or diffing.

### Training log format consumed by `dataset.py`

One JSON file per game (written by Plan B or produced by the Plan A export tool):

```json
{
  "game_id": "g123...",
  "game_spec": {
    "board_name": "tharsis",
    "player_count": 2,
    "expansions": ["corpEra", "venus", "prelude"],
    "variants": {"draftVariant": true},
    "created_at": "2026-05-01T10:00:00.000Z"
  },
  "players": [
    {"playerId": "p1", "name": "Alice", "isAI": false},
    {"playerId": "p2", "name": "Bot", "isAI": true}
  ],
  "turns": [
    {
      "step": 0,
      "playerId": "p1",
      "generation": 3,
      "phase": "action",
      "state": {"game": {"..."}, "player": {"..."}},
      "waitingFor": {"type": "or", "title": "Take action", "options": ["..."]},
      "input_response": {"type": "or", "responses": [{"index": 1}]},
      "is_human": true
    }
  ],
  "final_result": {
    "endGeneration": 14,
    "playerResults": [
      {"playerId": "p1", "tr": 67, "vp_total": 95, "rank": 1},
      {"playerId": "p2", "tr": 55, "vp_total": 82, "rank": 2}
    ]
  }
}
```

`dataset.py` filters on `is_human: true` for supervised pretraining and uses `final_result` to compute per-player rewards.

---

## Deployment

### Local (CPU inference)

```bash
uv run uvicorn tm_ai_server.main:app --reload --host 0.0.0.0 --port 8000
```

### Cloud (RunPod / Vast / Synpix)

- CPU inference image: `pytorch/pytorch` + FastAPI server only
- GPU training image: add CUDA, stable-baselines3
- Entry points: `run_inference.sh`, `run_training.sh`
- Checkpoints written to cloud volume or S3-compatible storage after N million steps

### Model versioning

```
models/
  policy_value_v0.1.0.pt
  policy_value_v0.1.0.config.json   # state_dim, hidden_sizes, action_space_size
  checkpoint_latest.pt              # symlink to latest
```

Version format: `v<major>.<minor>.<patch>`. Bump minor on architecture changes, patch on weight-only updates. Validate config compatibility on load.

---

## LLM Player (`llm_player.py`)

An alternative to the trained neural net that uses an LLM for strategic decision-making. Activated via `USE_LLM=true`; `inference.py` routes to it before any NN logic. Supports two providers via `LLM_PROVIDER`:

- **`ollama`** (default) — local inference, no API cost, requires Ollama daemon running
- **`gemini`** — Google Gemini cloud API, fast (~1s/move), free tier available

### Architecture

```
Game start (initialCards / prelude)
    → configured LLM, think=True (chain-of-thought for opening decisions)
    → outputs: corporation/card selection + strategy document (100-200 words)

All subsequent decisions
    → configured LLM, think=False (fast direct answer)
    → strategy document passed as system context
    → outputs: action choice + optional strategy revision
```

Strategy documents are stored per `game_id` in `_game_strategies` dict for the server lifetime.

### Env vars

| Var | Default | Description |
|-----|---------|-------------|
| `USE_LLM` | `false` | Enable LLM player |
| `LLM_PROVIDER` | `ollama` | `ollama` or `gemini` |
| `LLM_DEBUG` | `false` | Log full prompts and raw responses |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server base URL |
| `OLLAMA_MODEL` | `qwen3:4b` | Ollama model tag |
| `OLLAMA_TIMEOUT` | `600` | Ollama request timeout (seconds) |
| `GEMINI_API_KEY` | _(required for Gemini)_ | Google AI Studio API key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model name |

### Think mode

Setup phase uses `think=True` (chain-of-thought for opening decisions); action phase uses `think=False` (direct fast answer).

- **Ollama**: passed as `"think": true/false` in `/api/chat` payload — effective on qwen3 family.
- **Gemini**: maps to `ThinkingConfig(thinking_budget=1024)` (on) or `thinking_budget=0` (off).

### Key functions in `llm_player.py`

- `select_action_llm(state, waiting_for)` — public entry point, routes to setup or action
- `_call_llm(system, user, think)` — provider router; logs prompts/response if `LLM_DEBUG`
- `_call_ollama(system, user, think)` — Ollama `/api/chat` backend
- `_call_gemini(system, user, think)` — Google Gemini backend (lazy-init client)
- `_select_setup(state, waiting_for, game_id)` — rich prompt for `initialCards`/`prelude`; injects board/expansion context
- `_select_action(state, waiting_for, strategy, game_id)` — compact action prompt with strategy; injects board context + card descriptions for card-selection decisions
- `_parse_setup_response(text, waiting_for, game_id)` — extracts CORPORATION/BUY_CARDS/STRATEGY
- `_parse_action_response(text, options, waiting_for, game_id)` — extracts CHOICE/STRATEGY_UPDATE
- `_extract_card_names(waiting_for)` — recursively collects card names from a `card`-type decision node

### Game Knowledge Database (`game_knowledge.py`)

A companion module that provides structured game context injected into LLM prompts.

**Sources:**
- `CARD_DB` — loaded at startup from `data/card_db.json` (970 cards). Generated by
  `npx tsx src/server/tools/extract_card_db.ts > tm-ai/data/card_db.json` in the TM server repo.
  Contains `{name, type, cost, tags, description, victoryPoints}` for every card.
- `BOARD_INFO` — hand-written descriptions of all 12 boards (special tiles, milestones, awards, strategic notes).
- `EXPANSION_INFO` — one-paragraph summaries of all 9 expansions.

**Key functions:**
- `format_card_context(card_names, header, max_cards=20)` — returns a compact block of card descriptions for injection into prompts when cards are being selected.
- `format_game_context(board_name, expansions)` — returns a `=== GAME CONFIGURATION ===` block with board special tiles, milestones, awards, notes, and active expansion summaries.

**Injection points in `llm_player.py`:**
- **Setup phase system prompt**: `format_game_context(board, expansions)` prepended so the AI knows which milestones/awards are in play.
- **Setup prompt user message**: each card (corporation, project, prelude, CEO) shown with its description and tags inline.
- **Action phase system prompt**: `format_game_context` repeated so board context is always present.
- **Action prompt user message**: when the decision involves selecting from cards (waitingFor.type == `"card"`), `format_card_context` injects descriptions of those specific cards.

**Board/expansion data source:** `stateMapping.ts` now includes `boardName` and `expansions[]` in the game context sent with every `/move` request. `schemas.py` exposes them as `GameContext.boardName` and `GameContext.expansions`.

### Ollama local models

Install: https://ollama.com — then `ollama pull <model>`.

| Model | RAM | Action time | Notes |
|-------|-----|-------------|-------|
| `qwen3:4b` | 2.5 GB | ~30–60s | **Recommended** — built-in think mode, free |
| `phi4-mini` | 4 GB | ~20–40s | Fast, strong reasoning |
| `gemma4:e4b` | 9.9 GB | ~90–120s | Larger, slower; needs 14 GB RAM total |

```bash
ollama pull qwen3:4b   # one-time download
```

### Gemini free tier setup

1. Go to **https://aistudio.google.com/apikey** → "Create API key" (no credit card needed)
2. Free limits for `gemini-2.5-flash`: **1,500 req/day**, 15 RPM
3. Set `GEMINI_API_KEY=<key>` when starting the server

### Running

```bash
# Ollama (local, free — Ollama daemon must be running)
cd tm-ai-server && USE_LLM=true LLM_PROVIDER=ollama OLLAMA_MODEL=qwen3:4b LLM_DEBUG=true \
  uv run uvicorn tm_ai_server.main:app --host 0.0.0.0 --port 8000 >> /tmp/ai-server.log 2>&1 &

# Gemini (cloud, ~1s/move, free tier — get key at aistudio.google.com/apikey)
cd tm-ai-server && USE_LLM=true LLM_PROVIDER=gemini GEMINI_API_KEY=<key> LLM_DEBUG=true \
  uv run uvicorn tm_ai_server.main:app --host 0.0.0.0 --port 8000 >> /tmp/ai-server.log 2>&1 &
```

---

## Cloud LLM Provider Comparison

Cost basis: 200 moves/game × 1,000 input + 100 output tokens = 200K input / 20K output per game.

| Provider / Model | Input $/1M | Output $/1M | TTFT | Speed | $/game | $/1K games | Free tier |
|---|---|---|---|---|---|---|---|
| Groq Llama 3.1 8B Instant | $0.05 | $0.08 | <0.3s | 660 t/s | $0.001 | $1.20 | 1K req/day |
| Groq Llama 4 Scout | $0.11 | $0.34 | <0.4s | 447 t/s | $0.003 | $2.90 | 1K req/day |
| Cerebras Llama 3.1 8B | $0.10 | $0.10 | <0.2s | 2,326 t/s | $0.002 | $2.20 | 1M tok/day |
| **Gemini 2.5 Flash-Lite** | **$0.10** | **$0.40** | **0.3–0.6s** | 393 t/s | **$0.003** | **$2.80** | 1.5K req/day |
| GPT-4.1-nano | $0.10 | $0.40 | ~0.9s | 80 t/s | $0.003 | $2.80 | none |
| DeepSeek V3 | $0.14 | $0.28 | ~1.0s | 100 t/s | $0.003 | $3.40 | 5M tok free |
| GPT-4o-mini | $0.15 | $0.60 | ~0.9s | 80 t/s | $0.004 | $4.20 | none |
| **Gemini 2.5 Flash** | **$0.30** | **$2.50** | **~0.6s** | 220 t/s | **$0.011** | **$11.00** | **1.5K req/day** |
| Groq Llama 3.3 70B | $0.59 | $0.79 | <0.5s | 276 t/s | $0.013 | $13.40 | 1K req/day |
| Cerebras Llama 3.3 70B | $0.60 | $0.60 | <0.3s | 1,800 t/s | $0.013 | $13.20 | 1M tok/day |
| Claude Haiku 4.5 | $1.00 | $5.00 | ~0.8s | 98 t/s | $0.030 | $30.00 | none |
| Claude Sonnet 4.6 | $3.00 | $15.00 | ~1.2s | 75 t/s | $0.090 | $90.00 | none |

**Recommendations by scenario:**
- **Interactive play (single game)**: Gemini 2.5 Flash free tier — 1,500 req/day, sub-1s responses, zero cost to start
- **PPO training at scale**: Groq Llama 4 Scout ($2.90/1K games) or Groq Llama 3.1 8B ($1.20/1K games)
- **Best quality/cost for training**: Gemini 2.5 Flash-Lite ($2.80/1K games, frontier quality)
- **Avoid**: reasoning models (o4-mini, DeepSeek-R1) — 5–30s/move; Claude Sonnet — 30–75× more expensive

### Gemini Free Tier Setup

1. Go to **https://aistudio.google.com/apikey** → create API key (no credit card required)
2. Free limits for `gemini-2.5-flash`: **1,500 req/day**, 15 RPM, 1M TPM
3. Start the AI server:
   ```bash
   cd tm-ai-server && USE_LLM=true LLM_PROVIDER=gemini \
     GEMINI_API_KEY=<your-key> LLM_DEBUG=true \
     uv run uvicorn tm_ai_server.main:app --host 0.0.0.0 --port 8000
   ```
4. Monitor: `tail -f /tmp/ai-server.log` — prompts and responses logged when `LLM_DEBUG=true`

---

## Implementation Status

| Component | Status | Notes |
|---|---|---|
| Project structure + `pyproject.toml` | ✅ Done | src-layout, setuptools build system, dev deps |
| `main.py` (root shim) + `src/tm_ai_server/main.py` | ✅ Done | Correct `/move`, `/health`, `/version` endpoints |
| `schemas.py` | ✅ Done | Correct camelCase schemas matching TM server output |
| `config.py` | ✅ Done | STATE_DIM=55, constants for phases/boards/expansions/tags |
| `model.py` | ✅ Done | PolicyValueNet with LayerNorm + Dropout |
| `encoding.py` | ✅ Done | encode_state, flatten_options, index_to_response, response_to_index |
| `inference.py` | ✅ Done | load_model, select_action; routes to LLM if USE_LLM=true, else random fallback |
| `llm_player.py` | ✅ Done | LLM player (Ollama + Gemini) — setup prompt + action prompt + strategy doc per game; injects board/card context |
| `game_knowledge.py` | ✅ Done | 970-card DB + board/expansion descriptions; `format_card_context` + `format_game_context` |
| `data/card_db.json` | ✅ Done | Generated by `extract_card_db.ts` from TM server card manifests |
| `training/dataset.py` | ✅ Done | TMDataset from Plan-B training log format |
| `training/train_supervised.py` | ✅ Done | Cross-entropy + MSE, checkpoint_best/latest |
| `training/env_tm.py` | ✅ Skeleton | Gymnasium env; requires TM server Phase-2 HTTP endpoints |
| `training/train_ppo.py` | ✅ Skeleton | MaskablePPO; requires env_tm and sb3-contrib |
| Tests (test_encoding, test_schemas) | ✅ Done | 20 tests, all passing |

---

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| API paradigm | `provide_input` (send full `PlayerInputModel`) | Avoids the hard `PlayerInput`-flattening problem on the TM server side; AI server handles it |
| Action space | Flatten tree to padded list (max 128), mask invalid | Simple; upgrade to attention-based if needed later |
| State features | Essential only first (~150 dims) | Add extended features after baseline converges |
| Reward function | Relative scoring `(vp - mean_opponent_vp) / ref` | Gives signal to all players; switch to rank-based after stabilisation |
| Model versioning | `v<major>.<minor>.<patch>` | Minor bump on architecture, patch on weights |
| Error fallback | Log error, no action (current) → improve to random legal option | Safe default; prevents silent bad moves |
| Training data | Plan A (re-export DB) + Plan B (real-time hook) | Both needed: Plan A for 82 historical games, Plan B for all future games |
