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
      inference.py      # Model loading and action selection
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
  "input_response": {"type": "or", "responses": [{"index": 2}]},
  "debug": {
    "policy_logits": [1.2, -0.3, 0.8],
    "value_estimate": 0.35
  }
}
```

`input_response` is passed directly to `player.process()` on the TM server. `debug` is optional and ignored by the game.

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

## Implementation Status

| Component | Status | Notes |
|---|---|---|
| Project structure + `pyproject.toml` | ✅ Done | All deps declared |
| FastAPI skeleton (`main.py`) | ⚠️ Bug | Duplicate `if __name__ == "__main__"` block; second calls undefined `main()` — remove it |
| `schemas.py` | ❌ Wrong format | Uses old snake_case + nested `global_` field; replace entirely with schemas above |
| `encoding.py` | ❌ Not started | Blocks all downstream work |
| `model.py` | ❌ Not started | |
| `inference.py` | ❌ Not started | |
| `config.py` | ❌ Not started | |
| `dataset.py` | ❌ Not started | |
| `train_supervised.py` | ❌ Not started | |
| `env_tm.py` + `train_ppo.py` | ❌ Not started | Phase 2 |

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
