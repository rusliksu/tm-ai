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
      "oceanCount": 5,
      "boardName": "tharsis",
      "expansions": ["venus", "prelude"],
      "availableMilestones": [{"name": "Terraformer", "description": "TR ≥ 35"}],
      "availableAwards": [{"name": "Landlord", "description": "Most tiles on board"}],
      "gameVariants": {"draftVariant": true},
      "recentLog": ["Alice played Nuclear Power", "Bob placed a greenery on hex-15 and received 2 plant"]
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
      "cardsInHand": ["Nuclear Power", "Asteroid"],
      "production": {"megacredits": 4, "steel": 1, "titanium": 0, "plants": 2, "heat": 0, "energy": 1},
      "tags": {"science": 2, "building": 3, "space": 1},
      "isAI": true,
      "playedCards": ["Power Grid", "Soletta"],
      "corporations": ["Thorgate"]
    },
    "opponents": [{"id": "p2", "name": "Bob", "terraformRating": 38, "handSize": 3, ...}],
    "board": [{"id": "H05", "x": 3, "y": 2, "tileType": 0, "playerColor": "blue"}],
    "milestones": [{"name": "Terraformer", "playerId": "p456..."}],
    "awards": [{"name": "Landlord", "playerId": "p456..."}],
    "waitingFor": {"type": "or", "title": "Take action", "options": ["..."]}
  },
  "legal_actions": [
    {
      "action_id": "provide_input",
      "type": "or",
      "title": "Take action",
      "payload": {"input": {"type": "or", "title": "Take action", "options": ["..."]}}
    }
  ],
  "metadata": {"schema_version": 1},
  "last_error": "You do not have enough steel to pay for this card"
}
```

`last_error` is optional; set when the previous AI `input_response` was rejected by `player.process()`, enabling the AI to correct its choice or payment on retry.

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

### `POST /advise`

Called by the TM server's `ApiAiAdvice` route when the human player in an AI-trainer-enabled game wants coaching. Shares the same request structure as `/move`, plus an optional `user_question`.

**Request body:** same as `/move`, plus:
```json
{
  "user_question": "Should I focus on science tags this turn?"
}
```

**Response:**
```json
{
  "advice_text": "Given your limited budget, passing this turn would let you accumulate 8 MC before the next phase. However, playing Nuclear Power now would lock in 2 heat production...",
  "recommendation": {"type": "or", "index": 0, "response": {"type": "option"}},
  "debug": null
}
```

`advice_text` is the human-readable coaching paragraph(s). `recommendation` is a valid `input_response` dict the client can submit directly via `/api/ai/play-recommendation`.

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

### `POST /player/register`

Called by the TM server or play script before a game starts, once per AI player, to assign a specific model to that player. If not called, the first `/move` request auto-creates a player with the default model (with a warning).

**Request body:**
```json
{"player_id": "p456...", "game_id": "g123...", "model": "anthropic/claude-opus-4-7"}
```

`model` is optional — omit to use `OPENROUTER_MODEL` (or `OLLAMA_MODEL` if no API key).

**Response body:**
```json
{"ok": true, "player_id": "p456...", "model": "anthropic/claude-opus-4-7"}
```

### `POST /game-done`

Called after game end to flush the per-player token/cost summary to the log. Idempotent — second call for the same `game_id` is silently ignored.

**Request body:**
```json
{"game_id": "g123..."}
```

**Response body:**
```json
{"ok": true, "game_id": "g123..."}
```

---

## Pydantic Schemas (`schemas.py`)

Current `schemas.py` — matches the actual TM server camelCase output:

```python
class GameContext(BaseModel):
    id: str; phase: str; generation: int; oxygen: int; temperature: int; oceanCount: int
    boardName: str = "tharsis"; expansions: List[str] = []
    availableMilestones: List[Dict[str, str]] = []; availableAwards: List[Dict[str, str]] = []
    gameVariants: Dict[str, Any] = {}; recentLog: List[str] = []

class PlayerProduction(BaseModel):
    megacredits: int; steel: int; titanium: int; plants: int; heat: int; energy: int

class PlayerContext(BaseModel):
    id: str; name: str; color: str; terraformRating: int
    megacredits: int; steel: int; titanium: int; plants: int; energy: int; heat: int
    handSize: int; production: PlayerProduction; tags: Dict[str, int] = {}
    isAI: bool = False; playedCards: List[str] = []; playedCardCount: int = 0
    corporations: List[str] = []; cardResources: Dict[str, Any] = {}
    boardTiles: Dict[str, int] = {}; cardsInHand: List[str] = []
    victoryPoints: Optional[int] = None

class MoveRequestState(BaseModel):
    game: GameContext; player: PlayerContext; waitingFor: Optional[Dict[str, Any]] = None
    opponents: List[Dict[str, Any]] = []; milestones: List[Dict[str, Any]] = []
    awards: List[Dict[str, Any]] = []; board: List[Dict[str, Any]] = []
    boardSpaces: List[Dict[str, Any]] = []

class MoveRequest(BaseModel):
    game_id: str; player_id: str; state: MoveRequestState
    legal_actions: List[LegalAction]; metadata: Metadata
    last_error: Optional[str] = None

class MoveResponse(BaseModel):
    input_response: Dict[str, Any]; debug: Optional[MoveDebug] = None

class AdviceRequest(MoveRequest):
    user_question: Optional[str] = None

class AdviceResponse(BaseModel):
    advice_text: str; recommendation: Dict[str, Any]; debug: Optional[MoveDebug] = None

class PlayerRegisterRequest(BaseModel):
    player_id: str; game_id: str; model: Optional[str] = None

class PlayerRegisterResponse(BaseModel):
    ok: bool; player_id: str; model: str
```

---

## State Encoding (`encoding.py`)

Converts the JSON state from the TM server into a fixed-size float32 feature vector.

### Current architecture — flat 492-dim state, fixed 128-slot policy

`STATE_DIM = 492` (see `config.py`):

| Block | Dims | Contents |
|---|---|---|
| Global | 9 | generation, temperature, oxygen, oceans, phase (one-hot × 5) |
| Self | 230 | 7 resources + 6 production + 13 tags + 1 handSize + 199 per-card resources + 1 played_count + 3 board_tiles |
| Opponent | 230 | same shape as self (handSize is public) |
| Milestones/Awards | 4 | ms_self, ms_total, aw_self, aw_total |
| Config | 19 | player_count + 5 board one-hot + 13 expansion flags (zeroed when game_spec=None to keep train/inference consistent) |

`flatten_options()` walks the `waitingFor` decision tree depth-first into a list of leaf choices. Padded to `MAX_ACTIONS = 128`; the policy emits 128 logits, mask is applied before softmax.

### Planned architecture — Option C (card embeddings + spatial board + per-option scoring)

The flat MLP has two design flaws that motivate a planned refactor (Part B / Phase 4 of the active plan; see also [`ai-model-design.md`](../ai-model-design.md) for the full rationale and tensor layout):

1. **Per-card resources are 199 sparse slots** — *Tardigrades* and *Ants* live in totally separate dimensions. The 970-card vocabulary collapses to a learnable `nn.Embedding(971, 32)` (index 0 reserved for unknown/pad). Played, hand, and resource-bearing cards are gathered into the same table and pooled into three 32-dim summaries.
2. **The board is not encoded at all** — `state.board` is in the request but `encode_state()` ignores tile positions. Add `board_grid: tensor[6, 9, 9]` with channels {greenery, city, ocean, special, mine, opponent}.
3. **Rotating slot meanings.** Today a fixed `Linear(D, 128)` policy head emits one logit per slot, but slot 7 is "Greenery" one turn and "Search for Life" the next. Replace with **per-option scoring**: each leaf option carries a 64-dim feature vector (type one-hot, parsed cost, x/y, tile-type, card embedding), and a shared `Linear(state+option, 256) → 1` MLP scores each slot. Slot index becomes meaningless; reordering options leaves predictions unchanged.

**Planned constants** (`config.py`):
```
CARD_VOCAB_SIZE     = 971       # 970 cards + pad index 0
CARD_EMBEDDING_DIM  = 32
MAX_PLAYED_CARDS    = 80
MAX_HAND_CARDS      = 40
MAX_ACTIONS         = 128       # bumpable to 256 if histogram demands
OPTION_FEATURE_DIM  = 64
```

`encode_state` becomes a dict (state_other, played/hand IDs, resource card IDs + counts, board grid). `flatten_options` emits `{title, index, node, features: tensor[64]}` per leaf. See [`ai-model-design.md`](../ai-model-design.md) for the tensor pipeline.

---

## Model Architecture (`model.py`)

### Current — flat MLP `PolicyValueNet`

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

Config: `state_dim=492`, `hidden_sizes=[512, 512, 512]`, `action_space_size=128`.

### Planned — Option C

```python
class PolicyValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.card_embedding = nn.Embedding(971, 32, padding_idx=0)
        self.board_conv = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(2), nn.Flatten(),
        )                                              # → 128-dim board feature
        # state_other (~120) + 4×card_emb (4×32) + board (128) ≈ 376
        self.backbone = nn.Sequential(
            *[block for _ in range(4) for block in (
                nn.Linear(768 if _ else 376, 768), nn.LayerNorm(768),
                nn.ReLU(), nn.Dropout(0.1),
            )],
        )
        self.scoring_head = nn.Sequential(             # Option C: shared per-option scorer
            nn.Linear(768 + 64, 256), nn.ReLU(), nn.Linear(256, 1),
        )
        self.value_head = nn.Linear(768, 1)
```

Backbone is unchanged in spirit but wider (768) and deeper (4 layers). The 128-output `policy_head` is **removed**; logits come from running `scoring_head` once per option slot with shared weights — going from 128 → 256 slots adds zero parameters, just FLOPs.

**Migration constraint:** the refactor refuses to load any 492-dim checkpoint. Current `models/checkpoint_best.pt` is backed up as `models/checkpoint_supervised_492dim.pt.bak` before the swap; PPO has to restart from a fresh supervised pretrain.

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
- `reset()`: POST `/api/ai/new-game`, return initial state
- `step(action)`: POST `/api/ai/step` with `input_response`, receive next state + reward
- **Reward**: relative scoring = `(player_vp - mean_opponent_vp) / reference_vp`
  - Gives learning signal to all players, not just the winner
  - Switch to rank-based reward once agent stabilises
- Self-play games persist to the TM-server DB only — no per-game JSONL is written (see `TM-adaption.md`)

Stable-Baselines3 PPO hyperparameters (starting values):

| Parameter | Value |
|---|---|
| `learning_rate` | 3e-4 |
| `gamma` | 0.99 |
| `gae_lambda` | 0.95 |
| `clip_range` | 0.2 |
| `n_steps` | collect several thousand steps per update |

Initialise PPO from the supervised model checkpoint.

#### Compute profile

PPO training for TM has an unusual bottleneck: **CPU + TM-server throughput**, not GPU compute, because:
- Each PPO "step" is one decision in a live TM game; the TM Node.js server processes decisions serially over HTTP.
- The neural net is tiny (~3M params even after the Option C upgrade); inference and gradient updates take microseconds.
- A single TM game ≈ 800–1500 decisions × ~50–100 ms/decision = 1–3 min/game serial.
- 5M PPO steps ≈ 5000–7000 games.

Implication: **a single modest GPU is plenty.** The leverage is parallelizing TM-server instances via sb3's `SubprocVecEnv`. Target: 4–8 parallel envs on an 8–16 vCPU host with one T4 / L4 / 3090-class GPU.

#### Cloud GPU platform comparison

| Platform | Cheapest GPU class | On-demand | Spot / Interruptible | Notes |
|---|---|---|---|---|
| **Vast.ai** | RTX 3090 (24 GB) | ~$0.18–0.30/h | n/a | Cheapest. Community GPUs; reliability varies; pay by minute. |
| **RunPod (community)** | RTX 4090 (24 GB) | ~$0.34/h | ~$0.20/h | Reliable. Pay by second. Friendly web UI. |
| **Google Cloud (GCE)** | NVIDIA T4 | $0.35/h | **$0.07–0.10/h** | Managed, deep-learning VM, logging/monitoring; Spot 30 s preempt. |
| **Google Cloud (GCE)** | NVIDIA L4 (24 GB) | $0.65/h | ~$0.20/h | Newer Ada-gen, faster than T4. |
| **Google Vertex AI Training** | A100 40 GB | ~$3.67/h | — | Managed; overkill for this workload. |
| **AWS g4dn.xlarge** | NVIDIA T4 | $0.52/h | ~$0.16/h | Mature; frequent spot preemption. |
| **Lambda Labs** | A10 (24 GB) | $0.75/h | — | Premium hobby cloud, no spot. |
| **Modal** | A10G serverless | $0.59/h | — | Pay-per-second, zero-config. |

Recommended paths: **Vast.ai 3090 spot** for cheapest (~$0.20/h), **GCE Spot T4 + n1-standard-8** for most managed (~$0.50/h total). See [Phase 7 of the active plan](../ai-model-design.md) for end-to-end gcloud setup commands and the Spot-preemption recovery flow.

#### Estimated experiment budget

Single 5M-step run, 4 parallel envs, ~4–6 h wall:

| Setup | Per-hour total | 5 h run | 5-run experiment |
|---|---|---|---|
| Vast.ai 3090 (24 GB) | ~$0.25 | ~$1.25 | ~$6–10 |
| GCE Spot T4 + n1-standard-8 | ~$0.50 | ~$2.50 | ~$12–25 |
| GCE Spot L4 + n1-standard-8 | ~$0.60 | ~$3.00 | ~$15–30 |
| RunPod 4090 community | ~$0.34 | ~$1.70 | ~$8–15 |

Realistic budget incl. hyperparam search + verdict iteration: **€10–30 on Vast.ai, €25–50 on GCE.** Even at 50M steps (10× longer), GCE Spot stays under €100. Compute is not the limiting factor here.

### Phase 2.5 — LLM bootstrap training data (planned, ~500 games / €25–50)

Goal: dense supervised pretraining without relying on the limited human-game pool (62 games / 8235 turns).

**Pipeline:**
- `tools/run_llm_self_play.py` calls `/api/ai/new-game` + `/api/ai/step` with `USE_LLM=true` set on the AI server, concurrency = 4, randomized configs per `ApiAiSelfPlay.ts` defaults
- Output: per-game JSONLs under `logs/training/llm/<run_id>/<game_id>.jsonl`
- Run target: 500 successful games or €100 spent
- Expected with Phase 1 token optimisations (caching + per-gen restate + description elision): **~€50 total, ~24 h wall clock**

**Cost basis:** see "Cloud LLM Provider Comparison" below; ~$0.003/game on Gemini 2.5 Flash-Lite (frontier quality at flash-lite tier). Even without caching the upper bound stays under €100; with caching it lands closer to €25.

---

## Training Data

### Sources

| Source | Where | What |
|---|---|---|
| **Plan B — live per-turn capture** | `logs/training/*.jsonl` (`AI_TRAINING_LOG_DIR`) | Real-time `(state, waitingFor, input_response)` triples for **human and AI vs human** games. `Player.process()` writes this for every turn. Skipped entirely when `game.isSelfPlay === true`. |
| **Plan A — engine replay** | `export_training_data.ts` re-runs the engine on each DB save and infers actions from save-diffs + log-diffs. | Recovers historical games. Tool now skips any `gameId` whose JSONL already exists, so re-runs are cheap. After the May-2026 DB purge: 62 games / 8235 turns. |
| **Self-play (PPO)** | TM server DB only (`db/game.db`) | Games created via `/api/ai/new-game` (`isSelfPlay=true`) persist to the DB. No JSONL is written. Run `export_training_data.ts` against the DB if/when supervised retraining wants these games. |
| **LLM bootstrap (planned)** | `logs/training/llm/<run_id>/` | Phase 5: ~500 LLM-played self-play games via `tools/run_llm_self_play.py`. ~€25–50 on Gemini 2.5 Flash-Lite with caching. |

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

### Cloud (training)

Two Docker images on one network: a TM server + a PPO trainer.

**`terraforming-mars/Dockerfile`** (TM server):
```dockerfile
FROM node:20-bookworm-slim
WORKDIR /app
COPY package*.json ./
RUN npm ci --omit=dev
COPY . .
RUN npm run build:server
EXPOSE 8080
ENV NODE_ENV=production
CMD ["node", "build/src/server/server.js"]
```

**`tm-ai/tm-ai-server/Dockerfile.train`** (PPO trainer with CUDA):
```dockerfile
FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY . .
ENV PYTHONPATH=/app/src TM_SERVER_URL=http://tm-server:8080
ENTRYPOINT ["uv", "run", "python", "-m", "tm_ai_server.training.train_ppo"]
```

**`tm-ai/docker-compose.cloud.yml`** orchestrates both, mounts `logs/selfplay` + `models` as volumes, passes `WANDB_API_KEY`. The Vast.ai default images and GCE Deep-Learning VMs both ship with `nvidia-container-toolkit` pre-installed.

**Suggested entry points:** Vast.ai 3090 spot (cheapest), GCE Spot T4 + n1-standard-8 (most managed; budget alerts via `gcloud billing budgets create`). Spot preemption is survivable: with `--checkpoint-interval=50`, at most ~50 games are lost between checkpoints; `--checkpoint=checkpoint_latest.pt` resumes cleanly.

Full step-by-step (gcloud commands, Artifact Registry setup, GCS bucket for checkpoints, `gcsfuse` mount, budget alerts) is in [the active plan, Phase 7](../ai-model-design.md).

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

An alternative to the trained neural net that uses an LLM for strategic decision-making. Activated via `USE_LLM=true`; `inference.py` routes to it before any NN logic. Supports two providers:

- **OpenRouter** (cloud) — unified API gateway at `https://openrouter.ai/api/v1`; gives access to Anthropic Claude, OpenAI, xAI Grok, Google Gemini, and hundreds of other models via one API key and the standard `openai` Python SDK. Set `OPENROUTER_API_KEY`.
- **Ollama** (local) — local inference via direct HTTP; free, no API key. Model name has no `/` (e.g. `qwen3:4b`).

**Provider is inferred from model name**: any model containing `/` routes through OpenRouter; bare `name:tag` models route through Ollama.

### Architecture — One `LLMPlayer` instance per AI player

```
_player_registry: dict[player_id → LLMPlayer]
_game_players:    dict[game_id  → list[player_id]]

Game start (POST /player/register)
    → register_player(player_id, game_id, model)
      creates LLMPlayer, stores in _player_registry

First decision (initialCards)
    → player.init_session(system, user, think=True)
      TM rules + board/expansion/milestone/award context sent ONCE
      → corp/card selection + 150-250 word strategy doc with TABLEAU section

All subsequent decisions
    → player.continue_session(user)
      No rules re-sent; session messages[] grows organically
      Prompt: resources, hand cards with costs/tags/desc, opponent info (+ hand size),
      recent game log, numbered options annotated with card-action descriptions,
      payment section for projectCard/payment decisions
      → CHOICE: N  [PAYMENT: MC=N, STEEL=N, ...]

Per-generation boundary
    → _per_generation_strategy_update(player, generation, state)
      Structured restate prompt: standing, engine, milestone/award targets, next-gen priority
      Response stored in player.strategy; no session rebuild
```

**LLMPlayer class** (`player_id`, `game_id`, `model`, `provider`, `session`, `strategy`, `token_usage`): encapsulates all state for one AI player across the entire game. Session is a standard `messages[]` list (works identically for OpenRouter and Ollama).

**Model capability detection**: `_get_model_capabilities(model)` returns `{caching: bool, thinking: bool}`. 25+ models are pre-seeded in `_KNOWN_CAPABILITIES`; unknown models are probed with two tiny API calls on first use. If a feature is rejected mid-session, it is disabled and subsequent calls skip it without retry.

- **Anthropic caching** (OpenRouter): `cache_control: {"type": "ephemeral"}` on system message content + `anthropic-beta: prompt-caching-2024-07-31` header. System prompt billed once per game.
- **Anthropic thinking** (OpenRouter): `extra_body={"thinking": {"type": "enabled", "budget_tokens": N}}` + `anthropic-beta: interleaved-thinking-2025-05-14` header. `_extract_text` strips thinking blocks from the response.
- **OpenAI caching**: automatic for prompts >1024 tokens; no configuration needed.
- **Ollama thinking**: `"think": true` in payload — effective on qwen3 family.

**Token/cost tracking** — per-player, not per-game: `player.token_usage` accumulates `calls`, `input`, `output`, `cache_read`, `cache_write`. Cost is estimated from the OpenRouter pricing API (`GET /api/v1/models`, fetched once and cached). `log_game_token_summary(game_id)` iterates `_game_players[game_id]` and calls `player.log_token_summary()` for each.

**Context trimming**: when session exceeds `_MAX_SESSION_MESSAGES` (62), the session is rebuilt as: system message + strategy reminder + last 40 messages. For Ollama, strategy is captured first via an extra API call.

**Session recovery**: if no session exists (server restart), `player.recover_session(user)` re-initialises from the stored strategy.

**Description elision**: for `wf_type == "card"` (discard/keep/draft) where all candidate cards are already shown in the hand block, descriptions are replaced with `"(see hand above)"` — saves ~50–200 tokens per such turn.

**Error feedback**: when `last_error` is set, the action prompt prepends `⚠ Your previous response was rejected: "..."`.

**Payment**: for `projectCard` and `payment` decisions, the prompt shows available resources and asks for `PAYMENT: MC=N[, STEEL=N][, TITANIUM=N]...`. `_correct_payment` validates and clamps the parsed payment before submitting.

**Effective card cost**: hand display shows cost after steel/titanium discounts: e.g. `Nuclear Power [cost: 10 MC, effective: 4 MC with 3 steel]`.

**Opponent hand size**: action prompt includes opponent hand count: e.g. `Bob: 14 MC, TR:25, hand: 6 cards`.

**Transient error retry**: `_with_retry` retries on 503/429 with exponential backoff (5s, 10s, 3 attempts). Parses `Retry-After` header when present.

### Env vars

| Var | Default | Description |
|-----|---------|-------------|
| `USE_LLM` | `false` | Enable LLM player |
| `LLM_DEBUG` | `false` | Log prompts (`>` prefix) and responses (`<` prefix) |
| `OPENROUTER_API_KEY` | _(required for OpenRouter)_ | API key from openrouter.ai |
| `OPENROUTER_MODEL` | `anthropic/claude-opus-4-7` | Default model for new players |
| `OPENROUTER_THINKING_BUDGET` | `512` | Thinking tokens for capable models (setup/prelude always use 1024) |
| `OPENROUTER_MAX_OUTPUT_TOKENS` | `350` | Cap on action response length |
| `OPENROUTER_MAX_TURNS` | `80` | Trim session after this many turns |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server base URL |
| `OLLAMA_MODEL` | `qwen3:4b` | Default Ollama model |
| `OLLAMA_TIMEOUT` | `600` | Ollama request timeout (seconds) |

### Think mode

Think is enabled on every turn (setup, prelude, action, per-generation reflection) for models that support it. Setup/prelude always use budget 1024; action turns use `OPENROUTER_THINKING_BUDGET` (default 512). Models without thinking capability fall back gracefully.

### AI Trainer (`select_action_advise`)

The AI Trainer is a coaching sidebar that advises a **human** player. It is **opt-in per player** — no game-wide flag; each player toggles the trainer panel from their own UI (`localStorage[ai_trainer_visible:<participantId>]`).

**Per-player session isolation**: trainer sessions are registered as `trainer:<player_id>` in `_player_registry`. Each player gets a fresh, isolated `LLMPlayer` instance — the LLM never sees the other player's tableau or strategy. The model used is inherited from the human player's registered model (or the default).

**Setup-phase coaching** (`_select_setup_advise`): handles `initialCards` and `prelude` — recommends corp + cards, wrapping output in `<recommendation>` for the Play Recommendation button.

**Dual-format response**: plain text coaching (1–3 sentences, no markdown), then `<recommendation>\nCHOICE: N\n[PAYMENT: ...]\n</recommendation>`. `select_action_advise` strips the block from `advice_text` and parses it via `index_to_response`.

**Requires `USE_LLM=true`** — `inference.py` raises `RuntimeError` otherwise.

### Key functions in `llm_player.py`

- `register_player(player_id, game_id, model)` — create and store an `LLMPlayer`; called by `POST /player/register`
- `get_or_create_player(player_id, game_id)` — return existing player or auto-create with default model
- `log_game_token_summary(game_id)` — log per-player token/cost summary; idempotent
- `select_action_llm(state, waiting_for, game_id, player_id, last_error)` — public entry point; routes to `_select_setup` or `_select_action`
- `LLMPlayer.init_session(system, user, think)` — start new session, store in `self.session`
- `LLMPlayer.continue_session(user, max_output_tokens)` — append user turn, call API, append response
- `LLMPlayer.recover_session(user)` — re-initialise from `self.strategy` when session is lost
- `LLMPlayer.trim_session()` — rebuild session as system + strategy + last 40 messages
- `_get_model_capabilities(model)` — `{caching, thinking}` — pre-seeded table + lazy probe + in-session fallback
- `_get_openrouter_pricing(model)` — `(input_usd/tok, output_usd/tok)` — cached from `/api/v1/models`
- `_extract_text(response)` — strip Anthropic thinking blocks, return plain text
- `_strip_cache_control(messages)` — remove `cache_control` fields for retry without caching
- `_per_generation_strategy_update(player, generation, state)` — send restate prompt, store response in `player.strategy`
- `_maybe_per_generation_update(player, generation, state)` — fire on generation bump; no-op otherwise
- `_select_setup(state, waiting_for, player)` — rich prompt for `initialCards`/`prelude`
- `_select_action(state, waiting_for, player, last_error)` — compact action prompt
- `_build_action_prompt(state, waiting_for, options, last_error, player)` — assemble action prompt with hand, opponents, options
- `_build_setup_prompt(state, waiting_for)` — corp/card/prelude/CEO options with descriptions
- `_parse_setup_response(text, waiting_for, player)` — extract CORPORATION/BUY_CARDS/PRELUDE_CARDS/CEO_CARD/STRATEGY
- `_parse_action_response(text, options, waiting_for, player_id, player)` — extract CHOICE + PAYMENT
- `_correct_payment(payment, wf, player)` — clamp/validate payment against available resources
- `_get_trainer_player(game_id, player_id, state)` — return or create trainer session (`trainer:<player_id>`)
- `select_action_advise(state, waiting_for, game_id, player_id, user_question)` — AI Trainer entry point

### Game Knowledge Database (`game_knowledge.py`)

- `CARD_DB` — 970 cards loaded from `data/card_db.json`. Regenerate: `cd terraforming-mars && npx tsx src/server/tools/extract_card_db.ts > ../tm-ai/data/card_db.json`. `extract_card_db.ts` traverses the CardRenderer `renderData` tree to extract descriptions for prelude, CEO, and event cards that lack a plain `description` string.
- `BOARD_INFO`, `EXPANSION_INFO`, `GAME_VARIANT_DESCRIPTIONS` — hand-written strategic descriptions.
- `format_card_context(names, header, max_cards)` — compact card description block for prompt injection.
- `format_config_context(game)` — full `=== GAME CONFIGURATION ===` block using the live game state: board, expansions, game variants, and the ACTUAL milestones/awards for the game (supports randomised MA).

**Injection:** setup system prompt calls `format_config_context(game_state)`. Action prompt shows hand cards via `format_card_context(cardsInHand)` and annotates card-action options inline. Card descriptions for `card`-type selection decisions (research/discard) still use `_extract_card_names` + `format_card_context`.

### Ollama local models

| Model | RAM | Action time | Notes |
|-------|-----|-------------|-------|
| `qwen3:4b` | 2.5 GB | ~30–60s | **Recommended** — built-in think mode, free |
| `qwen3:8b` | 5 GB | ~60–120s | Better reasoning, still free |
| `llama3.2:3b` | 2 GB | ~15–30s | Fast, no thinking |

### Running

```bash
# Ollama (local, free — Ollama daemon must be running)
cd tm-ai-server && USE_LLM=true OLLAMA_MODEL=qwen3:4b LLM_DEBUG=true \
  uv run uvicorn tm_ai_server.main:app --host 0.0.0.0 --port 8000 >> /tmp/ai-server.log 2>&1 &

# OpenRouter — Claude Opus (cloud; get key at openrouter.ai)
source /home/pmunk/workspace/tm-ai/.env
cd tm-ai-server && USE_LLM=true OPENROUTER_API_KEY=$OPENROUTER_API_KEY \
  OPENROUTER_MODEL=anthropic/claude-opus-4-7 LLM_DEBUG=true \
  uv run uvicorn tm_ai_server.main:app --host 0.0.0.0 --port 8000 >> /tmp/ai-server.log 2>&1 &

# OpenRouter — 4-player multi-model game (register each player after game creation):
# curl -X POST http://localhost:8000/player/register \
#   -H 'Content-Type: application/json' \
#   -d '{"player_id":"p1abc","game_id":"g123","model":"anthropic/claude-opus-4-7"}'
# (repeat for p2 with openai/gpt-4o, p3 with x-ai/grok-3, p4 with google/gemini-2.5-pro)
```

---

## Cloud LLM Provider Comparison (via OpenRouter)

All models are accessed through `https://openrouter.ai/api/v1` with a single `OPENROUTER_API_KEY`. Cost basis: 200 moves/game × ~1,000 input + 100 output tokens = ~200K input / 20K output per game. Anthropic models benefit from prompt caching — system prompt billed once per game, subsequent turns at 10% of input rate.

| Model (OpenRouter ID) | Input $/1M | Output $/1M | Speed | $/game | $/1K games | Caching | Thinking |
|---|---|---|---|---|---|---|---|
| `google/gemini-2.5-flash-lite` | $0.10 | $0.40 | fast | $0.003 | $2.80 | no | no |
| `openai/gpt-4o-mini` | $0.15 | $0.60 | fast | $0.004 | $4.20 | auto | no |
| `google/gemini-2.5-flash` | $0.30 | $2.50 | fast | $0.011 | $11.00 | no | no |
| `openai/gpt-4o` | $2.50 | $10.00 | fast | $0.065 | $65.00 | auto | no |
| `x-ai/grok-3` | $3.00 | $15.00 | fast | $0.090 | $90.00 | no | no |
| `anthropic/claude-haiku-4-5` | $1.00 | $5.00 | fast | ~$0.005 | ~$5.00 | yes | no |
| `anthropic/claude-sonnet-4-6` | $3.00 | $15.00 | medium | ~$0.015 | ~$15.00 | yes | yes |
| `anthropic/claude-opus-4-7` | $15.00 | $75.00 | medium | ~$0.075 | ~$75.00 | yes | yes |

Anthropic estimates use caching: effective input cost is ~10× lower for cached tokens.

**Recommendations by use case:**
- **Low-cost batch / training data**: `google/gemini-2.5-flash-lite` ($2.80/1K games)
- **Best quality AI player**: `anthropic/claude-opus-4-7` with caching (~$0.075/game)
- **Multi-model 4-player game**: register each `player_id` with a different model via `POST /player/register`
- **Local/free**: Ollama `qwen3:4b` (no API cost, ~30–60s/move)

### OpenRouter Setup

1. Sign up at **https://openrouter.ai** and create an API key
2. Add to `tm-ai/.env`: `OPENROUTER_API_KEY=sk-or-...`
3. Start the AI server:
   ```bash
   source /home/pmunk/workspace/tm-ai/.env
   cd tm-ai-server && USE_LLM=true OPENROUTER_API_KEY=$OPENROUTER_API_KEY \
     OPENROUTER_MODEL=anthropic/claude-opus-4-7 LLM_DEBUG=true \
     uv run uvicorn tm_ai_server.main:app --host 0.0.0.0 --port 8000
   ```
4. Monitor: `tail -f /tmp/ai-server.log` — prompts/responses logged when `LLM_DEBUG=true`

---

## Implementation Status

| Component | Status | Notes |
|---|---|---|
| Project structure + `pyproject.toml` | ✅ Done | src-layout, setuptools build system, dev deps |
| `main.py` (root shim) + `src/tm_ai_server/main.py` | ✅ Done | `/move`, `/advise`, `/health`, `/version`, `/player/register`, `/game-done` |
| `schemas.py` | ✅ Done | camelCase schemas; AdviceRequest + PlayerRegisterRequest/Response |
| `config.py` | ✅ Done | STATE_DIM=492, 199-card resource vocab, normalisation caps |
| `model.py` | ✅ Done | Flat MLP PolicyValueNet (LayerNorm + Dropout); Option C refactor planned |
| `encoding.py` | ✅ Done | encode_state, flatten_options, index_to_response, response_to_index |
| `inference.py` | ✅ Done | load_model, select_action, select_advice; routes to LLM if USE_LLM=true |
| `llm_player.py` | ✅ Done | LLM player (OpenRouter + Ollama); LLMPlayer class per AI player; capability detection; per-player token/cost tracking; per-generation strategy update |
| `game_knowledge.py` | ✅ Done | 970-card DB + board/expansion descriptions; `format_card_context`, `format_config_context`, `format_board_layout` |
| `data/card_db.json` | ✅ Done | Generated by `extract_card_db.ts` from TM server card manifests |
| `training/dataset.py` | ✅ Done | TMDataset from Plan-B training log format |
| `training/train_supervised.py` | ✅ Done | Cross-entropy + MSE, checkpoint_best/latest |
| `training/env_tm.py` | ✅ Done | Gymnasium env over `/api/ai/new-game` + `/api/ai/step` |
| `training/train_ppo.py` | ✅ Done | MaskablePPO with manifest/metrics; checkpoints per N games |
| Tests (test_encoding, test_schemas, test_llm_player) | ✅ Done | 46 tests, all passing |
| Option C model refactor | 🗒 Planned (Phase 4 of plan) | Card embeddings + spatial board + per-option scoring; see `ai-model-design.md` |
| LLM bootstrap data run | 🗒 Planned (Phase 5 of plan) | 500-game self-play dataset, ~€25–50 |
| Docker / Cloud deployment | 🗒 Planned (Phase 7 of plan) | Vast.ai or GCE Spot; budget alerts; gcsfuse for checkpoint persistence |

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
