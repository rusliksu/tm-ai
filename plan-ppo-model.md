# PART B — PLAN ONLY, DO NOT EXECUTE

The following phases are fully designed but should **not** be implemented in the same session as Part A. They are the future PPO experiment ("is this game even learnable for our specialized NN with Option C?"). When ready, treat this as the input to a fresh planning round.

## Phase 3 (planned) — Wipe stale PPO state and instrument

- `rm -rf /home/pmunk/workspace/tm-ai/logs/selfplay/*` (14 run directories)
- `rm -rf /home/pmunk/workspace/tm-ai/models/selfplay/*` (matching 14 checkpoint dirs)
- Keep `models/checkpoint_best.pt` and `checkpoint_latest.pt` (supervised baseline) but back them up as `checkpoint_supervised_492dim.pt.bak` — these are 492-dim and incompatible with the new architecture
- Keep `logs/training/*.jsonl` (1540 historical games) — supervised training data
- Instrument: write `tools/measure_action_space.py` that loads each JSONL in `logs/training/` and computes histogram of `len(flatten_options(waiting_for))`. Reports max + 99th percentile. Use this to confirm `MAX_ACTIONS = 128` is sufficient or bump to 256.

## Phase 4 (planned) — Architecture refactor (Option C: card embeddings + spatial board + per-option scoring)

### 4.1 Card embeddings + state encoder

**`tm_ai_server/config.py`:**
- Build `CARD_VOCAB: dict[str, int]` from `data/card_db.json` (970 cards) at import time. Reserve index 0 as "unknown / pad"; real cards 1..970.
- Add constants:
  ```
  CARD_VOCAB_SIZE = 971
  CARD_EMBEDDING_DIM = 32
  MAX_PLAYED_CARDS = 80
  MAX_HAND_CARDS = 40
  MAX_ACTIONS = 128       # bump to 256 if Phase 3 histogram demands
  OPTION_FEATURE_DIM = 64
  ```
- Keep `CARD_RESOURCE_VOCAB` (199) for resource-count aggregation only
- New `STATE_DIM` (the dense "state_other" portion) ≈ 120 — confirm after refactor

**`tm_ai_server/encoding.py`:**
- Refactor `encode_state` to return a dict:
  ```python
  {
    "state_other": tensor[~120],
    "played_ids_self": LongTensor[80],
    "hand_ids_self": LongTensor[40],
    "played_ids_opp": LongTensor[80],
    "resource_card_ids": LongTensor[199],
    "resource_card_counts": FloatTensor[199],
    "board_grid": FloatTensor[6, 9, 9],
  }
  ```
- Card-name → ID lookup; unknown names log a warning and use index 0
- Add `encode_board(state.board)` → `FloatTensor[6, 9, 9]`. Channels: greenery, city, ocean, special, mine, opponent. Bucket each tile into the (channel, x, y) cell.

### 4.2 Per-option features + flatten_options refactor

**`tm_ai_server/encoding.py:flatten_options`:**
- Each emitted entry becomes `{"title", "index", "node", "features": FloatTensor[64]}`
- Feature vector composition per option type:
  - Type one-hot (~12 dims): `[is_select_option, is_card, is_project_card, is_space, is_amount, is_player, is_colony, is_delegate, is_party, is_resource, is_or, is_and]`
  - Numeric (~4 dims): parsed cost if extractable, normalized amount, x, y
  - Card embedding (~32 dims): looked up via the same 971-table when the option is card-related; zeros otherwise
  - Tile type one-hot (~8 dims): when space-related; zeros otherwise
  - Player color one-hot (~8 dims): when player/delegate-related; zeros otherwise
- Total ~64 dims (pad as needed to hit exact `OPTION_FEATURE_DIM`)
- Padded slots (slots > len(options)) all zeros; masked anyway

**Helper functions needed:**
- `_extract_cost(title: str) -> int | None` for "Greenery: pay 23" style options
- `_extract_card_for_option(node: dict) -> str | None`
- `_extract_coords_for_space(node: dict) -> tuple[int, int] | None`

### 4.3 PolicyValueNet refactor

**`tm_ai_server/model.py`:**

```python
class PolicyValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.card_embedding = nn.Embedding(971, 32, padding_idx=0)
        self.board_conv = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(2), nn.Flatten(),
        )  # → 128-dim board feature
        # state_other (~120) + 4 × card_emb (4×32=128) + board (128) = ~376
        self.backbone = nn.Sequential(
            nn.Linear(376, 768), nn.LayerNorm(768), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(768, 768), nn.LayerNorm(768), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(768, 768), nn.LayerNorm(768), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(768, 768), nn.LayerNorm(768), nn.ReLU(), nn.Dropout(0.1),
        )
        # Per-option scoring head (Option C)
        self.scoring_head = nn.Sequential(
            nn.Linear(768 + 64, 256), nn.ReLU(), nn.Linear(256, 1),
        )
        self.value_head = nn.Linear(768, 1)
    
    def forward(self, batch_dict, option_features, mask):
        # ... (see Tensor Layout section in Design Rationale)
        ...
```

- Backward compatibility: refuse to load any 492-dim checkpoint. Clear error message pointing to `checkpoint_supervised_492dim.pt.bak` rollback.

### 4.4 Dataset + supervised training updates

**`tm_ai_server/training/dataset.py`:**
- Update `__getitem__` to return the encoded state-dict plus `option_features: FloatTensor[128, 64]`, plus the chosen action index and reward
- Add `human_only` → renamed `source_filter: Literal["human", "ai", "all"] = "human"` so we can opt in to AI-generated games
- Cache option-features computation per row (it's expensive; consider precomputing once during dataset load)

**`tm_ai_server/training/train_supervised.py`:**
- New CLI flags: `--data-dirs path1 path2 …`, `--include-ai-games`
- Update training loop to unpack the dict-state + option_features into `model.forward(...)`
- Loss unchanged (cross-entropy on policy logits + MSE on value)

### 4.5 PPO env + training updates

**`tm_ai_server/training/env_tm.py`:**
- `observation_space` becomes a `gym.spaces.Dict` matching the encoder output
- `step()` computes option features at every call and includes them in the observation
- Action masks unchanged

**`tm_ai_server/training/train_ppo.py`:**
- MaskablePPO policy needs a custom feature extractor that consumes the Dict observation (sb3 supports this via `policy_kwargs.features_extractor_class`)
- Hyperparams unchanged for first run

### 4.6 Tests

- `tests/test_encoding.py`: cover dict-style return, card-ID gather + padding, board grid construction, option-feature extraction for each type
- `tests/test_model.py` (new): test forward pass shapes, gradient flow through embedding table, action masking
- Update `tests/test_schemas.py` as needed if schemas change

## Phase 5 (planned) — Generate LLM bootstrap training data (500 games)

- Build `tools/run_llm_self_play.py`: thin runner that calls `/api/ai/new-game` + `/api/ai/step` with `USE_LLM=true` AI server, parallel concurrency = 4, randomized configs per existing `ApiAiSelfPlay.ts` defaults
- Output: `logs/training/llm/<run_id>/<game_id>.jsonl`
- Run until 500 successful games or €100 spent
- Expected with Phase 1 optimizations: ~€50 total, ~24h wall clock

## Phase 6 (planned) — Supervised retrain on combined data

```bash
cd tm-ai-server && uv run python -m tm_ai_server.training.train_supervised \
  --data-dirs ../logs/training ../logs/training/llm \
  --include-ai-games \
  --output-dir ../models \
  --epochs 80
```

- ~50–80K training samples expected
- Backup before overwriting: `cp models/checkpoint_best.pt models/checkpoint_supervised_492dim.pt.bak`
- Watch val loss; expect a lower floor than the old 1.2255 baseline due to denser features

## Phase 7 (planned) — PPO self-play + cloud deployment + verdict

### 7.1 Compute profile

PPO training for TM has an unusual bottleneck: **CPU + TM-server throughput**, not GPU compute, because:

- Each PPO "step" is one decision in a live TM game; the TM Node.js server processes decisions serially over HTTP
- The neural net is tiny (~3M params); inference and gradient updates take microseconds
- A single TM game ≈ 800–1500 decisions × ~50–100 ms/decision = 1–3 min/game serial
- 5M PPO steps ≈ 5000–7000 games

Implication: **a single modest GPU is plenty.** The leverage comes from parallelizing TM-server instances via sb3's `SubprocVecEnv`. Target: 4–8 parallel envs on an 8–16 vCPU host with one T4 / L4 / 3090-class GPU.

### 7.2 Cloud GPU platform comparison

| Platform | Cheapest GPU class | On-demand | Spot / Interruptible | Notes |
|---|---|---|---|---|
| **Vast.ai** | RTX 3090 (24 GB) | ~$0.18–0.30/h | n/a | Cheapest. Community GPUs, reliability varies, occasional host reboots. Pay-by-minute. |
| **RunPod (community)** | RTX 4090 (24 GB) | ~$0.34/h | ~$0.20/h | Reliable. Pay-by-second. Web UI is friendly. |
| **Google Cloud (GCE)** | NVIDIA T4 | $0.35/h | **$0.07–0.10/h** | Managed, deep-learning VM image, integrated logging/monitoring. Spot preemptible with 30 s notice. |
| **Google Cloud (GCE)** | NVIDIA L4 (24 GB) | $0.65/h | ~$0.20/h | Newer Ada-gen, faster than T4. |
| **Google Vertex AI Training** | A100 40 GB | ~$3.67/h | — | Submit a container, fully managed. Overkill for this workload. |
| **AWS g4dn.xlarge** | NVIDIA T4 | $0.52/h | ~$0.16/h | Mature ecosystem; frequent preemption on spot. |
| **Lambda Labs** | A10 (24 GB) | $0.75/h | — | Premium hobby cloud, no spot tier. |
| **Modal** | A10G serverless | $0.59/h | — | Pay-per-second, zero-config. |

**Recommended paths for this project:**
- **Cheapest:** Vast.ai 3090 at ~$0.20/h
- **Most managed (and what the user explicitly wants included):** Google Cloud GCE Spot T4 at ~$0.10/h GPU + ~$0.40/h for n1-standard-8

### 7.3 Estimated cost for the verdict experiment

Single 5M-step run, 4 parallel envs, ~4–6 h wall:

| Setup | Per-hour total | 5 h run | 5-run experiment |
|---|---|---|---|
| Vast.ai 3090 (24 GB) | ~$0.25 | ~$1.25 | ~$6–10 |
| GCE Spot T4 + n1-standard-8 | ~$0.50 | ~$2.50 | ~$12–25 |
| GCE Spot L4 + n1-standard-8 | ~$0.60 | ~$3.00 | ~$15–30 |
| RunPod 4090 community | ~$0.34 | ~$1.70 | ~$8–15 |

Realistic total budget including hyperparam search + verdict iteration: **€10–30 on Vast.ai, €25–50 on GCE.** Even at 50M steps (10× longer), GCE Spot stays under €100. Compute is not the limiting factor here.

### 7.4 Dockerfiles

Two containers on one Docker network. Trainer talks to TM server at `http://tm-server:8080`.

**`terraforming-mars/Dockerfile`** (new — TM server image):
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

**`tm-ai/tm-ai-server/Dockerfile.train`** (new — PPO trainer image):
```dockerfile
FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY . .
ENV PYTHONPATH=/app/src TM_SERVER_URL=http://tm-server:8080
ENTRYPOINT ["uv", "run", "python", "-m", "tm_ai_server.training.train_ppo"]
```

**`tm-ai/docker-compose.cloud.yml`** (new — orchestrates both):
```yaml
services:
  tm-server:
    image: tm-server:latest
    build: ../terraforming-mars
    expose: ["8080"]
    volumes:
      - ./logs/selfplay:/app/logs/selfplay
    restart: on-failure

  ai-trainer:
    image: tm-ai-trainer:latest
    build:
      context: ./tm-ai-server
      dockerfile: Dockerfile.train
    runtime: nvidia
    depends_on: [tm-server]
    environment:
      TM_SERVER_URL: http://tm-server:8080
      WANDB_API_KEY: ${WANDB_API_KEY}
    volumes:
      - ./logs/selfplay:/app/logs/selfplay
      - ./models:/app/models
    command:
      - "--checkpoint=/app/models/checkpoint_best.pt"
      - "--output-dir=/app/models"
      - "--log-dir=/app/logs/selfplay"
      - "--total-steps=5000000"
      - "--checkpoint-interval=50"
```

GPU access requires `nvidia-container-toolkit` on the host. Both the Vast.ai default images and GCE Deep-Learning VMs ship with it preinstalled.

### 7.5 Setup path A — Google Cloud (managed)

**One-time GCP setup:**
1. Install `gcloud` CLI locally; `gcloud auth login`; `gcloud config set project <PROJECT_ID>`
2. Enable APIs:
   ```bash
   gcloud services enable compute.googleapis.com artifactregistry.googleapis.com logging.googleapis.com
   ```
3. Create an Artifact Registry repo and push images:
   ```bash
   gcloud artifacts repositories create tm-ai \
     --repository-format=docker --location=us-central1
   gcloud auth configure-docker us-central1-docker.pkg.dev
   IMG=us-central1-docker.pkg.dev/<PROJECT>/tm-ai
   docker build -t $IMG/tm-server:latest ../terraforming-mars
   docker build -t $IMG/tm-ai-trainer:latest -f tm-ai-server/Dockerfile.train tm-ai-server
   docker push $IMG/tm-server:latest
   docker push $IMG/tm-ai-trainer:latest
   ```
4. Create a GCS bucket for checkpoints + a service account with `Storage Object Admin` on it
5. Set a budget alert at €30 to catch runaway costs:
   ```bash
   gcloud billing budgets create --billing-account=<BA_ID> \
     --display-name=tm-ai-training --budget-amount=30EUR \
     --threshold-rule=percent=0.5 --threshold-rule=percent=0.9
   ```

**Launch training VM (Spot T4):**
```bash
gcloud compute instances create tm-ppo-train \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=common-cu123 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-balanced \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --metadata=install-nvidia-driver=True \
  --scopes=cloud-platform \
  --service-account=<SA_EMAIL>
```

The `common-cu123` deep-learning image bundles Docker, NVIDIA drivers, CUDA 12.3, and the NVIDIA container toolkit.

**SSH in and start training:**
```bash
gcloud compute ssh tm-ppo-train --zone=us-central1-a

# Inside the VM
gcloud auth configure-docker us-central1-docker.pkg.dev
git clone https://github.com/<user>/tm-ai.git
git clone https://github.com/<user>/terraforming-mars.git
cd tm-ai
# Mount the GCS bucket for checkpoint persistence across Spot preemption
gcsfuse <your-bucket> /home/$USER/tm-ai/models
WANDB_API_KEY=<your_key> docker compose -f docker-compose.cloud.yml up -d
docker compose logs -f ai-trainer
```

**Surviving Spot preemption:** Spot VMs get a 30-second warning then stop. With `--checkpoint-interval=50`, you lose at most ~50 games. Resume with:
```bash
gcloud compute instances start tm-ppo-train --zone=us-central1-a
gcloud compute ssh tm-ppo-train -- 'cd tm-ai && docker compose -f docker-compose.cloud.yml up -d'
```
The trainer auto-loads `checkpoint_latest.pt` and continues.

**Stop and clean up after the run:**
```bash
gcloud compute instances delete tm-ppo-train --zone=us-central1-a
```

### 7.6 Setup path B — Vast.ai (cheapest)

**1. Push images to a public registry** (Docker Hub is fine, or use GCR if you've already set it up):
```bash
docker push <username>/tm-ai-trainer:latest
docker push <username>/tm-server:latest
```

**2. Launch instance from Vast.ai web UI:**
- **Filters:** GPU = RTX 3090 (24 GB), CUDA ≥ 12.1, RAM ≥ 32 GB, disk ≥ 100 GB, reliability > 95 %
- **Interruptible:** off for stability, on (~50 % discount) if you tolerate restarts
- **Image:** `<username>/tm-ai-trainer:latest`
- **On-start script:**
  ```bash
  cd /workspace
  git clone https://github.com/<user>/tm-ai.git
  git clone https://github.com/<user>/terraforming-mars.git
  cd tm-ai
  apt-get update && apt-get install -y docker-compose-plugin
  docker compose -f docker-compose.cloud.yml up -d
  ```
- **Env vars to set in Vast.ai job config:** `WANDB_API_KEY=<your_key>`

**3. Monitor and back up:**
- Vast.ai gives an SSH command on the running instance page
- Tail logs: `ssh -p <port> root@<ip> 'cd /workspace/tm-ai && docker compose logs -f ai-trainer'`
- Pull checkpoints down hourly via cron on your laptop:
  ```bash
  rsync -av -e 'ssh -p <port>' root@<ip>:/workspace/tm-ai/models/checkpoint_*.pt ./models/cloud/
  ```

**Caveat:** community machines may be reclaimed by their owner with ~minutes' notice. Use `--checkpoint-interval=50` and the hourly rsync — the worst case is losing ~30 minutes of training.

### 7.7 Monitoring (works on either platform)

**Layer 1 — `metrics.jsonl` tail (always available)**
- The `SelfPlayCallback` writes `logs/selfplay/<run_id>/metrics.jsonl` each checkpoint interval
- Watch from your laptop: `ssh <host> 'tail -f /path/to/tm-ai/logs/selfplay/*/metrics.jsonl'`
- Key columns: `mean_generation`, `win_rate`, `mean_vp`, `mean_reward`

**Layer 2 — TensorBoard (port-forward over SSH)**
- Already integrated via sb3
- GCE: `gcloud compute ssh tm-ppo-train -- -L 6006:localhost:6006 'docker exec -it $(docker compose ps -q ai-trainer) tensorboard --logdir /app/logs/selfplay --bind_all --port 6006'`
- Vast.ai: vast.ai gives a forwarded port on the instance page; use `-L 6006:localhost:6006` form the same way
- Open `http://localhost:6006` locally

**Layer 3 — Weights & Biases (recommended)**
- Free tier covers this scale
- Add to `pyproject.toml`: `wandb = "^0.17"`
- Update `train_ppo.py`:
  ```python
  import wandb
  from wandb.integration.sb3 import WandbCallback
  run = wandb.init(project="tm-ai-ppo", sync_tensorboard=True,
                   config={"total_steps": args.total_steps, ...})
  callback = CallbackList([SelfPlayCallback(...), WandbCallback()])
  model.learn(total_timesteps=args.total_steps, callback=callback)
  ```
- Set `WANDB_API_KEY` via docker-compose env
- Persistent dashboard at `https://wandb.ai/<user>/tm-ai-ppo` — survives Spot preemption, share-able link

**Layer 4 (GCE only) — Cloud Logging + alerting**
- Container stdout streams automatically to Cloud Logging when the VM has `--scopes=cloud-platform`
- Set an alerting policy:
  ```bash
  gcloud alpha monitoring policies create --policy-from-file=alert-no-metrics.yaml
  ```
  where `alert-no-metrics.yaml` watches for "no log entry in `tm-ai-trainer` container in last 30 min" → email
- Useful for catching silent hangs after preemption-restart

### 7.8 Training command (same on both platforms)

```bash
# Inside the running container, or via docker-compose command (see 7.4):
uv run python -m tm_ai_server.training.train_ppo \
  --checkpoint /app/models/checkpoint_best.pt \
  --output-dir /app/models \
  --log-dir /app/logs/selfplay \
  --total-steps 5_000_000 \
  --checkpoint-interval 50 \
  --n-envs 4
```

`--n-envs 4` is the new flag for `SubprocVecEnv` parallelism — add to `train_ppo.py` argparser as part of Phase 4.

### 7.9 Verdict criteria

From `logs/selfplay/<run_id>/metrics.jsonl`:
- `mean_generation < 20` **and** `win_rate > 60 %` by game 2000 → architecture is learnable; commit to a longer run (50 M steps, ~€10–20 more on GCE Spot)
- `mean_generation` stuck > 40 by game 2000 → still insufficient; pivot away from specialized NN before more compute
- Anything in between → run to game 5000 (~10 h), decide then

### 7.10 Cost-control checklist

- [ ] GCP budget alert set at €30 (or your chosen ceiling)
- [ ] Spot/Interruptible enabled (saves ~75–90 %)
- [ ] Vast.ai max-spend limit set in account settings
- [ ] Checkpoint interval ≤ 50 games (loses ≤ 30 min on preemption)
- [ ] `gcsfuse` or rsync configured so checkpoints survive VM stop
- [ ] Trainer container has a clean exit on `model.learn()` completion (kills GPU billing)
- [ ] Stop/delete VM immediately after run: `gcloud compute instances delete tm-ppo-train`

## Phase 8 (planned) — Spec doc sync

- `specs/TM-AI.md`: State Encoding, Model Architecture, Cloud LLM Provider Comparison
- `CLAUDE.md`: STATE_DIM constants, embedding architecture, bootstrap-from-LLM flow
- `TODO.md`: mark Phase 5 line 69 (spatial board encoding) ✅, mark line 117 (payment) still open

---

## Critical files


### Part B — modified (when executed later)

| File | Phase | Change |
|---|---|---|
| `tm-ai-server/src/tm_ai_server/config.py` | 4 | card vocab build, embedding constants, MAX_HAND_CARDS=40, MAX_PLAYED_CARDS=80 |
| `tm-ai-server/src/tm_ai_server/model.py` | 4 | card embeddings, board conv, scoring head, value head |
| `tm-ai-server/src/tm_ai_server/encoding.py` | 4 | dict-state return, per-option features, board encoder |
| `tm-ai-server/src/tm_ai_server/training/dataset.py` | 4 | dict-state items, source filter |
| `tm-ai-server/src/tm_ai_server/training/train_supervised.py` | 4 | multi-dir, AI-inclusion flag |
| `tm-ai-server/src/tm_ai_server/training/env_tm.py` | 4 | dict observation space |
| `tm-ai-server/src/tm_ai_server/training/train_ppo.py` | 4 | custom feature extractor |
| `tm-ai-server/tests/test_encoding.py`, `test_model.py` | 4 | new tests |
| `tm-ai-server/tools/run_llm_self_play.py` (new) | 5 | bootstrap runner |
| `tm-ai-server/tools/measure_action_space.py` (new) | 3 | histogram instrumentation |
| `terraforming-mars/Dockerfile` (new) | 7 | TM server container image |
| `tm-ai-server/Dockerfile.train` (new) | 7 | PPO trainer container image |
| `tm-ai/docker-compose.cloud.yml` (new) | 7 | Two-container orchestration for cloud VMs |
| `tm-ai-server/training/train_ppo.py` | 7 | Add `--n-envs` flag + WandbCallback integration |
| `tm-ai-server/pyproject.toml` | 7 | Add `wandb` dependency |
