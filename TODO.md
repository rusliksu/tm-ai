# TM-AI Development Checklist

## Overview
Build the AI server for a Terraforming Mars AI agent per `specs/TM-AI.md` and `specs/TM-adaption.md`.

---

## Phase 1: Project Setup ✅ Complete

- [x] Python project with `uv`, src-layout, `pyproject.toml`
- [x] Dependencies: fastapi, uvicorn, pydantic, torch, numpy, stable-baselines3, gymnasium
- [x] `config.py` — STATE_DIM=55, constants for phases/boards/expansions/tags
- [x] `schemas.py` — correct camelCase Pydantic models matching TM server output
- [x] `main.py` — FastAPI app with `/move`, `/health`, `/version`
- [x] `model.py` — PolicyValueNet (MLP with LayerNorm + Dropout)
- [x] `encoding.py` — encode_state, flatten_options, index_to_response, response_to_index
- [x] `inference.py` — load_model, select_action, random fallback
- [x] Tests: 20 tests passing (`test_encoding.py`, `test_schemas.py`)

---

## Phase 2: TM Server Integration ✅ Complete

- [x] `isAI` flag on `Player`, serialization, game creation API
- [x] `setWaitingFor()` — captures pendingTrainingState for ALL players (human + AI)
- [x] `process()` — calls logTrainingTurn for ALL players
- [x] `requestAiMove()` — calls AI server, fallback to last SelectOption on failure
- [x] `aiFallbackResponse()` — finds last SelectOption in OrOptions
- [x] `takeAction(saveBeforeTakingAction)` — fixed to actually gate game.save()
- [x] `stateMapping.ts` — full state: player + opponents + board + milestones + awards + playedCards
- [x] `TrainingLogger.ts` — writeMeta / appendTurn / writeResult; per-game JSONL
- [x] `ApiCreateGame.ts` — writeMeta at game creation
- [x] `Game.ts` — writeResult at game end (gotoEndGame)
- [x] `CreateGameForm.vue` — AI player checkbox in game setup UI
- [x] `ServerModel.ts` — isAI exposed in game/player models
- [x] `_aiMoveInProgress` flag — prevents infinite retry loop when `process()` throws InputError
- [x] Re-trigger `requestAiMove()` after successful `process()` — fixes AI getting stuck when `process()` chains into a new `setWaitingFor()` while the flag is still set

---

## Phase 3: Training Data ✅ Complete (initial dataset)

- [x] Plan B logging wired end-to-end (all future games produce JSONL training logs)
- [x] `dataset.py` — reads JSONL format, extracts (state, mask, action, reward) tuples
- [x] **Plan A**: `export_training_data.ts` — re-run engine on 82 historical DB saves
  - Captures research + drafting phases via cardsInHand/draftedCards diffs
  - Captures action phase via log message matching (globalInitialize fix + inferResponseFromLogs)
  - 62 games / 8235 turns / 6888 usable training samples; output in logs/training/
  - `dataset.py` passes `None` for game_spec (train/inference consistency)

---

## Phase 4: Training ✅ Complete (supervised baseline)

- [x] `train_supervised.py` — cross-entropy policy loss + MSE value loss, checkpointing
- [x] Supervised training run on 6888 samples (50 epochs) → `models/checkpoint_best.pt`
- [~] Evaluate trained model vs random policy — skipped; self-play will reveal win rate naturally
- [~] Collect 5–10 more live games before retraining — skipped; PPO self-play replaces this

---

## Phase 5: Extended State Features ✅ Complete

- [x] Extend `encode_state()` to use `state.opponents` — opponent slot added (230 dims, same as self); handSize included (visible to all players)
- [x] Encode per-card resource counts — 199-card vocabulary; each card gets its own slot instead of type aggregates
- [x] Encode milestones/awards state — 4 dims (ms_self, ms_total, aw_self, aw_total)
- [x] Encode board tile counts per player — 3 dims (greenery, city, special) in each player slot
- [x] Update `STATE_DIM` to 492 and retrain from clean model (best val_loss=1.2255 at epoch 9)
- [ ] Extend `encode_state()` to encode `state.board` full spatial tile positions (x/y grid)

---

## Phase 6: PPO Self-Play (Phase 2 Training) ✅ Complete

- [x] `env_tm.py` — Full Gymnasium env; uses `/api/ai/new-game` + `/api/ai/step`; plays all positions
- [x] `train_ppo.py` — MaskablePPO with full logging, manifest, per-game checkpoints
- [x] TM server: `POST /api/ai/new-game` — creates 2-player self-play game, returns initial state
- [x] TM server: `POST /api/ai/step` — applies InputResponse, returns next state/player or done+result
- [x] `game.isSelfPlay` flag — suppresses auto `requestAiMove()` trigger for self-play games
- [x] `sb3-contrib` added to dependencies (MaskablePPO)

### Logging (all artifacts preserved, nothing overwritten):
- [x] **Game logs**: `logs/selfplay/<run_id>/<game_id>.jsonl` via Plan B (TM server TrainingLogger)
- [x] **Model checkpoints**: `models/selfplay/<run_id>/checkpoint_<N>.pt` every `--checkpoint-interval` games + `checkpoint_best.pt` + `checkpoint_latest.pt`
- [x] **Metrics**: `logs/selfplay/<run_id>/metrics.jsonl` — win rate, mean reward, mean game length after each checkpoint interval
- [x] **Run manifest**: `logs/selfplay/<run_id>/manifest.json` at run start (timestamp, base checkpoint, hyperparams, TM server URL)

### To start training:
```bash
# 1. Start TM server (if not already running)
cd /home/pmunk/workspace/terraforming-mars && node build/src/server/server.js &

# 2. Run PPO training (from tm-ai-server/)
uv run python -m tm_ai_server.training.train_ppo \
    --checkpoint ../models/checkpoint_best.pt \
    --output-dir ../models \
    --log-dir ../logs/selfplay \
    --total-steps 5_000_000 \
    --checkpoint-interval 100
```

---

## Phase 7: Docker / Deployment

- [ ] `Dockerfile` for CPU inference (FastAPI server only)
- [ ] `Dockerfile` for GPU training (CUDA + stable-baselines3)
- [ ] `docker-compose.yml` for local stack (TM server + AI server)
- [ ] Cloud deployment scripts



## Known Issues / Future Work

- [ ] **Error feedback for specialized AI (NN) training**: The error-retry loop in `Player.ts:requestAiMove` (added for LLM player) should also be incorporated into training data collection. When a move is rejected with an error, the (state, bad_action, error, retry_action) tuple is valuable training signal. Currently the training logger only records the successful response. Add a `last_error` field to the training JSONL turn records, and teach the supervised/PPO trainer to use rejection signals as negative examples.

- [ ] **Payment type handling in NN encoder**: `encoding.py:index_to_response` still uses `_mc_payment` (MC-only) for `projectCard` and `payment` types in the NN path. When the NN is trained to handle payment decisions, the encoder and response builder need to be extended to output per-resource payment amounts as part of the action.


- [ ] **self-training biased**: in ApiAiSelfPlay the game setup is quite deterministic. I want the ai to train with 2-4 player but 80% of games with 2 players only. I want it to use random official boards. I want it to use random awards and milestones. I want the prelude, prelude 2, and venus packs enabled always, and in 20% of cases the promo packs. I want the terraforming world government disabled and express mode enabled always. 

- [ ] **board information missing**: the AI has no information of the board. add a generic description of the board (what do the tile placement IDs mean, which IDs have which neighbours etc.) and for each board the placement bonus per placement tile, the placement tile type (normal, ocean, volcano, special - with description) and add this information to the initial prompt with the selected board.

## Future ideas
- use the llm as trainer for a human.
  - exnted the new game ui with a switch called AI Trainer, default off.
  - if this switch is on, the add a chat window with configurable width (30% by default) and 100% height on the main game screen
  - use the given system prompts and game details as for AI mode to feed a lokal or cloud llm but instead of playing directly, the llm should provide its results in a human readable from like a traininer in the chat window. 
  - in chat window, user can as questions and chat directly with ai
  - below chat window next to the send new message, there should be a button "Play recommondation" that takes the LLM response and uses the API to play the recommendation from the AI directly. So we have to request the LLM to output a human readable result and a machine readable result, the latter is hidden from the user.