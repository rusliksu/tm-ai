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

---

## Phase 3: Training Data ✅ Complete (initial dataset)

- [x] Plan B logging wired end-to-end (all future games produce JSONL training logs)
- [x] `dataset.py` — reads JSONL format, extracts (state, mask, action, reward) tuples
- [x] **Plan A**: `export_training_data.ts` — re-run engine on 82 historical DB saves
  - Captures research + drafting phases via cardsInHand/draftedCards diffs
  - **Now also captures action phase** via log message matching (globalInitialize fix + inferResponseFromLogs)
  - Produces 62 games / 8235 turns / 6888 usable training samples; output in logs/training/
  - fix: `dataset.py` now passes `None` for game_spec (train/inference consistency)

---

## Phase 4: Training ⏳ In Progress

- [x] `train_supervised.py` — cross-entropy policy loss + MSE value loss, checkpointing
- [x] Run Phase 1 supervised training — **6888 samples** (research + drafting + action phase)
  - Training running now: `uv run python -m tm_ai_server.training.train_supervised --data-dir ../logs/training --output-dir ../models --epochs 50`
- [ ] Evaluate trained model vs random policy

---

## Phase 5: Extended Features

- [ ] Extend `encode_state()` to use `state.opponents` (already in request, not yet encoded)
- [ ] Extend `encode_state()` to encode `state.board` tile positions
- [ ] Extend `encode_state()` to encode milestones/awards state
- [ ] Update `STATE_DIM` and retrain when extending

---

## Phase 6: PPO Self-Play (Phase 2 Training) 🔜 Future

- [x] `env_tm.py` — Gymnasium env skeleton
- [x] `train_ppo.py` — MaskablePPO skeleton
- [ ] TM server: `/api/ai/new-game` endpoint (start game, return initial state)
- [ ] TM server: `/api/ai/step` endpoint (send InputResponse, return next state + done + result)
- [ ] Run PPO training on GPU (RunPod/Vast/Synpix), initialise from supervised checkpoint

---

## Phase 7: Docker / Deployment

- [ ] `Dockerfile` for CPU inference (FastAPI server only)
- [ ] `Dockerfile` for GPU training (CUDA + stable-baselines3)
- [ ] `docker-compose.yml` for local stack (TM server + AI server)
- [ ] Cloud deployment scripts
