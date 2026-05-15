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


- [x] **self-training biased**: ApiAiSelfPlay now randomises player count (80%×2, 10%×3, 10%×4), board (tharsis/hellas/elysium), randomMA, prelude+prelude2+venus always, 20% promo, fastModeOption=true, solarPhaseOption=false.

- [x] **board information missing**: `buildAllBoardSpaces()` adds all board spaces (id, x, y, type, bonuses, volcanic) to state. `format_board_layout()` injects board layout into initial LLM system prompt. Space selection options annotated with position + bonuses (fixes AI confusing space IDs with option numbers).

- [x] **cardsInHand silently dropped** (CRITICAL bug found in game g786ed2285373): Pydantic `PlayerContext` was missing `cardsInHand` field — AI could never see its own hand. Fixed by adding `cardsInHand: List[str] = []` to schema.

- [x] **recentLog silently dropped** (CRITICAL bug): `GameContext` was missing `recentLog` field — AI never saw recent game events. Fixed by adding `recentLog: List[str] = []` to schema.

- [x] **victoryPoints added**: `buildPlayerSnapshot()` now emits `victoryPoints` (current VP total). Shown in action prompt for both self and opponents so AI can assess score standing.

- [ ] **Context accumulation / tableau memory**: Over a 15-generation game the Gemini session accumulates ~150+ messages. The AI was asked to memorise its tableau but later turns may attend less to early session content. Consider re-injecting a tableau summary each turn OR compressing at each generation boundary.

- [ ] **Terraforming urgency signal**: AI needs to know if the game is ending soon. Add `estGenerationsLeft` (estimated remaining generations based on current global parameter pace) to the game state or action prompt.

- [ ] **Opponent engine summary**: AI has opponent resources/tags but no engine-type summary. Consider adding a brief "opponent strategy" tag to help the AI decide what to deny or race.

## Future ideas
- ✅ **AI Trainer** (human coaching sidebar) — implemented.
  - ✅ `aiTrainerEnabled` game option (default off), checkbox in CreateGameForm.
  - ✅ `AiTrainerChat.vue` sidebar (fixed-position, resizable, width persisted to localStorage).
  - ✅ Auto-fetches coaching advice on each new decision point; human can ask follow-up questions.
  - ✅ LLM responds with human-readable coaching + hidden `<recommendation>` block.
  - ✅ "Play Recommendation" button submits the recommendation via `/api/ai/play-recommendation`.
  - ✅ Session namespace `trainer:<game_id>` — isolated from AI-player session in same game.
  - ✅ Requires `USE_LLM=true` on the AI server.


  # after llm test run
  - clear all ppo training runs from the db - keep all human played games, even if played against ai. identify all games to keep by checking if player names ["Sandra", "Peter"] (ignore case) are part of the game
  - remove the feature to log the ppo training runs - that is not required, we can always use the export feature to get the same information from the database - correct? If not correct, argue why and keep feature
  - src/server/tools/export_training_data.ts shoudl skip all games where the jsonl log files already exist in the folder  
  - export all human game logs again
    - analyse my latest play in ga097581101aa - copy the exported game log to logs/llm-test/ga097581101aa - ai and tm server logs are already there. 
    - why does the ai not fund the milestone when it is available? system prompt indicates milestones are important! 
    - it seesm the ai playes well until generation 8, then it blunders and does not perform. what happend? context window full? why does the automatic context reset not work - check the logs if that was even triggered.
    - make the llm model selection choosable - gemini-3-pro gemini-3-flash gemini-3-flash-light along with the 2.5 versions shall be selectable via env vars
    - game log still contains last 20-30 moves. crop to the last moves of the opponents
    - I think it might be good to ask the llm to re-iterate over its strategy at the end of each generation and force an output (also for debugging) - makes sense? if yes, implement!
    - add to system prompt the benefits you get for specific level of temperature, oxygen and venus scale increasing - check the terraforming-mars implementation for details
- update the specification then implement the changes


# ai trainer fix
- ai trainer side is a player-specific toogle button, not a game wide one.
- from the chat it seems the llm has a hard time to separate both players. make sure that each player get's a separat ai trainer
- update prompt of ai trainer to make output shorter
- add to prompt not to use markdown, just plain text for formatting
- play recommendation should trigger a refresh of the page in a similar way as if the the "play" button is pressed by the user
- ai trainer did not recommend to take any of the initial cards
- it seems some keys like s or d make the page jump to a specific location - this prevents chatting. disable this feature when ai chat is on
- update the specification then implement the changes



