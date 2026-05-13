# AI Integration for the Terraforming Mars Server

## Goal

Add AI player support to the local Terraforming Mars server (Vue 3 + Node.js):
- `isAI` flag on the player model
- HTTP request to an external AI server for each AI player decision
- Real-time logging of game states and decisions (Plan B) for training
- Historical export from the SQLite DB (Plan A) — remaining work

The existing game loop must remain intact; AI behavior is a clean extension point.

---

## Implementation Status

| Component | Status | Location |
|---|---|---|
| `isAI` flag on `Player` | ✅ Done | `src/server/Player.ts` |
| Auto-trigger AI on `setWaitingFor` | ✅ Done | `src/server/Player.ts` — `setWaitingFor()` |
| `aiFallbackResponse()` (OrOptions → last SelectOption) | ✅ Done | `src/server/Player.ts` |
| `requestAiMove()` with fallback on failure | ✅ Done | `src/server/Player.ts` |
| `AiClient.ts` HTTP client | ✅ Done | `src/server/ai/AiClient.ts` |
| `stateMapping.ts` — extended state (opponents, board, milestones, awards) | ✅ Done | `src/server/ai/stateMapping.ts` |
| `TrainingLogger.ts` — per-game JSONL with meta/turn/result records | ✅ Done | `src/server/ai/TrainingLogger.ts` |
| Plan B: `setWaitingFor()` captures `pendingTrainingState` for all players | ✅ Done | `src/server/Player.ts` |
| Plan B: `process()` calls `logTrainingTurn()` for all players | ✅ Done | `src/server/Player.ts` |
| `writeMeta()` called at game creation | ✅ Done | `src/server/routes/ApiCreateGame.ts` |
| `writeResult()` called at game end | ✅ Done | `src/server/Game.ts` — `gotoEndGame()` |
| `takeAction(saveBeforeTakingAction)` flag fixed | ✅ Done | `src/server/Player.ts` — gates `game.save()` correctly |
| `isAI` flag from game creation API | ✅ Done | `src/server/routes/ApiCreateGame.ts` |
| AI toggle in `CreateGameForm.vue` | ✅ Done | `src/client/components/create/CreateGameForm.vue` |
| `isAI` exposed in `ServerModel` / `PlayerModel` | ✅ Done | `src/server/models/ServerModel.ts` |
| Plan A: `export_training_data.ts` (re-run engine on DB saves) | ✅ Done | `src/server/tools/export_training_data.ts` |
| Self-play `POST /api/ai/new-game` + `/api/ai/step` | ✅ Done | `src/server/routes/ApiAiSelfPlay.ts` |
| `game.isSelfPlay` flag (suppresses auto AI trigger) | ✅ Done | `src/server/Game.ts`, `IGame.ts`, `Player.ts` |

---

## API Format

The TM server uses a **`provide_input` paradigm**: it sends the full `PlayerInputModel` decision tree to the AI server, which returns a raw `InputResponse`.

### Request (TM Server → AI Server)

```json
{
  "game_id": "g123...",
  "player_id": "p456...",
  "state": {
    "game": {"id": "g123...", "phase": "action", "generation": 7, "oxygen": 8, "temperature": -12, "oceanCount": 5},
    "player": {
      "id": "p456...", "name": "Alice", "color": "blue",
      "terraformRating": 42, "megacredits": 25, "steel": 3, "titanium": 1,
      "plants": 5, "energy": 2, "heat": 6, "handSize": 4,
      "production": {"megacredits": 4, "steel": 1, "titanium": 0, "plants": 2, "heat": 0, "energy": 1},
      "tags": {"science": 2, "building": 3, "space": 1},
      "isAI": true, "playedCards": [...], "corporations": [...]
    },
    "opponents": [{"id": "p2", "terraformRating": 38, ...}],
    "board": [{"id": "H05", "x": 3, "y": 2, "tileType": 0, "playerColor": "blue"}, ...],
    "milestones": [{"name": "Terraformer", "playerId": "p456..."}],
    "awards": [{"name": "Landlord", "playerId": "p456..."}],
    "waitingFor": {"type": "or", "title": "Take action", "options": [...]}
  },
  "legal_actions": [
    {
      "action_id": "provide_input",
      "type": "or",
      "title": "Take action",
      "payload": {"input": {"type": "or", "title": "Take action", "options": [...]}}
    }
  ],
  "metadata": {"schema_version": 1}
}
```

`legal_actions` always contains exactly one entry. The `PlayerInputModel` tree is in both `state.waitingFor` and `legal_actions[0].payload.input`.

### Response (AI Server → TM Server)

```json
{
  "input_response": {"type": "or", "index": 2, "response": {"type": "option"}},
  "debug": {"policy_logits": [1.2, -0.3, 0.8], "value_estimate": 0.35}
}
```

`input_response` is passed directly to `player.process()`. `debug` is optional and ignored by the game.

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

### Error Handling

- Timeout or non-OK HTTP status: fallback to `aiFallbackResponse()` (last `SelectOption` in `OrOptions`, or first option as best-effort)
- Timeout is controlled by `AI_TIMEOUT_MS` env var (default 600000ms / 10 min). A custom `undici.Agent` is used with matching `headersTimeout`/`bodyTimeout` to prevent Node.js undici's internal 300s headers timeout from firing before `AbortController`.
- If no fallback found: error logged, no action applied (AI player stalls)

---

## State Mapping (`stateMapping.ts`)

`buildAiRequestState()` extracts a full-game view via `buildPlayerSnapshot()` and `buildBoardState()`:

- **`game`**: id, phase, generation, oxygen, temperature, oceanCount
- **`player`** (active player): all resources, production, tags, handSize, isAI, playedCards (card names), corporations
- **`opponents`** (all other players): same snapshot fields as player
- **`board`**: placed tiles — spaceId, x, y, tileType, playerColor
- **`milestones`**: claimed milestones — name, playerId
- **`awards`**: funded awards — name, playerId
- **`waitingFor`**: the full `PlayerInputModel` for the current decision

---

## Plan B: Real-time Decision Logging

Logging is wired in `Player.ts` for all players (human and AI):

1. **`setWaitingFor()`** — captures `{step, state, waitingFor}` into `this.pendingTrainingState` for every player before each decision
2. **`process()`** — calls `logTrainingTurn(input)` which pairs the captured state with the chosen `InputResponse` and appends it to the game's JSONL file
3. **`ApiCreateGame.ts`** — calls `TrainingLogger.writeMeta()` at game creation to write the game_spec and player list
4. **`Game.gotoEndGame()`** — calls `TrainingLogger.writeResult()` to write final VP and rankings

### JSONL File Format

One file per game: `ai_training_logs/{game_id}.jsonl` (env var: `AI_TRAINING_LOG_DIR`)

Three record types, one per line:

```jsonl
{"type":"meta", "game_id":"g123", "game_spec":{"board_name":"tharsis","player_count":2,"expansions":["corpEra","venus"],"variants":{...},"created_at":"..."}, "players":[{"playerId":"p1","name":"Alice","isAI":false},...]}
{"type":"turn", "step":0, "playerId":"p1", "generation":3, "phase":"action", "timestamp":"...", "state":{...}, "waitingFor":{...}, "input_response":{"type":"or","index":1,"response":{"type":"option"}}, "is_human":true}
{"type":"result", "endGeneration":14, "playerResults":[{"playerId":"p1","name":"Alice","tr":67,"vp_total":95,"rank":1},...]}
```

`dataset.py` reads this format: it requires all three record types; games missing meta or result are skipped.

---

## Self-Play API (Phase 2)

Two endpoints for PPO self-play training, implemented in `src/server/routes/ApiAiSelfPlay.ts`.

### `POST /api/ai/new-game`
Creates a 2-player self-play game (both players have `isAI=true`, `game.isSelfPlay=true`).
`isSelfPlay` suppresses the auto `requestAiMove()` trigger in `setWaitingFor()`.

Request body (all optional):
```json
{"boardName": "tharsis", "logDir": "logs/selfplay/<run_id>"}
```

Response:
```json
{
  "game_id": "g...",
  "player_id": "p...",
  "state": {...},
  "waitingFor": {...},
  "game_spec": {...}
}
```

### `POST /api/ai/step`
Applies one player's `InputResponse`, returns the next player's state or end-of-game result.

Request:
```json
{"game_id": "g...", "player_id": "p...", "input_response": {...}}
```

Response (mid-game):
```json
{"done": false, "player_id": "p...", "state": {...}, "waitingFor": {...}, "result": null}
```

Response (game over):
```json
{
  "done": true, "player_id": null, "state": null, "waitingFor": null,
  "result": {"endGeneration": 14, "playerResults": [{"playerId":"p...","tr":67,"vp_total":95,"rank":1},...]}
}
```

The Python env (`env_tm.py`) calls these endpoints sequentially; the model plays both AI players.

---

## Plan A: Re-run Engine on DB Saves

**Goal:** Extract training tuples from the 82 existing historical games in the SQLite DB.

**Approach:** `Game.deserialize()` re-triggers `player.setWaitingFor()` automatically, so the `waitingFor` tree is available after loading any save. Consecutive save pairs (N, N+1) reveal what decision was made by diffing the game log messages between saves.

**Key prerequisite already done:** `takeAction(saveBeforeTakingAction=false)` is now respected — the export tool can load saves without writing spurious DB entries.

**Steps for `export_training_data.ts`:**
1. Iterate all games from the DB
2. For each game, iterate consecutive save pairs (saveId N → N+1)
3. Deserialize save N → `activePlayer.waitingFor` is set automatically
4. Capture `waitingFor.toModel(player)` and state via `buildAiRequestState()`
5. Diff game log messages between save N and N+1 to infer the chosen action
6. Write the training turn record
7. At the last save, compute final VP and write the result record

**Remaining risks:** Inferring the chosen action from log message diffs is imprecise — some saves may be mid-deferred-action rather than clean decision points. Filter by checking if `activePlayer` or `phase` changed.

---

## PlayerInput Model Reference

The `waitingFor` object is a `PlayerInputModel` — a recursive decision tree defined in `src/common/models/PlayerInputModel.ts`.

Full union of types: `OrOptions | AndOptions | SelectInitialCards | SelectOption | SelectProjectCardToPlay | SelectCard | SelectAmount | SelectColony | SelectDelegate | SelectParty | SelectPayment | SelectPlayer | SelectProductionToLose | SelectSpace | ShiftAresGlobalParameters | SelectGlobalEvent | SelectPolicy | SelectResource | SelectResources | SelectClaimedUndergroundToken`

---

## File Reference

| File | Purpose | Status |
|---|---|---|
| `src/server/Player.ts` | isAI flag, setWaitingFor, process, requestAiMove, fallback, Plan B logging | ✅ Done |
| `src/server/Game.ts` | writeResult at game end | ✅ Done |
| `src/server/ai/AiClient.ts` | HTTP client to AI server | ✅ Done |
| `src/server/ai/stateMapping.ts` | Full state build (player + opponents + board + milestones + awards) | ✅ Done |
| `src/server/ai/TrainingLogger.ts` | writeMeta / appendTurn / writeResult; per-game JSONL | ✅ Done |
| `src/server/ai/index.ts` | Re-exports | ✅ Done |
| `src/server/routes/ApiCreateGame.ts` | isAI on player create, writeMeta at game start | ✅ Done |
| `src/server/models/ServerModel.ts` | isAI exposed in game/player models | ✅ Done |
| `src/common/game/NewGameConfig.ts` | isAI field on NewPlayerModel | ✅ Done |
| `src/client/components/create/CreateGameForm.vue` | AI player checkbox in game setup UI | ✅ Done |
| `src/server/tools/export_all_logs.ts` | Exports game display logs (not training data) | ✅ Done |
| `src/server/tools/export_training_data.ts` | Re-run engine on DB saves (Plan A) | ✅ Done |
| `src/server/routes/ApiAiSelfPlay.ts` | POST /api/ai/new-game + /api/ai/step | ✅ Done |
| `src/server/IGame.ts` | `isSelfPlay: boolean` field | ✅ Done |
| `src/common/app/paths.ts` | `API_AI_NEW_GAME` + `API_AI_STEP` path constants | ✅ Done |
