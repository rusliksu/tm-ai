# AI Integration for the Terraforming Mars Server

## Goal

Add LLM-player support to the local Terraforming Mars server (Vue 3 + Node.js) with the
**minimal** adaption from upstream `main`:
- an `isAI` flag on the player model,
- an HTTP request to the external AI server for each AI decision,
- a self-play driver API for headless all-LLM games.

The existing game loop stays intact; AI behavior is a clean extension point. There is no
training-data logging and no in-game AI Trainer (both removed in the 0.2 reimplementation).

---

## Implementation Status

| Component | Location |
|---|---|
| `isAI` flag on `Player` | `src/server/Player.ts` |
| Auto-trigger AI on `setWaitingFor` (when `isAI && !isSelfPlay`) | `src/server/Player.ts` |
| `aiFallbackResponse()` (OrOptions → last SelectOption) | `src/server/Player.ts` |
| `requestAiMove()` with retry + `last_error` feedback, fallback on failure | `src/server/Player.ts` |
| `AiClient.ts` HTTP client (`requestMove`) | `src/server/ai/AiClient.ts` |
| `stateMapping.ts` — full state (player + opponents + board + boardSpaces + milestones/awards + recentLog + config) | `src/server/ai/stateMapping.ts` |
| `isAI` from game-creation API + UI checkbox | `src/server/routes/ApiCreateGame.ts`, `src/client/components/create/CreateGameForm.vue` |
| `isAI` exposed in `ServerModel` / `PlayerModel` | `src/server/models/ServerModel.ts` |
| Self-play `POST /api/ai/new-game` + `/api/ai/step` | `src/server/routes/ApiAiSelfPlay.ts` |
| `game.isSelfPlay` flag (suppresses auto AI trigger) | `src/server/Game.ts`, `IGame.ts`, `Player.ts` |
| `extract_card_db.ts` — renderData traversal for prelude/CEO descriptions | `src/server/tools/extract_card_db.ts` |

---

## API Format

The TM server uses a **`provide_input` paradigm**: it sends the full `PlayerInputModel`
decision tree to the AI server, which returns a raw `InputResponse`.

### Request (TM Server → AI Server)

```json
{
  "game_id": "g123...",
  "player_id": "p456...",
  "state": {
    "game": {"id","phase","generation","oxygen","temperature","oceanCount",
             "boardName","expansions","availableMilestones","availableAwards",
             "gameVariants","recentLog"},
    "player": {"...resources/production/tags...","handSize","isAI","playedCards",
               "corporations","cardResources","boardTiles","victoryPoints","cardsInHand"},
    "opponents": [{"...","handSize"}],
    "board": [{"id","x","y","tileType","playerColor"}],
    "boardSpaces": [{"id","x","y","t","b","v?","tile?","pc?"}],
    "milestones": [{"name","playerId"}], "awards": [{"name","playerId"}],
    "waitingFor": {"type":"or","title":"Take action","options":[...]}
  },
  "legal_actions": [{"action_id":"provide_input","type":"or","title":"Take action",
                     "payload":{"input": {...}}}],
  "metadata": {"schema_version": 1},
  "last_error": "optional — set when the previous response was rejected"
}
```

`legal_actions` always contains exactly one entry; the tree is in both `state.waitingFor` and
`legal_actions[0].payload.input`.

### Response (AI Server → TM Server)

```json
{"input_response": {"type": "or", "index": 2, "response": {"type": "option"}}, "debug": {...}}
```

`input_response` is passed directly to `player.process()`. `debug` is optional and ignored.

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

- Timeout / non-OK HTTP status → `aiFallbackResponse()` (last `SelectOption` in `OrOptions`,
  else best-effort first option).
- Timeout is `AI_TIMEOUT_MS` (default 600000ms / 10 min). A custom `undici.Agent` with matching
  `headersTimeout`/`bodyTimeout` prevents undici's internal 300s timeout firing early.
- `player.process()` failure (e.g. "not enough resources") → `requestAiMove` retries up to 2×
  with `last_error` set so the AI can correct, then falls back to `aiFallbackResponse()`.

---

## State Mapping (`stateMapping.ts`)

`buildAiRequestState()` builds the full-game view via `buildPlayerSnapshot()`,
`buildBoardState()`, and `buildAllBoardSpaces()`:

- **`game`**: id, phase, generation, oxygen, temperature, oceanCount, boardName, expansions,
  availableMilestones / availableAwards (name + description), gameVariants, `recentLog`
  (serialized log since the start of the current generation, cap 25).
- **`player`** (active): resources, production, tags, handSize, isAI, playedCards,
  corporations, cardResources (per-card `{name:count}`), boardTiles, victoryPoints, and
  `cardsInHand` (card names — **self only**, opponents' hands are secret).
- **`opponents`**: same snapshot minus `cardsInHand`; `handSize` is a count only.
- **`board`**: placed tiles — id, x, y, tileType, playerColor.
- **`boardSpaces`**: every hex — id, x, y, `t` (spaceType), `b` (bonus names), `v?` (volcanic),
  and `tile?`/`pc?` for occupied hexes. Drives the AI server's live-board rendering + adjacency.
- **`milestones`** / **`awards`**: claimed/funded — name, playerId.
- **`waitingFor`**: the full `PlayerInputModel` for the current decision.

`recentLog` serialization (`serializeLogMessage`) substitutes `${N}` placeholders: `PLAYER` →
name (by color), `TILE_TYPE` → tile name, `SPACE_BONUS` → resource name, `SPACE` → `hex-<id>`,
`CARDS` → comma-joined. `getRecentLog` **includes the AI's own moves** as well as opponents'
and system messages — the AI is stateless per turn, so the log is how it learns what it did
this generation.

---

## Self-Play API

Two endpoints in `src/server/routes/ApiAiSelfPlay.ts` drive headless all-LLM ("death match")
games. `game.isSelfPlay=true` suppresses the auto `requestAiMove()` trigger in
`setWaitingFor()`; the driver advances the game explicitly. Games persist to the DB.

### `POST /api/ai/new-game`
Creates a self-play game (all players `isAI=true`, `isSelfPlay=true`).
Request (all optional): `{"boardName":"tharsis","playerCount":2,"playerNames":["Claude","GPT"]}`.
Response: `{game_id, player_id, spectator_id, state, waitingFor, game_spec}`.
`spectator_id` → `http://localhost:8080/spectator?id=<spectator_id>`. `play_game.py` writes
this URL to `/tmp/current-game.url`, which the start scripts open in Chrome.

### `POST /api/ai/step`
Applies one player's `InputResponse`, returns the next state or the end result.
Request: `{game_id, player_id, input_response}`.
- Mid-game: `{done:false, player_id, state, waitingFor, result:null}`.
- Game over: `{done:true, player_id:null, state:null, waitingFor:null, result:{endGeneration, playerResults:[{playerId,name,tr,vp_total,rank},...]}}`.

### `scripts/play_game.py` — multi-LLM driver
- Calls `POST /player/register` before the game to assign one LLM per seat
  (`--models "a/m1,b/m2,..."`).
- Writes the spectator URL to `/tmp/current-game.url` after `POST /api/ai/new-game`.
- On a TM-server `POST /api/ai/step` rejection (HTTP 400): re-calls `POST /move` with
  `last_error` and retries `/step` up to `_MAX_STEP_RETRIES=2` before aborting.
- Calls `POST /game-done` at game end to flush per-player token/cost logs.

---

## PlayerInput Model Reference

`waitingFor` is a `PlayerInputModel` — a recursive decision tree
(`src/common/models/PlayerInputModel.ts`). Full union: `OrOptions | AndOptions |
SelectInitialCards | SelectOption | SelectProjectCardToPlay | SelectCard | SelectAmount |
SelectColony | SelectDelegate | SelectParty | SelectPayment | SelectPlayer |
SelectProductionToLose | SelectSpace | ShiftAresGlobalParameters | SelectGlobalEvent |
SelectPolicy | SelectResource | SelectResources | SelectClaimedUndergroundToken`.

---

## File Reference

| File | Purpose |
|---|---|
| `src/server/Player.ts` | isAI flag, setWaitingFor trigger, process, requestAiMove + retry/fallback |
| `src/server/Game.ts`, `src/server/IGame.ts` | `isSelfPlay` field |
| `src/server/ai/AiClient.ts` | HTTP client (`requestMove`); `MoveRequestPayload` includes optional `last_error` |
| `src/server/ai/stateMapping.ts` | Full state payload (see above) |
| `src/server/ai/index.ts` | Re-exports (`AiClient`, `stateMapping`) |
| `src/server/routes/ApiAiSelfPlay.ts` | `POST /api/ai/new-game` + `/api/ai/step` |
| `src/server/routes/ApiCreateGame.ts` | `isAI` on player create |
| `src/server/models/ServerModel.ts` | `isAI` exposed in game/player models |
| `src/common/game/NewGameConfig.ts` | `isAI` field on NewPlayerModel |
| `src/client/components/create/CreateGameForm.vue` | AI player checkbox in game setup |
| `src/server/tools/extract_card_db.ts` | Extract card DB (descriptions/tags) into `tm-ai/data/card_db.json` |
| `src/common/app/paths.ts` | `API_AI_NEW_GAME` + `API_AI_STEP` path constants |
