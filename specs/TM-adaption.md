## AI Integration for the Terraforming Mars Server

### Goal

Add AI player support to the local Terraforming Mars server (Vue 3 + Node.js):
- `isAI` flag on the player model
- HTTP request to an external AI server for each AI player decision
- Logging of game states and decisions in a training-friendly format for both human and AI players

The existing game loop must remain intact; AI behavior is a clean extension point.

---

## Implementation Status

| Component | Status | Location |
|---|---|---|
| `isAI` flag on `Player` | ✅ Done | `src/server/Player.ts:258` |
| Auto-trigger AI on `setWaitingFor` | ✅ Done | `src/server/Player.ts:1718–1720` |
| `requestAiMove()` | ✅ Done | `src/server/Player.ts:1747–1795` |
| `AiClient.ts` HTTP client | ✅ Done | `src/server/ai/AiClient.ts` |
| `stateMapping.ts` state builder | ✅ Minimal | `src/server/ai/stateMapping.ts` |
| `TrainingLogger.ts` | ✅ Stub | `src/server/ai/TrainingLogger.ts` |
| `isAI` flag from game creation API | ✅ Done | `src/server/routes/ApiCreateGame.ts:103` |
| Human game decision logging | ❌ Not started | — |
| Training data export from existing games | ❌ Not started | — |
| AI toggle in `CreateGameForm.vue` | ❌ Not started | — |

---

## Actual API Format

> **Note:** The implemented API diverges from the original flat `actionId` design. The actual implementation uses a `provide_input` paradigm: the TM server sends the full `PlayerInput` decision tree, and the AI server returns a raw `InputResponse`.

### Request (TM Server → AI Server)

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

`legal_actions` always contains exactly one entry with `action_id: "provide_input"`. The full `PlayerInputModel` decision tree is in both `state.waitingFor` and `legal_actions[0].payload.input`.

### Response (AI Server → TM Server)

```json
{
  "input_response": {"type": "or", "responses": [{"index": 2}]},
  "debug": {
    "policy_logits": [1.2, -0.3, 0.8],
    "value_estimate": 0.35
  }
}
```

`input_response` is passed directly to `player.process(input_response)` in the TM server. It must be a valid serialized response to the `PlayerInput` type that was sent. `debug` is optional and logged but otherwise ignored.

### Error Handling (current behavior)

- Timeout (> 5000 ms): error logged, no action applied (AI player stalls)
- Non-OK HTTP status: error logged, no action applied
- `input_response` undefined: error logged, no action applied

No automatic fallback to a random or default action is currently implemented.

---

## State Mapping (`stateMapping.ts`)

`buildAiRequestState()` currently extracts a minimal single-player view:

- **`game`**: id, phase, generation, oxygen, temperature, oceanCount
- **`player`** (active player only): id, name, color, terraformRating, megacredits, steel, titanium, plants, heat, energy, handSize, production (all 6 resources), tags (all counts), isAI
- **`waitingFor`**: the full `PlayerInputModel` for the current decision

**Not yet included:**
- Other players' resources, production, tags, played cards
- Board state (tile placement, ocean/city/greenery positions)
- Milestones and awards (claimed, funded, scores)
- Active player's played cards

---

## PlayerInput Model Reference

The `waitingFor` and `payload.input` objects are `PlayerInputModel` instances — a recursive decision tree. Key types:

| Type | Description |
|---|---|
| `OrOptions` | Player chooses one of N options |
| `SelectCard` | Player selects from a list of cards |
| `SelectSpace` | Player selects a board hex |
| `SelectPlayer` | Player selects another player |
| `SelectAmount` | Player enters a number within a range |
| `AndOptions` | Sequence of inputs resolved in order |

The AI must return an `input_response` compatible with the active `PlayerInput` type. This is the core complexity of the AI implementation — there is no flat list of named actions.

---

## Training Data

### Current Situation

**The 80 exported JSON files in `logs/json/` are not usable for training.** The `export_all_logs.ts` script exports only `version.gameLog` (an array of human-readable display messages). It does not export game states, player resources, or actions.

The SQLite database (`db/game.db`) contains the full `SerializedGame` JSON at every save point:
- 82 games, 11,192 total save rows
- Largest game: 282 saves (~282 decision points)
- Each row contains complete state: players, board, cards, resources, game options

### Plan A: Re-run Engine During Export (for existing games)

**Concept:** `Game.deserialize()` already re-triggers the active player's turn at the end (line 1793 of `Game.ts`), which internally calls `player.setWaitingFor()`. This means after loading any save from the DB, `player.waitingFor` is populated automatically — legal actions are available without extra work.

**Steps:**
1. Iterate consecutive save pairs (N, N+1) for each game
2. Deserialize save N → `game.activePlayer.waitingFor` is set automatically
3. Capture `player.waitingFor.toModel(player)` as the `PlayerInputModel`
4. Compare game logs between save N and N+1 to determine which option was chosen
5. Extract state features from `SerializedGame`

**Risks and mitigations:**

| Risk | Detail | Mitigation |
|---|---|---|
| `game.save()` fires during deserialization | `takeAction()` calls `game.save()` when `actionsTakenThisRound === 0`. The `saveBeforeTakingAction` parameter is currently `@ts-ignore`d and does nothing. This means loading a save for export would write spurious saves to the DB. | Patch `takeAction()` to honour the flag during export, or use a read-only DB connection |
| Chosen action inference is hard | No chosen action is stored; must be inferred by diffing consecutive saves or parsing log message deltas | GameLog messages between saves describe what happened; build a parser or diff player state |
| Non-decision saves | Some consecutive saves are mid-deferred-action (not at a clean decision boundary) | Filter: only emit a training tuple when `activePlayer` changes or `phase` changes |
| Undo pollution | DB stores saves after undo operations, creating contradictory sequences | Detect and skip games with `undoCount > 0`, or cross-reference `lastSaveId` monotonicity |
| `PlayerInputModel` is a tree, not a flat list | The `waitingFor` model has variable depth and structure | Accept the tree as-is; flatten to a canonical list during dataset preprocessing in Python |

### Plan B: Real-time Hook (for future games)

Add logging in `Player.setWaitingFor()` (captures state + waitingFor before decision) and `Player.process()` (captures the chosen `InputResponse` after decision) for **both AI and human players**.

This produces clean `(state, waitingFor, input_response, is_human)` tuples at every decision point without any inference or diffing.

**Recommended:** Implement Plan A for historical data and Plan B simultaneously so all future games produce training data automatically.

### Training Log Format

Per-game log file (one file per game):

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

---

## Remaining Work

- [ ] Fix `takeAction()` to honour `saveBeforeTakingAction=false` (required for Plan A)
- [ ] Create `src/server/tools/export_training_data.ts` (Plan A: re-run engine on DB saves)
- [ ] Add logging hook to `Player.setWaitingFor()` + `Player.process()` (Plan B)
- [ ] Extend `stateMapping.ts` to include opponent states, played cards, board
- [ ] Extend `TrainingLogger.ts` to write per-game files (not per-record files)
- [ ] Add AI toggle to `CreateGameForm.vue`
- [ ] Add fallback action when AI server fails (random legal option or pass)

---

## File Reference

| File | Purpose | Status |
|---|---|---|
| `src/server/Player.ts` | AI trigger on `setWaitingFor`, `requestAiMove` | ✅ Done |
| `src/server/ai/AiClient.ts` | HTTP client to AI server | ✅ Done |
| `src/server/ai/stateMapping.ts` | Build state payload | ✅ Minimal |
| `src/server/ai/TrainingLogger.ts` | Log training records (per-record, AI only) | ✅ Stub |
| `src/server/ai/index.ts` | Re-exports | ✅ Done |
| `src/server/routes/ApiCreateGame.ts` | Set `isAI` on player create | ✅ Done |
| `src/server/tools/export_all_logs.ts` | Exports game log messages only | ⚠️ Not training data |
| `src/server/tools/export_training_data.ts` | Re-run engine export (Plan A) | ❌ To create |
| `src/client/components/create/CreateGameForm.vue` | AI toggle in game setup UI | ❌ To add |
