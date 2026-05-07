## AI integration adaptation for the local Terraforming Mars repo

CAUTION: this specification was already implemented in the TM repo and is contained here for reference only.

### Goal

Add a lightweight AI player integration to the local Terraforming Mars repo that:

- supports AI players via a new `isAI` flag on the player model,
- makes an HTTP request for each AI decision to an external AI server,
- logs game states, legal actions, and chosen moves in a training-friendly JSON format.

This should keep the existing game loop intact and add AI behavior as a clean extension point.

### Repo context

This repo is a Vue 3 + Node.js implementation of Terraforming Mars.
Key integration points for AI support are:

- `src/server/Player.ts` — player action flow and waiting-for-input logic,
- `src/server/Game.ts` — game lifecycle and end-of-game state,
- `src/server/routes/ApiCreateGame.ts` — new-game endpoint and player creation,
- `src/common/game/NewGameConfig.ts` — client/server game creation payload definitions,
- `src/client/components/create/CreateGameForm.vue` — game setup UI,
- `src/server/database/LocalFilesystem.ts` — existing JSON persistence.

The cleanest implementation is a boolean `isAI` flag rather than a separate player enum.

## Implementation plan

### 1. Add AI player support in game setup

- Extend `src/common/game/NewGameConfig.ts` so `NewPlayerModel` includes `isAI?: boolean`.
- Update `src/client/components/create/CreateGameForm.vue` to add a per-player AI toggle in the setup UI.
- Preserve the flag in the serialized `NewGameConfig` payload.
- Update `src/server/routes/ApiCreateGame.ts` to set `player.isAI = obj.isAI === true` when new players are created.
- Add `public isAI: boolean = false;` to `src/server/Player.ts` and include it in `serialize()` / deserialization if needed.

This keeps the new feature minimal and compatible with existing human-player setup.

### 2. Intercept AI turns in the server

The server already computes player decisions using `Player.takeAction()` and `Player.waitingFor`. AI integration should hook there.

- Detect AI players during the turn flow by checking `player.isAI`.
- When the AI player is waiting for input, serialize the current game state and available legal actions.
- Send that payload to the configured AI server endpoint.
- Apply the returned action through the game engine using the existing `InputResponse` / action-processing path.

This approach avoids replacing the entire game loop and keeps AI behavior localized to player decision handling.

### 3. Serialize game state and legal actions for the AI server

Build a mapper from the current `Game` / `Player` state into an AI payload.
Required payload fields should include:

- global state: generation, temperature, oxygen, oceans, phase,
- each player: TR, resources, production, tags, played cards,
- current player: hand cards, active projects, board state context,
- legal actions: flattened action IDs with explicit parameters.

The repo uses `PlayerInput` objects such as `OrOptions`, `SelectCard`, `SelectSpace`, etc. Build a serializer layer that converts those inputs into a canonical `legal_actions` list and a reverse mapper from AI response back into a `PlayerInput` response.

### 4. Add AI client and configuration

Create a simple AI client in `src/server/ai/AiClient.ts`:

- read `AI_SERVER_URL`, `AI_TIMEOUT_MS`, and `LOG_DIR` from environment variables,
- POST to `AI_SERVER_URL/move`,
- parse the AI response into `{ action_id, parameters }`,
- throw on timeout or non-OK response.

Also add `src/server/ai/index.ts` to export a shared client instance.

This keeps the external dependency isolated and reusable.

### 5. Log training data

Use the existing JSON persistence model to add training logs without breaking the current database.

- Keep the existing `LocalFilesystem` persistence for saved games.
- Add a training log writer in `src/server/ai/TrainingLogger.ts`.
- Write one JSON file per game or per training episode under `LOG_DIR`.
- Record for each decision:
  - `game_id`, `player_id`, generation, phase,
  - serialized state,
  - legal actions,
  - chosen action + parameters,
  - `is_human` / `is_ai`.
- At game end, record final results per player and the full turn history.

Example log structure:

```json
{
  "game_id": "...",
  "players": [ ... ],
  "turns": [
    {
      "step": 0,
      "player_id": "...",
      "state": { ... },
      "legal_actions": [ ... ],
      "action": { "action_id": "...", "parameters": { ... } },
      "is_human": true
    }
  ],
  "final_result": {
    "player_results": [
      { "player_id": "...", "tr": 67, "vp_total": 95, "rank": 1 }
    ]
  }
}
```

This output is suitable for supervised learning and reinforcement learning analysis.

## Repo-specific review

### What is already available

- `src/server/Player.ts` is the main hook for player action flow and waiting-for-input semantics.
- `src/server/routes/ApiCreateGame.ts` currently builds players from `NewGameConfig`.
- The client `CreateGameForm.vue` already serializes players and can be extended with an AI flag.
- `src/server/database/LocalFilesystem.ts` already writes serialized game JSON, so training log support can be added on top.

### Key technical gap

The main challenge is mapping the repo's internal `PlayerInput` tree into a flat `legal_actions` representation and back. This is the core serialization work for the AI integration.

## Suggested file changes

- `src/common/game/NewGameConfig.ts`
- `src/client/components/create/CreateGameForm.vue`
- `src/server/routes/ApiCreateGame.ts`
- `src/server/Player.ts`
- `src/server/ai/AiClient.ts`
- `src/server/ai/index.ts`
- `src/server/ai/TrainingLogger.ts`
- `src/server/ai/stateMapping.ts`
- `src/server/database/LocalFilesystem.ts` (optional: hook training logs)
- `tests/server/ai/*.spec.ts`

## Example AI client and flow

### AI client implementation

```ts
// src/server/ai/AiClient.ts
import { URL } from "node:url";

const AI_SERVER_URL = process.env.AI_SERVER_URL ?? "http://localhost:8000";
const AI_TIMEOUT_MS = Number(process.env.AI_TIMEOUT_MS ?? "5000");

export interface GameStatePayload {
  generation: number;
  temperature: number;
  oxygen: number;
  oceans: number;
  currentPlayerId: string;
  players: Array<{
    playerId: string;
    tr: number;
    resources: Record<string, number>;
    production: Record<string, number>;
    tags: Record<string, number>;
    playedCards: string[];
  }>;
  hand: Array<{ cardId: string; cost: number; tags: string[] }>;
  phase: string;
}

export interface LegalAction {
  actionId: string;
  parameters?: Record<string, unknown>;
}

export interface MoveRequestPayload {
  game_id: string;
  player_id: string;
  state: GameStatePayload;
  legal_actions: LegalAction[];
  metadata: { schema_version: number };
}

export interface MoveResponsePayload {
  action_id: string;
  parameters?: Record<string, unknown>;
  debug?: { policy_logits?: number[]; value_estimate?: number };
}

export class AiClient {
  private readonly baseUrl: string;

  constructor(baseUrl = AI_SERVER_URL) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
  }

  async requestMove(payload: MoveRequestPayload): Promise<MoveResponsePayload> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), AI_TIMEOUT_MS);

    try {
      const url = new URL("/move", this.baseUrl).toString();
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      if (!res.ok) {
        throw new Error(`AI server returned status ${res.status}`);
      }

      return (await res.json()) as MoveResponsePayload;
    } finally {
      clearTimeout(timeout);
    }
  }
}
```

### AI decision hook example

```ts
// repo-specific example: server code should call this when an AI player waits for input
import { aiClient } from "./ai/index";

export async function handleAiPlayerTurn(game: Game, player: Player): Promise<void> {
  const waitingFor = player.waitingFor;
  if (!player.isAI || !waitingFor) {
    return;
  }

  const state = buildGameStatePayload(game, player);
  const legalActions = flattenPlayerInputToLegalActions(waitingFor);

  const request = {
    game_id: game.id,
    player_id: player.id,
    state,
    legal_actions: legalActions,
    metadata: { schema_version: 1 },
  };

  try {
    const response = await aiClient.requestMove(request);
    const chosen = legalActions.find((a) => a.actionId === response.action_id);

    if (!chosen) {
      return applyFallbackAction(game, player, legalActions);
    }

    const inputResponse = buildInputResponse(response, chosen);
    player.process(inputResponse);
  } catch (err) {
    applyFallbackAction(game, player, legalActions);
  }
}
```

This is the first iteration: once the AI client is in place, the remaining work is to make sure the action mapping covers the repo's actual `PlayerInput` types.

## Testing note

Mark the test section for future implementation. The AI integration should be verified with:

- unit tests for serialization and response validation,
- integration tests with a dummy AI server,
- end-to-end tests where AI and human players run through a small game.

> The actual implementation should be repository-specific and use the existing `Player` / `Game` classes and serialization patterns. Keep tests as planned future work and focus first on the AI hook, state mapping, and logging paths.
