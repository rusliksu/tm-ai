# LLM Player State Persistence

**Status:** PROPOSED — awaiting user review
**Owner:** tm-ai (Python AI server)
**Scope:** `tm-ai-server/` only. No changes to the TypeScript TM server.

## Problem

`_player_registry` and per-`LLMPlayer` state live in process memory only
(`llm_player.py:75-93`, `llm_player.py:384-407`). When the tm-ai server restarts
mid-game, the next `/move` call falls through `get_or_create_player`
(`llm_player.py:856-861`) and constructs a fresh `LLMPlayer` with the env-default
model and empty memory. Concretely the following is lost:

1. **Model assignment** — multi-LLM death-match players collapse to the env
   default model.
2. **Strategy (Part 2 memory)** — the coarse inter-generation engine/milestone
   plan. Heals at the next generation bump but is empty until then.
3. **Tactical (Part 1 memory)** — short next-steps list for the current
   generation. Lost for one turn, then re-written by the model.
4. **Token / cost counters** — the final `/game-done` summary is partial.
5. **`last_generation`** — `_maybe_per_generation_update` skips the per-gen
   refresh on the resume turn (treats it as the first turn instead of a bump).

The action phase itself is stateless by design (`single_shot` per turn, full
state snapshot from the TM server every call), so the game can keep running.
But the play quality drops, and for graded multi-model runs the lost model
identity is a correctness bug, not just a quality regression.

## Goal

On graceful shutdown of the tm-ai server, persist each in-flight LLM player's
state to a JSON file keyed by `player_id`. On the next startup, restore that
state lazily when the TM server next calls `/move` for that player. Delete
the file when the game finishes (so old files don't accumulate).

Non-goal: surviving `kill -9`. Best-effort graceful-shutdown persistence only.
(An optional incremental-save mode is sketched at the end but is out of scope
for the first cut.)

## Storage layout

```
logs/llm-state/
  <player_id>.json        # one file per AI player
```

* Directory configurable via `LLM_STATE_DIR` env var; default
  `<repo-root>/logs/llm-state` (resolved relative to the tm-ai repo root, so
  invocation cwd doesn't matter — match the existing `logs/training` and
  `logs/selfplay` convention).
* `player_id` is the canonical filename — TM server assigns unique random
  player IDs across all games, so no collisions across in-flight games.
* Trainer players (`player_id` starting with `trainer:`) are **not**
  persisted. They are tied to live human UI sessions; on restart the user
  simply reopens the chat sidebar and a fresh trainer session is started.

### File schema

```json
{
  "schema_version": 1,
  "saved_at": "2026-06-01T10:34:12Z",
  "player_id": "p4f3aab12",
  "game_id":   "g8c91dd03",
  "model":     "anthropic/claude-sonnet-4-6",
  "strategy":  "STANDING: ...\nENGINE: ...\n... (full Part 2 memory)",
  "tactical":  "1. Convert 8 heat. 2. Play Capital. ...",
  "last_generation": 4,
  "hand_shown_generation": 4,
  "token_usage": {
    "calls": 37, "input": 142000, "output": 11800,
    "cache_read": 98000, "cache_write": 5400, "thinking": 9200
  },
  "summary_logged": false
}
```

**Not persisted:**

* `session` — only used in the setup phase (`initialCards` / `prelude`). Setup
  happens once at game start; the chance of a restart landing mid-setup is
  vanishingly low, and even if it does, `recover_session` already handles the
  fallback (`llm_player.py:484`).
* `action_system` — rebuilt deterministically from `state` on first call
  (`_ensure_action_system`, `llm_player.py:1671`). Byte-identical to the
  pre-restart version, so prompt caching warms up again.
* `provider`, `base_system`, capability cache — re-derived from `model`.

## Behavior

### Save: on graceful shutdown (FastAPI lifespan)

In `main.py`'s `lifespan` context, after `yield`, call
`llm_player.save_all_active_players()`. Inside, iterate `_player_registry`:

* Skip `player_id.startswith("trainer:")`.
* Skip players whose `game_id` is in `_game_summary_logged` (game already
  finished — no point persisting).
* Atomically write `<player_id>.json` via `tmp + os.replace()`.
* Log one summary line: `LLM state persisted: N players → <dir>`.

Errors during save log a warning per player but do not crash shutdown.

### Restore: lazy, on first `/move` after restart

Modify `get_or_create_player(player_id, game_id)` (`llm_player.py:856`):

1. If `player_id in _player_registry` → return it (unchanged).
2. Else, try `try_load_player_state(player_id, game_id)`:
   * Read `logs/llm-state/<player_id>.json`.
   * If file missing → fall through to current behavior (auto-create + warn).
   * If `game_id` in file ≠ `game_id` argument → stale; delete the file, fall
     through to auto-create. (TM server creates a fresh game with a fresh
     `player_id`, so a stale file with a matching `player_id` but mismatched
     `game_id` should never happen in practice; treat it defensively.)
   * If `schema_version` is newer than what we know → log warning, fall
     through to auto-create (don't crash).
   * Otherwise hydrate a new `LLMPlayer`, set `strategy`, `tactical`,
     `last_generation`, `hand_shown_generation`, `token_usage`,
     `summary_logged`. Insert into `_player_registry` and `_game_players`.
     Log: `LLM state restored: player=<id> game=<id> model=<m> gen=<n>`.

`action_system` and `session` stay empty — they regenerate naturally on the
first `/move`.

### Cleanup: on `/game-done`

Modify `log_game_token_summary(game_id)` (`llm_player.py:864`): after logging
the summary, for each `player_id` in `_game_players[game_id]` delete
`<player_id>.json` if it exists. One log line:
`LLM state cleaned up: N files removed for game=<id>`.

Errors are warnings, not fatal.

### Stale-file pruning (cold-start hygiene)

On lifespan startup (before `yield`), scan `logs/llm-state/` and delete files
older than `LLM_STATE_MAX_AGE_DAYS` (default 7). Cheap insurance against
games that crashed without `/game-done` ever firing.

## Files to change

| File | Change |
|------|--------|
| `tm-ai-server/src/tm_ai_server/llm_player.py` | Add `_LLM_STATE_DIR`, `_state_path()`, `LLMPlayer.to_dict()` / `from_dict()`, `save_all_active_players()`, `clear_player_state(player_id)`, `try_load_player_state(player_id, game_id)`, `prune_stale_state(max_age_days)`. Modify `get_or_create_player` (lazy restore) and `log_game_token_summary` (delete files). |
| `tm-ai-server/src/tm_ai_server/main.py` | In `lifespan`: call `prune_stale_state()` before `yield`; call `save_all_active_players()` after `yield`. |
| `tm-ai-server/tests/test_llm_state_persistence.py` | New test file (see below). |
| `CLAUDE.md` | Document `LLM_STATE_DIR` + `LLM_STATE_MAX_AGE_DAYS` env vars and the persistence behavior in the LLM Player section. |
| `specs/LLM-state-persistence.md` | This file. |

No changes to `terraforming-mars/`.

## API additions (internal Python only)

```python
# llm_player.py
def save_all_active_players() -> int: ...
def try_load_player_state(player_id: str, game_id: str) -> LLMPlayer | None: ...
def clear_player_state(player_id: str) -> None: ...
def prune_stale_state(max_age_days: int = 7) -> int: ...

class LLMPlayer:
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, data: dict) -> "LLMPlayer": ...
```

No new HTTP endpoints.

## Env vars

| Var | Default | Description |
|-----|---------|-------------|
| `LLM_STATE_DIR` | `<repo-root>/logs/llm-state` | Where per-player state files live |
| `LLM_STATE_MAX_AGE_DAYS` | `7` | Files older than this are pruned at startup |
| `LLM_STATE_PERSIST` | `true` | Master switch — set `false` to disable entirely |

## Tests (`tm-ai-server/tests/test_llm_state_persistence.py`)

1. **Roundtrip**: build an `LLMPlayer`, populate strategy/tactical/last_generation/
   token_usage, call `to_dict()` → JSON → `from_dict()`, assert all persisted
   fields match. Confirm `session` and `action_system` are empty in the
   reconstructed instance.
2. **Save and restore via registry**: register two players, mutate their state,
   call `save_all_active_players()`, clear the registry, call
   `get_or_create_player(player_id, game_id)` and assert the player is
   restored with full state (no warning logged).
3. **Trainer players skipped**: register `trainer:p1`, call save, assert no
   file written.
4. **Finished games skipped**: register `p1` in game `g1`, add `g1` to
   `_game_summary_logged`, save, assert no file written for `p1`.
5. **Stale `game_id` discarded**: write a state file with `game_id=gX`, call
   `get_or_create_player(player_id, "gY")`, assert auto-create fallback and
   the stale file is deleted.
6. **`/game-done` cleanup**: register player, save, call
   `log_game_token_summary(game_id)`, assert the file is gone.
7. **`prune_stale_state`**: write two state files with mtimes 1 day and 30
   days old, run prune with `max_age_days=7`, assert only the old one is
   deleted.

Tests use `tmp_path` fixture and monkeypatch `_LLM_STATE_DIR` so they don't
touch the real `logs/llm-state/` directory.

## Edge cases & risks

* **`kill -9` / crash**: lifespan does not run → no save. Accepted. (Optional
  future work: incremental save after each `_per_generation_strategy_update`
  and after `_capture_tactical`. Disk overhead is tiny — one JSON write per
  generation per player. Out of scope for v1.)
* **Concurrent shutdowns**: lifespan is single-threaded; no concurrency.
* **Disk full / permission denied**: log warning, continue; never crash
  shutdown or the next `/move`.
* **Schema evolution**: include `schema_version`. On bump, restore code checks
  the version and decides whether to migrate or fall back to auto-create.
* **Multiple games in parallel**: each player has a unique `player_id` from
  the TM server → no filename collisions.
* **Session-based setup mid-restart**: not handled (see Non-goal). On restart
  during setup, `recover_session` continues to fire and produces a generic
  stub strategy. The persisted `strategy` field is normally empty during
  setup anyway, so there's nothing useful to restore.
* **Trainer sessions**: explicitly skipped — the human UI session is the
  source of truth, and trainer state is cheap to rebuild.

## Out of scope / follow-ups

* Incremental save during play (vs. only on shutdown). Would survive
  `kill -9`. Add later if shutdowns are commonly ungraceful.
* Persisting the setup-phase `session`. Only useful if restart-during-setup
  becomes a real concern.
* TM-server-side persistence of `recentLog` beyond the 25-entry cap. Not
  needed for this feature — recovery quality is acceptable with the existing
  log window plus restored strategy.
