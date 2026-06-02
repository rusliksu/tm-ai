# LLM Player State Persistence

**Status:** IMPLEMENTED (`tm_llm/player.py`, `tm_llm/registry.py`, `tm_llm/app.py`).
**Scope:** `tm-ai-server/` only. No changes to the TypeScript TM server.

## Problem

The player registry and per-`LLMPlayer` state live in process memory. When the AI server
restarts mid-game, the next `/move` would otherwise construct a fresh `LLMPlayer` with the
env-default model and empty memory, losing:

1. **Model assignment** — a multi-LLM death-match seat collapses to the default model
   (a correctness bug for graded runs, not just quality).
2. **Strategy** (coarse inter-generation memory) — empty until the next generation bump.
3. **Tactical** (intra-generation next-steps) — lost for a turn, then re-written.
4. **Token / cost counters** — the `/game-done` summary would be partial.
5. **`last_generation`** — the per-gen refresh would be skipped on the resume turn.

The action phase is stateless by design (one `single_shot` per turn with a full state
snapshot), so the game keeps running — but play quality drops and the model identity is lost.

## Goal

On graceful shutdown, persist each in-flight player's state to a JSON file keyed by
`player_id`. On the next `/move`, restore it lazily. Delete it when the game finishes.
Non-goal: surviving `kill -9` (best-effort graceful-shutdown persistence only).

## Storage layout

```
logs/llm-state/<player_id>.json     # one file per AI player
```

Directory via `LLM_STATE_DIR` (default `<repo-root>/logs/llm-state`). `player_id` is unique
across games (TM assigns random IDs), so no collisions. `:` and `/` in an id are sanitised in
the filename (`registry._state_path`).

### File schema (`LLMPlayer.to_dict`)

```json
{
  "schema_version": 1,
  "saved_at": "2026-06-02T10:34:12Z",
  "player_id": "p4f3aab12",
  "game_id":   "g8c91dd03",
  "model":     "anthropic/claude-sonnet-4-6",
  "strategy":  "STANDING: ...\nENGINE: ... (full inter-generation memory)",
  "tactical":  "convert 8 heat, then play Capital, then pass",
  "last_generation": 4,
  "token_usage": {"calls": 37, "input": 142000, "output": 11800, "cache_read": 98000, "cache_write": 5400},
  "summary_logged": false
}
```

**Not persisted:** `action_system` (rebuilt deterministically from `state` on the first
`/move`, byte-identical so prompt caching re-warms). Capabilities/pricing are module-level in
`openrouter.py` and re-derived from `model`.

## Behavior

- **Save** — `app.lifespan` calls `registry.save_all_active_players()` after `yield`. It
  iterates the registry, skips players whose `game_id` is already in `_game_summary_logged`
  (game finished), and atomically writes `<player_id>.json` (`tmp + os.replace`). Errors log a
  warning per player; shutdown never crashes.
- **Restore** — `registry.get_or_create_player(player_id, game_id)` first tries
  `try_load_player_state`. Missing file → auto-create + warn. Mismatched `game_id` → delete
  the stale file, auto-create. Newer `schema_version` → warn, auto-create. Otherwise hydrate
  an `LLMPlayer` via `from_dict` and insert it into the registry. `action_system` stays empty
  and regenerates on the first `/move`.
- **Cleanup** — `registry.log_game_token_summary(game_id)` (called by `/game-done`) deletes
  each finished game's state files after logging the summary.
- **Prune** — `app.lifespan` calls `registry.prune_stale_state()` before `yield`, deleting
  files older than `LLM_STATE_MAX_AGE_DAYS` (cold-start hygiene for games that crashed without
  `/game-done`).

## API (internal Python only)

```python
# registry.py
def save_all_active_players() -> int: ...
def try_load_player_state(player_id: str, game_id: str) -> LLMPlayer | None: ...
def clear_player_state(player_id: str) -> bool: ...
def prune_stale_state(max_age_days: int | None = None) -> int: ...

# player.py
class LLMPlayer:
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, data: dict) -> "LLMPlayer": ...
```

No new HTTP endpoints.

## Env vars

| Var | Default | Description |
|-----|---------|-------------|
| `LLM_STATE_PERSIST` | `true` | Master switch — set `false` to disable entirely |
| `LLM_STATE_DIR` | `<repo-root>/logs/llm-state` | Where per-player state files live |
| `LLM_STATE_MAX_AGE_DAYS` | `7` | Files older than this are pruned at startup |

## Edge cases

- **`kill -9` / crash**: lifespan does not run → no save. Accepted.
- **Disk full / permission denied**: log a warning, continue; never crash shutdown or `/move`.
- **Schema evolution**: `schema_version` gates restore; a newer version falls back to
  auto-create rather than crashing.
- **Setup phase**: `strategy` is normally empty until the opening decision completes; there is
  little to restore mid-setup, and the setup phase is stateless `single_shot` anyway.
