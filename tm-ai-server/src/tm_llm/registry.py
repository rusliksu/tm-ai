"""Player registry, per-game token summaries, and cross-restart state persistence.

The registry maps player_id → LLMPlayer. On a registry miss, `get_or_create_player` first
tries to restore state persisted from a previous server run; otherwise it auto-creates with
the default model. Persistence (save on graceful shutdown, lazy restore, prune) is here too
since it operates directly on the registry. See specs/LLM-state-persistence.md.
"""
from __future__ import annotations
import json
import logging
import os
import time
from pathlib import Path

from . import config
from .player import LLMPlayer

logger = logging.getLogger(__name__)

_registry: dict[str, LLMPlayer] = {}
_game_players: dict[str, list[str]] = {}
_game_summary_logged: set[str] = set()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def register_player(player_id: str, game_id: str, model: str | None = None) -> LLMPlayer:
    if model is None:
        model = config.OPENROUTER_MODEL
    player = LLMPlayer(player_id=player_id, game_id=game_id, model=model)
    _registry[player_id] = player
    _game_players.setdefault(game_id, [])
    if player_id not in _game_players[game_id]:
        _game_players[game_id].append(player_id)
    logger.info("Registered player %s (game=%s model=%s)", player_id, game_id, model)
    return player


def get_or_create_player(player_id: str, game_id: str) -> LLMPlayer:
    if player_id not in _registry:
        restored = try_load_player_state(player_id, game_id)
        if restored is not None:
            _registry[player_id] = restored
            _game_players.setdefault(game_id, [])
            if player_id not in _game_players[game_id]:
                _game_players[game_id].append(player_id)
            logger.info("LLM state restored: player=%s game=%s model=%s gen=%d",
                        restored.player_id, restored.game_id, restored.model, restored.last_generation)
        else:
            logger.warning("Player %s not pre-registered — auto-creating with default model", player_id)
            register_player(player_id, game_id)
    return _registry[player_id]


def log_game_token_summary(game_id: str) -> None:
    if game_id in _game_summary_logged:
        return
    _game_summary_logged.add(game_id)
    player_ids = _game_players.get(game_id, [])
    if not player_ids:
        logger.info("TOKEN SUMMARY game=%s (no registered players)", game_id)
        return
    totals = {"calls": 0, "input": 0, "output": 0}
    for pid in player_ids:
        player = _registry.get(pid)
        if player:
            player.log_token_summary()
            for k in totals:
                totals[k] += player.token_usage[k]
    logger.info("TOKEN SUMMARY game=%s | players=%d | calls=%d in=%d out=%d",
                game_id, len(player_ids), totals["calls"], totals["input"], totals["output"])
    removed = sum(1 for pid in player_ids if clear_player_state(pid))
    if removed:
        logger.info("LLM state cleaned up: %d files removed for game=%s", removed, game_id)


def game_finished(game_id: str) -> bool:
    return game_id in _game_summary_logged


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _state_path(player_id: str) -> Path:
    safe = player_id.replace(":", "_").replace("/", "_")
    return config.LLM_STATE_DIR / f"{safe}.json"


def save_all_active_players() -> int:
    if not config.LLM_STATE_PERSIST:
        return 0
    try:
        config.LLM_STATE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.warning("Could not create LLM state dir %s: %s", config.LLM_STATE_DIR, exc)
        return 0
    written = 0
    for pid, player in list(_registry.items()):
        if player.game_id in _game_summary_logged:
            continue
        try:
            path = _state_path(pid)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(player.to_dict(), indent=2))
            os.replace(tmp, path)
            written += 1
        except Exception as exc:
            logger.warning("Failed to persist LLM state for player %s: %s", pid, exc)
    if written:
        logger.info("LLM state persisted: %d players → %s", written, config.LLM_STATE_DIR)
    return written


def try_load_player_state(player_id: str, game_id: str) -> LLMPlayer | None:
    if not config.LLM_STATE_PERSIST:
        return None
    path = _state_path(player_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        logger.warning("Could not read LLM state %s: %s", path, exc)
        return None
    if int(data.get("schema_version", 0)) > config.LLM_STATE_SCHEMA_VERSION:
        logger.warning("LLM state %s has newer schema — ignoring", path)
        return None
    if data.get("game_id") != game_id:
        logger.warning("LLM state %s has stale game_id=%s (expected %s) — discarding",
                       path, data.get("game_id"), game_id)
        try:
            path.unlink()
        except Exception:
            pass
        return None
    try:
        return LLMPlayer.from_dict(data)
    except Exception as exc:
        logger.warning("Could not deserialize LLM state %s: %s", path, exc)
        return None


def clear_player_state(player_id: str) -> bool:
    path = _state_path(player_id)
    try:
        if path.exists():
            path.unlink()
            return True
    except Exception as exc:
        logger.warning("Could not delete LLM state %s: %s", path, exc)
    return False


def prune_stale_state(max_age_days: int | None = None) -> int:
    if not config.LLM_STATE_PERSIST or not config.LLM_STATE_DIR.exists():
        return 0
    cutoff_days = config.LLM_STATE_MAX_AGE_DAYS if max_age_days is None else max_age_days
    cutoff = time.time() - cutoff_days * 86400
    removed = 0
    for path in config.LLM_STATE_DIR.glob("*.json"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except Exception as exc:
            logger.warning("Could not prune LLM state %s: %s", path, exc)
    if removed:
        logger.info("LLM state pruned: %d files older than %d days removed", removed, cutoff_days)
    return removed
