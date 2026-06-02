"""Tests for cross-restart LLM player state persistence (tm_llm.registry + player)."""
import pytest

from tm_llm import config, registry
from tm_llm.player import LLMPlayer


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    """Point state files at a temp dir and reset registry globals around each test."""
    monkeypatch.setattr(config, "LLM_STATE_DIR", tmp_path / "llm-state")
    registry._registry.clear()
    registry._game_players.clear()
    registry._game_summary_logged.clear()
    yield
    registry._registry.clear()
    registry._game_players.clear()
    registry._game_summary_logged.clear()


def test_to_from_dict_roundtrip():
    p = LLMPlayer("p1", "g1", "anthropic/claude-sonnet-4-6")
    p.strategy = "ENGINE: science"
    p.tactical = "convert heat, play card"
    p.last_generation = 4
    p.token_usage = {"calls": 3, "input": 100, "output": 20, "cache_read": 50, "cache_write": 5}

    restored = LLMPlayer.from_dict(p.to_dict())
    assert restored.player_id == "p1" and restored.game_id == "g1"
    assert restored.model == "anthropic/claude-sonnet-4-6"
    assert restored.strategy == "ENGINE: science"
    assert restored.tactical == "convert heat, play card"
    assert restored.last_generation == 4
    assert restored.token_usage["cache_read"] == 50
    assert restored.action_system == ""  # not persisted; regenerated


def test_save_and_restore_via_registry():
    p = registry.register_player("p1", "g1", "deepseek/deepseek-v4-flash")
    p.strategy = "BACKUP: pivot to greenery sprint"
    p.last_generation = 6
    assert registry.save_all_active_players() == 1

    registry._registry.clear()  # simulate restart
    restored = registry.get_or_create_player("p1", "g1")
    assert restored.model == "deepseek/deepseek-v4-flash"
    assert restored.strategy == "BACKUP: pivot to greenery sprint"
    assert restored.last_generation == 6


def test_finished_game_not_persisted():
    registry.register_player("p1", "g1", "openai/gpt-4o-mini")
    registry._game_summary_logged.add("g1")
    assert registry.save_all_active_players() == 0


def test_stale_game_id_discarded():
    p = registry.register_player("p1", "gX", "openai/gpt-4o-mini")
    p.strategy = "stale"
    registry.save_all_active_players()
    registry._registry.clear()

    # Asking for the same player_id under a DIFFERENT game → stale file discarded.
    fresh = registry.get_or_create_player("p1", "gY")
    assert fresh.game_id == "gY"
    assert fresh.strategy == ""  # auto-created, not restored
    assert not registry._state_path("p1").exists()  # stale file removed


def test_game_done_cleans_up_state_files():
    registry.register_player("p1", "g1", "openai/gpt-4o-mini")
    registry.save_all_active_players()
    assert registry._state_path("p1").exists()
    registry.log_game_token_summary("g1")
    assert not registry._state_path("p1").exists()
