"""Tests for LLM player state persistence (specs/LLM-state-persistence.md)."""

import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import tm_ai_server.llm_player as llm


@pytest.fixture
def tmp_state_dir(tmp_path):
    """Redirect _LLM_STATE_DIR to a fresh temp dir and reset registries."""
    state_dir = tmp_path / "llm-state"
    with patch.object(llm, '_LLM_STATE_DIR', state_dir), \
         patch.object(llm, '_player_registry', {}), \
         patch.object(llm, '_game_players', {}), \
         patch.object(llm, '_game_summary_logged', set()), \
         patch.object(llm, '_LLM_STATE_PERSIST', True):
        yield state_dir


def _populate(player, *, strategy="primary plan", tactical="step 1",
              last_gen=3, calls=12, in_tok=4000, out_tok=300):
    """Mutate a player so we can assert roundtrip fidelity."""
    player.strategy = strategy
    player.tactical = tactical
    player.last_generation = last_gen
    player.hand_shown_generation = last_gen
    player.token_usage = {
        "calls": calls, "input": in_tok, "output": out_tok,
        "cache_read": 100, "cache_write": 50, "thinking": 200,
    }
    player.summary_logged = False
    return player


# ---------------------------------------------------------------------------
# 1. Roundtrip
# ---------------------------------------------------------------------------

def test_to_dict_from_dict_roundtrip(tmp_state_dir):
    original = _populate(llm.LLMPlayer("p1", "g1", "anthropic/claude-sonnet-4-6"))
    payload  = original.to_dict()

    # Sanity: payload must be JSON-serializable
    data = json.loads(json.dumps(payload))

    restored = llm.LLMPlayer.from_dict(data)
    assert restored.player_id == "p1"
    assert restored.game_id   == "g1"
    assert restored.model     == "anthropic/claude-sonnet-4-6"
    assert restored.provider  == "openrouter"
    assert restored.strategy  == "primary plan"
    assert restored.tactical  == "step 1"
    assert restored.last_generation       == 3
    assert restored.hand_shown_generation == 3
    assert restored.token_usage["calls"]  == 12
    assert restored.token_usage["input"]  == 4000
    assert restored.token_usage["output"] == 300

    # Non-persisted fields stay empty
    assert restored.session == []
    assert restored.action_system == ""


# ---------------------------------------------------------------------------
# 2. Save + restore via the registry
# ---------------------------------------------------------------------------

def test_save_all_and_lazy_restore_via_get_or_create(tmp_state_dir):
    p1 = llm.register_player("p1", "g1", "anthropic/claude-sonnet-4-6")
    p2 = llm.register_player("p2", "g1", "openai/gpt-4o-mini")
    _populate(p1, strategy="p1 strategy", last_gen=4)
    _populate(p2, strategy="p2 strategy", last_gen=4, calls=7)

    written = llm.save_all_active_players()
    assert written == 2
    assert (tmp_state_dir / "p1.json").exists()
    assert (tmp_state_dir / "p2.json").exists()

    # Simulate a restart: drop all in-memory state
    llm._player_registry.clear()
    llm._game_players.clear()

    restored = llm.get_or_create_player("p1", "g1")
    assert restored.model    == "anthropic/claude-sonnet-4-6"
    assert restored.strategy == "p1 strategy"
    assert restored.last_generation == 4
    assert "p1" in llm._player_registry
    assert "p1" in llm._game_players["g1"]


# ---------------------------------------------------------------------------
# 3. Trainer players skipped
# ---------------------------------------------------------------------------

def test_save_skips_trainer_players(tmp_state_dir):
    llm.register_player("trainer:p1", "g1", "anthropic/claude-sonnet-4-6")
    llm.register_player("p1", "g1", "anthropic/claude-sonnet-4-6")

    written = llm.save_all_active_players()

    assert written == 1
    assert     (tmp_state_dir / "p1.json").exists()
    assert not (tmp_state_dir / "trainer_p1.json").exists()


# ---------------------------------------------------------------------------
# 4. Finished games skipped
# ---------------------------------------------------------------------------

def test_save_skips_finished_games(tmp_state_dir):
    llm.register_player("p1", "g1", "anthropic/claude-sonnet-4-6")
    llm._game_summary_logged.add("g1")

    written = llm.save_all_active_players()

    assert written == 0
    assert not (tmp_state_dir / "p1.json").exists()


# ---------------------------------------------------------------------------
# 5. Stale game_id is discarded and the file deleted
# ---------------------------------------------------------------------------

def test_stale_game_id_is_discarded(tmp_state_dir, caplog):
    tmp_state_dir.mkdir(parents=True, exist_ok=True)
    stale = {
        "schema_version": 1,
        "saved_at": "2026-06-01T00:00:00Z",
        "player_id": "p1",
        "game_id":   "gOLD",
        "model":     "anthropic/claude-sonnet-4-6",
        "strategy":  "stale",
        "tactical":  "",
        "last_generation":       2,
        "hand_shown_generation": 2,
        "token_usage": {},
        "summary_logged": False,
    }
    (tmp_state_dir / "p1.json").write_text(json.dumps(stale))

    import logging
    with patch.object(llm, '_OPENROUTER_API_KEY', ''), \
         patch.object(llm, '_OLLAMA_MODEL', 'qwen3:4b'), \
         caplog.at_level(logging.WARNING, logger="tm_ai_server.llm_player"):
        player = llm.get_or_create_player("p1", "gNEW")

    # Auto-create fallback path → default Ollama model, not the stale model
    assert player.model == "qwen3:4b"
    # File was treated as stale and deleted
    assert not (tmp_state_dir / "p1.json").exists()
    assert any("stale game_id" in r.message for r in caplog.records)
    assert any("not pre-registered" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 6. /game-done cleanup
# ---------------------------------------------------------------------------

def test_log_game_token_summary_removes_state_files(tmp_state_dir):
    llm.register_player("p1", "g1", "anthropic/claude-sonnet-4-6")
    llm.register_player("p2", "g1", "openai/gpt-4o-mini")
    llm.save_all_active_players()
    assert (tmp_state_dir / "p1.json").exists()
    assert (tmp_state_dir / "p2.json").exists()

    llm.log_game_token_summary("g1")

    assert not (tmp_state_dir / "p1.json").exists()
    assert not (tmp_state_dir / "p2.json").exists()


# ---------------------------------------------------------------------------
# 7. prune_stale_state
# ---------------------------------------------------------------------------

def test_prune_stale_state_removes_only_old_files(tmp_state_dir):
    tmp_state_dir.mkdir(parents=True, exist_ok=True)
    young = tmp_state_dir / "young.json"
    old   = tmp_state_dir / "old.json"
    young.write_text("{}")
    old.write_text("{}")

    now = time.time()
    os.utime(young, (now - 1 * 86400, now - 1 * 86400))  # 1 day old
    os.utime(old,   (now - 30 * 86400, now - 30 * 86400))  # 30 days old

    removed = llm.prune_stale_state(max_age_days=7)

    assert removed == 1
    assert     young.exists()
    assert not old.exists()


# ---------------------------------------------------------------------------
# Bonus: persistence disabled via LLM_STATE_PERSIST=false
# ---------------------------------------------------------------------------

def test_persistence_disabled(tmp_state_dir):
    with patch.object(llm, '_LLM_STATE_PERSIST', False):
        llm.register_player("p1", "g1", "anthropic/claude-sonnet-4-6")
        written = llm.save_all_active_players()

    assert written == 0
    assert not (tmp_state_dir / "p1.json").exists()
