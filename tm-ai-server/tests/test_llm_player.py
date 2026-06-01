import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from unittest.mock import MagicMock, patch
import pytest

import tm_ai_server.llm_player as llm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_player(player_id="p1", game_id="g1", model="qwen3:4b"):
    """Return a fresh LLMPlayer without touching the global registry."""
    return llm.LLMPlayer(player_id=player_id, game_id=game_id, model=model)


# ---------------------------------------------------------------------------
# test_register_player
# ---------------------------------------------------------------------------

def test_register_player_creates_correct_provider():
    """register_player sets provider='openrouter' for slash-model, 'ollama' for bare model."""
    registry: dict = {}
    game_players: dict = {}

    with patch.object(llm, '_player_registry', registry), \
         patch.object(llm, '_game_players', game_players):

        p_or = llm.register_player("p_or", "g1", "anthropic/claude-opus-4-7")
        assert p_or.provider == "openrouter"
        assert p_or.model    == "anthropic/claude-opus-4-7"
        assert p_or.game_id  == "g1"

        p_ol = llm.register_player("p_ol", "g1", "qwen3:4b")
        assert p_ol.provider == "ollama"

        assert "p_or" in registry
        assert "p_ol" in registry
        assert "p_or" in game_players["g1"]
        assert "p_ol" in game_players["g1"]


def test_register_player_default_model_uses_openrouter_key():
    """When no model is given and OPENROUTER_API_KEY is set, default to OPENROUTER_MODEL."""
    registry: dict = {}
    game_players: dict = {}

    with patch.object(llm, '_player_registry', registry), \
         patch.object(llm, '_game_players', game_players), \
         patch.object(llm, '_OPENROUTER_API_KEY', 'fake-key'), \
         patch.object(llm, '_OPENROUTER_MODEL', 'anthropic/claude-sonnet-4-6'):

        p = llm.register_player("p_default", "g2")
        assert p.model    == "anthropic/claude-sonnet-4-6"
        assert p.provider == "openrouter"


def test_get_or_create_player_auto_creates_with_warning(caplog):
    """get_or_create_player auto-creates and warns when player not pre-registered."""
    registry: dict = {}
    game_players: dict = {}

    with patch.object(llm, '_player_registry', registry), \
         patch.object(llm, '_game_players', game_players), \
         patch.object(llm, '_OPENROUTER_API_KEY', ''), \
         patch.object(llm, '_OLLAMA_MODEL', 'qwen3:4b'):

        import logging
        with caplog.at_level(logging.WARNING, logger="tm_ai_server.llm_player"):
            p = llm.get_or_create_player("p_new", "g1")

        assert p.player_id == "p_new"
        assert any("not pre-registered" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# test_capability_resolution
# ---------------------------------------------------------------------------

def test_known_capabilities_returned_without_probing():
    """Models in _KNOWN_CAPABILITIES are returned directly (no HTTP probe)."""
    caps_cache: dict = {}

    with patch.object(llm, '_model_capabilities', caps_cache):
        caps = llm._get_model_capabilities("anthropic/claude-opus-4-7")

    assert caps["caching"] is True
    assert caps["thinking"] is True
    # Should be stored in cache now
    assert "anthropic/claude-opus-4-7" in caps_cache


def test_known_caps_for_openai_model():
    caps_cache: dict = {}
    with patch.object(llm, '_model_capabilities', caps_cache):
        caps = llm._get_model_capabilities("openai/gpt-4o")
    assert caps["caching"] is False
    assert caps["thinking"] is False


def test_provider_for_slash_model():
    assert llm._provider_for("anthropic/claude-opus-4-7") == "openrouter"
    assert llm._provider_for("openai/gpt-4o") == "openrouter"


def test_provider_for_bare_model():
    assert llm._provider_for("qwen3:4b") == "ollama"
    assert llm._provider_for("llama3.2:3b") == "ollama"


# ---------------------------------------------------------------------------
# test_per_generation_strategy_update
# ---------------------------------------------------------------------------

def test_per_generation_strategy_update_sends_prompt_and_stores():
    """_maybe_per_generation_update sends a restate prompt and stores the response."""
    player = _make_player()
    player.last_generation = 3

    captured_prompt: list[str] = []

    def fake_shot(system, user, max_output_tokens=None, thinking_budget=None):
        captured_prompt.append(user)
        return "1. STANDING: ahead. 2. ENGINE: Jovian. 3. MILESTONE: Rim Settler."

    player.single_shot = fake_shot

    llm._maybe_per_generation_update(player, 4, {
        "game": {"generation": 4, "temperature": -20, "oxygen": 0, "oceanCount": 0},
        "player": {"terraformRating": 22, "heat": 0, "plants": 0,
                   "production": {"megacredits": 2, "steel": 0, "titanium": 0,
                                  "plants": 0, "energy": 0, "heat": 0}},
    })

    assert player.strategy.startswith("1. STANDING")
    assert player.last_generation == 4
    assert captured_prompt, "single_shot was never called"
    assert "Generation 4" in captured_prompt[0]
    assert "MILESTONE" in captured_prompt[0]


def test_per_generation_update_not_triggered_same_generation():
    """_maybe_per_generation_update does nothing when generation is unchanged."""
    player = _make_player()
    player.last_generation = 5
    player.strategy = "original"

    calls: list = []
    player.single_shot = lambda s, u, **kw: calls.append(u) or "new strategy"

    llm._maybe_per_generation_update(player, 5, {})

    assert not calls, "single_shot should not have been called"
    assert player.strategy == "original"


def test_per_generation_update_skipped_before_first_action():
    """No update fires if last_generation == -1 (player never had a turn yet)."""
    player = _make_player()
    player.last_generation = -1
    calls: list = []
    player.single_shot = lambda s, u, **kw: calls.append(u) or ""

    llm._maybe_per_generation_update(player, 1, {})

    assert not calls
    assert player.last_generation == 1


# ---------------------------------------------------------------------------
# test_hand_card_descriptions_skipped
# ---------------------------------------------------------------------------

def test_hand_card_descriptions_skipped():
    """`_is_card_decision_about_hand` returns True when all decision cards are in hand."""
    hand = ["Tardigrades", "Ants", "Search for Life"]
    assert llm._is_card_decision_about_hand(["Tardigrades", "Ants"], hand) is True
    assert llm._is_card_decision_about_hand(["Tardigrades", "Unknown Card"], hand) is False


def test_hand_card_descriptions_empty_hand():
    """`_is_card_decision_about_hand` returns False when hand is empty."""
    assert llm._is_card_decision_about_hand(["Tardigrades"], []) is False


def test_hand_card_descriptions_empty_decision():
    """`_is_card_decision_about_hand` returns False when no cards in decision."""
    assert llm._is_card_decision_about_hand([], ["Tardigrades"]) is False


# ---------------------------------------------------------------------------
# test_correct_payment
# ---------------------------------------------------------------------------

def test_correct_payment_clamps_excess_mc():
    """_correct_payment clamps MC that exceeds available balance."""
    payment = {"megacredits": 50, "steel": 0, "titanium": 0, "heat": 0, "plants": 0,
               "microbes": 0, "floaters": 0, "lunaArchivesScience": 0, "spireScience": 0,
               "seeds": 0, "auroraiData": 0, "graphene": 0, "kuiperAsteroids": 0}
    wf = {"type": "payment", "amount": 14}
    player = {"megacredits": 20, "steel": 0, "titanium": 0, "heat": 0, "plants": 0}
    result = llm._correct_payment(payment, wf, player)
    assert result["megacredits"] == 20  # clamped to available


def test_correct_payment_clamps_steel_and_tops_up_mc():
    """_correct_payment clamps steel to available, tops up MC if still short."""
    payment = {"megacredits": 0, "steel": 10, "titanium": 0, "heat": 0, "plants": 0,
               "microbes": 0, "floaters": 0, "lunaArchivesScience": 0, "spireScience": 0,
               "seeds": 0, "auroraiData": 0, "graphene": 0, "kuiperAsteroids": 0}
    wf = {"type": "projectCard", "card": {"calculatedCost": 14}}
    player = {"megacredits": 10, "steel": 3, "titanium": 0, "heat": 0, "plants": 0}
    result = llm._correct_payment(payment, wf, player)
    assert result["steel"] == 3                                  # clamped from 10 to 3
    assert result["steel"] * 2 + result["megacredits"] >= 14    # covers cost


def test_correct_payment_no_change_when_valid():
    """_correct_payment leaves a valid payment unchanged."""
    payment = {"megacredits": 5, "steel": 3, "titanium": 0, "heat": 0, "plants": 0,
               "microbes": 0, "floaters": 0, "lunaArchivesScience": 0, "spireScience": 0,
               "seeds": 0, "auroraiData": 0, "graphene": 0, "kuiperAsteroids": 0}
    wf = {"type": "projectCard", "card": {"calculatedCost": 11}}
    player = {"megacredits": 10, "steel": 5, "titanium": 0, "heat": 0, "plants": 0}
    result = llm._correct_payment(payment, wf, player)
    assert result["megacredits"] == 5
    assert result["steel"] == 3


def test_correct_payment_blocks_steel_on_non_project_card():
    """_correct_payment zeroes steel/titanium for payment-type (not projectCard)."""
    payment = {"megacredits": 5, "steel": 3, "titanium": 2, "heat": 0, "plants": 0,
               "microbes": 0, "floaters": 0, "lunaArchivesScience": 0, "spireScience": 0,
               "seeds": 0, "auroraiData": 0, "graphene": 0, "kuiperAsteroids": 0}
    wf = {"type": "payment", "amount": 8}
    player = {"megacredits": 10, "steel": 5, "titanium": 5, "heat": 0, "plants": 0}
    result = llm._correct_payment(payment, wf, player)
    assert result["steel"] == 0
    assert result["titanium"] == 0


# ---------------------------------------------------------------------------
# test_select_action_advise
# ---------------------------------------------------------------------------

_MINIMAL_STATE = {
    "game": {"generation": 2, "temperature": -20, "oxygen": 0, "oceanCount": 0,
             "boardName": "tharsis", "expansions": {}, "recentLog": [],
             "availableMilestones": [], "availableAwards": [], "gameVariants": {}},
    "player": {"megacredits": 30, "steel": 0, "titanium": 0, "plants": 0, "energy": 0,
               "heat": 0, "megacreditProduction": 0, "steelProduction": 0,
               "titaniumProduction": 0, "plantsProduction": 0, "energyProduction": 0,
               "heatProduction": 0, "tags": {}, "cardsInHand": [], "playedCards": [],
               "cardResources": {}, "corporations": [], "victoryPoints": 0,
               "production": {"megacredits": 0, "steel": 0, "titanium": 0,
                              "plants": 0, "energy": 0, "heat": 0}},
    "opponents": [],
    "board": [],
    "milestones": [],
    "awards": [],
}

_MINIMAL_WAITING_FOR = {
    "type": "or",
    "title": "Select action",
    "options": [
        {"type": "option", "title": "Pass", "index": 0},
        {"type": "option", "title": "Standard Project", "index": 1},
    ],
}


def _make_trainer_player(trainer_id="trainer:p1", game_id="g1"):
    """Make a trainer LLMPlayer whose session is pre-seeded so continue_session is called."""
    p = _make_player(player_id=trainer_id, game_id=game_id, model="qwen3:4b")
    p.session = [{"role": "system", "content": "rules"}]
    p.last_generation = 2  # same as minimal state gen → no per-gen update
    return p


def test_select_action_advise_dual_format():
    """select_action_advise splits LLM response into advice text + parsed recommendation dict."""
    trainer = _make_trainer_player()
    llm_response = (
        "You should pass this turn to conserve resources.\n"
        "<recommendation>\n"
        "CHOICE: 1\n"
        "</recommendation>"
    )
    trainer.continue_session = lambda u, **kw: llm_response

    registry = {"trainer:p1": trainer}

    with patch.object(llm, '_player_registry', registry), \
         patch.object(llm, '_game_players', {}):

        advice, recommendation = llm.select_action_advise(
            _MINIMAL_STATE, _MINIMAL_WAITING_FOR, "g1", "p1",
        )

    assert "conserve resources" in advice
    assert "<recommendation>" not in advice
    # CHOICE: 1 → 0-based index 0 (Pass); or-wrapper from index_to_response
    assert recommendation["type"] == "or"
    assert recommendation["index"] == 0


def test_advice_session_isolated_per_player():
    """select_action_advise uses 'trainer:<player_id>' as session key."""
    llm_response = "Short coaching.\n<recommendation>\nCHOICE: 2\n</recommendation>"

    player_id  = "pAAA"
    trainer_id = f"trainer:{player_id}"
    trainer    = _make_trainer_player(trainer_id=trainer_id, game_id="g_iso_test")
    trainer.continue_session = lambda u, **kw: llm_response

    registry = {trainer_id: trainer}

    with patch.object(llm, '_player_registry', registry), \
         patch.object(llm, '_game_players', {}):

        llm.select_action_advise(_MINIMAL_STATE, _MINIMAL_WAITING_FOR, "g_iso_test", player_id)

    # Registry key is trainer:<player_id> only — NOT trainer:<game_id>:<player_id>
    assert trainer_id in registry
    assert f"trainer:g_iso_test:{player_id}" not in registry


def test_advice_sessions_separate_for_two_players():
    """Two players in the same game produce two distinct trainer sessions."""
    llm_response = "Short.\n<recommendation>\nCHOICE: 1\n</recommendation>"

    registry: dict = {}
    game_players: dict = {}

    def fake_init_session(self, system, user, think=True):
        self.session = [{"role": "system", "content": system}]
        return llm_response

    def fake_continue_session(self, user, max_output_tokens=None):
        return llm_response

    with patch.object(llm, '_player_registry', registry), \
         patch.object(llm, '_game_players', game_players), \
         patch.object(llm, '_OPENROUTER_API_KEY', ''), \
         patch.object(llm, '_OLLAMA_MODEL', 'qwen3:4b'), \
         patch.object(llm.LLMPlayer, 'init_session', fake_init_session), \
         patch.object(llm.LLMPlayer, 'continue_session', fake_continue_session):

        llm.select_action_advise(_MINIMAL_STATE, _MINIMAL_WAITING_FOR, "g1", "pSandra")
        llm.select_action_advise(_MINIMAL_STATE, _MINIMAL_WAITING_FOR, "g1", "pPeter")

    assert "trainer:pSandra" in registry
    assert "trainer:pPeter" in registry


# ---------------------------------------------------------------------------
# test_log_game_token_summary
# ---------------------------------------------------------------------------

def test_log_game_token_summary_aggregates_players():
    """log_game_token_summary calls log_token_summary on each player in the game."""
    p1 = _make_player("p1", "g_sum")
    p1.token_usage = {"calls": 3, "input": 100, "output": 50,
                      "cache_read": 0, "cache_write": 0, "thinking": 0}
    p2 = _make_player("p2", "g_sum")
    p2.token_usage = {"calls": 5, "input": 200, "output": 80,
                      "cache_read": 0, "cache_write": 0, "thinking": 0}

    logged: list[str] = []
    original_log = llm.LLMPlayer.log_token_summary

    def capture_log(self):
        logged.append(self.player_id)
        self.summary_logged = True

    registry    = {"p1": p1, "p2": p2}
    game_players = {"g_sum": ["p1", "p2"]}
    already_logged: set = set()

    with patch.object(llm, '_player_registry', registry), \
         patch.object(llm, '_game_players', game_players), \
         patch.object(llm, '_game_summary_logged', already_logged), \
         patch.object(llm.LLMPlayer, 'log_token_summary', capture_log):

        llm.log_game_token_summary("g_sum")

    assert set(logged) == {"p1", "p2"}


def test_log_game_token_summary_not_called_twice():
    """log_game_token_summary is idempotent — second call for same game is skipped."""
    logged: list[str] = []

    def capture_log(self):
        logged.append(self.player_id)

    p1 = _make_player("p1", "g_dup")
    registry    = {"p1": p1}
    game_players = {"g_dup": ["p1"]}
    already_logged: set = {"g_dup"}  # already in the set

    with patch.object(llm, '_player_registry', registry), \
         patch.object(llm, '_game_players', game_players), \
         patch.object(llm, '_game_summary_logged', already_logged), \
         patch.object(llm.LLMPlayer, 'log_token_summary', capture_log):

        llm.log_game_token_summary("g_dup")

    assert not logged, "log_token_summary should not be called for an already-logged game"


# ---------------------------------------------------------------------------
# test_extract_text
# ---------------------------------------------------------------------------

def test_extract_text_plain_string():
    """_extract_text handles a plain string content response."""
    choice = MagicMock()
    choice.message.content = "Hello, world!"
    response = MagicMock()
    response.choices = [choice]
    assert llm._extract_text(response) == "Hello, world!"


def test_extract_text_filters_thinking_blocks():
    """_extract_text strips thinking blocks and returns only text blocks."""
    text_block   = MagicMock(type="text",     text="The answer is 42.")
    think_block  = MagicMock(type="thinking", thinking="Internal reasoning…")

    choice = MagicMock()
    choice.message.content = [think_block, text_block]
    response = MagicMock()
    response.choices = [choice]

    result = llm._extract_text(response)
    assert result == "The answer is 42."
    assert "Internal reasoning" not in result


def test_extract_text_multiple_text_blocks():
    """_extract_text joins multiple text blocks."""
    b1 = MagicMock(type="text", text="Part A.")
    b2 = MagicMock(type="text", text="Part B.")
    choice = MagicMock()
    choice.message.content = [b1, b2]
    response = MagicMock()
    response.choices = [choice]
    assert llm._extract_text(response) == "Part A.\nPart B."


# ---------------------------------------------------------------------------
# test_strip_cache_control
# ---------------------------------------------------------------------------

def test_strip_cache_control_removes_blocks():
    """_strip_cache_control converts system cache_control list to plain string."""
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "Rule A.", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "Rule B."},
            ],
        },
        {"role": "user", "content": "What to do?"},
    ]
    result = llm._strip_cache_control(messages)
    assert result[0]["role"] == "system"
    assert isinstance(result[0]["content"], str)
    assert "cache_control" not in str(result[0]["content"])
    assert "Rule A." in result[0]["content"]
    # User message unchanged
    assert result[1] == messages[1]


# ---------------------------------------------------------------------------
# Retry logic (network-outage handling)
# ---------------------------------------------------------------------------

def test_is_transient_matches_openai_connection_error():
    import openai
    # APIConnectionError requires a 'request' kwarg in modern openai SDK; mock it.
    exc = openai.APIConnectionError.__new__(openai.APIConnectionError)
    Exception.__init__(exc, "Connection error.")
    assert llm._is_transient_error(exc)


def test_is_transient_matches_openai_timeout():
    import openai
    exc = openai.APITimeoutError.__new__(openai.APITimeoutError)
    Exception.__init__(exc, "Request timed out")
    assert llm._is_transient_error(exc)


def test_is_transient_matches_requests_connection_error():
    import requests
    exc = requests.ConnectionError("HTTPSConnectionPool: max retries exceeded")
    assert llm._is_transient_error(exc)


def test_is_transient_matches_substring_keywords():
    """Pure RuntimeError carrying a network error string should be transient too."""
    assert llm._is_transient_error(RuntimeError("Connection refused"))
    assert llm._is_transient_error(RuntimeError("Network is unreachable"))
    assert llm._is_transient_error(RuntimeError("Temporary failure in name resolution"))
    assert llm._is_transient_error(RuntimeError("503 Service Unavailable"))
    assert llm._is_transient_error(RuntimeError("429 Too Many Requests"))


def test_is_transient_rejects_non_network_errors():
    assert not llm._is_transient_error(ValueError("bad payload"))
    assert not llm._is_transient_error(KeyError("missing"))
    assert not llm._is_transient_error(RuntimeError("Authentication failed (401)"))


def test_retry_delay_schedule():
    """Schedule is 1s, 5s, 10s, then 30s forever."""
    assert llm._retry_delay_for_attempt(0) == 1.0
    assert llm._retry_delay_for_attempt(1) == 5.0
    assert llm._retry_delay_for_attempt(2) == 10.0
    assert llm._retry_delay_for_attempt(3) == 30.0
    assert llm._retry_delay_for_attempt(10) == 30.0
    assert llm._retry_delay_for_attempt(1_000_000) == 30.0


def test_with_retry_raises_immediately_on_non_transient():
    """Non-transient errors must raise on the first attempt."""
    calls = {"n": 0}
    def fn():
        calls["n"] += 1
        raise ValueError("bad payload")

    with patch.object(llm.time, "sleep") as sleep_mock:
        with pytest.raises(ValueError):
            llm._with_retry(fn)
    assert calls["n"] == 1
    sleep_mock.assert_not_called()


def test_with_retry_retries_transient_until_success():
    """Transient errors retry; the call eventually succeeds."""
    calls = {"n": 0}
    def fn():
        calls["n"] += 1
        if calls["n"] < 4:
            raise RuntimeError("Connection refused")
        return "ok"

    with patch.object(llm.time, "sleep") as sleep_mock:
        result = llm._with_retry(fn)

    assert result == "ok"
    assert calls["n"] == 4
    # 3 sleeps for the 3 failed attempts: 1s, 5s, 10s
    assert [c.args[0] for c in sleep_mock.call_args_list] == [1.0, 5.0, 10.0]


def test_with_retry_follows_schedule_then_30s_forever():
    """After the 1/5/10s ramp every subsequent retry is 30s."""
    calls = {"n": 0}
    def fn():
        calls["n"] += 1
        if calls["n"] < 7:
            raise RuntimeError("Connection refused")
        return "ok"

    with patch.object(llm.time, "sleep") as sleep_mock:
        result = llm._with_retry(fn)

    assert result == "ok"
    assert calls["n"] == 7
    assert [c.args[0] for c in sleep_mock.call_args_list] == [1.0, 5.0, 10.0, 30.0, 30.0, 30.0]


def test_with_retry_respects_max_attempts_cap_for_tests():
    """max_attempts caps the number of attempts (mainly for tests)."""
    def fn():
        raise RuntimeError("Connection refused")

    with patch.object(llm.time, "sleep"):
        with pytest.raises(RuntimeError):
            llm._with_retry(fn, max_attempts=3)
