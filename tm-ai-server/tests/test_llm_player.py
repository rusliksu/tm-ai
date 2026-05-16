import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from unittest.mock import MagicMock, patch, call
import pytest

import tm_ai_server.llm_player as llm


# ---------------------------------------------------------------------------
# test_cached_content_used
# ---------------------------------------------------------------------------

def test_cached_content_used():
    """_init_gemini_session uses cached_content when cache creation succeeds."""
    fake_cache = MagicMock()
    fake_cache.name = "cachedContents/abc123"

    fake_response = MagicMock()
    fake_response.text = "CORPORATION: TestCorp\nBUY_CARDS: none\nSTRATEGY: Test strategy."

    fake_client = MagicMock()
    fake_client.caches.create.return_value = fake_cache
    fake_client.models.generate_content.return_value = fake_response
    fake_chat = MagicMock()
    fake_client.chats.create.return_value = fake_chat

    with patch.object(llm, '_gemini_client', fake_client), \
         patch.object(llm, '_GEMINI_MODEL', 'gemini-2.5-flash-lite'), \
         patch.object(llm, '_GEMINI_API_KEY', 'fake-key'), \
         patch.object(llm, '_game_cache_info', {}), \
         patch.object(llm, '_game_chat_sessions', {}):

        llm._init_gemini_session("game1", "system prompt", "user message", think=False)

        # Cache was created
        fake_client.caches.create.assert_called_once()
        create_kwargs = fake_client.caches.create.call_args[1]
        assert create_kwargs['model'] == 'gemini-2.5-flash-lite'
        assert create_kwargs['config'].system_instruction == "system prompt"

        # generate_content used cached_content, not system_instruction
        gen_cfg = fake_client.models.generate_content.call_args[1]['config']
        assert gen_cfg.cached_content == "cachedContents/abc123"
        assert not hasattr(gen_cfg, 'system_instruction') or gen_cfg.system_instruction is None

        # Chat was also created with cached_content
        chat_cfg = fake_client.chats.create.call_args[1]['config']
        assert chat_cfg.cached_content == "cachedContents/abc123"


def test_cached_content_falls_back_on_error():
    """_init_gemini_session falls back to inline system_instruction when cache creation fails."""
    fake_response = MagicMock()
    fake_response.text = "CORPORATION: TestCorp\nBUY_CARDS: none\nSTRATEGY: Test."

    fake_client = MagicMock()
    fake_client.caches.create.side_effect = Exception("Cache not supported")
    fake_client.models.generate_content.return_value = fake_response
    fake_chat = MagicMock()
    fake_client.chats.create.return_value = fake_chat

    with patch.object(llm, '_gemini_client', fake_client), \
         patch.object(llm, '_GEMINI_MODEL', 'gemini-2.5-flash-lite'), \
         patch.object(llm, '_GEMINI_API_KEY', 'fake-key'), \
         patch.object(llm, '_game_cache_info', {}), \
         patch.object(llm, '_game_chat_sessions', {}):

        llm._init_gemini_session("game2", "system prompt", "user message", think=False)

        # Falls back: generate_content should use system_instruction, not cached_content
        gen_cfg = fake_client.models.generate_content.call_args[1]['config']
        assert gen_cfg.system_instruction == "system prompt"
        assert not gen_cfg.cached_content


# ---------------------------------------------------------------------------
# test_per_generation_strategy_update
# ---------------------------------------------------------------------------

def test_per_generation_strategy_update_captures_and_preserves_chat():
    """_maybe_per_generation_update sends a restate prompt to the existing chat and
    captures the response — without rebuilding the chat session."""
    fake_strategy_response = MagicMock()
    fake_strategy_response.text = "1. STANDING: ahead by 6 VP. 2. ENGINE: Jovian. 3. MILESTONE: Rim Settler."

    fake_chat = MagicMock()
    fake_chat.send_message.return_value = fake_strategy_response

    game_id = "game_pergen_test"
    game_sessions = {game_id: fake_chat}
    strategies = {}
    last_gen = {game_id: 3}

    with patch.object(llm, '_game_chat_sessions', game_sessions), \
         patch.object(llm, '_game_strategies', strategies), \
         patch.object(llm, '_gemini_last_generation', last_gen):

        # Generation bumps from 3 → 4
        llm._maybe_per_generation_update(game_id, 4, {})

        # Strategy was captured and stored
        assert strategies[game_id] == fake_strategy_response.text

        # The existing chat was used (no rebuild)
        assert game_sessions[game_id] is fake_chat
        fake_chat.send_message.assert_called_once()

        # The restate prompt was sent
        sent_prompt = fake_chat.send_message.call_args[0][0]
        assert "Generation 3" in sent_prompt and "Generation 4" in sent_prompt
        assert "MILESTONE" in sent_prompt

        # Last generation was updated
        assert last_gen[game_id] == 4


def test_per_generation_update_not_triggered_same_generation():
    """_maybe_per_generation_update does nothing when generation is unchanged."""
    fake_chat = MagicMock()
    game_id = "game_no_update"
    game_sessions = {game_id: fake_chat}
    last_gen = {game_id: 5}

    with patch.object(llm, '_game_chat_sessions', game_sessions), \
         patch.object(llm, '_gemini_last_generation', last_gen):

        llm._maybe_per_generation_update(game_id, 5, {})  # Same generation

        # Chat untouched
        assert game_sessions[game_id] is fake_chat
        fake_chat.send_message.assert_not_called()


# ---------------------------------------------------------------------------
# test_hand_card_descriptions_skipped
# ---------------------------------------------------------------------------

def test_hand_card_descriptions_skipped():
    """`_is_card_decision_about_hand` returns True when all decision cards are in hand."""
    hand = ["Tardigrades", "Ants", "Search for Life"]
    # All decision cards are in hand
    assert llm._is_card_decision_about_hand(["Tardigrades", "Ants"], hand) is True
    # A card not in hand → False
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
    # Card costs 14 MC, player has 3 steel (=6 MC value) and 10 MC
    wf = {"type": "projectCard", "card": {"calculatedCost": 14}}
    player = {"megacredits": 10, "steel": 3, "titanium": 0, "heat": 0, "plants": 0}
    result = llm._correct_payment(payment, wf, player)
    assert result["steel"] == 3           # clamped from 10 to 3
    assert result["steel"] * 2 + result["megacredits"] >= 14  # covers cost


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
# test_select_action_advise_dual_format
# ---------------------------------------------------------------------------

_MINIMAL_STATE = {
    "game": {"generation": 2, "temperature": -20, "oxygen": 0, "oceanCount": 0,
             "boardName": "tharsis", "expansions": {}, "recentLog": [],
             "availableMilestones": [], "availableAwards": [], "gameVariants": {}},
    "player": {"megacredits": 30, "steel": 0, "titanium": 0, "plants": 0, "energy": 0,
               "heat": 0, "megacreditProduction": 0, "steelProduction": 0,
               "titaniumProduction": 0, "plantsProduction": 0, "energyProduction": 0,
               "heatProduction": 0, "tags": {}, "cardsInHand": [], "playedCards": [],
               "cardResources": {}, "corporations": [], "victoryPoints": 0},
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


def test_select_action_advise_dual_format():
    """select_action_advise splits LLM response into advice text + parsed recommendation dict."""
    llm_response = (
        "You should pass this turn to conserve resources.\n"
        "<recommendation>\n"
        "CHOICE: 1\n"
        "</recommendation>"
    )

    with patch.object(llm, '_LLM_PROVIDER', 'ollama'), \
         patch.object(llm, '_game_sessions', {}), \
         patch.object(llm, '_session_base_system', {}), \
         patch.object(llm, '_call_llm_init', return_value=llm_response) as mock_init:

        advice, recommendation = llm.select_action_advise(
            _MINIMAL_STATE, _MINIMAL_WAITING_FOR, "game_advise_test", "pdummy123",
        )

    assert "conserve resources" in advice
    assert "<recommendation>" not in advice
    # CHOICE: 1 → 0-based index 0 (Pass); or-wrapper from index_to_response
    assert recommendation["type"] == "or"
    assert recommendation["index"] == 0


def test_advice_session_isolated_per_player():
    """select_action_advise uses 'trainer:<game_id>:<player_id>' namespace."""
    llm_response = "Short coaching.\n<recommendation>\nCHOICE: 2\n</recommendation>"

    game_id = "g_iso_test"
    player_id = "pAAA"
    expected_session = f"trainer:{game_id}:{player_id}"

    with patch.object(llm, '_LLM_PROVIDER', 'ollama'), \
         patch.object(llm, '_game_sessions', {}), \
         patch.object(llm, '_session_base_system', {}), \
         patch.object(llm, '_call_llm_init', return_value=llm_response) as mock_init:

        llm.select_action_advise(_MINIMAL_STATE, _MINIMAL_WAITING_FOR, game_id, player_id)

        # _call_llm_init was called with the per-player trainer namespace
        called_session = mock_init.call_args[0][0]
        assert called_session == expected_session
        assert called_session != game_id
        assert called_session != f"trainer:{game_id}"


def test_advice_sessions_separate_for_two_players():
    """Two players in the same game produce two distinct trainer sessions."""
    llm_response = "Short.\n<recommendation>\nCHOICE: 1\n</recommendation>"

    sessions: dict = {}

    with patch.object(llm, '_LLM_PROVIDER', 'ollama'), \
         patch.object(llm, '_game_sessions', sessions), \
         patch.object(llm, '_session_base_system', {}), \
         patch.object(llm, '_call_llm_init', return_value=llm_response) as mock_init:

        def fake_init(game_id, system, user, think=False):
            sessions[game_id] = [{"role": "system", "content": system}]
            return llm_response

        mock_init.side_effect = fake_init

        llm.select_action_advise(_MINIMAL_STATE, _MINIMAL_WAITING_FOR, "g1", "pSandra")
        llm.select_action_advise(_MINIMAL_STATE, _MINIMAL_WAITING_FOR, "g1", "pPeter")

    assert "trainer:g1:pSandra" in sessions
    assert "trainer:g1:pPeter" in sessions
