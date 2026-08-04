"""Contract tests for explicit, structured Terraforming Mars actions."""

import asyncio

import pytest
from fastapi import HTTPException

from tm_llm.action_contract import (
    ACTION_CONTRACT_VERSION,
    ActionContractError,
    ActionPlan,
    build_input_response,
    compile_action_candidates,
    parse_action_plan,
    render_action_contract,
    validate_action_plan,
)
from tm_llm.options import _default_response
from tm_llm import config, engine
from tm_llm import app as app_module
from tm_llm import prompts


def stormcraft_waiting_for(target: int = 8, heat: int = 8, floaters: int = 4) -> dict:
    return {
        "type": "and",
        "title": {
            "message": "Select how to spend ${0} heat",
            "data": [{"value": target}],
        },
        "options": [
            {"type": "amount", "title": "Heat", "min": 0, "max": heat},
            {
                "type": "amount",
                "title": "Stormcraft Incorporated Floaters (2 heat each)",
                "min": 0,
                "max": floaters,
            },
        ],
    }


def stormcraft_server_accepts(response: dict, target: int) -> bool:
    """Independent oracle matching the public StormCraftIncorporated callback."""
    try:
        heat = response["responses"][0]["amount"]
        floaters = response["responses"][1]["amount"]
    except (KeyError, IndexError, TypeError):
        return False
    if heat + 2 * floaters < target:
        return False
    if heat > 0 and heat - 1 + 2 * floaters >= target:
        return False
    if floaters > 0 and heat + 2 * (floaters - 1) >= target:
        return False
    return True


def test_legacy_stormcraft_default_loses_both_decisions():
    waiting_for = stormcraft_waiting_for()

    response = _default_response(waiting_for)

    assert response == {
        "type": "and",
        "responses": [
            {"type": "amount", "amount": 0},
            {"type": "amount", "amount": 0},
        ],
    }
    assert stormcraft_server_accepts(response, target=8) is False


def test_stormcraft_compiles_two_required_amount_slots_and_builds_exact_wire_response():
    waiting_for = stormcraft_waiting_for()

    candidates = compile_action_candidates(waiting_for)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.id == "root"
    assert [(slot.id, slot.kind, slot.minimum, slot.maximum) for slot in candidate.slots] == [
        ("v:0", "amount", 0, 8),
        ("v:1", "amount", 0, 4),
    ]

    plan = ActionPlan(candidate="root", values={"v:0": 4, "v:1": 2})
    response = build_input_response(waiting_for, plan)

    assert response == {
        "type": "and",
        "responses": [
            {"type": "amount", "amount": 4},
            {"type": "amount", "amount": 2},
        ],
    }
    assert stormcraft_server_accepts(response, target=8) is True


@pytest.mark.parametrize(
    ("values", "reason"),
    [
        ({"v:0": 8}, "missing_value"),
        ({"v:0": 9, "v:1": 0}, "value_out_of_bounds"),
        ({"v:0": 8, "v:1": 0, "v:2": 1}, "unknown_value"),
        ({"v:0": "8", "v:1": 0}, "value_type"),
    ],
)
def test_stormcraft_structural_validation_fails_closed(values: dict, reason: str):
    waiting_for = stormcraft_waiting_for()
    candidate = compile_action_candidates(waiting_for)[0]

    with pytest.raises(ActionContractError) as exc:
        validate_action_plan(candidate, ActionPlan(candidate="root", values=values))

    assert exc.value.reason == reason


def test_or_project_card_keeps_original_indices_and_excludes_disabled_cards():
    waiting_for = {
        "type": "or",
        "title": "Select action",
        "options": [
            {
                "type": "projectCard",
                "title": "Standard projects",
                "cards": [
                    {"name": "Disabled:SP", "calculatedCost": 0, "isDisabled": True},
                    {"name": "Aquifer:SP", "calculatedCost": 18},
                    {"name": "City:SP", "calculatedCost": 25},
                ],
            },
            {"type": "option", "title": "Pass"},
        ],
    }

    candidates = compile_action_candidates(waiting_for)

    assert [candidate.id for candidate in candidates] == ["p:0", "p:1"]
    project_slot = candidates[0].slots[0]
    assert project_slot.id == "v:0"
    assert [(choice.value, choice.label) for choice in project_slot.choices] == [
        (1, "Aquifer:SP"),
        (2, "City:SP"),
    ]
    assert build_input_response(
        waiting_for,
        ActionPlan(candidate="p:0", values={"v:0": 1}),
    ) == {
        "type": "or",
        "index": 0,
        "response": {
            "type": "projectCard",
            "card": "Aquifer:SP",
            "payment": {
                "megacredits": 18,
                "steel": 0,
                "titanium": 0,
                "heat": 0,
                "plants": 0,
                "microbes": 0,
                "floaters": 0,
                "lunaArchivesScience": 0,
                "seeds": 0,
                "graphene": 0,
                "kuiperAsteroids": 0,
                "auroraiData": 0,
                "spireScience": 0,
            },
        },
    }


def test_multiselect_card_requires_explicit_original_indices():
    waiting_for = {
        "type": "card",
        "title": "Keep cards",
        "min": 1,
        "max": 2,
        "cards": [
            {"name": "A"},
            {"name": "Disabled", "isDisabled": True},
            {"name": "C"},
        ],
    }

    response = build_input_response(
        waiting_for,
        ActionPlan(candidate="root", values={"v:root": [0, 2]}),
    )

    assert response == {"type": "card", "cards": ["A", "C"]}


def test_nested_or_inside_and_requires_branch_and_active_child_value_only():
    waiting_for = {
        "type": "and",
        "title": "Composite",
        "options": [
            {
                "type": "or",
                "title": "Pick resource",
                "options": [
                    {"type": "amount", "title": "Heat", "min": 1, "max": 3},
                    {"type": "resource", "title": "Cube", "include": ["microbes", "floaters"]},
                ],
            },
            {"type": "option", "title": "Confirm"},
        ],
    }
    candidate = compile_action_candidates(waiting_for)[0]
    branch = candidate.slots[0]

    assert branch.kind == "branch"
    assert branch.id == "v:0"
    assert [choice.value for choice in branch.choices] == [0, 1]

    response = build_input_response(
        waiting_for,
        ActionPlan(candidate="root", values={"v:0": 0, "v:0.0": 2}),
    )

    assert response == {
        "type": "and",
        "responses": [
            {
                "type": "or",
                "index": 0,
                "response": {"type": "amount", "amount": 2},
            },
            {"type": "option"},
        ],
    }


@pytest.mark.parametrize(
    ("waiting_for", "value", "expected"),
    [
        (
            {"type": "space", "title": "Place", "spaces": ["01", "02"]},
            "02",
            {"type": "space", "spaceId": "02"},
        ),
        (
            {"type": "player", "title": "Player", "players": ["red", "blue"]},
            "blue",
            {"type": "player", "player": "blue"},
        ),
        (
            {"type": "delegate", "title": "Delegate", "players": ["neutral", "red"]},
            "red",
            {"type": "delegate", "player": "red"},
        ),
        (
            {"type": "party", "title": "Party", "parties": ["reds", "greens"]},
            "greens",
            {"type": "party", "partyName": "greens"},
        ),
        (
            {"type": "resource", "title": "Resource", "include": ["microbes", "floaters"]},
            "floaters",
            {"type": "resource", "resource": "floaters"},
        ),
        (
            {
                "type": "colony",
                "title": "Colony",
                "coloniesModel": [{"name": "Luna"}, {"name": "Triton"}],
            },
            1,
            {"type": "colony", "colonyName": "Triton"},
        ),
        (
            {"type": "globalEvent", "title": "Event", "globalEventNames": ["Dry Deserts", "War"]},
            "War",
            {"type": "globalEvent", "globalEventName": "War"},
        ),
    ],
)
def test_single_value_node_types_build_exact_wire_response(waiting_for: dict, value, expected: dict):
    candidate = compile_action_candidates(waiting_for)[0]

    response = build_input_response(
        waiting_for,
        ActionPlan(candidate="root", values={candidate.slots[0].id: value}),
    )

    assert response == expected


def test_nested_resource_builds_canonical_server_response():
    waiting_for = {
        "type": "and",
        "options": [
            {"type": "resource", "include": ["microbes", "floaters"]},
            {"type": "option", "title": "Confirm"},
        ],
    }

    response = build_input_response(
        waiting_for,
        ActionPlan(candidate="root", values={"v:0": "microbes"}),
    )

    assert response == {
        "type": "and",
        "responses": [
            {"type": "resource", "resource": "microbes"},
            {"type": "option"},
        ],
    }


def test_canonical_server_resource_shape_rejects_legacy_field():
    def server_accepts(response: dict, include: list[str]) -> bool:
        return (
            set(response) == {"type", "resource"}
            and response.get("type") == "resource"
            and response.get("resource") in include
        )

    include = ["microbes", "floaters"]
    accepted = build_input_response(
        {"type": "resource", "include": include},
        ActionPlan(candidate="root", values={"v:root": "floaters"}),
    )

    assert server_accepts(accepted, include) is True
    assert server_accepts(
        {"type": "resource", "resourceType": "floaters"}, include
    ) is False


def test_static_option_has_no_slots_and_needs_no_hidden_default():
    waiting_for = {"type": "option", "title": "Confirm"}
    candidate = compile_action_candidates(waiting_for)[0]

    assert candidate.slots == ()
    assert build_input_response(waiting_for, ActionPlan(candidate="root", values={})) == {
        "type": "option"
    }


def test_choice_bound_fails_closed_instead_of_silently_dropping_legal_values():
    waiting_for = {
        "type": "space",
        "title": "Place",
        "spaces": [f"space-{index}" for index in range(201)],
    }

    with pytest.raises(ActionContractError) as exc:
        compile_action_candidates(waiting_for)

    assert exc.value.reason == "too_many_choices"


def test_action_json_parser_and_rendered_contract_are_bounded_and_explicit():
    candidates = compile_action_candidates(stormcraft_waiting_for())
    text = render_action_contract(candidates)

    assert ACTION_CONTRACT_VERSION in text
    assert "v:0 amount [0..8] Heat" in text
    assert "v:1 amount [0..4] Stormcraft Incorporated Floaters" in text
    assert "ignore the legacy CHOICE line" in text

    plan = parse_action_plan(
        'Reasoning\nACTION: {"candidate":"root","values":{"v:0":4,"v:1":2}}'
    )
    assert plan == ActionPlan(candidate="root", values={"v:0": 4, "v:1": 2})
    assert parse_action_plan(
        '**ACTION:** {"candidate":"root","values":{"v:0":4,"v:1":2}}'
    ) == plan


@pytest.mark.parametrize(
    "text",
    [
        "CHOICE: 1",
        "ACTION: not-json",
        'ACTION: {"candidate":"root","values":{},"input_response":{"type":"option"}}',
        'ACTION: {"candidate":1,"values":{}}',
        'ACTION: {"candidate":"root","values":[]}',
    ],
)
def test_action_json_parser_rejects_missing_malformed_or_arbitrary_wire_payload(text: str):
    with pytest.raises(ActionContractError):
        parse_action_plan(text)


class FakePlayer:
    def __init__(self, responses: list[str]):
        self.player_id = "p1"
        self.game_id = "g1"
        self.action_system = ""
        self.last_generation = -1
        self.strategy = ""
        self.tactical = ""
        self.responses = list(responses)
        self.user_prompts: list[str] = []

    def single_shot(self, system: str, user: str, **kwargs) -> str:
        self.user_prompts.append(user)
        return self.responses.pop(0)


def test_engine_v2_uses_explicit_stormcraft_values_and_never_legacy_default(monkeypatch):
    player = FakePlayer(
        ['ACTION: {"candidate":"root","values":{"v:0":4,"v:1":2}}']
    )
    monkeypatch.setattr(config, "ACTION_CONTRACT_MODE", "v2", raising=False)
    monkeypatch.setattr(config, "MAX_ACTION_RETRIES", 0)
    monkeypatch.setattr(engine.registry, "get_or_create_player", lambda *_: player)

    response, debug = engine.select_action_llm(
        state={"game": {"generation": 1}, "player": {"name": "A", "color": "red"}},
        waiting_for=stormcraft_waiting_for(),
        game_id="g1",
        player_id="p1",
    )

    assert response["responses"][0]["amount"] == 4
    assert response["responses"][1]["amount"] == 2
    assert debug == {
        "llm_phase": "action",
        "action_contract": ACTION_CONTRACT_VERSION,
        "candidate": "root",
    }
    assert "ACTION CONTRACT" in player.user_prompts[0]
    assert "ignore the legacy CHOICE line" in player.user_prompts[0]


def test_engine_v2_exhaustion_fails_closed_instead_of_sending_zero_zero(monkeypatch):
    player = FakePlayer(['ACTION: {"candidate":"root","values":{"v:0":0}}'])
    monkeypatch.setattr(config, "ACTION_CONTRACT_MODE", "v2", raising=False)
    monkeypatch.setattr(config, "MAX_ACTION_RETRIES", 0)
    monkeypatch.setattr(engine.registry, "get_or_create_player", lambda *_: player)

    with pytest.raises(ActionContractError) as exc:
        engine.select_action_llm(
            state={"game": {"generation": 1}, "player": {"name": "A", "color": "red"}},
            waiting_for=stormcraft_waiting_for(),
            game_id="g1",
            player_id="p1",
        )

    assert exc.value.reason == "validation_retries_exhausted"


def test_engine_compare_mode_keeps_legacy_response_and_emits_safe_summary(monkeypatch):
    player = FakePlayer(["CHOICE: 1"])
    monkeypatch.setattr(config, "ACTION_CONTRACT_MODE", "compare", raising=False)
    monkeypatch.setattr(config, "MAX_ACTION_RETRIES", 0)
    monkeypatch.setattr(engine.registry, "get_or_create_player", lambda *_: player)

    response, debug = engine.select_action_llm(
        state={"game": {"generation": 1}, "player": {"name": "A", "color": "red"}},
        waiting_for=stormcraft_waiting_for(),
        game_id="g1",
        player_id="p1",
    )

    assert response == _default_response(stormcraft_waiting_for())
    assert debug["action_contract_compare"] == {
        "version": ACTION_CONTRACT_VERSION,
        "candidateCount": 1,
        "requiredSlotCount": 2,
        "parity": "requires_values",
    }
    assert "ACTION CONTRACT" not in player.user_prompts[0]


def test_engine_compare_mode_proves_zero_slot_legacy_parity(monkeypatch):
    waiting_for = {
        "type": "or",
        "options": [
            {"type": "option", "title": "Confirm"},
            {"type": "option", "title": "Pass"},
        ],
    }
    player = FakePlayer(["CHOICE: 2"])
    monkeypatch.setattr(config, "ACTION_CONTRACT_MODE", "compare", raising=False)
    monkeypatch.setattr(config, "MAX_ACTION_RETRIES", 0)
    monkeypatch.setattr(engine.registry, "get_or_create_player", lambda *_: player)

    response, debug = engine.select_action_llm(
        state={"game": {"generation": 1}, "player": {"name": "A", "color": "red"}},
        waiting_for=waiting_for,
        game_id="g1",
        player_id="p1",
    )

    assert response == {
        "type": "or",
        "index": 1,
        "response": {"type": "option"},
    }
    assert debug["action_contract_compare"]["parity"] == "equivalent"


def test_engine_v2_provider_failure_is_not_misclassified_or_defaulted(monkeypatch):
    player = FakePlayer([])

    def fail_provider(*args, **kwargs):
        raise RuntimeError("provider unavailable")

    player.single_shot = fail_provider
    monkeypatch.setattr(config, "ACTION_CONTRACT_MODE", "v2", raising=False)
    monkeypatch.setattr(engine.registry, "get_or_create_player", lambda *_: player)

    with pytest.raises(RuntimeError, match="provider unavailable"):
        engine.select_action_llm(
            state={"game": {"generation": 1}, "player": {"name": "A", "color": "red"}},
            waiting_for=stormcraft_waiting_for(),
            game_id="g1",
            player_id="p1",
        )


def test_engine_v2_retries_with_allowlisted_reason_and_does_not_echo_model_text(monkeypatch):
    rejected = 'ACTION: {"candidate":"root","values":{"v:0":0}}'
    player = FakePlayer(
        [rejected, 'ACTION: {"candidate":"root","values":{"v:0":8,"v:1":0}}']
    )
    monkeypatch.setattr(config, "ACTION_CONTRACT_MODE", "v2", raising=False)
    monkeypatch.setattr(config, "MAX_ACTION_RETRIES", 1)
    monkeypatch.setattr(engine.registry, "get_or_create_player", lambda *_: player)

    response, _ = engine.select_action_llm(
        state={"game": {"generation": 1}, "player": {"name": "A", "color": "red"}},
        waiting_for=stormcraft_waiting_for(),
        game_id="g1",
        player_id="p1",
    )

    assert response["responses"] == [
        {"type": "amount", "amount": 8},
        {"type": "amount", "amount": 0},
    ]
    assert len(player.user_prompts) == 2
    assert "Reason: missing_value" in player.user_prompts[1]
    assert rejected not in player.user_prompts[1]


def test_action_line_is_not_persisted_as_tactical_or_strategy_memory():
    text = (
        "TACTICAL: raise temperature, then pass\n"
        'ACTION: {"candidate":"root","values":{"v:0":8,"v:1":0}}'
    )

    assert prompts.capture_tactical(text) == "raise temperature, then pass"
    assert "ACTION" not in prompts.sanitize_strategy(text)


def test_move_endpoint_returns_sanitized_422_for_action_contract_failure(monkeypatch):
    class FakeState:
        waitingFor = stormcraft_waiting_for()

        def model_dump(self):
            return {"waitingFor": self.waitingFor}

    class FakeRequest:
        state = FakeState()
        game_id = "g1"
        player_id = "p1"
        last_error = None

    def fail_selection(*args, **kwargs):
        raise ActionContractError("validation_retries_exhausted")

    monkeypatch.setattr(app_module, "select_action_llm", fail_selection)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(app_module.move(FakeRequest()))

    assert exc.value.status_code == 422
    assert exc.value.detail == {
        "stage": "action_contract",
        "reason": "validation_retries_exhausted",
    }


def test_version_endpoint_publishes_contract_and_mode(monkeypatch):
    monkeypatch.setattr(config, "ACTION_CONTRACT_MODE", "compare", raising=False)

    result = asyncio.run(app_module.version())

    assert result.config["action_contract"] == ACTION_CONTRACT_VERSION
    assert result.config["action_contract_mode"] == "compare"
