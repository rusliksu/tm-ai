import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from tm_ai_server.config import STATE_DIM, ACTION_SPACE_SIZE
from tm_ai_server.encoding import (
    encode_state, flatten_options, build_mask,
    index_to_response, response_to_index,
)

SAMPLE_STATE = {
    "game": {"id": "g1", "phase": "action", "generation": 5, "oxygen": 8, "temperature": -12, "oceanCount": 4},
    "player": {
        "id": "p1", "name": "Alice", "color": "blue",
        "terraformRating": 25, "megacredits": 30, "steel": 3, "titanium": 1,
        "plants": 5, "energy": 2, "heat": 4, "handSize": 4,
        "production": {"megacredits": 3, "steel": 1, "titanium": 0, "plants": 2, "energy": 1, "heat": 0},
        "tags": {"science": 2, "building": 3, "space": 1},
        "isAI": True,
    },
}

SAMPLE_OR = {
    "type": "or",
    "title": "Take action",
    "options": [
        {"type": "option", "title": "Pass", "buttonLabel": "Pass"},
        {"type": "option", "title": "Use standard project", "buttonLabel": "Select"},
        {"type": "option", "title": "Claim milestone", "buttonLabel": "Select"},
    ],
}


def test_encode_state_shape():
    vec = encode_state(SAMPLE_STATE)
    assert vec.shape == (STATE_DIM,)
    assert vec.dtype == np.float32


def test_encode_state_with_spec():
    spec = {"board_name": "tharsis", "player_count": 2, "expansions": ["corpEra", "venus"]}
    vec = encode_state(SAMPLE_STATE, spec)
    assert vec.shape == (STATE_DIM,)


def test_encode_state_values_in_range():
    vec = encode_state(SAMPLE_STATE)
    # Most encoded values should be in [0, 2] (some normalizations may slightly exceed 1)
    assert np.all(vec >= -1.0)
    assert np.all(vec <= 2.0)


def test_flatten_or_options():
    options = flatten_options(SAMPLE_OR)
    assert len(options) == 3
    assert options[0]["title"] == "Pass"
    assert options[0]["index"] == 0


def test_flatten_card_options():
    node = {"type": "card", "cards": [{"name": "Flooding"}, {"name": "IceAsteroid"}], "min": 1, "max": 1}
    options = flatten_options(node)
    assert len(options) == 2
    assert options[0]["title"] == "Flooding"


def test_flatten_amount():
    node = {"type": "amount", "min": 2, "max": 5, "maxByDefault": False}
    options = flatten_options(node)
    assert len(options) == 4
    assert options[0]["title"] == "2"


def test_flatten_space():
    node = {"type": "space", "spaces": ["H05", "H06", "H07"]}
    options = flatten_options(node)
    assert len(options) == 3


def test_build_mask():
    mask = build_mask(5)
    assert mask.shape == (ACTION_SPACE_SIZE,)
    assert mask[:5].all()
    assert not mask[5:].any()


def test_index_to_response_or():
    resp = index_to_response(SAMPLE_OR, 1)
    assert resp["type"] == "or"
    assert resp["index"] == 1
    assert resp["response"]["type"] == "option"


def test_index_to_response_card():
    node = {"type": "card", "cards": [{"name": "Flooding"}, {"name": "IceAsteroid"}], "min": 1, "max": 1}
    resp = index_to_response(node, 1)
    assert resp == {"type": "card", "cards": ["IceAsteroid"]}


def test_index_to_response_amount():
    node = {"type": "amount", "min": 3, "max": 8, "maxByDefault": False}
    resp = index_to_response(node, 2)
    assert resp == {"type": "amount", "amount": 5}


def test_index_to_response_space():
    node = {"type": "space", "spaces": ["H05", "H06", "H07"]}
    resp = index_to_response(node, 2)
    assert resp == {"type": "space", "spaceId": "H07"}


def test_response_to_index_or():
    resp = {"type": "or", "index": 2, "response": {"type": "option"}}
    assert response_to_index(SAMPLE_OR, resp) == 2


def test_response_to_index_card():
    node = {"type": "card", "cards": [{"name": "Flooding"}, {"name": "IceAsteroid"}], "min": 1, "max": 1}
    resp = {"type": "card", "cards": ["IceAsteroid"]}
    assert response_to_index(node, resp) == 1


def test_response_to_index_amount():
    node = {"type": "amount", "min": 3, "max": 8, "maxByDefault": False}
    resp = {"type": "amount", "amount": 6}
    assert response_to_index(node, resp) == 3


def test_response_roundtrip():
    for i in range(3):
        resp = index_to_response(SAMPLE_OR, i)
        idx = response_to_index(SAMPLE_OR, resp)
        assert idx == i


# ---------------------------------------------------------------------------
# _default_response — OR fallback should prefer the last 'option' (Pass)
# ---------------------------------------------------------------------------

from tm_ai_server.encoding import _default_response


def test_default_response_or_prefers_last_option_type():
    """For OrOptions, _default_response should pick the LAST bare option
    (conventionally "Pass" in TM), matching TM's aiFallbackResponse."""
    node = {
        "type": "or",
        "options": [
            {"type": "projectCard", "cards": []},
            {"type": "option", "title": "Use action"},
            {"type": "option", "title": "Pass"},
        ],
    }
    resp = _default_response(node)
    assert resp["type"] == "or"
    assert resp["index"] == 2          # last 'option'-typed sub-option
    assert resp["response"] == {"type": "option"}


def test_default_response_or_no_option_type_falls_back_to_zero():
    """If no bare 'option' sub-option exists, fall through to first sub-option's default."""
    node = {
        "type": "or",
        "options": [
            {"type": "amount", "min": 0, "max": 5},
        ],
    }
    resp = _default_response(node)
    assert resp["index"] == 0
    assert resp["response"]["type"] == "amount"


def test_default_response_or_empty_options():
    """Empty options list should produce a safe placeholder."""
    node = {"type": "or", "options": []}
    resp = _default_response(node)
    assert resp == {"type": "or", "index": 0, "response": {"type": "option"}}


# ---------------------------------------------------------------------------
# Nested-OR expansion (regression: "Fund an award" auto-resolved to Benefactor
# instead of letting the AI pick which award)
# ---------------------------------------------------------------------------

def test_flatten_options_expands_nested_or_of_leaf_options():
    """A nested OR whose children are all bare 'option' leaves should expand
    inline so the LLM picks the specific sub-choice directly."""
    waiting_for = {
        "type": "or",
        "title": "Take action",
        "options": [
            {
                "type": "or",
                "title": "Fund an award (14 M€)",
                "options": [
                    {"type": "option", "title": "Benefactor"},
                    {"type": "option", "title": "Desert Settler"},
                    {"type": "option", "title": "Scientist"},
                ],
            },
            {"type": "option", "title": "Pass for this generation"},
        ],
    }
    opts = flatten_options(waiting_for)
    titles = [o["title"] for o in opts]
    assert titles == [
        "Fund an award (14 M€): Benefactor",
        "Fund an award (14 M€): Desert Settler",
        "Fund an award (14 M€): Scientist",
        "Pass for this generation",
    ]
    # index = flat position (used for the displayed CHOICE number)
    assert [o["index"] for o in opts] == [0, 1, 2, 3]
    # path = walk into the tree
    assert [o["path"] for o in opts] == [[0, 0], [0, 1], [0, 2], [1]]


def test_index_to_response_walks_nested_path():
    """index_to_response(wf, [parent, child]) builds a nested OR response."""
    waiting_for = {
        "type": "or",
        "options": [
            {
                "type": "or",
                "options": [
                    {"type": "option", "title": "Benefactor"},
                    {"type": "option", "title": "Desert Settler"},
                    {"type": "option", "title": "Scientist"},
                ],
            },
            {"type": "option", "title": "Pass"},
        ],
    }
    # Picking "Desert Settler" via path [0, 1]
    resp = index_to_response(waiting_for, [0, 1])
    assert resp == {
        "type": "or",
        "index": 0,
        "response": {"type": "or", "index": 1, "response": {"type": "option"}},
    }


def test_index_to_response_accepts_bare_int_for_backcompat():
    """index_to_response(wf, 2) still works (treated as path=[2])."""
    waiting_for = {
        "type": "or",
        "options": [
            {"type": "option", "title": "A"},
            {"type": "option", "title": "B"},
            {"type": "option", "title": "C"},
        ],
    }
    resp = index_to_response(waiting_for, 2)
    assert resp == {"type": "or", "index": 2, "response": {"type": "option"}}


def test_flatten_options_does_not_expand_deep_nesting():
    """A nested OR whose children are themselves non-leaf (e.g. another OR)
    is NOT expanded — the AI picks the top-level option and the sub-decision
    appears in the next /move call as a fresh waitingFor."""
    waiting_for = {
        "type": "or",
        "options": [
            {
                "type": "or",
                "title": "Use blue action",
                "options": [
                    {"type": "or", "title": "Sub-action", "options": [
                        {"type": "option", "title": "Sub-A"},
                        {"type": "option", "title": "Sub-B"},
                    ]},
                ],
            },
            {"type": "option", "title": "Pass"},
        ],
    }
    opts = flatten_options(waiting_for)
    titles = [o["title"] for o in opts]
    # The "Use blue action" OR is NOT expanded (its child is another OR, not a leaf option)
    assert titles == ["Use blue action", "Pass"]
    assert [o["path"] for o in opts] == [[0], [1]]
