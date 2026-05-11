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
