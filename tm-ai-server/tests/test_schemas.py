import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from tm_ai_server.schemas import MoveRequest, MoveResponse, HealthResponse

SAMPLE_REQUEST = {
    "game_id": "g123",
    "player_id": "p456",
    "state": {
        "game": {"id": "g123", "phase": "action", "generation": 7, "oxygen": 8, "temperature": -12, "oceanCount": 5},
        "player": {
            "id": "p456", "name": "Alice", "color": "blue",
            "terraformRating": 42, "megacredits": 25, "steel": 3, "titanium": 1,
            "plants": 5, "energy": 2, "heat": 6, "handSize": 4,
            "production": {"megacredits": 4, "steel": 1, "titanium": 0, "plants": 2, "heat": 0, "energy": 1},
            "tags": {"science": 2, "building": 3},
            "isAI": True,
        },
        "waitingFor": {"type": "or", "title": "Take action", "options": []},
    },
    "legal_actions": [
        {"action_id": "provide_input", "type": "or", "title": "Take action",
         "payload": {"input": {"type": "or", "title": "Take action", "options": []}}}
    ],
    "metadata": {"schema_version": 1},
}


def test_move_request_parses():
    req = MoveRequest(**SAMPLE_REQUEST)
    assert req.game_id == "g123"
    assert req.state.player.terraformRating == 42
    assert req.state.game.oceanCount == 5


def test_move_request_defaults():
    req = MoveRequest(**SAMPLE_REQUEST)
    assert req.state.player.isAI is True
    assert req.state.player.tags == {"science": 2, "building": 3}


def test_move_response_valid():
    resp = MoveResponse(input_response={"type": "or", "index": 0, "response": {"type": "option"}})
    assert resp.debug is None
    d = resp.model_dump()
    assert d["input_response"]["type"] == "or"


def test_health_response():
    h = HealthResponse()
    assert h.status == "ok"
