"""Tests for response parsing, payment validation, and single-sourced data in tm_llm."""
from tm_llm.options import flatten_options
from tm_llm.prompts import (
    capture_tactical, parse_action_response, standard_project_cost,
    STANDARD_PROJECT_COSTS,
)
from tm_llm.payment import check_payment_valid, parse_payment_line


def test_capture_tactical():
    text = "Reasoning here.\nTACTICAL: convert 8 heat, then play Soletta, then pass\nCHOICE: 2"
    assert capture_tactical(text) == "convert 8 heat, then play Soletta, then pass"


def test_capture_tactical_absent():
    assert capture_tactical("CHOICE: 1") is None


def test_parse_action_picks_choice():
    wf = {"type": "or", "options": [
        {"type": "option", "title": "Play card"},
        {"type": "option", "title": "Pass"},
    ]}
    options = flatten_options(wf)
    text = "I will pass.\nTACTICAL: hold\nCHOICE: 2"
    response, debug = parse_action_response(text, options, wf, "p1")
    assert response == {"type": "or", "index": 1, "response": {"type": "option"}}
    assert debug["llm_choice"] == 2


def test_parse_action_project_card_payment():
    wf = {"type": "projectCard", "card": {"name": "UnknownCard", "calculatedCost": 10},
          "cards": [{"name": "UnknownCard", "calculatedCost": 10}]}
    options = flatten_options(wf)
    text = "CHOICE: 1\nPAYMENT: MC=10"
    player = {"megacredits": 20, "steel": 0, "titanium": 0}
    response, _ = parse_action_response(text, options, wf, "p1", player=player)
    assert response["payment"]["megacredits"] == 10


def test_payment_line_parsing():
    p = parse_payment_line("PAYMENT: MC=5, STEEL=3, TITANIUM=1")
    assert p["megacredits"] == 5 and p["steel"] == 3 and p["titanium"] == 1


def test_check_payment_underfunded():
    wf = {"type": "projectCard", "card": {"name": "UnknownCard", "calculatedCost": 12}}
    player = {"megacredits": 5, "steel": 0, "titanium": 0}
    # total 5 < 12 → an error message is returned
    err = check_payment_valid({"card": "UnknownCard", "payment": {"megacredits": 5}}, [], wf, player)
    assert err is not None and "insufficient" in err


def test_check_payment_sufficient():
    wf = {"type": "projectCard", "card": {"name": "UnknownCard", "calculatedCost": 12}}
    player = {"megacredits": 12}
    err = check_payment_valid({"card": "UnknownCard", "payment": {"megacredits": 12}}, [], wf, player)
    assert err is None


def test_standard_project_cost_single_sourced():
    assert standard_project_cost("City:SP") == STANDARD_PROJECT_COSTS["city:sp"]
    assert standard_project_cost("Standard projects: Asteroid:SP".split(":", 1)[1].strip()) == 14
    assert standard_project_cost("Sell patents") is None
