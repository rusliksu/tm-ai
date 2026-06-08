"""Tests for response parsing, payment validation, and single-sourced data in tm_llm."""
from tm_llm.options import flatten_options
from tm_llm.prompts import (
    capture_tactical, parse_action_response, standard_project_cost, find_choice, find_choices,
    STANDARD_PROJECT_COSTS, sanitize_strategy, game_end_proximity,
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


def test_find_choice_plain():
    assert find_choice("blah\nCHOICE: 3\n") == 3


def test_find_choice_markdown_variants():
    # Models often wrap the label in markdown emphasis or headings; all must still parse.
    assert find_choice("**CHOICE:** 1") == 1
    assert find_choice("**CHOICE**: 2") == 2
    assert find_choice("### CHOICE: 3") == 3
    assert find_choice("`CHOICE:` 4") == 4
    assert find_choice("- CHOICE:5") == 5


def test_find_choice_absent():
    assert find_choice("no decision here") is None


def test_parse_action_markdown_choice():
    wf = {"type": "or", "options": [
        {"type": "option", "title": "Play card"},
        {"type": "option", "title": "Pass"},
    ]}
    options = flatten_options(wf)
    text = "Reasoning.\n**TACTICAL:** hold the line\n**CHOICE:** 2"
    response, debug = parse_action_response(text, options, wf, "p1")
    assert response == {"type": "or", "index": 1, "response": {"type": "option"}}
    assert debug["llm_choice"] == 2


def test_capture_tactical_markdown():
    text = "Reasoning.\n**TACTICAL:** convert heat, then pass\n**CHOICE:** 2"
    assert capture_tactical(text) == "convert heat, then pass"


def test_payment_line_markdown():
    p = parse_payment_line("**PAYMENT:** MC=5, STEEL=3")
    assert p["megacredits"] == 5 and p["steel"] == 3


def test_standard_project_cost_single_sourced():
    assert standard_project_cost("City:SP") == STANDARD_PROJECT_COSTS["city:sp"]
    assert standard_project_cost("Standard projects: Asteroid:SP".split(":", 1)[1].strip()) == 14
    assert standard_project_cost("Sell patents") is None


def test_find_choices_multi():
    assert find_choices("CHOICE: 2,3") == [2, 3]
    assert find_choices("**CHOICE:** 1 and 4") == [1, 4]
    assert find_choices("CHOICE: 2") == [2]
    assert find_choices("CHOICE: 1, 3, 4") == [1, 3, 4]
    # prose after the answer must NOT be swept in (the hex number stays out)
    assert find_choices("CHOICE: 2 and place an ocean on hex-61") == [2]
    assert find_choices("no choice here") == []
    # find_choice keeps returning the first number for single-select callers
    assert find_choice("CHOICE: 2,3") == 2


def test_parse_action_multi_card_buy():
    # Regression: 'CHOICE: 2,3' on a multi-select card decision must buy BOTH cards, not just
    # the first (the research-phase 'Select card(s) to buy' bug).
    wf = {"type": "card", "title": "Select card(s) to buy", "min": 0, "max": 4,
          "cards": [{"name": "Imported GHG"}, {"name": "Water to Venus"},
                    {"name": "Bribed Committee"}, {"name": "Stratopolis"}]}
    options = flatten_options(wf)
    text = "Buy both for TR.\nTACTICAL: play them.\nCHOICE: 2,3"
    resp, _dbg = parse_action_response(text, options, wf, "pX", player={})
    assert resp == {"type": "card", "cards": ["Water to Venus", "Bribed Committee"]}


def test_parse_action_card_take_none():
    wf = {"type": "card", "title": "Select card(s) to buy", "min": 0, "max": 4,
          "cards": [{"name": "A"}, {"name": "B"}]}
    options = flatten_options(wf)
    resp, _dbg = parse_action_response("Too expensive, skip.\nCHOICE: none", options, wf, "pX", player={})
    assert resp == {"type": "card", "cards": []}


def test_parse_action_single_card_keep_unchanged():
    # A max=1 'keep one' draft decision still resolves to exactly the chosen card.
    wf = {"type": "card", "title": "Select a card to keep", "min": 1, "max": 1,
          "cards": [{"name": "A"}, {"name": "B"}]}
    options = flatten_options(wf)
    resp, _dbg = parse_action_response("Keep B.\nCHOICE: 2", options, wf, "pX", player={})
    assert resp == {"type": "card", "cards": ["B"]}


def test_sanitize_strategy_strips_turn_artefacts():
    raw = (
        "----- PRIOR STRATEGY -----\n"
        "**Reasoning:** Build an MC engine and target Builder.\n"
        "**TACTICAL:** Play Research Network, then a city.\n"
        "**CHOICE:** 15\n"
        "**PAYMENT:** MC=8\n"
        "--------------------------"
    )
    out = sanitize_strategy(raw)
    assert "CHOICE" not in out
    assert "PAYMENT" not in out
    assert "PRIOR STRATEGY" not in out
    assert "-----" not in out
    assert "MC engine" in out
    assert "Research Network" in out


def test_game_end_proximity_banner():
    # 0/3 and 1/3 maxed → no banner; ≥2/3 → banner that names what is still rising.
    assert game_end_proximity({"temperature": -10, "oxygen": 3, "oceanCount": 2}) is None
    assert game_end_proximity({"temperature": 8, "oxygen": 3, "oceanCount": 2}) is None
    banner = game_end_proximity({"temperature": 8, "oxygen": 14, "oceanCount": 7})
    assert banner is not None
    assert "GAME-END IMMINENT" in banner
    assert "oceans 7/9" in banner
