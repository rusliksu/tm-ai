"""Tests for the decision-tree handling lifted into tm_llm.options."""
from tm_llm.options import flatten_options, index_to_response, _default_response


def test_flatten_simple_or():
    wf = {"type": "or", "options": [
        {"type": "option", "title": "Play card"},
        {"type": "option", "title": "Pass"},
    ]}
    opts = flatten_options(wf)
    assert [o["title"] for o in opts] == ["Play card", "Pass"]
    assert [o["index"] for o in opts] == [0, 1]


def test_flatten_expands_nested_or_of_leaf_options():
    wf = {"type": "or", "options": [
        {"type": "or", "title": "Fund an award", "options": [
            {"type": "option", "title": "Banker"},
            {"type": "option", "title": "Scientist"},
        ]},
        {"type": "option", "title": "Pass"},
    ]}
    opts = flatten_options(wf)
    titles = [o["title"] for o in opts]
    assert "Fund an award: Banker" in titles
    assert "Fund an award: Scientist" in titles
    assert "Pass" in titles
    # The nested award option carries a 2-element path so we resolve the sub-choice.
    banker = next(o for o in opts if o["title"].endswith("Banker"))
    assert banker["path"] == [0, 0]


def test_flatten_expands_standard_projects_project_card_menu():
    # "Standard projects" is a single projectCard node listing every SP. It must expand to one
    # option per project so the LLM can pick e.g. Aquifer — not silently get the first (Power
    # Plant). Regression for the gen-10 "wanted an ocean, got a power plant" bug.
    wf = {"type": "or", "options": [
        {"type": "option", "title": "Pass"},
        {"type": "projectCard", "title": "Standard projects", "cards": [
            {"name": "Power Plant:SP", "calculatedCost": 11},
            {"name": "Aquifer:SP", "calculatedCost": 18},
            {"name": "City:SP", "calculatedCost": 25},
        ]},
    ]}
    opts = flatten_options(wf)
    titles = [o["title"] for o in opts]
    assert "Standard projects: Aquifer:SP" in titles
    assert "Standard projects: City:SP" in titles
    aquifer = next(o for o in opts if o["title"].endswith("Aquifer:SP"))
    assert aquifer["path"] == [1, 1]
    # The parent projectCard node is retained so payment resolution can find the cost.
    assert aquifer["node"].get("type") == "projectCard"
    # Picking Aquifer resolves to the Aquifer card, not the first one.
    resp = index_to_response(wf, aquifer["path"])
    assert resp == {"type": "or", "index": 1,
                    "response": {"type": "projectCard", "card": "Aquifer:SP",
                                 "payment": resp["response"]["payment"]}}
    assert resp["response"]["payment"]["megacredits"] == 18


def test_flatten_keeps_single_card_project_node_collapsed():
    # A projectCard child with only one card need not be expanded (nothing to choose).
    wf = {"type": "or", "options": [
        {"type": "option", "title": "Pass"},
        {"type": "projectCard", "title": "Sell patents", "cards": [{"name": "X", "calculatedCost": 0}]},
    ]}
    opts = flatten_options(wf)
    assert [o["title"] for o in opts] == ["Pass", "Sell patents"]


def test_index_to_response_or_nested():
    wf = {"type": "or", "options": [
        {"type": "or", "title": "Fund", "options": [
            {"type": "option", "title": "Banker"},
            {"type": "option", "title": "Scientist"},
        ]},
        {"type": "option", "title": "Pass"},
    ]}
    resp = index_to_response(wf, [0, 1])
    assert resp == {"type": "or", "index": 0,
                    "response": {"type": "or", "index": 1, "response": {"type": "option"}}}


def test_index_to_response_project_card():
    wf = {"type": "projectCard", "cards": [{"name": "Tardigrades", "calculatedCost": 4}]}
    resp = index_to_response(wf, 0)
    assert resp["type"] == "projectCard"
    assert resp["card"] == "Tardigrades"
    assert resp["payment"]["megacredits"] == 4


def test_default_response_or_prefers_last_bare_option():
    # Pass (the last bare option) should be chosen as the safe default.
    wf = {"type": "or", "options": [
        {"type": "projectCard", "cards": [{"name": "X", "calculatedCost": 3}]},
        {"type": "option", "title": "Pass"},
    ]}
    resp = _default_response(wf)
    assert resp == {"type": "or", "index": 1, "response": {"type": "option"}}


def test_card_min_selection():
    wf = {"type": "card", "min": 1, "max": 1,
          "cards": [{"name": "A"}, {"name": "B"}]}
    resp = index_to_response(wf, 1)
    assert resp == {"type": "card", "cards": ["B"]}
