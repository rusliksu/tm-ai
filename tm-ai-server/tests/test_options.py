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
