"""Tests for live-board rendering and hex adjacency (must match TM's algorithm)."""
from tm_llm.board import _BoardIndex, render_space_choices, render_live_board


def _grid(max_y=8):
    """A dense synthetic board covering every (x,y) for y in 0..max_y, x in 0..max_y."""
    spaces = []
    for y in range(max_y + 1):
        for x in range(max_y + 1):
            spaces.append({"id": f"{x}-{y}", "x": x, "y": y, "t": "land", "b": []})
    return spaces


def _coords(neighbours):
    return {(n["x"], n["y"]) for n in neighbours}


def test_adjacency_middle_row():
    # Middle row (y == maxY/2 == 4): bottom_right and top_right shift +1 in x.
    idx = _BoardIndex(_grid())
    space = idx.by_id["3-4"]
    got = _coords(idx.neighbours(space))
    assert got == {(3, 3), (4, 3), (4, 4), (4, 5), (3, 5), (2, 4)}


def test_adjacency_top_half():
    # y < 4: bottom_left shifts -1, top_right shifts +1.
    idx = _BoardIndex(_grid())
    space = idx.by_id["3-2"]
    got = _coords(idx.neighbours(space))
    assert got == {(3, 1), (4, 1), (4, 2), (3, 3), (2, 3), (2, 2)}


def test_adjacency_bottom_half():
    # y > 4: bottom_right shifts +1, top_left shifts -1.
    idx = _BoardIndex(_grid())
    space = idx.by_id["3-6"]
    got = _coords(idx.neighbours(space))
    assert got == {(2, 5), (3, 5), (4, 6), (4, 7), (3, 7), (2, 6)}


def test_render_space_choices_flags_own_city():
    # A candidate hex with YOUR city as a neighbour should be described as such.
    spaces = _grid()
    # Place a city owned by 'red' next to candidate 3-4 (its neighbour 4-4).
    for s in spaces:
        if s["id"] == "4-4":
            s["tile"] = 2  # city
            s["pc"] = "red"
        if s["id"] == "3-4":
            s["b"] = ["plant", "plant"]
    state = {
        "player": {"color": "red", "name": "Me"},
        "opponents": [],
        "boardSpaces": spaces,
    }
    options = [{"index": 0, "title": "3-4", "node": {"spaceId": "3-4"}}]
    out = render_space_choices(state, options)
    assert "hex-3-4" in out
    assert "YOUR city" in out
    assert "plant" in out


def test_render_live_board_groups_by_owner():
    spaces = _grid()
    for s in spaces:
        if s["id"] == "1-1":
            s["tile"], s["pc"] = 2, "red"      # my city
        if s["id"] == "2-2":
            s["tile"], s["pc"] = 0, "green"    # opp greenery
        if s["id"] == "5-5":
            s["tile"], s["pc"] = 1, "red"      # an ocean (owner irrelevant)
    state = {
        "player": {"color": "red", "name": "Me"},
        "opponents": [{"color": "green", "name": "Foe"}],
        "boardSpaces": spaces,
    }
    out = render_live_board(state)
    assert "YOU: city@hex-1-1" in out
    assert "Foe: greenery@hex-2-2" in out
    assert "Oceans: 1 placed" in out
