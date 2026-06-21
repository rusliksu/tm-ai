"""Live board rendering and hex adjacency for the LLM prompts.

The TM server sends the full board every turn but never a human-readable view of it. This
module turns `state["boardSpaces"]` (every hex: id, x, y, bonuses, current tile + owner)
into:
  • render_live_board(state)   — a compact "who owns what, where" summary (normal turns)
  • render_space_choices(...)  — per-candidate adjacency detail for a `space` decision,
                                 so the model picks placements by described neighbours
                                 instead of doing offset-hex geometry itself.

Adjacency is computed exactly the way the TM server does it
(`Board.computeAdjacentSpaces`): a pointy-top hex grid where the bottom-left/top-right (top
half), or bottom-right/top-left (bottom half), neighbour shifts by one column, with the
middle row shifting both bottom-right and top-right. Reproducing it verbatim guarantees the
neighbours we describe match the ones the game enforces.
"""
from __future__ import annotations

# Numeric TileType → short name (mirrors stateMapping.ts TILE_TYPE_NAMES).
TILE_NAMES: dict[int, str] = {
    0: "greenery", 1: "ocean", 2: "city", 3: "capital", 4: "commercial",
    5: "ecological zone", 6: "industrial center", 7: "lava flows", 8: "mining area",
    9: "mining rights", 10: "mohole area", 11: "natural preserve", 12: "nuclear zone",
    13: "restricted area", 14: "deimos down", 15: "great dam", 16: "magnetic field gen.",
    17: "biofertilizer", 18: "metallic asteroid", 19: "solar farm",
    20: "ocean city", 21: "ocean farm", 22: "ocean sanctuary",
}

_CITY_TYPES = {2, 3, 20}  # city, capital, ocean city all count as cities for adjacency


def _tile_name(t) -> str:
    if t is None:
        return ""
    try:
        return TILE_NAMES.get(int(t), "special")
    except (TypeError, ValueError):
        return "special"


def _color_labels(state: dict) -> dict[str, str]:
    """Map player color → display label ('YOU' for the active player, else opponent name)."""
    labels: dict[str, str] = {}
    p = state.get("player", {})
    if p.get("color"):
        labels[p["color"]] = "YOU"
    for opp in state.get("opponents") or []:
        if opp.get("color"):
            labels[opp["color"]] = opp.get("name", "Opp")
    return labels


class _BoardIndex:
    """Indexes board spaces by (x, y) and reproduces TM's hex adjacency."""

    def __init__(self, board_spaces: list[dict]):
        self.by_xy: dict[tuple[int, int], dict] = {}
        self.by_id: dict[str, dict] = {}
        self.max_y = 0
        for s in board_spaces:
            x, y = s.get("x"), s.get("y")
            if x is None or y is None or x < 0:
                continue
            self.by_xy[(x, y)] = s
            self.by_id[str(s.get("id"))] = s
            self.max_y = max(self.max_y, y)

    def neighbours(self, space: dict) -> list[dict]:
        x, y = space.get("x"), space.get("y")
        if x is None or y is None:
            return []
        middle = self.max_y / 2
        left = [x - 1, y]
        right = [x + 1, y]
        top_left = [x, y - 1]
        top_right = [x, y - 1]
        bottom_left = [x, y + 1]
        bottom_right = [x, y + 1]
        if y < middle:
            bottom_left[0] -= 1
            top_right[0] += 1
        elif y == middle:
            bottom_right[0] += 1
            top_right[0] += 1
        else:
            bottom_right[0] += 1
            top_left[0] -= 1
        out: list[dict] = []
        for cx, cy in (top_left, top_right, right, bottom_right, bottom_left, left):
            adj = self.by_xy.get((cx, cy))
            if adj is not None and adj is not space:
                out.append(adj)
        return out


def render_live_board(state: dict) -> str:
    """Compact 'placed tiles by owner' summary for normal (non-placement) turns."""
    spaces = state.get("boardSpaces") or []
    if not spaces:
        return ""
    labels = _color_labels(state)
    by_owner: dict[str, list[str]] = {}
    oceans = 0
    for s in spaces:
        if s.get("tile") is None:
            continue
        name = _tile_name(s.get("tile"))
        if name == "ocean":
            oceans += 1
            continue
        owner = labels.get(s.get("pc"), s.get("pc") or "neutral")
        by_owner.setdefault(owner, []).append(f"{name}@hex-{s.get('id')}")
    if not by_owner and not oceans:
        return ""
    lines = ["Placed tiles on the board:"]
    if oceans:
        lines.append(f"  Oceans: {oceans} placed")
    # YOU first, then others
    for owner in sorted(by_owner, key=lambda o: (o != "YOU", o)):
        tiles = by_owner[owner]
        lines.append(f"  {owner}: {', '.join(tiles)}")
    return "\n".join(lines)


def render_space_choices(state: dict, options: list[dict]) -> str:
    """For a `space` decision, describe each offered candidate hex: placement bonus +
    adjacent tiles (whose city/greenery/ocean), so the model can pick for VP/bonuses.
    """
    spaces = state.get("boardSpaces") or []
    if not spaces:
        return ""
    index = _BoardIndex(spaces)
    labels = _color_labels(state)

    lines: list[str] = ["Candidate spaces (placement bonus + adjacent tiles):"]
    for opt in options:
        sid = str((opt.get("node") or {}).get("spaceId") or opt.get("title") or "")
        space = index.by_id.get(sid)
        if space is None:
            continue
        bonuses = space.get("b") or []
        bonus_str = ("+".join(bonuses)) if bonuses else "none"
        own_cities = own_greeneries = opp_cities = opp_greeneries = adj_oceans = 0
        specials: list[str] = []
        for adj in index.neighbours(space):
            t = adj.get("tile")
            if t is None:
                continue
            try:
                tid = int(t)
            except (TypeError, ValueError):
                tid = -1
            name = _tile_name(t)
            owner = labels.get(adj.get("pc"), "")
            who = "YOUR" if owner == "YOU" else (f"{owner}'s" if owner else "neutral")
            if name == "ocean":
                adj_oceans += 1
            elif tid in _CITY_TYPES:
                if owner == "YOU":
                    own_cities += 1
                else:
                    opp_cities += 1
            elif name == "greenery":
                if owner == "YOU":
                    own_greeneries += 1
                else:
                    opp_greeneries += 1
            else:
                # Special tiles (nuclear zone, natural preserve, etc.) — name + owner so the
                # model can read placement restrictions and adjacency value for either side.
                specials.append(f"{who} {name}")
        parts = []
        if own_cities:
            parts.append(f"YOUR city ×{own_cities}")
        if own_greeneries:
            parts.append(f"YOUR greenery ×{own_greeneries}")
        if opp_cities:
            parts.append(f"opponent city ×{opp_cities}")
        if opp_greeneries:
            parts.append(f"opponent greenery ×{opp_greeneries}")
        if adj_oceans:
            parts.append(f"ocean ×{adj_oceans}")
        parts.extend(specials)
        adj_str = "; ".join(parts) if parts else "no adjacent tiles"
        lines.append(f"  hex-{sid}: bonus={bonus_str}; adjacent → {adj_str}")
    return "\n".join(lines) if len(lines) > 1 else ""
