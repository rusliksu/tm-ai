"""
State encoding and action space handling.

InputResponse wire format (from TM server InputResponse.ts):
  OrOptions:    {type:'or',   index:N, response:<InputResponse>}
  AndOptions:   {type:'and',  responses:[<InputResponse>, ...]}
  SelectOption: {type:'option'}
  SelectCard:   {type:'card', cards:[<CardName>, ...]}
  SelectProjectCardToPlay: {type:'projectCard', card:<CardName>, payment:{...}}
  SelectSpace:  {type:'space',  spaceId:<SpaceId>}
  SelectAmount: {type:'amount', amount:N}
  SelectPlayer: {type:'player', player:<Color>}
  SelectColony: {type:'colony', colonyName:<ColonyName>}
  SelectDelegate: {type:'delegate', player:<Color>}
  SelectParty:  {type:'party',  partyName:<PartyName>}
  SelectResource: {type:'resource', resourceType:<ResourceType>}
"""

from __future__ import annotations
import numpy as np
from .config import (
    PHASES, BOARDS, EXPANSION_FLAGS, TAG_TYPES,
    CARD_RESOURCE_VOCAB, CARD_RESOURCE_CAP,
    RESOURCE_CAPS, PRODUCTION_CAPS,
    ACTION_SPACE_SIZE, STATE_DIM,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cap(value: float | int | None, cap: float) -> float:
    return min(float(value or 0), cap) / cap


def _encode_player(vec: np.ndarray, pos: int, p: dict) -> int:
    """Encode a player snapshot into vec starting at pos. Returns new pos."""

    # Resources (7) — including terraformRating
    vec[pos] = _cap(p.get("terraformRating", 20), RESOURCE_CAPS["terraformRating"])
    pos += 1
    for res in ("megacredits", "steel", "titanium", "plants", "energy", "heat"):
        vec[pos] = _cap(p.get(res, 0), RESOURCE_CAPS[res])
        pos += 1

    # Production (6)
    prod = p.get("production", {})
    mc_prod = prod.get("megacredits", 0) or 0
    cap_mc = PRODUCTION_CAPS["megacredits"]
    vec[pos] = (mc_prod + 5) / (cap_mc + 5)   # -5..+60 → 0..1
    pos += 1
    for res in ("steel", "titanium", "plants", "energy", "heat"):
        cap = PRODUCTION_CAPS[res]
        vec[pos] = min(float(prod.get(res, 0) or 0), cap) / cap
        pos += 1

    # Tags (13)
    tags = p.get("tags", {})
    for tag in TAG_TYPES:
        vec[pos] = _cap(tags.get(tag) or 0, 20)
        pos += 1

    # Hand size (1) — visible to all players
    vec[pos] = _cap(p.get("handSize", 0), 10)
    pos += 1

    # Per-card resource counts (199) — one slot per card in vocab
    card_res = p.get("cardResources", {})
    for card_name in CARD_RESOURCE_VOCAB:
        vec[pos] = _cap(card_res.get(card_name, 0), CARD_RESOURCE_CAP)
        pos += 1

    # Played card count (1)
    vec[pos] = _cap(p.get("playedCardCount", len(p.get("playedCards", []))), 30)
    pos += 1

    # Board tiles (3): greenery, city, special
    bt = p.get("boardTiles", {})
    vec[pos] = _cap(bt.get("greenery", 0), 10)
    pos += 1
    vec[pos] = _cap(bt.get("city", 0), 10)
    pos += 1
    vec[pos] = _cap(bt.get("special", 0), 10)
    pos += 1

    return pos


# ---------------------------------------------------------------------------
# State encoding
# ---------------------------------------------------------------------------

def encode_state(state: dict, game_spec: dict | None = None) -> np.ndarray:
    """Encode game state dict to float32 vector of shape (STATE_DIM,)."""
    vec = np.zeros(STATE_DIM, dtype=np.float32)
    pos = 0

    game = state.get("game", {})
    player = state.get("player", {})

    # --- Global (9 dims) ---
    vec[pos] = (game.get("generation", 1) or 1) / 20.0
    pos += 1
    vec[pos] = ((game.get("temperature", -30) or -30) + 30) / 38.0  # -30..+8 → 0..1
    pos += 1
    vec[pos] = (game.get("oxygen", 0) or 0) / 14.0
    pos += 1
    vec[pos] = (game.get("oceanCount", 0) or 0) / 9.0
    pos += 1
    phase = game.get("phase", "action")
    if phase in PHASES:
        vec[pos + PHASES.index(phase)] = 1.0
    pos += len(PHASES)

    # --- Active player (230 dims) ---
    pos = _encode_player(vec, pos, player)

    # --- One opponent slot (230 dims) ---
    from .config import _PLAYER_DIMS
    opponents = state.get("opponents", [])
    if opponents:
        # Use the strongest opponent (highest TR) as the representative
        opp = max(opponents, key=lambda o: o.get("terraformRating", 0) or 0)
        pos = _encode_player(vec, pos, opp)
    else:
        pos += _PLAYER_DIMS  # all zeros — solo or no opponent data

    # --- Milestones / Awards (4 dims) ---
    player_id = player.get("id", "")
    milestones = state.get("milestones", [])
    awards = state.get("awards", [])
    ms_self = sum(1 for m in milestones if m.get("playerId") == player_id)
    ms_total = len(milestones)
    aw_self = sum(1 for a in awards if a.get("playerId") == player_id)
    aw_total = len(awards)
    vec[pos] = ms_self / 3.0      # max 3 milestones per player
    pos += 1
    vec[pos] = ms_total / 5.0     # max 5 milestones total
    pos += 1
    vec[pos] = aw_self / 3.0
    pos += 1
    vec[pos] = aw_total / 5.0
    pos += 1

    # --- Game config (19 dims) ---
    if game_spec:
        vec[pos] = (game_spec.get("player_count", 1) or 1) / 4.0
        pos += 1
        board = game_spec.get("board_name", "tharsis")
        if board in BOARDS:
            vec[pos + BOARDS.index(board)] = 1.0
        pos += len(BOARDS)
        expansions = set(game_spec.get("expansions", []))
        for exp in EXPANSION_FLAGS:
            vec[pos] = 1.0 if exp in expansions else 0.0
            pos += 1
    else:
        pos += 1 + len(BOARDS) + len(EXPANSION_FLAGS)

    assert pos == STATE_DIM, f"encode_state: wrote {pos} dims, expected {STATE_DIM}"
    return vec


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

def flatten_options(waiting_for: dict, max_actions: int = ACTION_SPACE_SIZE) -> list[dict]:
    """
    Enumerate the immediate selectable choices from a PlayerInputModel node.
    Each returned dict has {title, index, node} where node is the sub-model
    for the chosen option (used to build the sub-response via index_to_response).
    """
    node_type = waiting_for.get("type", "")
    options: list[dict] = []

    if node_type in ("or", "and", "initialCards"):
        for i, opt in enumerate(waiting_for.get("options", [])):
            if len(options) >= max_actions:
                break
            options.append({"title": _node_title(opt, i), "index": i, "node": opt})

    elif node_type == "card":
        cards = waiting_for.get("cards", [])
        if not cards or waiting_for.get("max", 1) == 0:
            # No selection possible (empty list or max=0 notification like "You cannot afford any cards")
            options.append({"title": waiting_for.get("title", "OK") or "OK", "index": 0, "node": {}})
        else:
            for i, card in enumerate(cards):
                if len(options) >= max_actions:
                    break
                options.append({"title": card.get("name", f"Card {i}"), "index": i, "node": card})

    elif node_type == "projectCard":
        for i, card in enumerate(waiting_for.get("cards", [])):
            if len(options) >= max_actions:
                break
            options.append({"title": card.get("name", f"Card {i}"), "index": i, "node": card})

    elif node_type == "space":
        for i, space_id in enumerate(waiting_for.get("spaces", [])):
            if len(options) >= max_actions:
                break
            options.append({"title": str(space_id), "index": i, "node": {"spaceId": space_id}})

    elif node_type == "amount":
        min_val = waiting_for.get("min", 0)
        max_val = min(waiting_for.get("max", min_val), min_val + max_actions - 1)
        for i, v in enumerate(range(min_val, max_val + 1)):
            options.append({"title": str(v), "index": i, "node": {"amount": v}})

    elif node_type == "player":
        for i, p in enumerate(waiting_for.get("players", [])):
            if len(options) >= max_actions:
                break
            options.append({"title": str(p), "index": i, "node": {"player": p}})

    elif node_type == "colony":
        for i, colony in enumerate(waiting_for.get("coloniesModel", [])):
            if len(options) >= max_actions:
                break
            options.append({"title": colony.get("name", f"Colony {i}"), "index": i, "node": colony})

    elif node_type == "delegate":
        for i, p in enumerate(waiting_for.get("players", [])):
            if len(options) >= max_actions:
                break
            options.append({"title": str(p), "index": i, "node": {"player": p}})

    elif node_type == "party":
        for i, party in enumerate(waiting_for.get("parties", [])):
            if len(options) >= max_actions:
                break
            options.append({"title": str(party), "index": i, "node": {"partyName": party}})

    elif node_type == "resource":
        for i, res in enumerate(waiting_for.get("resources", [])):
            if len(options) >= max_actions:
                break
            options.append({"title": str(res), "index": i, "node": {"resourceType": res}})

    else:
        # Leaf node (option, payment, etc.) — single choice
        options.append({"title": _node_title(waiting_for, 0), "index": 0, "node": waiting_for})

    return options


def build_mask(num_valid: int, max_size: int = ACTION_SPACE_SIZE) -> np.ndarray:
    """Bool array of shape (max_size,) with first num_valid slots True."""
    mask = np.zeros(max_size, dtype=bool)
    mask[:min(num_valid, max_size)] = True
    return mask


def index_to_response(waiting_for: dict, index: int) -> dict:
    """Construct a valid InputResponse for choosing option at `index`."""
    node_type = waiting_for.get("type", "option")

    if node_type == "or":
        options = waiting_for.get("options", [])
        chosen = options[index] if index < len(options) else {}
        return {"type": "or", "index": index, "response": _default_response(chosen)}

    elif node_type == "initialCards":
        options = waiting_for.get("options", [])
        return {"type": "initialCards", "responses": [_default_response(opt) for opt in options]}

    elif node_type == "and":
        options = waiting_for.get("options", [])
        return {"type": "and", "responses": [_default_response(opt) for opt in options]}

    elif node_type == "card":
        cards = waiting_for.get("cards", [])
        if not cards or waiting_for.get("max", 1) == 0:
            return {"type": "card", "cards": []}
        min_count = max(waiting_for.get("min", 1), 1)
        n = len(cards)
        # Pick min_count cards starting at index, wrapping around
        selected = [cards[(index + i) % n].get("name", "") for i in range(min(min_count, n))]
        return {"type": "card", "cards": selected}

    elif node_type == "projectCard":
        cards = waiting_for.get("cards", [])
        chosen = cards[index] if index < len(cards) else {}
        cost = chosen.get("calculatedCost", 0)
        return {"type": "projectCard", "card": chosen.get("name", ""), "payment": _mc_payment(cost)}

    elif node_type == "space":
        spaces = waiting_for.get("spaces", [])
        space_id = spaces[index] if index < len(spaces) else ""
        return {"type": "space", "spaceId": space_id}

    elif node_type == "amount":
        return {"type": "amount", "amount": waiting_for.get("min", 0) + index}

    elif node_type == "player":
        players = waiting_for.get("players", [])
        return {"type": "player", "player": players[index] if index < len(players) else ""}

    elif node_type == "colony":
        colonies = waiting_for.get("coloniesModel", [])
        chosen = colonies[index] if index < len(colonies) else {}
        return {"type": "colony", "colonyName": chosen.get("name", "")}

    elif node_type == "delegate":
        players = waiting_for.get("players", [])
        return {"type": "delegate", "player": players[index] if index < len(players) else "neutral"}

    elif node_type == "party":
        parties = waiting_for.get("parties", [])
        return {"type": "party", "partyName": parties[index] if index < len(parties) else ""}

    elif node_type == "resource":
        resources = waiting_for.get("resources", ["megacredits"])
        res = resources[index] if index < len(resources) else resources[0]
        return {"type": "resource", "resourceType": res}

    else:
        return _default_response(waiting_for)


def response_to_index(waiting_for: dict, input_response: dict) -> int | None:
    """Extract chosen option index from an input_response (used in training)."""
    node_type = waiting_for.get("type", "")
    resp_type = input_response.get("type", "")

    if node_type == "or" and resp_type == "or":
        return input_response.get("index")

    elif node_type == "card" and resp_type == "card":
        chosen = input_response.get("cards", [])
        if not chosen:
            return None
        name = chosen[0]
        for i, card in enumerate(waiting_for.get("cards", [])):
            if card.get("name") == name:
                return i

    elif node_type == "projectCard" and resp_type == "projectCard":
        name = input_response.get("card", "")
        for i, card in enumerate(waiting_for.get("cards", [])):
            if card.get("name") == name:
                return i

    elif node_type == "space" and resp_type == "space":
        space_id = input_response.get("spaceId", "")
        spaces = waiting_for.get("spaces", [])
        if space_id in spaces:
            return spaces.index(space_id)

    elif node_type == "amount" and resp_type == "amount":
        return max(0, input_response.get("amount", 0) - waiting_for.get("min", 0))

    elif node_type == "player" and resp_type == "player":
        color = input_response.get("player", "")
        players = waiting_for.get("players", [])
        if color in players:
            return players.index(color)

    elif node_type in ("option", "payment") and resp_type == node_type:
        return 0

    elif node_type == "colony" and resp_type == "colony":
        name = input_response.get("colonyName", "")
        for i, c in enumerate(waiting_for.get("coloniesModel", [])):
            if c.get("name") == name:
                return i

    elif node_type == "delegate" and resp_type == "delegate":
        color = input_response.get("player", "")
        players = waiting_for.get("players", [])
        if color in players:
            return players.index(color)

    elif node_type == "party" and resp_type == "party":
        name = input_response.get("partyName", "")
        parties = waiting_for.get("parties", [])
        if name in parties:
            return parties.index(name)

    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node_title(node: dict, fallback: int) -> str:
    title = node.get("title", "")
    if isinstance(title, str) and title:
        return title
    if isinstance(title, dict):
        msg = title.get("message", f"Option {fallback}")
        data = title.get("data")
        if data and isinstance(data, list):
            import re
            def _sub(m: re.Match) -> str:
                idx = int(m.group(1))
                entry = data[idx] if idx < len(data) else None
                return str(entry.get("value", m.group(0))) if isinstance(entry, dict) else m.group(0)
            msg = re.sub(r'\$\{(\d+)\}', _sub, msg)
        return msg
    return f"Option {fallback}"


def _default_response(node: dict) -> dict:
    """Heuristic: return the first/minimum valid response for any node type."""
    t = node.get("type", "option")

    if t == "option":
        return {"type": "option"}
    elif t == "or":
        opts = node.get("options", [])
        sub = _default_response(opts[0]) if opts else {"type": "option"}
        return {"type": "or", "index": 0, "response": sub}
    elif t == "initialCards":
        opts = node.get("options", [])
        return {"type": "initialCards", "responses": [_default_response(opt) for opt in opts]}
    elif t == "and":
        return {"type": "and", "responses": [_default_response(o) for o in node.get("options", [])]}
    elif t == "card":
        min_count = node.get("min", 0)
        cards = node.get("cards", [])
        return {"type": "card", "cards": [c.get("name", "") for c in cards[:min_count]]}
    elif t == "projectCard":
        cards = node.get("cards", [])
        if cards:
            c = cards[0]
            return {"type": "projectCard", "card": c.get("name", ""), "payment": _mc_payment(c.get("calculatedCost", 0))}
        return {"type": "option"}
    elif t == "space":
        spaces = node.get("spaces", [])
        return {"type": "space", "spaceId": spaces[0] if spaces else ""}
    elif t == "amount":
        return {"type": "amount", "amount": node.get("min", 0)}
    elif t == "payment":
        return {"type": "payment", "payment": _mc_payment(node.get("amount", 0))}
    elif t == "player":
        players = node.get("players", [])
        return {"type": "player", "player": players[0] if players else ""}
    elif t == "colony":
        cols = node.get("coloniesModel", [])
        return {"type": "colony", "colonyName": cols[0].get("name", "") if cols else ""}
    elif t == "delegate":
        players = node.get("players", [])
        return {"type": "delegate", "player": players[0] if players else "neutral"}
    elif t == "party":
        parties = node.get("parties", [])
        return {"type": "party", "partyName": parties[0] if parties else ""}
    elif t == "globalEvent":
        events = node.get("globalEventNames", [])
        return {"type": "globalEvent", "globalEventName": events[0] if events else ""}
    elif t == "policy":
        return {"type": "policy", "policyId": node.get("policyId", "")}
    elif t == "resource":
        resources = node.get("resources", ["megacredits"])
        return {"type": "resource", "resourceType": resources[0] if resources else "megacredits"}
    else:
        return {"type": t}


def _mc_payment(amount: int) -> dict:
    return {
        "megacredits": max(0, amount),
        "steel": 0, "titanium": 0, "heat": 0, "plants": 0,
        "microbes": 0, "floaters": 0, "lunaArchivesScience": 0,
        "seeds": 0, "graphene": 0, "kuiperAsteroids": 0,
        "auroraiData": 0, "spireScience": 0,
    }
