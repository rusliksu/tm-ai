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
"""

from __future__ import annotations
import numpy as np
from .config import PHASES, BOARDS, EXPANSION_FLAGS, TAG_TYPES, ACTION_SPACE_SIZE, STATE_DIM


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
    vec[pos] = game.get("generation", 1) / 20.0
    pos += 1
    vec[pos] = (game.get("temperature", -30) + 30) / 38.0   # range -30..+8 → 0..1
    pos += 1
    vec[pos] = game.get("oxygen", 0) / 14.0
    pos += 1
    vec[pos] = game.get("oceanCount", 0) / 9.0
    pos += 1
    phase = game.get("phase", "action")
    if phase in PHASES:
        vec[pos + PHASES.index(phase)] = 1.0
    pos += len(PHASES)

    # --- Player resources (7 dims) ---
    vec[pos] = player.get("terraformRating", 20) / 100.0
    pos += 1
    vec[pos] = min(player.get("megacredits", 0), 100) / 100.0
    pos += 1
    vec[pos] = min(player.get("steel", 0), 20) / 20.0
    pos += 1
    vec[pos] = min(player.get("titanium", 0), 20) / 20.0
    pos += 1
    vec[pos] = min(player.get("plants", 0), 20) / 20.0
    pos += 1
    vec[pos] = min(player.get("energy", 0), 20) / 20.0
    pos += 1
    vec[pos] = min(player.get("heat", 0), 20) / 20.0
    pos += 1

    # --- Player production (6 dims) ---
    prod = player.get("production", {})
    vec[pos] = (prod.get("megacredits", 0) + 5) / 25.0     # range -5..+20 → 0..1
    pos += 1
    vec[pos] = min(prod.get("steel", 0), 10) / 10.0
    pos += 1
    vec[pos] = min(prod.get("titanium", 0), 10) / 10.0
    pos += 1
    vec[pos] = min(prod.get("plants", 0), 10) / 10.0
    pos += 1
    vec[pos] = min(prod.get("energy", 0), 10) / 10.0
    pos += 1
    vec[pos] = min(prod.get("heat", 0), 10) / 10.0
    pos += 1

    # --- Player tags (13 dims) ---
    tags = player.get("tags", {})
    for tag in TAG_TYPES:
        vec[pos] = min(tags.get(tag) or 0, 20) / 20.0
        pos += 1

    # --- Hand size (1 dim) ---
    vec[pos] = min(player.get("handSize", 0), 10) / 10.0
    pos += 1

    # --- Game config (19 dims) ---
    if game_spec:
        vec[pos] = game_spec.get("player_count", 1) / 4.0
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
        for i, card in enumerate(waiting_for.get("cards", [])):
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

    if node_type in ("or", "initialCards"):
        options = waiting_for.get("options", [])
        chosen = options[index] if index < len(options) else {}
        return {"type": node_type, "index": index, "response": _default_response(chosen)}

    elif node_type == "and":
        options = waiting_for.get("options", [])
        return {"type": "and", "responses": [_default_response(opt) for opt in options]}

    elif node_type == "card":
        cards = waiting_for.get("cards", [])
        chosen = cards[index] if index < len(cards) else {}
        return {"type": "card", "cards": [chosen.get("name", "")]}

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

    else:
        return _default_response(waiting_for)


def response_to_index(waiting_for: dict, input_response: dict) -> int | None:
    """Extract chosen option index from an input_response (used in training)."""
    node_type = waiting_for.get("type", "")
    resp_type = input_response.get("type", "")

    if node_type in ("or", "initialCards") and resp_type in ("or", "initialCards"):
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
        return str(title.get("message", f"Option {fallback}"))
    return f"Option {fallback}"


def _default_response(node: dict) -> dict:
    """Heuristic: return the first/minimum valid response for any node type."""
    t = node.get("type", "option")

    if t == "option":
        return {"type": "option"}
    elif t in ("or", "initialCards"):
        opts = node.get("options", [])
        sub = _default_response(opts[0]) if opts else {"type": "option"}
        return {"type": t, "index": 0, "response": sub}
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
    else:
        return {"type": t}


def _mc_payment(amount: int) -> dict:
    return {
        "megaCredits": max(0, amount),
        "steel": 0, "titanium": 0, "heat": 0, "plants": 0,
        "microbes": 0, "floaters": 0, "lunaArchivesScience": 0,
        "seeds": 0, "graphene": 0, "kuiperAsteroids": 0,
        "auroraiData": 0, "spireScience": 0,
    }
