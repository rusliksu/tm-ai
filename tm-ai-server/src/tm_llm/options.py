"""Decision-tree handling — enumerate selectable choices and build InputResponses.

These are pure, NN-free functions lifted from the original encoding.py. They turn a
TM `waitingFor` (PlayerInputModel) node into a flat list of options the LLM picks from,
and turn a picked option back into a valid InputResponse for `player.process()`.

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

from .action_contract import MAX_OPTIONS, node_title as _node_title
from .payment import mc_payment


def flatten_options(waiting_for: dict, max_actions: int = MAX_OPTIONS) -> list[dict]:
    """Enumerate the immediate selectable choices from a PlayerInputModel node.

    Each returned dict has:
      - title: display string (parent-prefixed for expanded nested ORs)
      - index: 0-based position in the flat output (= displayed CHOICE number minus 1)
      - path:  list[int] walking the original tree, passed to index_to_response
               to build the nested InputResponse
      - node:  the leaf node for this choice (for card/desc extraction)

    Nested OR options whose children are all leaf 'option' types are expanded inline so
    the LLM picks the specific sub-choice directly (e.g. a specific award under "Fund an
    award"). Nested projectCard menus (a "Standard projects" list, or an inline "Play a
    card" list) are likewise expanded into one option per card, so the LLM names the
    specific project instead of the engine silently defaulting to the first card.
    """
    node_type = waiting_for.get("type", "")
    options: list[dict] = []

    def _emit(title: str, path: list[int], node: dict) -> bool:
        if len(options) >= max_actions:
            return False
        options.append({"title": title, "index": len(options), "path": path, "node": node})
        return True

    if node_type in ("or", "and", "initialCards"):
        for i, opt in enumerate(waiting_for.get("options", [])):
            if len(options) >= max_actions:
                break
            child_opts = opt.get("options") or []
            child_cards = opt.get("cards") or []
            if (
                opt.get("type") == "or"
                and child_opts
                and all(c.get("type") == "option" for c in child_opts)
            ):
                parent_title = _node_title(opt, i).strip()
                for j, child in enumerate(child_opts):
                    if len(options) >= max_actions:
                        break
                    child_title = _node_title(child, j)
                    full_title = f"{parent_title}: {child_title}" if parent_title else child_title
                    _emit(full_title, [i, j], child)
            elif opt.get("type") == "projectCard" and child_cards:
                # A nested project-card menu (e.g. "Standard projects" listing every standard
                # project, or an inline "Play a card" listing the hand). Expand each card into
                # its own option so the LLM picks the specific one — otherwise the engine
                # silently defaults to the FIRST card (historically always Power Plant). The
                # parent projectCard node is kept as `node` so payment auto-generation and
                # validation can still resolve each card's cost from its `.cards`.
                enabled_cards = [
                    (j, card) for j, card in enumerate(child_cards)
                    if card.get("isDisabled") is not True
                ]
                if len(child_cards) == 1 and len(enabled_cards) == 1:
                    _emit(_node_title(opt, i), [i], opt)
                else:
                    parent_title = _node_title(opt, i).strip()
                    for j, card in enabled_cards:
                        if len(options) >= max_actions:
                            break
                        cname = card.get("name", f"Card {j}")
                        full_title = f"{parent_title}: {cname}" if parent_title else cname
                        _emit(full_title, [i, j], opt)
            else:
                _emit(_node_title(opt, i), [i], opt)

    elif node_type == "card":
        cards = waiting_for.get("cards", [])
        if not cards or waiting_for.get("max", 1) == 0:
            _emit(waiting_for.get("title", "OK") or "OK", [0], {})
        else:
            for i, card in enumerate(cards):
                if len(options) >= max_actions:
                    break
                _emit(card.get("name", f"Card {i}"), [i], card)

    elif node_type == "projectCard":
        for i, card in enumerate(waiting_for.get("cards", [])):
            if len(options) >= max_actions:
                break
            if card.get("isDisabled") is True:
                continue
            _emit(card.get("name", f"Card {i}"), [i], card)

    elif node_type == "space":
        for i, space_id in enumerate(waiting_for.get("spaces", [])):
            if len(options) >= max_actions:
                break
            _emit(str(space_id), [i], {"spaceId": space_id})

    elif node_type == "amount":
        min_val = waiting_for.get("min", 0)
        max_val = min(waiting_for.get("max", min_val), min_val + max_actions - 1)
        for i, v in enumerate(range(min_val, max_val + 1)):
            _emit(str(v), [i], {"amount": v})

    elif node_type == "player":
        for i, p in enumerate(waiting_for.get("players", [])):
            if len(options) >= max_actions:
                break
            _emit(str(p), [i], {"player": p})

    elif node_type == "colony":
        for i, colony in enumerate(waiting_for.get("coloniesModel", [])):
            if len(options) >= max_actions:
                break
            _emit(colony.get("name", f"Colony {i}"), [i], colony)

    elif node_type == "delegate":
        for i, p in enumerate(waiting_for.get("players", [])):
            if len(options) >= max_actions:
                break
            _emit(str(p), [i], {"player": p})

    elif node_type == "party":
        for i, party in enumerate(waiting_for.get("parties", [])):
            if len(options) >= max_actions:
                break
            _emit(str(party), [i], {"partyName": party})

    elif node_type == "resource":
        for i, res in enumerate(waiting_for.get("resources", [])):
            if len(options) >= max_actions:
                break
            _emit(str(res), [i], {"resourceType": res})

    else:
        _emit(_node_title(waiting_for, 0), [0], waiting_for)

    return options


def index_to_response(waiting_for: dict, index) -> dict:
    """Construct a valid InputResponse for choosing option at `index`.

    `index` may be an int (top-level pick) or a list[int] path through nested decisions.
    """
    if isinstance(index, (list, tuple)):
        path = list(index)
    else:
        path = [int(index)]
    head = path[0] if path else 0
    rest = path[1:]

    node_type = waiting_for.get("type", "option")

    if node_type == "or":
        options = waiting_for.get("options", [])
        chosen = options[head] if head < len(options) else {}
        sub_response = index_to_response(chosen, rest) if rest else _default_response(chosen)
        return {"type": "or", "index": head, "response": sub_response}

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
        selected = [cards[(head + i) % n].get("name", "") for i in range(min(min_count, n))]
        return {"type": "card", "cards": selected}

    elif node_type == "projectCard":
        cards = waiting_for.get("cards", [])
        chosen = cards[head] if head < len(cards) else {}
        cost = chosen.get("calculatedCost", 0)
        return {"type": "projectCard", "card": chosen.get("name", ""), "payment": mc_payment(cost)}

    elif node_type == "space":
        spaces = waiting_for.get("spaces", [])
        space_id = spaces[head] if head < len(spaces) else ""
        return {"type": "space", "spaceId": space_id}

    elif node_type == "amount":
        return {"type": "amount", "amount": waiting_for.get("min", 0) + head}

    elif node_type == "player":
        players = waiting_for.get("players", [])
        return {"type": "player", "player": players[head] if head < len(players) else ""}

    elif node_type == "colony":
        colonies = waiting_for.get("coloniesModel", [])
        chosen = colonies[head] if head < len(colonies) else {}
        return {"type": "colony", "colonyName": chosen.get("name", "")}

    elif node_type == "delegate":
        players = waiting_for.get("players", [])
        return {"type": "delegate", "player": players[head] if head < len(players) else "neutral"}

    elif node_type == "party":
        parties = waiting_for.get("parties", [])
        return {"type": "party", "partyName": parties[head] if head < len(parties) else ""}

    elif node_type == "resource":
        resources = waiting_for.get("resources", ["megacredits"])
        res = resources[head] if head < len(resources) else resources[0]
        return {"type": "resource", "resourceType": res}

    else:
        return _default_response(waiting_for)


def _default_response(node: dict) -> dict:
    """Heuristic: return the first/minimum valid response for any node type.

    For OR nodes, prefer the LAST bare-option sub-option (conventionally Pass) to mirror
    TM's own aiFallbackResponse — picking option 0 historically produced junk moves.
    """
    t = node.get("type", "option")

    if t == "option":
        return {"type": "option"}
    elif t == "or":
        opts = node.get("options", [])
        if not opts:
            return {"type": "or", "index": 0, "response": {"type": "option"}}
        for i in range(len(opts) - 1, -1, -1):
            if opts[i].get("type") == "option":
                return {"type": "or", "index": i, "response": {"type": "option"}}
        return {"type": "or", "index": 0, "response": _default_response(opts[0])}
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
            return {"type": "projectCard", "card": c.get("name", ""), "payment": mc_payment(c.get("calculatedCost", 0))}
        return {"type": "option"}
    elif t == "space":
        spaces = node.get("spaces", [])
        return {"type": "space", "spaceId": spaces[0] if spaces else ""}
    elif t == "amount":
        return {"type": "amount", "amount": node.get("min", 0)}
    elif t == "payment":
        return {"type": "payment", "payment": mc_payment(node.get("amount", 0))}
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
