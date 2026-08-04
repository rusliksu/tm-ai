"""Pure structured-action compiler, validator, parser, and wire-response builder.

The legacy mapper chooses a flat option and fills every remaining child with a hidden
default.  This module instead makes every player-controlled value explicit.  It validates
only structure exposed by PlayerInputModel; Terraforming Mars remains the authority for
semantic constraints implemented inside server callbacks.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from .payment import mc_payment


ACTION_CONTRACT_VERSION = "action-contract/v2"
MAX_OPTIONS = 200


class ActionContractError(ValueError):
    """Fail-closed error with an allowlisted reason and no model/server payload."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class ActionChoice:
    value: int | str
    label: str


@dataclass(frozen=True)
class ActionBranch:
    value: int
    slots: tuple["ActionValueSlot", ...]


@dataclass(frozen=True)
class ActionValueSlot:
    id: str
    kind: str
    title: str
    minimum: int | None = None
    maximum: int | None = None
    min_count: int | None = None
    max_count: int | None = None
    choices: tuple[ActionChoice, ...] = ()
    branches: tuple[ActionBranch, ...] = ()


@dataclass(frozen=True)
class ActionCandidate:
    id: str
    title: str
    path: tuple[int, ...]
    slots: tuple[ActionValueSlot, ...]


@dataclass(frozen=True)
class ActionPlan:
    candidate: str
    values: dict[str, Any]


def node_title(node: dict, fallback: int) -> str:
    """Render a public PlayerInputModel title with numbered message substitutions."""
    title = node.get("title", "")
    if isinstance(title, str) and title:
        return title
    if isinstance(title, dict):
        message = title.get("message", f"Option {fallback}")
        data = title.get("data")
        if data and isinstance(data, list):
            def substitute(match: re.Match) -> str:
                index = int(match.group(1))
                entry = data[index] if index < len(data) else None
                return str(entry.get("value", match.group(0))) if isinstance(entry, dict) else match.group(0)

            message = re.sub(r"\$\{(\d+)\}", substitute, message)
        return message
    return f"Option {fallback}"


def _path_token(path: tuple[int, ...]) -> str:
    return ".".join(str(part) for part in path) if path else "root"


def _candidate_id(path: tuple[int, ...]) -> str:
    return f"p:{_path_token(path)}" if path else "root"


def _slot_id(path: tuple[int, ...]) -> str:
    return f"v:{_path_token(path)}"


def _enabled_indexed(items: list[dict]) -> list[tuple[int, dict]]:
    return [(index, item) for index, item in enumerate(items) if item.get("isDisabled") is not True]


def _is_selectable(node: dict) -> bool:
    node_type = node.get("type", "option")
    if node_type == "projectCard":
        return bool(_enabled_indexed(node.get("cards") or []))
    if node_type == "card":
        enabled = _enabled_indexed(node.get("cards") or [])
        return bool(enabled) or int(node.get("min", 0) or 0) == 0
    if node_type == "or":
        return any(_is_selectable(child) for child in node.get("options") or [])
    if node_type == "resource":
        return bool(node.get("include") or node.get("resources") or [])
    if node_type in ("space", "player", "delegate", "party", "colony", "globalEvent"):
        source_key = {
            "space": "spaces",
            "player": "players",
            "delegate": "players",
            "party": "parties",
            "colony": "coloniesModel",
            "globalEvent": "globalEventNames",
        }[node_type]
        return bool(node.get(source_key) or [])
    return True


def compile_action_candidates(waiting_for: dict) -> list[ActionCandidate]:
    """Compile the root decision into bounded candidates and recursive typed slots."""
    if not isinstance(waiting_for, dict):
        raise ActionContractError("invalid_tree")

    if waiting_for.get("type") == "or":
        candidates: list[ActionCandidate] = []
        for index, child in enumerate(waiting_for.get("options") or []):
            if not _is_selectable(child):
                continue
            if len(candidates) >= MAX_OPTIONS:
                raise ActionContractError("too_many_candidates")
            path = (index,)
            candidates.append(
                ActionCandidate(
                    id=_candidate_id(path),
                    title=node_title(child, index),
                    path=path,
                    slots=_compile_slots(child, path),
                )
            )
        if not candidates:
            raise ActionContractError("no_candidates")
        return candidates

    return [
        ActionCandidate(
            id="root",
            title=node_title(waiting_for, 0),
            path=(),
            slots=_compile_slots(waiting_for, ()),
        )
    ]


def _compile_slots(node: dict, path: tuple[int, ...]) -> tuple[ActionValueSlot, ...]:
    node_type = node.get("type", "option")
    title = node_title(node, path[-1] if path else 0)

    if node_type in ("option", "policy"):
        return ()

    if node_type in ("and", "initialCards"):
        slots: list[ActionValueSlot] = []
        for index, child in enumerate(node.get("options") or []):
            slots.extend(_compile_slots(child, path + (index,)))
        return tuple(slots)

    if node_type == "or":
        choices: list[ActionChoice] = []
        branches: list[ActionBranch] = []
        for index, child in enumerate(node.get("options") or []):
            if not _is_selectable(child):
                continue
            if len(choices) >= MAX_OPTIONS:
                raise ActionContractError("too_many_choices")
            choices.append(ActionChoice(index, node_title(child, index)))
            branches.append(ActionBranch(index, _compile_slots(child, path + (index,))))
        if not choices:
            raise ActionContractError("no_choices")
        return (
            ActionValueSlot(
                id=_slot_id(path),
                kind="branch",
                title=title,
                choices=tuple(choices),
                branches=tuple(branches),
            ),
        )

    if node_type == "amount":
        minimum = int(node.get("min", 0) or 0)
        maximum = int(node.get("max", minimum) if node.get("max") is not None else minimum)
        if maximum < minimum:
            raise ActionContractError("invalid_bounds")
        return (ActionValueSlot(_slot_id(path), "amount", title, minimum=minimum, maximum=maximum),)

    if node_type in ("card", "projectCard"):
        cards = _enabled_indexed(node.get("cards") or [])
        if len(cards) > MAX_OPTIONS:
            raise ActionContractError("too_many_choices")
        choices = tuple(ActionChoice(index, str(card.get("name", f"Card {index}"))) for index, card in cards)
        if node_type == "projectCard":
            if not choices:
                raise ActionContractError("no_choices")
            return (ActionValueSlot(_slot_id(path), "projectCard", title, choices=choices),)

        minimum = int(node.get("min", 0) or 0)
        maximum = int(node.get("max", 1) if node.get("max") is not None else 1)
        maximum = min(maximum, len(choices))
        if minimum < 0 or maximum < minimum:
            raise ActionContractError("invalid_bounds")
        return (
            ActionValueSlot(
                _slot_id(path),
                "card",
                title,
                min_count=minimum,
                max_count=maximum,
                choices=choices,
            ),
        )

    if node_type == "space":
        values = node.get("spaces") or []
        _ensure_bounded(values)
        choices = tuple(ActionChoice(str(value), str(value)) for value in values)
        return _single_choice_slot(path, node_type, title, choices)

    if node_type in ("player", "delegate"):
        values = node.get("players") or []
        _ensure_bounded(values)
        choices = tuple(ActionChoice(str(value), str(value)) for value in values)
        return _single_choice_slot(path, node_type, title, choices)

    if node_type == "colony":
        values = node.get("coloniesModel") or []
        _ensure_bounded(values)
        choices = tuple(
            ActionChoice(index, str(value.get("name", f"Colony {index}")))
            for index, value in enumerate(values)
        )
        return _single_choice_slot(path, node_type, title, choices)

    if node_type == "party":
        values = node.get("parties") or []
        _ensure_bounded(values)
        choices = tuple(ActionChoice(str(value), str(value)) for value in values)
        return _single_choice_slot(path, node_type, title, choices)

    if node_type == "resource":
        values = node.get("include") or node.get("resources") or []
        _ensure_bounded(values)
        choices = tuple(ActionChoice(str(value), str(value)) for value in values)
        return _single_choice_slot(path, node_type, title, choices)

    if node_type == "globalEvent":
        values = node.get("globalEventNames") or []
        _ensure_bounded(values)
        choices = tuple(ActionChoice(str(value), str(value)) for value in values)
        return _single_choice_slot(path, node_type, title, choices)

    raise ActionContractError("unsupported_node")


def _ensure_bounded(values: list) -> None:
    if len(values) > MAX_OPTIONS:
        raise ActionContractError("too_many_choices")


def _single_choice_slot(
    path: tuple[int, ...], kind: str, title: str, choices: tuple[ActionChoice, ...]
) -> tuple[ActionValueSlot, ...]:
    if not choices:
        raise ActionContractError("no_choices")
    return (ActionValueSlot(_slot_id(path), kind, title, choices=choices),)


def validate_action_plan(candidate: ActionCandidate, plan: ActionPlan) -> None:
    """Validate a plan against one compiled candidate without reading raw model text."""
    if plan.candidate != candidate.id:
        raise ActionContractError("unknown_candidate")
    if not isinstance(plan.values, dict):
        raise ActionContractError("value_type")

    used: set[str] = set()
    _validate_slots(candidate.slots, plan.values, used)
    if set(plan.values) - used:
        raise ActionContractError("unknown_value")


def _validate_slots(slots: tuple[ActionValueSlot, ...], values: dict[str, Any], used: set[str]) -> None:
    for slot in slots:
        if slot.id not in values:
            raise ActionContractError("missing_value")
        value = values[slot.id]
        used.add(slot.id)

        if slot.kind == "amount":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ActionContractError("value_type")
            if value < int(slot.minimum) or value > int(slot.maximum):
                raise ActionContractError("value_out_of_bounds")
            continue

        if slot.kind == "card":
            if not isinstance(value, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
                raise ActionContractError("value_type")
            if len(value) != len(set(value)):
                raise ActionContractError("duplicate_value")
            if len(value) < int(slot.min_count) or len(value) > int(slot.max_count):
                raise ActionContractError("value_out_of_bounds")
            allowed = {choice.value for choice in slot.choices}
            if any(item not in allowed for item in value):
                raise ActionContractError("value_not_allowed")
            continue

        allowed = {choice.value for choice in slot.choices}
        if value not in allowed or isinstance(value, bool):
            if slot.kind == "branch" and not isinstance(value, int):
                raise ActionContractError("value_type")
            raise ActionContractError("value_not_allowed")

        if slot.kind == "branch":
            branch = next((branch for branch in slot.branches if branch.value == value), None)
            if branch is None:
                raise ActionContractError("value_not_allowed")
            _validate_slots(branch.slots, values, used)


def build_input_response(waiting_for: dict, plan: ActionPlan) -> dict:
    """Validate a plan and build the existing Terraforming Mars wire response."""
    candidates = {candidate.id: candidate for candidate in compile_action_candidates(waiting_for)}
    candidate = candidates.get(plan.candidate)
    if candidate is None:
        raise ActionContractError("unknown_candidate")
    validate_action_plan(candidate, plan)

    if waiting_for.get("type") == "or":
        index = candidate.path[0]
        children = waiting_for.get("options") or []
        if index >= len(children):
            raise ActionContractError("invalid_tree")
        return {
            "type": "or",
            "index": index,
            "response": _build_node(children[index], (index,), plan.values),
        }
    return _build_node(waiting_for, (), plan.values)


def _build_node(node: dict, path: tuple[int, ...], values: dict[str, Any]) -> dict:
    node_type = node.get("type", "option")

    if node_type == "option":
        return {"type": "option"}
    if node_type == "policy":
        return {"type": "policy", "policyId": node.get("policyId", "")}
    if node_type == "or":
        index = values[_slot_id(path)]
        children = node.get("options") or []
        if index >= len(children):
            raise ActionContractError("invalid_tree")
        return {
            "type": "or",
            "index": index,
            "response": _build_node(children[index], path + (index,), values),
        }
    if node_type in ("and", "initialCards"):
        key = "responses"
        return {
            "type": node_type,
            key: [
                _build_node(child, path + (index,), values)
                for index, child in enumerate(node.get("options") or [])
            ],
        }
    if node_type == "amount":
        return {"type": "amount", "amount": values[_slot_id(path)]}
    if node_type == "card":
        cards = node.get("cards") or []
        indices = values[_slot_id(path)]
        return {"type": "card", "cards": [str(cards[index].get("name", "")) for index in indices]}
    if node_type == "projectCard":
        cards = node.get("cards") or []
        index = values[_slot_id(path)]
        card = cards[index]
        return {
            "type": "projectCard",
            "card": str(card.get("name", "")),
            "payment": mc_payment(int(card.get("calculatedCost", 0) or 0)),
        }
    if node_type == "space":
        return {"type": "space", "spaceId": values[_slot_id(path)]}
    if node_type == "player":
        return {"type": "player", "player": values[_slot_id(path)]}
    if node_type == "delegate":
        return {"type": "delegate", "player": values[_slot_id(path)]}
    if node_type == "colony":
        index = values[_slot_id(path)]
        colony = (node.get("coloniesModel") or [])[index]
        return {"type": "colony", "colonyName": str(colony.get("name", ""))}
    if node_type == "party":
        return {"type": "party", "partyName": values[_slot_id(path)]}
    if node_type == "resource":
        return {"type": "resource", "resource": values[_slot_id(path)]}
    if node_type == "globalEvent":
        return {"type": "globalEvent", "globalEventName": values[_slot_id(path)]}
    raise ActionContractError("unsupported_node")


_EMPH = r"[*_`~#]*"
_ACTION_LINE = re.compile(
    rf"^\s*{_EMPH}ACTION{_EMPH}\s*:{_EMPH}[ \t]*(\{{.*\}})\s*$",
    re.MULTILINE,
)


def parse_action_plan(text: str) -> ActionPlan:
    """Parse exactly one schema-limited ACTION JSON line without retaining model text."""
    matches = _ACTION_LINE.findall(text or "")
    if len(matches) != 1:
        raise ActionContractError("missing_action" if not matches else "multiple_actions")
    try:
        payload = json.loads(matches[0])
    except (TypeError, json.JSONDecodeError):
        raise ActionContractError("malformed_action") from None
    if not isinstance(payload, dict) or set(payload) != {"candidate", "values"}:
        raise ActionContractError("malformed_action")
    if not isinstance(payload.get("candidate"), str) or not isinstance(payload.get("values"), dict):
        raise ActionContractError("malformed_action")
    return ActionPlan(candidate=payload["candidate"], values=payload["values"])


def render_action_contract(candidates: list[ActionCandidate]) -> str:
    """Render a bounded model-facing schema; values stay explicit and machine-parseable."""
    lines = [
        f"ACTION CONTRACT: {ACTION_CONTRACT_VERSION}",
        "For this decision, ignore the legacy CHOICE line and use ACTION JSON.",
        "Candidates:",
    ]
    for candidate in candidates:
        lines.append(f"- {candidate.id}: {candidate.title}")
        _render_slots(candidate.slots, lines, indent="  ")
    lines.extend(
        [
            "Reply on one line:",
            'ACTION: {"candidate":"<candidate id>","values":{"<slot id>":<typed value>}}',
        ]
    )
    return "\n".join(lines)


def _render_slots(slots: tuple[ActionValueSlot, ...], lines: list[str], indent: str) -> None:
    for slot in slots:
        if slot.kind == "amount":
            detail = f"[{slot.minimum}..{slot.maximum}]"
        elif slot.kind == "card":
            choices = ", ".join(f"{choice.value}={choice.label}" for choice in slot.choices)
            detail = f"[{slot.min_count}..{slot.max_count}] from {{{choices}}}"
        else:
            choices = ", ".join(f"{choice.value}={choice.label}" for choice in slot.choices)
            detail = "{" + choices + "}"
        lines.append(f"{indent}- {slot.id} {slot.kind} {detail} {slot.title}")
        if slot.kind == "branch":
            for branch in slot.branches:
                lines.append(f"{indent}  when {slot.id}={branch.value}:")
                _render_slots(branch.slots, lines, indent + "    ")
