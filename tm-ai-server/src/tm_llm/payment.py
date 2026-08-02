"""Payment parsing, correction, validation, and auto-generation for project cards.

The LLM emits `PAYMENT: MC=n[, STEEL=n][, TITANIUM=n][, HEAT=n]...`. These helpers parse
that, clamp it to the player's actual resources and the card's tag rules (steel only for
building tags, titanium only for space tags), and validate it covers the cost.
"""
from __future__ import annotations
import logging
import re

from .knowledge import CARD_DB

logger = logging.getLogger(__name__)

# Models often wrap the mandatory response labels in markdown emphasis or headings
# (e.g. "**CHOICE:** 1", "**PAYMENT**: MC=10", "### TACTICAL:", "`CHOICE:`"). _EMPH
# matches such markers so label regexes tolerate them; label_prefix(label) matches the
# label plus its colon with emphasis allowed around the label and after the colon.
# Callers append their own value capture group. Kept here (the lowest-level module) so
# both prompts.py and payment.py can share one definition without an import cycle.
_EMPH = r"[*_`~#]*"


def label_prefix(label: str) -> str:
    return rf"{label}{_EMPH}\s*:{_EMPH}[ \t]*"


# MC value of one unit of each payable resource toward a card's cost.
PAYMENT_VALUES = {
    "steel": 2, "titanium": 3, "heat": 1, "plants": 3,
    "microbes": 2, "floaters": 3, "seeds": 5, "graphene": 4,
    "lunaArchivesScience": 1, "kuiperAsteroids": 1, "auroraiData": 3, "spireScience": 2,
}

# PAYMENT line key → payment dict field.
PAYMENT_KEYS = {
    "MC": "megacredits", "MEGACREDITS": "megacredits",
    "STEEL": "steel", "TITANIUM": "titanium", "HEAT": "heat", "PLANTS": "plants",
    "MICROBES": "microbes", "FLOATERS": "floaters", "SEEDS": "seeds",
    "GRAPHENE": "graphene", "LUNA": "lunaArchivesScience",
    "LUNAARCHIVESSCIENCE": "lunaArchivesScience", "KUIPER": "kuiperAsteroids",
    "KUIPERASTEROIDS": "kuiperAsteroids", "AURORA": "auroraiData",
    "AURORAIDATA": "auroraiData", "SPIRE": "spireScience", "SPIRESCIENCE": "spireScience",
}


def mc_payment(amount: int) -> dict:
    """A megacredits-only payment dict covering ``amount``."""
    return {
        "megacredits": max(0, amount),
        "steel": 0, "titanium": 0, "heat": 0, "plants": 0,
        "microbes": 0, "floaters": 0, "lunaArchivesScience": 0,
        "seeds": 0, "graphene": 0, "kuiperAsteroids": 0,
        "auroraiData": 0, "spireScience": 0,
    }


def card_resource_values(card_name: str) -> tuple[int, int]:
    """Return (steel_value, titanium_value) for a card based on its tags.

    Steel is worth 2 MC only for building-tagged cards; titanium 3 MC only for
    space-tagged. Unknown cards fail closed because the server model, not a local
    guess, is authoritative for payment eligibility.
    """
    if not card_name:
        return 0, 0
    info = CARD_DB.get(card_name, {})
    if not info:
        return 0, 0
    tags = info.get("tags", [])
    return (2 if "building" in tags else 0), (3 if "space" in tags else 0)


def empty_payment() -> dict:
    return {
        "megacredits": 0, "steel": 0, "titanium": 0, "heat": 0, "plants": 0,
        "microbes": 0, "floaters": 0, "lunaArchivesScience": 0, "spireScience": 0,
        "seeds": 0, "auroraiData": 0, "graphene": 0, "kuiperAsteroids": 0,
    }


def parse_payment_line(text: str) -> dict | None:
    m = re.search(label_prefix("PAYMENT") + r"(.+?)(?:\n|$)", text, re.IGNORECASE)
    if not m:
        return None
    parts = re.findall(r"([A-Z_]+)\s*=\s*(\d+)", m.group(1), re.IGNORECASE)
    if not parts:
        return None
    payment = empty_payment()
    for key, val in parts:
        field = PAYMENT_KEYS.get(key.upper())
        if field and field in payment:
            payment[field] = int(val)
    return payment


def _project_card_info(waiting_for: dict, card_name: str) -> dict:
    cards = waiting_for.get("cards", []) if isinstance(waiting_for, dict) else []
    card = next((c for c in cards if c.get("name") == card_name), None)
    if card is not None:
        return card
    direct = waiting_for.get("card", {}) if isinstance(waiting_for, dict) else {}
    return direct if direct.get("name") == card_name else {}


def _payment_context(waiting_for: dict, player: dict, card_name: str) -> tuple[dict, dict, dict]:
    """Return (allowed, available, values) for a server-shaped project-card input."""
    card = _project_card_info(waiting_for, card_name)
    payment_options = waiting_for.get("paymentOptions", {}) or {}
    standard_rules = card.get("standardProjectCanPayWith")
    entry = CARD_DB.get(card_name, {})
    tags = set(entry.get("tags", []) if entry else [])

    allowed = {field: False for field in empty_payment()}
    allowed["megacredits"] = True

    if isinstance(standard_rules, dict):
        allowed["steel"] = standard_rules.get("steel") is True
        allowed["titanium"] = (
            standard_rules.get("titanium") is True
            or payment_options.get("lunaTradeFederationTitanium") is True
        )
        allowed["heat"] = payment_options.get("heat") is True
        allowed["seeds"] = standard_rules.get("seeds") is True
        allowed["kuiperAsteroids"] = standard_rules.get("kuiperAsteroids") is True
        allowed["auroraiData"] = True
        allowed["spireScience"] = True
    else:
        allowed["steel"] = "building" in tags
        allowed["titanium"] = (
            "space" in tags
            or payment_options.get("lunaTradeFederationTitanium") is True
        )
        allowed["heat"] = payment_options.get("heat") is True
        allowed["plants"] = (
            "building" in tags and payment_options.get("plants") is True
        )
        allowed["microbes"] = "plant" in tags
        allowed["floaters"] = "venus" in tags
        allowed["lunaArchivesScience"] = "moon" in tags
        allowed["seeds"] = "plant" in tags
        allowed["graphene"] = bool(tags.intersection({"space", "city"}))
        allowed["kuiperAsteroids"] = "space" in tags

    available = {
        "megacredits": max(0, int(player.get("megacredits", 0) or 0)),
        "steel": max(0, int(player.get("steel", 0) or 0)),
        "titanium": max(0, int(player.get("titanium", 0) or 0)),
        "heat": max(0, int(player.get("heat", 0) or 0)),
        "plants": max(0, int(player.get("plants", 0) or 0)),
    }
    for field in empty_payment():
        if field not in available:
            available[field] = max(0, int(waiting_for.get(field, 0) or 0))

    values = dict(PAYMENT_VALUES)
    values["megacredits"] = 1
    values["steel"] = max(0, int(player.get("steelValue", 2) or 2))
    values["titanium"] = max(0, int(player.get("titaniumValue", 3) or 3))
    return allowed, available, values


def _project_card_cost(waiting_for: dict, card_name: str) -> int:
    card = _project_card_info(waiting_for, card_name)
    if card:
        return max(0, int(card.get("calculatedCost", 0) or 0))
    return max(0, int(waiting_for.get("amount", 0) or 0))


def _legal_payment_total(payment: dict, waiting_for: dict, player: dict, card_name: str) -> int:
    allowed, _available, values = _payment_context(waiting_for, player, card_name)
    return sum(
        max(0, int(payment.get(field, 0) or 0)) * values.get(field, 0)
        for field, is_allowed in allowed.items() if is_allowed
    )


def correct_payment(payment: dict, waiting_for: dict, player: dict, card_name: str = "") -> dict:
    """Clamp payment fields to available resources and ensure the total covers the cost."""
    if waiting_for.get("type", "") != "projectCard":
        result = dict(payment)
        result["steel"] = 0
        result["titanium"] = 0
        for field in ("heat", "plants"):
            result[field] = min(
                max(0, int(result.get(field, 0) or 0)),
                max(0, int(player.get(field, 0) or 0)),
            )
        mc_avail = max(0, int(player.get("megacredits", 0) or 0))
        result["megacredits"] = min(
            max(0, int(result.get("megacredits", 0) or 0)), mc_avail,
        )
        covered = sum(
            max(0, int(result.get(field, 0) or 0)) * value
            for field, value in PAYMENT_VALUES.items()
            if field not in ("steel", "titanium")
        )
        needed_mc = max(0, int(waiting_for.get("amount", 0) or 0) - covered)
        if result["megacredits"] < needed_mc:
            result["megacredits"] = min(needed_mc, mc_avail)
        return result

    allowed, available, values = _payment_context(waiting_for, player, card_name)
    result = empty_payment()
    for field, is_allowed in allowed.items():
        if is_allowed:
            result[field] = min(
                max(0, int(payment.get(field, 0) or 0)),
                available.get(field, 0),
            )

    cost = _project_card_cost(waiting_for, card_name)
    covered = sum(
        result[field] * values.get(field, 0)
        for field in result if field != "megacredits"
    )
    needed_mc = max(0, cost - covered)
    result["megacredits"] = min(needed_mc, available["megacredits"])
    if available["megacredits"] < needed_mc:
        logger.warning(
            "Payment underfunded: cost=%d covered_by_resources=%d need_mc=%d have_mc=%d",
            cost, covered, needed_mc, available["megacredits"],
        )
    return result


def check_payment_valid(response: dict, options: list[dict], waiting_for: dict, player: dict) -> str | None:
    """Return a human-readable error string if the response payment is underfunded, else None."""
    mc = player.get("megacredits", 0)
    st = player.get("steel", 0)
    ti = player.get("titanium", 0)

    def _msg(card_name: str, payment: dict, cost: int, project_node: dict) -> str | None:
        if cost <= 0:
            return None
        st_val, ti_val = card_resource_values(card_name)
        total = _legal_payment_total(payment, project_node, player, card_name)
        if total >= cost:
            return None
        st_note = f"Steel={st} (×{st_val}={st*st_val} MC, building-tag only)" if st_val else f"Steel={st} (not applicable — no building tag)"
        ti_note = f"Titanium={ti} (×{ti_val}={ti*ti_val} MC, space-tag only)" if ti_val else f"Titanium={ti} (not applicable — no space tag)"
        return (
            f"Your payment for {card_name!r} is insufficient: "
            f"total {total} MC but card costs {cost} MC. "
            f"Available resources: MC={mc}, {st_note}, {ti_note}. "
            f"Reply with PAYMENT totalling ≥{cost} MC, or choose a different option."
        )

    wf_type = waiting_for.get("type", "")

    if wf_type == "projectCard":
        cname = response.get("card", "card")
        return _msg(cname, response.get("payment", {}), _project_card_cost(waiting_for, cname), waiting_for)

    if wf_type == "payment":
        cost = waiting_for.get("amount", 0)
        payment = response.get("payment", {})
        total = max(0, int(payment.get("megacredits", 0) or 0)) + sum(
            max(0, int(payment.get(field, 0) or 0)) * value
            for field, value in PAYMENT_VALUES.items()
            if field not in ("steel", "titanium")
        )
        return None if total >= cost else (
            f"Your payment for 'standard project' is insufficient: total {total} MC "
            f"but card costs {cost} MC. Available resources: MC={mc}."
        )

    if response.get("type") == "or":
        inner = response.get("response", {})
        if inner.get("type") == "projectCard":
            card_name = inner.get("card", "")
            payment = inner.get("payment", {})
            chosen_idx = max(0, int(response.get("index", 0) or 0))
            parent_options = waiting_for.get("options", [])
            sub_node = parent_options[chosen_idx] if chosen_idx < len(parent_options) else {}
            cost = _project_card_cost(sub_node, card_name)
            return _msg(card_name, payment, cost, sub_node)

    return None


def auto_payment_for_card(card_name: str, sub_node: dict, player: dict) -> dict:
    """Generate a legal payment using the same server-owned policy as validation."""
    cost = _project_card_cost(sub_node, card_name)
    allowed, available, values = _payment_context(sub_node, player, card_name)
    payment = empty_payment()
    remaining = cost
    resource_fields = [
        field for field, is_allowed in allowed.items()
        if is_allowed and field != "megacredits" and values.get(field, 0) > 0
    ]
    resource_fields.sort(key=lambda field: values[field], reverse=True)
    for field in resource_fields:
        value = values[field]
        used = min(available[field], (remaining + value - 1) // value)
        payment[field] = used
        remaining = max(0, remaining - used * value)
    payment["megacredits"] = min(remaining, available["megacredits"])
    return payment
