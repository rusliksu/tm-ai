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


def card_resource_values(card_name: str) -> tuple[int, int]:
    """Return (steel_value, titanium_value) for a card based on its tags.

    Steel is worth 2 MC only for building-tagged cards; titanium 3 MC only for
    space-tagged. Unknown cards are treated permissively (both enabled) to avoid
    false rejections.
    """
    if not card_name:
        return 2, 3
    info = CARD_DB.get(card_name, {})
    if not info:
        return 2, 3
    tags = info.get("tags", [])
    return (2 if "building" in tags else 0), (3 if "space" in tags else 0)


def empty_payment() -> dict:
    return {
        "megacredits": 0, "steel": 0, "titanium": 0, "heat": 0, "plants": 0,
        "microbes": 0, "floaters": 0, "lunaArchivesScience": 0, "spireScience": 0,
        "seeds": 0, "auroraiData": 0, "graphene": 0, "kuiperAsteroids": 0,
    }


def parse_payment_line(text: str) -> dict | None:
    m = re.search(r"PAYMENT:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
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


def correct_payment(payment: dict, waiting_for: dict, player: dict, card_name: str = "") -> dict:
    """Clamp payment fields to available resources and ensure the total covers the cost."""
    wf_type = waiting_for.get("type", "")
    payment = dict(payment)

    mc_avail = player.get("megacredits", 0)
    st_avail = player.get("steel", 0)
    ti_avail = player.get("titanium", 0)
    ht_avail = player.get("heat", 0)
    pl_avail = player.get("plants", 0)

    is_project = wf_type == "projectCard"
    if not is_project:
        payment["steel"] = 0
        payment["titanium"] = 0
    else:
        st_val, ti_val = card_resource_values(card_name)
        if st_val == 0:
            payment["steel"] = 0
        if ti_val == 0:
            payment["titanium"] = 0

    for field, available in [
        ("steel", st_avail), ("titanium", ti_avail),
        ("heat", ht_avail), ("plants", pl_avail),
    ]:
        if payment.get(field, 0) > available:
            payment[field] = available

    if payment.get("megacredits", 0) > mc_avail:
        payment["megacredits"] = mc_avail

    if is_project:
        card_node = waiting_for.get("card", {})
        cost = card_node.get("calculatedCost", 0) if card_node else waiting_for.get("amount", 0)
    else:
        cost = waiting_for.get("amount", 0)

    st_val, ti_val = card_resource_values(card_name) if is_project else (0, 0)
    covered = (
        payment.get("steel", 0) * st_val +
        payment.get("titanium", 0) * ti_val +
        sum(payment.get(f, 0) * PAYMENT_VALUES.get(f, 0)
            for f in PAYMENT_VALUES if f not in ("steel", "titanium"))
    )
    needed_mc = max(0, cost - covered)
    if payment.get("megacredits", 0) < needed_mc:
        payment["megacredits"] = min(needed_mc, mc_avail)
        if mc_avail < needed_mc:
            logger.warning(
                "Payment underfunded: cost=%d covered_by_resources=%d need_mc=%d have_mc=%d",
                cost, covered, needed_mc, mc_avail,
            )

    return payment


def check_payment_valid(response: dict, options: list[dict], waiting_for: dict, player: dict) -> str | None:
    """Return a human-readable error string if the response payment is underfunded, else None."""
    mc = player.get("megacredits", 0)
    st = player.get("steel", 0)
    ti = player.get("titanium", 0)

    def _total(payment: dict, card_name: str = "") -> int:
        st_val, ti_val = card_resource_values(card_name)
        return (
            payment.get("megacredits", 0) +
            payment.get("steel", 0) * st_val +
            payment.get("titanium", 0) * ti_val +
            sum(payment.get(f, 0) * PAYMENT_VALUES[f]
                for f in PAYMENT_VALUES if f not in ("steel", "titanium") and f in payment)
        )

    def _msg(card_name: str, payment: dict, cost: int) -> str | None:
        if cost <= 0:
            return None
        st_val, ti_val = card_resource_values(card_name)
        total = _total(payment, card_name)
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
        card_node = waiting_for.get("card", {})
        cost = card_node.get("calculatedCost", 0) if card_node else waiting_for.get("amount", 0)
        cname = response.get("card", "card")
        return _msg(cname, response.get("payment", {}), cost)

    if wf_type == "payment":
        cost = waiting_for.get("amount", 0)
        return _msg("standard project", response.get("payment", {}), cost)

    if response.get("type") == "or":
        inner = response.get("response", {})
        if inner.get("type") == "projectCard":
            card_name = inner.get("card", "")
            payment = inner.get("payment", {})
            chosen_idx = response.get("index", 0)
            sub_option = next((o for o in options if o.get("index") == chosen_idx), {})
            sub_node = sub_option.get("node", {})
            cards = sub_node.get("cards", []) if isinstance(sub_node, dict) else []
            card_info = next((c for c in cards if c.get("name") == card_name), {})
            cost = card_info.get("calculatedCost", 0)
            return _msg(card_name, payment, cost)

    return None


def auto_payment_for_card(card_name: str, sub_node: dict, player: dict) -> dict:
    """Generate an optimal steel/titanium payment for a project card using CARD_DB tags."""
    cards = sub_node.get("cards", []) if isinstance(sub_node, dict) else []
    card_info = next((c for c in cards if c.get("name") == card_name), {})
    cost = card_info.get("calculatedCost", 0)
    entry = CARD_DB.get(card_name, {})
    tags = entry.get("tags", []) if entry else []

    ti_avail = player.get("titanium", 0) if "space" in tags else 0
    st_avail = player.get("steel", 0) if "building" in tags else 0

    ti_used = min(ti_avail, (cost + 2) // 3)
    remaining = max(0, cost - ti_used * 3)
    st_used = min(st_avail, (remaining + 1) // 2)
    remaining = max(0, remaining - st_used * 2)
    mc_used = min(remaining, player.get("megacredits", 0))

    payment = empty_payment()
    payment["megacredits"] = mc_used
    payment["steel"] = st_used
    payment["titanium"] = ti_used
    return payment
