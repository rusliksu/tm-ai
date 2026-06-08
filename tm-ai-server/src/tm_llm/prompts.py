"""Prompt construction and response parsing for the LLM player.

Layout:
  • RULES (cached, byte-identical per game): TM_RULES + STRATEGY_PRIMER + config + board.
  • Per-turn user prompt (volatile): state header, conditional one-line advisories, live
    board / candidate-space adjacency, tableau, compact hand, opponents, milestone & award
    status, recent log, the player's own memory (strategy + tactical), numbered options.

Single source of truth for standard-project costs and milestone thresholds lives in this
module (STANDARD_PROJECT_COSTS, MILESTONE_THRESHOLDS) and is reused by both the prose and
the runtime annotations, so they cannot drift.
"""
from __future__ import annotations
import logging
import re

from .knowledge import CARD_DB, card_brief, format_card_context, format_config_context, format_board_layout
from .options import _default_response, index_to_response
from .payment import (
    parse_payment_line, correct_payment, auto_payment_for_card, label_prefix, _EMPH,
)
from . import board as board_mod

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Single-sourced game data
# ---------------------------------------------------------------------------

# Standard-project MC costs (title slug emitted by TM → cost). Sell Patents is free.
STANDARD_PROJECT_COSTS: dict[str, int] = {
    "power plant:sp": 11,
    "asteroid:sp": 14,
    "aquifer:sp": 18,
    "greenery:sp": 23,
    "city:sp": 25,
    "buffer gas:sp": 16,
    "air scrapping:sp": 15,
    "lava flows:sp": 36,
    "asteroid mining:sp": 35,
}

# Base-game milestone numeric thresholds (substring-matched against milestone name) used to
# compute the player's/opponents' progress. Variants fall through with no progress line.
MILESTONE_THRESHOLDS: dict[str, int] = {
    "terraformer": 35, "mayor": 3, "gardener": 3, "builder": 8, "planner": 16,
}
_MILESTONE_RACE_GAP = 3


def standard_project_cost(title: str) -> int | None:
    if not title:
        return None
    t = re.sub(r"\s*\(\d+ ?m€\)\s*$", "", title.lower().strip())
    # Keys are all suffixed ':sp', but TM only adds that suffix to standard projects whose name
    # collides with a real card (Power Plant:SP, Asteroid:SP); others arrive bare (Aquifer,
    # Greenery, City...). Match either form so every standard project gets a cost tag.
    for candidate in (t, f"{t}:sp", t[:-3] if t.endswith(":sp") else t):
        if candidate in STANDARD_PROJECT_COSTS:
            return STANDARD_PROJECT_COSTS[candidate]
    return None


# ---------------------------------------------------------------------------
# Rules reference (cached system prefix) — RULES only, strategy split out below
# ---------------------------------------------------------------------------

TM_RULES = f"""
=== TERRAFORMING MARS — RULES ===
GOAL: most VP at game end. VP = 1/TR + 1/greenery + 1 per greenery adjacent to a city (any
owner) + milestones (5) + awards (5 for 1st, 2 for 2nd) + card VP.

GLOBAL PARAMETERS — game ENDS when temperature, oxygen AND oceans are all maxed; each step
raised = +1 TR (= +1 income AND +1 VP):
  • Temperature −30→+8°C (2°/step). Raise: 8 heat, Asteroid SP, cards.
  • Oxygen 0→14%. Raise: place greenery, cards.
  • Oceans 0→9 tiles. Place: Aquifer SP, cards.
  • Venus (Venus Next only) 0→30% (2%/step). Raise: Venus cards/SPs. Does NOT end the game.
THRESHOLDS (one-time, to the trigger): temp −24/−20°C → +1 heat prod; temp 0°C → 1 ocean;
O₂ 8% → temp +1 step; Venus 8% → draw card, 16% → +1 TR. Hitting a threshold yourself ≈ 10 MC.

RESOURCES (gained each production phase): MC — income = TR + MC-prod (MC-prod may go to −5,
other prods ≥0); Steel pays BUILDING-tag cards @2 MC/cube; Titanium pays SPACE-tag cards
@3 MC/cube; 8 Plants → greenery (+1 O₂ +1 TR); 8 Heat → +1 temp (+1 TR); spare Energy → Heat.

CARDS: GREEN (one-time effect, tag stays), BLUE (ongoing effect or 1×/gen action), RED event
(one-time, tag counts only when played).

EACH GENERATION: research (draw 4, buy any @3 MC) → actions (1–2 per turn until all pass) →
production.

ACTIONS: play a card; standard project; claim a milestone (8 MC, max 3/game — claim the
instant you qualify, opponents race you); fund an award (8/14/20 MC); blue-card action
(1×/gen); 8 plants→greenery; 8 heat→temp. AWARDS score ONLY at game END (1st=5, 2nd=2) — fund
only one you will still lead at the end; funding a lead you won't hold = 0 VP.

STANDARD PROJECTS: Sell patents (free, discard N cards→N MC — almost always bad); Power Plant
({STANDARD_PROJECT_COSTS['power plant:sp']} MC, +1 energy prod); Asteroid ({STANDARD_PROJECT_COSTS['asteroid:sp']} MC, +1 temp); Aquifer ({STANDARD_PROJECT_COSTS['aquifer:sp']} MC, ocean);
Greenery ({STANDARD_PROJECT_COSTS['greenery:sp']} MC, greenery); City ({STANDARD_PROJECT_COSTS['city:sp']} MC, city +1 MC prod).

TILE PLACEMENT: ocean only on reserved blue spaces (adjacent owners get +2 MC); greenery must
go next to your own tile if possible; city can't touch another city.

PLAYABILITY: the options below are already filtered to legal AND affordable moves — never
re-check requirements/affordability; pick the best and pay.

PAYMENT: cover the cost in MC, optionally substituting steel/titanium per the tags above; no
MC overpay, surplus cubes are lost.

RESPONSE FORMAT (every turn — terse and dense, no markdown headings, do not restate the state):
  <1–2 sentence reason tied to your strategy>
  TACTICAL: <ordered next steps for the rest of this gen — your ONLY memory to your next move; rewrite each turn>
  CHOICE: N   (option number, own line)
  PAYMENT: MC=<n>[, STEEL=<n>][, TITANIUM=<n>][, HEAT=<n>]   (only when playing a card; total ≥ shown cost)

SERVER AUTHORITY: the server is always right; on rejection, read the error and pick a
DIFFERENT valid option.
=== END RULES ===
"""

# Strategy guidance — condensed from the original essays. Cached alongside the rules.
STRATEGY_PRIMER = """
=== STRATEGY PRIMER ===
• TR is income AND VP — terraform actively; never pass with ≥8 spare heat/plants unconverted.
• Build MC production early; it compounds. City SP and production cards beat one-off effects.
• Throughput wins: 2–4 cards/gen. Fat hand + low MC = stalled engine; use steel/titanium
  discounts and cheap synergy cards. Buy drafts to match resources (titanium→space, steel→building);
  stockpiles with no matching cards are wasted.
• Greenery geometry: place greeneries next to YOUR cities (each adjacency = +1 VP); two cities
  two hexes apart share a 3-VP hex; never feed an opponent's city. First city by ~gen 3–4.
• Pace: ahead per gen → slow terraforming; ahead on TR but behind on VP → rush the game end
  before their engines mature. Read opponents (tableau/production/awards/milestones) — block and race.
• One-time setup perks (starting free city/tile, starting resources, preludes) already happened
  at game start — NOT recurring. Never plan a "free city" mid-game; only pursue moves listed in
  your options this turn, and drop any planned move that isn't listed.
=== END STRATEGY PRIMER ===
"""

_ACTION_SYSTEM_SUFFIX = (
    "\n\nYou are an expert Terraforming Mars player. Each turn you get the COMPLETE state plus "
    "YOUR MEMORY (STRATEGY + TACTICAL) — no chat history, so those notes are all you remember; "
    "keep them accurate. Answer in the RESPONSE FORMAT, terse and dense: brief reasoning, no "
    "markdown headers, no restating the state back."
)


def build_action_system(state: dict) -> str:
    """Stable per-game system prompt (rules + primer + config + static board layout)."""
    g = state.get("game", {})
    game_ctx = format_config_context(g)
    layout = format_board_layout(state.get("boardSpaces") or [])
    return (
        TM_RULES + "\n" + STRATEGY_PRIMER + "\n\n" + game_ctx + "\n\n"
        + (layout + "\n\n" if layout else "")
    ).rstrip() + _ACTION_SYSTEM_SUFFIX


# ---------------------------------------------------------------------------
# Title / message rendering
# ---------------------------------------------------------------------------

def _format_message(title: object) -> str:
    if isinstance(title, str):
        return title
    if isinstance(title, dict):
        msg = str(title.get("message", ""))
        for i, item in enumerate(title.get("data") or []):
            val = item.get("value") if isinstance(item, dict) else item
            msg = msg.replace(f"${{{i}}}", str(val))
        return msg
    return ""


def _node_title(node: dict, fallback: int) -> str:
    title = node.get("title", "")
    if isinstance(title, str) and title:
        return title
    if isinstance(title, dict):
        return _format_message(title) or f"Option {fallback}"
    return f"Option {fallback}"


def _color_to_name(state: dict) -> dict[str, str]:
    """Map each player's color → display name, so option/decision titles that identify a player
    only by colour (e.g. 'Remove 1 plants from orange') can name them."""
    out: dict[str, str] = {}
    p = state.get("player", {})
    if p.get("color"):
        out[str(p["color"]).lower()] = p.get("name", "you")
    for o in state.get("opponents") or []:
        if o.get("color"):
            out[str(o["color"]).lower()] = o.get("name", "opp")
    return out


def _annotate_colors(text: str, color_to_name: dict[str, str]) -> str:
    """Append the player name after a bare colour in a title — 'from orange' → 'from orange
    (Sandra)', or a bare 'orange' player-select option → 'orange (Sandra)'. Only known player
    colours are touched, and (outside the bare case) only after a preposition, so card names
    like 'Black Polar Dust' are left alone."""
    if not text or not color_to_name:
        return text
    if text.strip().lower() in color_to_name:
        return f"{text} ({color_to_name[text.strip().lower()]})"
    colors = "|".join(re.escape(c) for c in sorted(color_to_name, key=len, reverse=True))
    pat = re.compile(rf"\b(from|to|against|for|by)\s+({colors})\b", re.IGNORECASE)
    return pat.sub(lambda m: f"{m.group(1)} {m.group(2)} ({color_to_name[m.group(2).lower()]})", text)


# ---------------------------------------------------------------------------
# Milestone / award helpers
# ---------------------------------------------------------------------------

def _milestone_value(entity: dict, ms_lower: str) -> int | None:
    if "terraformer" in ms_lower:
        return entity.get("terraformRating", 20)
    if "mayor" in ms_lower:
        return (entity.get("boardTiles") or {}).get("city", 0)
    if "gardener" in ms_lower:
        return (entity.get("boardTiles") or {}).get("greenery", 0)
    if "builder" in ms_lower:
        return (entity.get("tags") or {}).get("building", 0)
    if "planner" in ms_lower:
        return entity.get("handSize", 0)
    return None


def _milestone_threshold(ms_lower: str) -> int | None:
    for key, val in MILESTONE_THRESHOLDS.items():
        if key in ms_lower:
            return val
    return None


def _milestone_claim_options(options: list[dict]) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for opt in options:
        m = re.match(r"^claim\s+milestone\s+(.+?)\s*$", str(opt.get("title", "")).strip(), re.IGNORECASE)
        if m:
            name = m.group(1).strip().rstrip(".")
            if name and name.lower() not in ("a milestone", "milestone"):
                out.append((opt["index"], name))
    return out


def _milestone_advisory(state: dict, options: list[dict]) -> list[str]:
    ms_opts = _milestone_claim_options(options)
    if not ms_opts:
        return []
    opponents = state.get("opponents") or []
    lines = [
        "",
        "⚠ MILESTONE CLAIMABLE (5 VP for 8 MC, max 3/game): default CLAIM IT NOW. Postpone only if "
        "no opponent is near any unclaimed milestone AND you have a >5-VP play this turn.",
    ]
    for idx, ms_name in ms_opts:
        ms_lower = ms_name.lower()
        threshold = _milestone_threshold(ms_lower)
        label = f"Option {idx + 1}: Claim '{ms_name}'"
        if threshold is None:
            lines.append(f"  {label} (5 VP for 8 MC). Default: CLAIM.")
            continue
        progress, any_close = [], False
        for opp in opponents:
            ov = _milestone_value(opp, ms_lower)
            if ov is None:
                continue
            gap = max(0, threshold - ov)
            progress.append(f"{opp.get('name', 'Opp')}={ov}/{threshold}(gap {gap})")
            any_close = any_close or gap <= _MILESTONE_RACE_GAP
        progress_str = "; ".join(progress) if progress else "no opponents tracked"
        if any_close:
            lines.append(f"  {label}: ⚠ an opponent is within {_MILESTONE_RACE_GAP} — CLAIM NOW. {progress_str}.")
        else:
            lines.append(f"  {label}: no opponent close ({progress_str}); may postpone for a higher-VP play.")
    return lines


def compute_milestone_status(state: dict) -> list[str]:
    """List every milestone with claim state + your progress, so the model never plans to
    claim one already taken or after the 3-claim cap."""
    g, p = state.get("game", {}), state.get("player", {})
    opps = state.get("opponents") or []
    claimed = state.get("milestones") or []
    available = g.get("availableMilestones") or []
    if not available and not claimed:
        return []

    my_id, my_name = p.get("id"), p.get("name", "You")
    id_to_name = {my_id: my_name} if my_id else {}
    for opp in opps:
        if opp.get("id"):
            id_to_name[opp["id"]] = opp.get("name", "Opp")

    claimed_by: dict[str, str] = {}
    for entry in claimed:
        ms_name = entry.get("name") if isinstance(entry, dict) else None
        pid = entry.get("playerId") if isinstance(entry, dict) else None
        if ms_name:
            claimed_by[ms_name] = f"you ({my_name})" if pid == my_id else id_to_name.get(pid, "?")

    n_claimed = len(claimed)
    phase_over = n_claimed >= 3
    if phase_over:
        lines = [f"Milestones ({n_claimed}/3 claimed — PHASE OVER, no further claims; drop any 'claim' plan):"]
    else:
        lines = [f"Milestones ({n_claimed}/3 claimed; {3 - n_claimed} more can be claimed):"]

    names: list[str] = []
    for ms in available:
        nm = ms.get("name") if isinstance(ms, dict) else None
        if nm and nm not in names:
            names.append(nm)
    for nm in claimed_by:
        if nm not in names:
            names.append(nm)

    for ms_name in names:
        if ms_name in claimed_by:
            lines.append(f"  ✗ {ms_name} — claimed by {claimed_by[ms_name]}")
            continue
        if phase_over:
            lines.append(f"  ✗ {ms_name} — unclaimed but CANNOT be claimed (3-claim cap reached)")
            continue
        ms_lower = ms_name.lower()
        threshold = _milestone_threshold(ms_lower)
        my_v = _milestone_value(p, ms_lower)
        if threshold is None or my_v is None:
            lines.append(f"  • {ms_name} — claimable; requirement varies by variant")
            continue
        opp_progress = [f"{o.get('name', 'Opp')}={_milestone_value(o, ms_lower)}"
                        for o in opps if _milestone_value(o, ms_lower) is not None]
        progress_str = ", ".join([f"you={my_v}"] + opp_progress)
        if my_v >= threshold:
            lines.append(f"  ✓ {ms_name} — requires ≥{threshold} (you ALREADY meet it; {progress_str}) — CLAIMABLE NOW")
        else:
            lines.append(f"  • {ms_name} — requires ≥{threshold} ({progress_str}; you need {threshold - my_v} more)")
    return lines


def compute_award_standings(state: dict) -> list[str]:
    """Every available award with its funded status (and funder) plus the current standings, so
    the model can both decide what to fund AND see where it stands on already-funded awards (it
    can still score 1st/2nd at game end)."""
    g, p = state.get("game", {}), state.get("player", {})
    opps = state.get("opponents") or []
    available = g.get("availableAwards") or []
    if not available:
        return []

    my_id, my_name = p.get("id"), p.get("name", "You")
    id_to_name = {my_id: f"you ({my_name})"} if my_id else {}
    for opp in opps:
        if opp.get("id"):
            id_to_name[opp["id"]] = opp.get("name", "Opp")
    funded_by: dict[str, str] = {}
    for a in state.get("awards", []):
        nm = a.get("name") if isinstance(a, dict) else None
        if nm:
            funded_by[nm] = id_to_name.get(a.get("playerId"), "?")

    def _val(entity: dict, award_name: str) -> int:
        nl = award_name.lower()
        prod, tags, bt = entity.get("production", {}), entity.get("tags", {}), entity.get("boardTiles", {})
        if "banker" in nl:
            return prod.get("megacredits", 0)
        if "scientist" in nl:
            return tags.get("science", 0)
        if "thermalist" in nl:
            return entity.get("heat", 0)
        if "miner" in nl:
            return entity.get("steel", 0) + entity.get("titanium", 0)
        if "industrialist" in nl:
            return entity.get("steel", 0) + entity.get("energy", 0)
        if "landlord" in nl or "settler" in nl or "estate" in nl:
            return sum(bt.values()) if isinstance(bt, dict) else 0
        return -1

    lines: list[str] = []
    for award_info in available:
        aname = award_info.get("name", "?")
        status = f"FUNDED by {funded_by[aname]}" if aname in funded_by else "unfunded"
        my_v = _val(p, aname)
        if my_v < 0:
            # Metric not modelled (e.g. a fan/randomised award) — still surface funded status.
            lines.append(f"  {aname} [{status}]")
            continue
        entries = [(my_name, my_v)] + [(o.get("name", "Opp"), _val(o, aname)) for o in opps]
        entries.sort(key=lambda x: x[1], reverse=True)
        rank = next((i + 1 for i, (n, _) in enumerate(entries) if n == my_name), len(entries))
        rank_str = {1: "1st", 2: "2nd", 3: "3rd"}.get(rank, f"{rank}th")
        standings = ", ".join(f"{n}={v}" for n, v in entries)
        lines.append(f"  {aname} [{status}]: you are {rank_str} ({standings})")
    return lines


def game_end_proximity(game: dict) -> str | None:
    """A banner when ≥2 of the 3 game-ending parameters are maxed — the game may end imminently
    (possibly on an opponent's turn), so unspent resources are wasted. Venus does NOT gate game
    end and is excluded."""
    temp = game.get("temperature", -30)
    oxygen = game.get("oxygen", 0)
    oceans = game.get("oceanCount", 0)
    done = [
        ("temperature", temp >= 8, f"{temp}/8°C"),
        ("oxygen", oxygen >= 14, f"{oxygen}/14%"),
        ("oceans", oceans >= 9, f"{oceans}/9"),
    ]
    n_done = sum(1 for _, ok, _ in done if ok)
    if n_done < 2:
        return None
    remaining = ", ".join(f"{name} {cur}" for name, ok, cur in done if not ok) or "none — ends now"
    return (
        f"⚠ GAME-END IMMINENT: {n_done}/3 ending parameters maxed (rising: {remaining}). The game may "
        "end THIS gen — even on an opponent's turn. Spend ALL MC/plants/heat on VP now (greeneries, "
        "VP cards, any TR step); bank NOTHING for next gen."
    )


# ---------------------------------------------------------------------------
# Option/card helpers
# ---------------------------------------------------------------------------

def _get_card_desc_for_option(opt: dict) -> str:
    name = (opt.get("card") or {}).get("name", "")
    if not name:
        return ""
    desc = CARD_DB.get(name, {}).get("description", "")
    return f"Use {name} action — {desc}" if desc else f"Use {name} action"


def _option_card_name(opt: dict) -> str:
    """The card name an option resolves to, if it maps to a single known card.

    Handles both a card-type option (node is the card dict) and an expanded projectCard menu
    option (node is the parent projectCard node; the card sits at the last path index)."""
    node = opt.get("node")
    if not isinstance(node, dict):
        return ""
    if node.get("name"):
        return node["name"]
    cards = node.get("cards")
    path = opt.get("path") or []
    if cards and path:
        j = path[-1]
        if isinstance(j, int) and 0 <= j < len(cards):
            return cards[j].get("name", "")
    return ""


def _extract_card_names(waiting_for: dict, max_depth: int = 3) -> list[str]:
    if max_depth <= 0:
        return []
    names: list[str] = []
    wf_type = waiting_for.get("type", "")
    if wf_type == "card":
        for c in waiting_for.get("cards", []):
            name = c.get("name", "") if isinstance(c, dict) else str(c)
            if name and name not in names:
                names.append(name)
    elif wf_type in ("or", "and"):
        for opt in waiting_for.get("options", []):
            for n in _extract_card_names(opt, max_depth - 1):
                if n not in names:
                    names.append(n)
    return names


def _is_card_decision_about_hand(card_names: list[str], cards_in_hand: list) -> bool:
    if not card_names or not cards_in_hand:
        return False
    hand_set = {c if isinstance(c, str) else c.get("name", "") for c in cards_in_hand}
    return all(name in hand_set for name in card_names)


# ---------------------------------------------------------------------------
# Action prompt builder
# ---------------------------------------------------------------------------

def build_action_prompt(state: dict, waiting_for: dict, options: list[dict], *,
                        strategy: str = "", tactical: str = "", last_error: str | None = None) -> str:
    g, p = state.get("game", {}), state.get("player", {})
    aw = state.get("awards", [])
    opponents = state.get("opponents") or []

    temp, oxygen, oceans = g.get("temperature", -30), g.get("oxygen", 0), g.get("oceanCount", 0)
    tr, mc = p.get("terraformRating", 20), p.get("megacredits", 0)
    mc_prod = p.get("production", {}).get("megacredits", 0)
    mc_income = tr + mc_prod
    wf_type = waiting_for.get("type", "")
    own_name, own_color = p.get("name", "?"), p.get("color", "?")

    lines: list[str] = []

    if last_error:
        lines += [
            "⚠ THE GAME SERVER REJECTED YOUR PREVIOUS MOVE:",
            f'  Error: "{last_error}"',
            "The server is always correct. Choose a DIFFERENT, valid action — do not repeat it.",
            "",
        ]

    lines += [
        f"Gen {g.get('generation', 1)} | Temp:{temp}°C O₂:{oxygen}% Oceans:{oceans}/9",
        f'You are "{own_name}" ({own_color}); log lines "You ({own_name})" are your own past actions.',
        f"You: TR:{tr} VP:{p.get('victoryPoints', '?')}  MC:{mc}(income:{mc_income})  "
        f"Steel:{p.get('steel', 0)} Ti:{p.get('titanium', 0)}  "
        f"Plants:{p.get('plants', 0)} Energy:{p.get('energy', 0)} Heat:{p.get('heat', 0)}",
    ]

    # Conditional one-line advisories (only when actionable)
    heat_now, plants_now = p.get("heat", 0), p.get("plants", 0)
    if heat_now >= 8 and temp < 8:
        lines.append(f">> {heat_now} heat ≥8 — Convert 8 heat = free +1 TR before passing.")
    if plants_now >= 8 and oxygen < 14:
        lines.append(f">> {plants_now} plants ≥8 — Convert 8 plants = greenery, +1 TR +1 VP.")
    elif plants_now >= 8 and oxygen >= 14:
        lines.append(f">> {plants_now} plants — O₂ maxed, but each greenery still scores +1 VP.")
    if temp >= 8:
        lines.append("⚠ Temp maxed — do NOT Convert Heat.")
    if oceans >= 9:
        lines.append("⚠ All 9 oceans placed.")
    end_banner = game_end_proximity(g)
    if end_banner:
        lines.append(end_banner)

    lines.extend(_milestone_advisory(state, options))

    prod = {k: v for k, v in p.get("production", {}).items() if v}
    tags = {k: v for k, v in p.get("tags", {}).items() if v}
    if prod:
        lines.append(f"Production: {prod}")
    if tags:
        lines.append(f"Tags: {tags}")

    if opponents:
        max_opp_income = max(o.get("terraformRating", 20) + o.get("production", {}).get("megacredits", 0) for o in opponents)
        if max_opp_income - mc_income >= 8:
            lines.append(f"⚠ INCOME GAP: you {mc_income}/gen vs best opponent {max_opp_income}/gen — prioritise MC production & City SP.")

    # Tableau (names + per-card resources; effects are passive and in the hand glossary)
    played = p.get("playedCards") or []
    corps = p.get("corporations") or []
    if played or corps:
        corp_str = f"  [corp: {', '.join(corps)}]" if corps else ""
        lines += ["", f"Your tableau ({len(played)} in play): {', '.join(played[:60]) or '(none)'}{corp_str}"]
        card_res = p.get("cardResources") or {}
        if card_res:
            lines.append("  Card resources: " + ", ".join(f"{n}={v}" for n, v in card_res.items()))

    # Hand — compact (1 line/card via knowledge.format_card_context)
    hand_cards = p.get("cardsInHand") or []
    if hand_cards and wf_type not in ("space", "payment", "amount"):
        lines += ["", format_card_context(hand_cards, header=f"Your hand ({len(hand_cards)} cards):", max_cards=30)]

    # Opponents — include corporation + their played tableau so the model can read each engine
    # (recurring income/VP cards, award threats) instead of guessing from tags alone.
    for i, opp in enumerate(opponents, 1):
        opp_prod = {k: v for k, v in opp.get("production", {}).items() if v}
        opp_tags = {k: v for k, v in opp.get("tags", {}).items() if v}
        nm = opp.get("name", f"Opponent{i}")
        vp = f" VP:{opp['victoryPoints']}" if opp.get("victoryPoints") is not None else ""
        hs = f"  hand:{opp['handSize']}" if opp.get("handSize") is not None else ""
        corp = opp.get("corporations") or []
        corp_str = f"  corp:{','.join(corp)}" if corp else ""
        lines.append(f"{nm}: TR:{opp.get('terraformRating', 20)}{vp}  MC:{opp.get('megacredits', 0)}{hs}{corp_str}  prod:{opp_prod}  tags:{opp_tags}")
        played = opp.get("playedCards") or []
        if played:
            # Describe each played card (effect/tags/VP) plus resources stored on it, so the model
            # can read the opponent's engine and award/milestone threats — a bare name list is
            # unreadable for strategy.
            lines.append(format_card_context(
                played, header=f"  {nm}'s tableau ({len(played)} cards):",
                max_cards=40, resources=opp.get("cardResources")))

    lines.extend(compute_milestone_status(state))

    standings = compute_award_standings(state)
    if standings:
        n_funded = len(aw) if aw else 0
        header = (f"Awards ({n_funded}/3 funded; scored at GAME END — you can still place 1st/2nd "
                  "on an already-funded award, so keep competing).")
        if n_funded >= 3:
            header += " FUNDING PHASE OVER — do not try to fund more."
        else:
            header += (" Leads erode as opponents grow: fund only a 1st-place lead you'll hold to "
                       "the end, rarely before ~gen 8, never a thin one.")
        lines.append(header)
        lines.extend(standings)

    # Live board (compact) on non-placement turns; full candidate adjacency on space turns
    if wf_type == "space":
        choices = board_mod.render_space_choices(state, options)
        if choices:
            lines += ["", choices]
    else:
        live = board_mod.render_live_board(state)
        if live:
            lines += ["", live]

    # Recent log
    recent_log = g.get("recentLog") or []
    if recent_log:
        lines += ["", f"This generation's events ({len(recent_log)}):"]
        for entry in recent_log:
            text = str(entry)
            if own_name and own_name != "?" and text.startswith(own_name + " "):
                text = f"You ({own_name}) " + text[len(own_name) + 1:]
            lines.append(f"  {text}")

    # Memory
    if strategy or tactical:
        lines += ["", "=== YOUR MEMORY (carries between turns) ==="]
        if strategy:
            lines += ["STRATEGY:", strategy]
        if tactical:
            lines += ["TACTICAL PLAN:", tactical]

    # Decision + options
    color_to_name = _color_to_name(state)
    title = _annotate_colors(_format_message(waiting_for.get("title")).strip() or "Select action", color_to_name)
    lines += ["", f"Decision: {title}", "Options:"]

    card_names_in_decision = _extract_card_names(waiting_for)
    is_hand_decision = _is_card_decision_about_hand(card_names_in_decision, hand_cards)
    hand_name_set = {c if isinstance(c, str) else c.get("name", "") for c in hand_cards}
    hand_shown = bool(hand_cards) and wf_type not in ("space", "payment", "amount")
    for opt in options:
        idx = opt["index"] + 1
        title2 = _annotate_colors(str(opt["title"]), color_to_name)
        desc = _get_card_desc_for_option(opt)
        sp_cost = standard_project_cost(
            title2.split(":", 1)[1].strip() if title2.lower().startswith("standard projects:") else title2
        )
        cost_tag = ""
        if sp_cost is not None:
            cost_tag = (f"  [{sp_cost} MC — NOT AFFORDABLE, you have {mc} MC]" if sp_cost > mc
                        else f"  [{sp_cost} MC; you have {mc} MC]")
        # Append the card's description unless it's already in the hand glossary above (avoids
        # repeating it) so every choosable card — drafts, buyable cards — explains itself.
        brief = ""
        cname = _option_card_name(opt)
        if cname and not (hand_shown and cname in hand_name_set):
            b = card_brief(cname)
            if b:
                brief = f"  — {b}"
        if wf_type == "card" and is_hand_decision and card_names_in_decision:
            lines.append(f"  {idx}. {title2[:70]} (see hand above){cost_tag}")
        elif desc:
            lines.append(f"  {idx}. {desc}{cost_tag}{brief}")
        else:
            lines.append(f"  {idx}. {title2[:70]}{cost_tag}{brief}")

    if wf_type == "card":
        cmax = waiting_for.get("max", 1) or 1
        cmin = waiting_for.get("min", 0)
        if cmax > 1:
            none_hint = " — or CHOICE: none to take none" if cmin <= 0 else ""
            lines.append(
                f"(You may pick up to {cmax}: list EVERY chosen number on the CHOICE line, "
                f"e.g. CHOICE: 1,3{none_hint}.)"
            )

    if wf_type in ("projectCard", "payment"):
        lines += _format_payment_section(waiting_for, p)

    return "\n".join(lines)


def _format_payment_section(waiting_for: dict, player: dict) -> list[str]:
    lines = ["", "Payment:"]
    if waiting_for.get("type") == "projectCard":
        card = waiting_for.get("card", {})
        name = card.get("name", "?")
        lines.append(f"  Card: {name}  Cost: {card.get('calculatedCost', '?')} MC")
        st, ti, mc = player.get("steel", 0), player.get("titanium", 0), player.get("megacredits", 0)
        card_tags = CARD_DB.get(name, {}).get("tags") or []
        avail = f"  Available: MC={mc}"
        if "building" in card_tags and st:
            avail += f" Steel={st}(worth {st * 2}MC)"
        if "space" in card_tags and ti:
            avail += f" Titanium={ti}(worth {ti * 3}MC)"
        lines += [avail,
                  "  Reply: PAYMENT: MC=<n>[, STEEL=<n>][, TITANIUM=<n>]...",
                  "  Steel only for building-tag, titanium only for space-tag."]
    else:
        lines += [f"  Amount: {waiting_for.get('amount', 0)} MC  Available MC: {player.get('megacredits', 0)}",
                  "  Reply: PAYMENT: MC=<n>[, HEAT=<n>]...  (no steel/titanium for standard projects)"]
    return lines


# ---------------------------------------------------------------------------
# Action response parsing
# ---------------------------------------------------------------------------

# Matches the CHOICE line's value: one number, or a list joined by commas / 'and' / '&' / '+'
# / '/'. The list only extends while each separator is followed by another number, so prose
# after the answer (e.g. "CHOICE: 2 and place an ocean on hex-61") is NOT swept in.
_CHOICE_LIST = re.compile(
    label_prefix("CHOICE") + r"(\d+(?:\s*(?:,|and|&|\+|/)\s*\d+)*)",
    re.IGNORECASE,
)


def find_choices(text: str) -> list[int]:
    """All 1-based option numbers on the `CHOICE:` line, in order, de-duplicated.

    Multi-select decisions (research-phase 'buy card(s)', 'keep N cards') let the model pick
    several, e.g. `CHOICE: 2,3` → [2, 3]. Single decisions just yield one element."""
    m = _CHOICE_LIST.search(text)
    if not m:
        return []
    out: list[int] = []
    for n in re.findall(r"\d+", m.group(1)):
        v = int(n)
        if v not in out:
            out.append(v)
    return out


def find_choice(text: str) -> int | None:
    """The first 1-based option number from a `CHOICE: N` line, tolerating markdown emphasis
    (e.g. `**CHOICE:** 1`), or None if absent. Case-sensitive on the label so prose like
    "my choice: ..." in the reasoning body is not mistaken for the answer."""
    choices = find_choices(text)
    return choices[0] if choices else None


def capture_tactical(text: str) -> str | None:
    m = re.search(label_prefix("TACTICAL") + r"(.*?)(?=\n\s*" + _EMPH + r"CHOICE|\n\s*" + _EMPH + r"PAYMENT|\Z)",
                  text, re.IGNORECASE | re.DOTALL)
    if m:
        tactical = m.group(1).strip()
        return tactical or None
    return None


# A strategy doc is long-term PROSE memory, but the per-gen reflection sometimes answers in the
# turn format (CHOICE:/PAYMENT:) or echoes the prompt's "----- PRIOR STRATEGY -----" framing.
# Saving those verbatim pollutes every later prompt with stale, meaningless option numbers
# (the observed "CHOICE: 15"). Drop those lines before persisting the strategy.
_STRATEGY_DROP = re.compile(
    r"^\s*(?:"
    + _EMPH + r"(?:CHOICE|PAYMENT)" + _EMPH + r"\s*:.*"   # turn-answer lines
    r"|-{3,}.*?-{3,}"                                       # ----- PRIOR STRATEGY ----- banners
    r"|-{3,}"                                               # plain divider rules
    r")\s*$",
    re.IGNORECASE,
)


def sanitize_strategy(text: str) -> str:
    """Strip action-format artefacts and template banners the model leaks into a strategy doc."""
    if not text:
        return text
    kept = [ln for ln in text.splitlines() if not _STRATEGY_DROP.match(ln)]
    out = "\n".join(kept).strip()
    # Safety net: the model sometimes parrots the meta-instruction labels inline (one paragraph,
    # so line-filtering above can't catch them). Drop the obvious echoes.
    out = re.sub(r"\*{0,3}\s*PROSE[ -]?ONLY\b\.?\*{0,3}", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\*{0,3}\s*DEFERRAL(?: CHECK)?\s*:?\*{0,3}", "", out, flags=re.IGNORECASE)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip()


def parse_action_response(text: str, options: list[dict], waiting_for: dict, player_id: str,
                          player: dict | None = None) -> tuple[dict, dict]:
    wf_type = waiting_for.get("type", "")
    p = player or {}

    choices = find_choices(text)
    if not choices:
        logger.warning("No CHOICE line (player=%s) — defaulting to option 1. Response: %.300s", player_id, text)
    choice = choices[0] if choices else None
    chosen = (choice - 1) if choice is not None else 0
    chosen = max(0, min(chosen, len(options) - 1))
    option = options[chosen]

    # Card decisions can be multi-select (research-phase 'buy card(s)', 'keep N cards') or allow
    # selecting nothing (min 0). The model expresses these as 'CHOICE: 2,3' or 'CHOICE: none'.
    if wf_type == "card":
        cmax = waiting_for.get("max", 1) or 1
        cmin = waiting_for.get("min", 0)
        positive = [c for c in choices if c > 0]
        wants_none = (cmin <= 0 and not positive
                      and re.search(label_prefix("CHOICE") + r"(?:0\b|none)", text, re.IGNORECASE))
        if wants_none:
            logger.info("Action card player=%s: take none", player_id)
            return {"type": "card", "cards": []}, {"llm_choice": "none", "llm_option": "(none)"}
        if cmax > 1 and len(positive) > 1:
            picked: list[str] = []
            for c in positive:
                idx = c - 1
                if 0 <= idx < len(options):
                    nm = (options[idx].get("node") or {}).get("name", "")
                    if nm and nm not in picked:
                        picked.append(nm)
            picked = picked[:cmax]
            if picked:
                logger.info("Action multi-card player=%s: %s", player_id, picked)
                return ({"type": "card", "cards": picked},
                        {"llm_choice": ",".join(map(str, positive)), "llm_option": ", ".join(picked)})

    logger.info("Action choice player=%s: %d. %s", player_id, chosen + 1, option["title"])
    response = index_to_response(waiting_for, option["path"])

    if wf_type in ("projectCard", "payment"):
        payment = parse_payment_line(text)
        if payment:
            wf_card_name = waiting_for.get("card", {}).get("name", "") if wf_type == "projectCard" else ""
            payment = correct_payment(payment, waiting_for, p, card_name=wf_card_name)
            response = {**response, "payment": payment}
        else:
            logger.warning("No PAYMENT line for %s (player=%s) — using default. %.200s", wf_type, player_id, text)
    elif response.get("type") == "or":
        inner = response.get("response", {})
        if inner.get("type") == "projectCard":
            sub_node = option.get("node", {})
            available_cards = sub_node.get("cards", []) if isinstance(sub_node, dict) else []
            card_name = inner.get("card", "")
            payment = parse_payment_line(text)
            if payment:
                text_lower = text.lower()
                for c in available_cards:
                    cn = c.get("name", "")
                    if cn and cn.lower() in text_lower:
                        card_name = cn
                        break
            if not card_name and available_cards:
                card_name = available_cards[0].get("name", "")
                logger.warning("or→projectCard: no card named (player=%s) — using first %r", player_id, card_name)
            if payment:
                card_info = next((c for c in available_cards if c.get("name") == card_name), {})
                stub = {"type": "projectCard", "amount": card_info.get("calculatedCost", 0)}
                payment = correct_payment(payment, stub, p, card_name=card_name)
            else:
                payment = auto_payment_for_card(card_name, sub_node, p)
                logger.info("Auto-payment for %r (player=%s): %s", card_name, player_id, payment)
            response = {**response, "response": {"type": "projectCard", "card": card_name, "payment": payment}}

    return response, {"llm_choice": chosen + 1, "llm_option": option["title"]}


# ---------------------------------------------------------------------------
# Setup phase (initialCards / prelude)
# ---------------------------------------------------------------------------

def build_setup_system(state: dict) -> str:
    g = state.get("game", {})
    game_ctx = format_config_context(g)
    layout = format_board_layout(state.get("boardSpaces") or [])
    return (
        TM_RULES + "\n" + STRATEGY_PRIMER + "\n\n" + game_ctx + "\n\n"
        + (layout + "\n\n" if layout else "")
        + "You are an expert Terraforming Mars strategist making the opening decisions. Think "
        "step by step about card synergies, engine building, the listed milestones/awards, and "
        "any active variants. During the game you receive the COMPLETE state fresh each turn plus "
        "your own STRATEGY notes — there is no chat history, so your STRATEGY is your long-term "
        "memory. Write a strategy strong enough to guide play from those notes alone. Follow the "
        "EXACT output format requested."
    )


def _find_option(options: list, keywords: tuple) -> dict | None:
    for opt in options:
        t = _node_title(opt, 0).lower()
        if any(k in t for k in keywords):
            return opt
    return None


def build_setup_prompt(state: dict, waiting_for: dict) -> str:
    wf_type = waiting_for.get("type", "")
    options = waiting_for.get("options", [])
    lines: list[str] = []

    if wf_type == "initialCards":
        corp_opt = _find_option(options, ("corporation",))
        prelude_opt = _find_option(options, ("prelude",))
        ceo_opt = _find_option(options, ("ceo",))
        project_opt = _find_option(options, ("project", "initial", "cards to buy"))
        corps = corp_opt.get("cards", []) if corp_opt else []
        buyable = project_opt.get("cards", []) if project_opt else []

        lines += ["# Terraforming Mars — Opening Decisions", "", "## Corporations (choose exactly 1)"]
        for i, c in enumerate(corps, 1):
            name = c.get("name", f"Corp {i}")
            entry = CARD_DB.get(name)
            desc = f" — {entry['description']}" if entry and entry.get("description") else ""
            lines.append(f"  {i}. {name}{desc}")

        if buyable:
            lines += ["", "## Project cards to add to hand (3 MC each; play cost in [brackets])",
                      "  Buy as many as you can afford that synergise with your corporation — a thin hand starves your engine."]
            for i, c in enumerate(buyable, 1):
                name = c.get("name", f"Card {i}")
                entry = CARD_DB.get(name)
                tags = entry.get("tags") or [] if entry else []
                tag_str = f" [{', '.join(tags)}]" if tags else ""
                desc = f" — {entry['description']}" if entry and entry.get("description") else ""
                lines.append(f"  {i}. {name}{tag_str}  [play cost: {c.get('calculatedCost', '?')} MC]{desc}")

        if prelude_opt:
            lines += ["", "## Prelude cards (choose 2)"]
            for i, c in enumerate(prelude_opt.get("cards", []), 1):
                name = c.get("name", f"Prelude {i}")
                entry = CARD_DB.get(name)
                desc = f" — {entry['description']}" if entry and entry.get("description") else ""
                lines.append(f"  {i}. {name}{desc}")

        if ceo_opt:
            lines += ["", "## CEO card (choose 1)"]
            for i, c in enumerate(ceo_opt.get("cards", []), 1):
                name = c.get("name", f"CEO {i}")
                entry = CARD_DB.get(name)
                desc = f" — {entry['description']}" if entry and entry.get("description") else ""
                lines.append(f"  {i}. {name}{desc}")

        lines += [
            "",
            "Choose the corporation that best synergises with the project cards and write a clear plan.",
            "Respond in EXACTLY this format (copy names exactly as listed):",
            "CORPORATION: <exact name>",
            "BUY_CARDS: <exact comma-separated names, or 'none'>",
        ]
        if prelude_opt:
            lines.append("PRELUDE_CARDS: <exact comma-separated names of 2 preludes>")
        if ceo_opt:
            lines.append("CEO_CARD: <exact name of 1 CEO>")
        lines += [
            "STRATEGY:",
            "<150-250 words: engine type (PRIMARY plan), priority tags, key synergies, pace plan.",
            " MILESTONE TARGET: name ONE specific milestone from the list above you will go for,",
            " the tag/tile/TR count it needs, and the generation you plan to claim it by (claim the",
            " moment you qualify — only 3 are ever claimed and opponents race you). AWARD TARGET:",
            " one award you can realistically win 1st by game end, or 'none'. End with a BACKUP",
            " PLAN: one sentence naming an alternative engine/scoring path if the primary stalls.>",
        ]

    elif wf_type == "prelude":
        p = state.get("player", {})
        prod = {k: v for k, v in p.get("production", {}).items() if v}
        lines += ["# Terraforming Mars — Prelude Selection",
                  f"State: MC={p.get('megacredits', 0)} Production={prod}", "", "## Prelude options"]
        for i, opt in enumerate(options, 1):
            if opt.get("type") == "card":
                lines.append(f"  {i}. " + ", ".join(c.get("name", "") for c in opt.get("cards", [])))
            else:
                lines.append(f"  {i}. {_node_title(opt, i)}")
        lines += ["", "Choose the prelude that best advances your strategy.",
                  "Respond in EXACTLY this format:", "CHOICE: <option number>",
                  "STRATEGY_UPDATE: <full updated strategy; repeat everything to keep — this REPLACES",
                  " your memory. Write 'no change' only if truly nothing changed.>"]

    return "\n".join(lines)


def parse_setup_response(text: str, waiting_for: dict, prior_strategy: str = "") -> tuple[dict, str]:
    wf_type = waiting_for.get("type", "")
    options = waiting_for.get("options", [])

    if wf_type == "initialCards":
        corp_opt = _find_option(options, ("corporation",))
        prelude_opt = _find_option(options, ("prelude",))
        ceo_opt = _find_option(options, ("ceo",))
        project_opt = _find_option(options, ("project", "initial", "cards to buy"))
        corps = [c.get("name", "") for c in (corp_opt or {}).get("cards", [])]
        buyable = [c.get("name", "") for c in (project_opt or {}).get("cards", [])]

        chosen_corp = corps[0] if corps else ""
        m = re.search(label_prefix("CORPORATION") + r"(.+)", text)
        if m:
            raw = m.group(1).strip().rstrip(".")
            for c in corps:
                if c.lower() in raw.lower() or raw.lower() in c.lower():
                    chosen_corp = c
                    break

        bought: list[str] = []
        m = re.search(label_prefix("BUY_CARDS") + r"(.+?)(?:\n" + _EMPH + r"PRELUDE|\n" + _EMPH + r"CEO|\n" + _EMPH + r"STRATEGY|\Z)", text, re.DOTALL | re.IGNORECASE)
        if m and m.group(1).strip().lower() not in ("none", "none.", ""):
            for raw_name in re.split(r",\s*|\n", m.group(1).strip()):
                raw_name = raw_name.strip().lstrip("-•").strip().rstrip(".")
                if not raw_name:
                    continue
                for b in buyable:
                    if (b.lower() in raw_name.lower() or raw_name.lower() in b.lower()) and b not in bought:
                        bought.append(b)
                        break

        prelude_chosen: list[str] = []
        if prelude_opt:
            preludes = [c.get("name", "") for c in prelude_opt.get("cards", [])]
            m2 = re.search(label_prefix("PRELUDE_CARDS") + r"(.+?)(?:\n" + _EMPH + r"CEO|\n" + _EMPH + r"STRATEGY|\Z)", text, re.DOTALL | re.IGNORECASE)
            if m2:
                for raw_name in re.split(r",\s*|\n", m2.group(1).strip()):
                    raw_name = raw_name.strip().rstrip(".")
                    for pr in preludes:
                        if (pr.lower() in raw_name.lower() or raw_name.lower() in pr.lower()) and pr not in prelude_chosen:
                            prelude_chosen.append(pr)
                            break
            if len(prelude_chosen) < 2:
                prelude_chosen = preludes[:2]

        ceo_chosen = ""
        if ceo_opt:
            ceos = [c.get("name", "") for c in ceo_opt.get("cards", [])]
            m3 = re.search(label_prefix("CEO_CARD") + r"(.+?)(?:\n" + _EMPH + r"STRATEGY|\Z)", text, re.DOTALL | re.IGNORECASE)
            if m3 and ceos:
                raw_ceo = m3.group(1).strip().rstrip(".")
                for c in ceos:
                    if c.lower() in raw_ceo.lower() or raw_ceo.lower() in c.lower():
                        ceo_chosen = c
                        break
            if not ceo_chosen and ceos:
                ceo_chosen = ceos[0]

        m4 = re.search(label_prefix("STRATEGY") + r"(.*)", text, re.DOTALL)
        strategy = m4.group(1).strip() if m4 else text.strip()
        logger.info("Setup parsed: corp=%r buy=%r prelude=%r ceo=%r", chosen_corp, bought, prelude_chosen, ceo_chosen)

        responses = []
        for opt in options:
            t = _node_title(opt, 0).lower()
            if "corporation" in t:
                responses.append({"type": "card", "cards": [chosen_corp] if chosen_corp else corps[:1]})
            elif "prelude" in t:
                responses.append({"type": "card", "cards": prelude_chosen})
            elif "ceo" in t:
                responses.append({"type": "card", "cards": [ceo_chosen] if ceo_chosen else []})
            elif any(k in t for k in ("project", "initial", "cards to buy")):
                responses.append({"type": "card", "cards": bought})
            else:
                responses.append(_default_response(opt))
        return {"type": "initialCards", "responses": responses}, strategy

    elif wf_type == "prelude":
        choice = find_choice(text)
        idx = max(0, min((choice - 1 if choice is not None else 0), len(options) - 1))
        strategy = prior_strategy or "Play balanced."
        m2 = re.search(label_prefix("STRATEGY_UPDATE") + r"(.*)", text, re.DOTALL)
        if m2:
            upd = m2.group(1).strip()
            if upd and not upd.lower().startswith("no change"):
                strategy = upd
        return index_to_response(waiting_for, idx), strategy

    return _default_response(waiting_for), prior_strategy or "Play a balanced game."


# ---------------------------------------------------------------------------
# Per-generation strategy update prompt
# ---------------------------------------------------------------------------

def build_pergen_prompt(state: dict, prior_strategy: str, generation: int) -> str:
    g, p = state.get("game", {}), state.get("player", {})
    temp, oxygen, oceans = g.get("temperature", -30), g.get("oxygen", 0), g.get("oceanCount", 0)
    tr = p.get("terraformRating", 20)
    mc_prod = p.get("production", {}).get("megacredits", 0)
    mc_income = tr + mc_prod

    status: list[str] = []
    if temp >= 8:
        status.append("temperature maxed — no Convert Heat")
    if oxygen >= 14:
        status.append("O₂ maxed — greenery gives no TR but still +1 VP")
    if oceans >= 9:
        status.append("all oceans placed")
    status_line = ("⚠ Global: " + "; ".join(status) + "\n") if status else ""
    end_banner = game_end_proximity(g)
    if end_banner:
        status_line += end_banner + "\n"

    heat_now, plants_now = p.get("heat", 0), p.get("plants", 0)
    notes = [f"Income this gen: MC {mc_income} (TR {tr} + prod {mc_prod:+d})."]
    if heat_now >= 8 and temp < 8:
        notes.append(f"You have {heat_now} heat — Convert 8 heat is a free +1 TR; do it first.")
    if plants_now >= 8 and oxygen < 14:
        notes.append(f"You have {plants_now} plants — Convert 8 plants is a free +1 TR; do it first.")

    ms_status = compute_milestone_status(state)
    aw_status = compute_award_standings(state)
    own_name = p.get("name", "?")

    return (
        f"=== End of Generation {generation - 1}, start of Generation {generation} ===\n"
        f'You are "{own_name}". {status_line}'
        + " ".join(notes) + "\n"
        + ("\n".join(ms_status) + "\n" if ms_status else "")
        + ("Unfunded award standings:\n" + "\n".join(aw_status) + "\n" if aw_status else "")
        + "\nLast gen's strategy notes (ALL you remember — state is supplied fresh each turn):\n"
        "----- PRIOR STRATEGY -----\n"
        f"{prior_strategy or '(none yet — first strategy update)'}\n"
        "--------------------------\n\n"
        "Reply with ONLY your rewritten strategy: ~120 words of dense prose. No CHOICE/PAYMENT "
        "lines, no option numbers, and do NOT echo any instruction label (e.g. 'DEFERRAL', "
        "'PROSE ONLY') — those are directions to you, not content. Cover, in order:\n"
        "1. STANDING: VP/TR vs each opponent — ahead, level, or behind?\n"
        "2. ENGINE (PRIMARY): your path to victory and how you score each gen.\n"
        "3. MILESTONE TARGET: a still-available one (never ✗); if you already meet a ✓, claim it FIRST this gen.\n"
        "4. AWARD TARGET: only one you can win 1st/close-2nd; else 'none'.\n"
        "5. NEXT-GEN PRIORITY: ordered actions; convert spare heat/plants FIRST. Re-do an undone "
        "goal from last gen only if it's actually offered this gen — drop one-time setup perks or a "
        "'free' placement you no longer have.\n"
        "6. BACKUP PLAN + SWITCH: keep primary, or switch (mandatory if >20 VP behind late)."
    )
