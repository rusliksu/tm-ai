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

from .knowledge import CARD_DB, format_card_context, format_config_context, format_board_layout
from .options import _default_response, index_to_response
from .payment import (
    parse_payment_line, correct_payment, auto_payment_for_card,
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
    t = title.lower().strip()
    if t in STANDARD_PROJECT_COSTS:
        return STANDARD_PROJECT_COSTS[t]
    t2 = re.sub(r"\s*\(\d+ ?m€\)\s*$", "", t)
    return STANDARD_PROJECT_COSTS.get(t2)


# ---------------------------------------------------------------------------
# Rules reference (cached system prefix) — RULES only, strategy split out below
# ---------------------------------------------------------------------------

TM_RULES = f"""
=== TERRAFORMING MARS — RULES REFERENCE ===

OBJECTIVE: most Victory Points (VP) at game end wins.
VP sources: Terraform Rating (1 VP/TR), greenery tiles (1 VP each),
city tiles (1 VP per adjacent greenery of ANY owner), milestones (5 VP, 8 MC to claim,
max 3 claimed in the whole game), awards (5 VP 1st / 2 VP 2nd, max 3 funded), card VP.

GLOBAL PARAMETERS (game ends when temperature, oxygen and oceans are all maxed):
  • Temperature: −30°C → +8°C (+2°C per step, 20 steps). Raise: 8 heat, Asteroid SP, cards.
  • Oxygen:       0% → 14% (14 steps). Raise: place greenery, or cards.
  • Oceans:       0 → 9 tiles. Place: Aquifer SP or cards.
  • Venus (Venus Next only): 0% → 30% (+2% per step, 15 steps). Raise: Venus cards/SPs.
  Each step raised = +1 TR (= +1 income AND +1 VP).

GLOBAL-PARAMETER THRESHOLD BONUSES (one-time, to the player who triggers the step):
  • Temperature −24°C and −20°C: +1 heat production each.
  • Temperature 0°C: place 1 ocean tile.
  • Oxygen 8%: temperature rises +1 step automatically (free TR for the raiser).
  • Venus 8%: draw 1 card.   Venus 16%: +1 TR.
  Timing a step to hit a threshold yourself is worth ~10 MC of value.

RESOURCES (collected each production phase):
  • MegaCredits (MC): currency. INCOME each generation = TR + MC-production.
  • Steel: pays for BUILDING-tag cards at 2 MC/cube.   Titanium: SPACE-tag cards at 3 MC/cube.
  • Plants: 8 → greenery tile (+1 O₂, +1 TR).   Energy: leftover converts to heat each gen.
  • Heat: 8 → raise temperature +1 step (+1 TR).   MC production may be negative (min −5);
    all other productions never go below 0.

CARD TYPES: GREEN (automated, one-time effect, tag persists), BLUE (active: ongoing effect
or a once-per-generation action), RED event (one-time, tag counts only when played).
Steel discounts building-tag cards; titanium discounts space-tag cards.

TURN STRUCTURE each generation: player order rotates; research phase (draw 4, buy any at
3 MC); action phase (each player takes 1–2 actions per turn until all pass); production phase.

ACTIONS (1–2 per turn): play a card; use a standard project; claim a milestone (8 MC + meet
requirement); fund an award (8/14/20 MC); use a blue card's action (once/gen); convert 8
plants → greenery; convert 8 heat → +1 temperature.
  AWARD RULE: you only SCORE an award if you place 1st (5 VP) or 2nd (2 VP) at game END.
  Funding an award you are not winning = paying MC for 0 VP. Only fund what you lead.

STANDARD PROJECTS (always available):
  • Sell patents (free): discard N cards → N MC. Almost always bad — cards are worth far more.
  • Power Plant ({STANDARD_PROJECT_COSTS['power plant:sp']} MC): +1 energy production. Weak; skip once you convert heat regularly.
  • Asteroid ({STANDARD_PROJECT_COSTS['asteroid:sp']} MC): +1 temperature (+1 TR). Good value.
  • Aquifer ({STANDARD_PROJECT_COSTS['aquifer:sp']} MC): place ocean (+1 TR + placement bonus). Good.
  • Greenery ({STANDARD_PROJECT_COSTS['greenery:sp']} MC): place greenery (+1 O₂, +1 TR).
  • City ({STANDARD_PROJECT_COSTS['city:sp']} MC): place city + 1 MC production. High value — TR + income + board VP.

PLAYABILITY: the option list below already contains ONLY cards/projects you can legally play
AND currently afford. You never need to check requirements or affordability yourself — if it
is listed, it is playable. Pick the best option; pay with PAYMENT.

TILE PLACEMENT: ocean only on reserved blue spaces (adjacent tile owners get +2 MC); greenery
must go next to your own tile if possible; cities cannot be adjacent to another city. At game
end greenery = 1 VP, city = 1 VP per adjacent greenery.

PAYMENT: pay a card's cost in MC, optionally substituting steel (building, 2 MC/cube) or
titanium (space, 3 MC/cube). You cannot overpay in MC; surplus steel/titanium is lost.

RESPONSE FORMAT — MANDATORY every turn:
  1. One or two sentences of reasoning tied to your strategy.
  2. TACTICAL: <your updated ordered next-steps for the rest of this generation — this is the
     ONLY note you carry to your next move; rewrite it from what just happened>.
  3. CHOICE: N   (the option number, on its own line)
  4. PAYMENT: MC=<n>[, STEEL=<n>][, TITANIUM=<n>][, HEAT=<n>]   (only when playing a card)
  Steel only counts for building-tag cards, titanium only for space-tag cards; PAYMENT must
  total at least the displayed cost.

SERVER AUTHORITY: the game server enforces all rules and is always correct. If it rejects a
move, read the error, pick a DIFFERENT valid option, and never repeat the invalid move.
=== END RULES REFERENCE ===
"""

# Strategy guidance — condensed from the original essays. Cached alongside the rules.
STRATEGY_PRIMER = """
=== STRATEGY PRIMER ===
• TR is income AND VP: terraforming is your primary objective, not an afterthought. Each +1
  TR pays back every remaining generation. Convert spare heat/plants before passing — ≥8 heat
  (temp < 8°C) or ≥8 plants (O₂ < 14%) is a free +1 TR; never waste it.
• Build MC production early — it compounds. City SP and production cards beat one-off effects.
• Card throughput wins: aim to play 2–4 project cards per generation. A fat hand with low MC
  means a stalled engine — use steel/titanium discounts and cheap synergy cards to unstall.
• Match draft buys to your resources: titanium → buy space cards, steel → buy building cards.
  Stockpiled resources with no matching cards are wasted production.
• Milestones are exceptional value (5 VP for 8 MC) and capped at 3 — claim the moment you
  qualify; opponents can race you. Awards: only fund ones you are winning, and not too early.
• City + greenery geometry: place greeneries adjacent to YOUR cities (each adjacency = +1 VP);
  two cities two hexes apart share a 3-VP greenery hex. Never place greenery next to an
  opponent's city. Place your first city by ~gen 3–4.
• Pace: if you out-score opponents per generation, slow terraforming; if you are ahead on TR
  but behind on VP, accelerate to end the game before their engines mature.
• Read opponents from their tableau, funded awards, and claimed milestones — block and race.
=== END STRATEGY PRIMER ===
"""

_ACTION_SYSTEM_SUFFIX = (
    "\n\nYou are an expert Terraforming Mars player. Each turn you receive the COMPLETE game "
    "state plus YOUR OWN MEMORY (a coarse STRATEGY and a short TACTICAL plan). There is NO chat "
    "history — those two notes are all you remember between turns, so keep them accurate and act "
    "on them. Always answer in the mandatory RESPONSE FORMAT (reasoning, TACTICAL, CHOICE, "
    "PAYMENT)."
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
        "⚠ MILESTONE CLAIM AVAILABLE: a 'Claim milestone' option means you already meet the "
        "requirement and can afford 8 MC. 5 VP for 8 MC, capped at 3 per game — default CLAIM IT "
        "NOW. Postpone only if every opponent is far from every unclaimed milestone AND you have "
        "an >5-VP play this turn.",
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
    g, p = state.get("game", {}), state.get("player", {})
    opps = state.get("opponents") or []
    funded_names = {a["name"] for a in state.get("awards", [])}
    available = g.get("availableAwards") or []
    if not available:
        return []

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

    my_name = p.get("name", "You")
    lines: list[str] = []
    for award_info in available:
        aname = award_info.get("name", "?")
        if aname in funded_names:
            continue
        my_v = _val(p, aname)
        if my_v < 0:
            continue
        entries = [(my_name, my_v)] + [(o.get("name", "Opp"), _val(o, aname)) for o in opps]
        entries.sort(key=lambda x: x[1], reverse=True)
        rank = next((i + 1 for i, (n, _) in enumerate(entries) if n == my_name), len(entries))
        rank_str = {1: "1st", 2: "2nd", 3: "3rd"}.get(rank, f"{rank}th")
        standings = ", ".join(f"{n}={v}" for n, v in entries)
        lines.append(f"  {aname}: you are {rank_str} ({standings})")
    return lines


# ---------------------------------------------------------------------------
# Option/card helpers
# ---------------------------------------------------------------------------

def _get_card_desc_for_option(opt: dict) -> str:
    name = (opt.get("card") or {}).get("name", "")
    if not name:
        return ""
    desc = CARD_DB.get(name, {}).get("description", "")
    return f"Use {name} action — {desc}" if desc else f"Use {name} action"


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
        f'You are "{own_name}" (color={own_color}). In the event log, lines starting '
        f'"You ({own_name})" are your own past actions.',
        f"You: TR:{tr} VP:{p.get('victoryPoints', '?')}  MC:{mc}(income:{mc_income})  "
        f"Steel:{p.get('steel', 0)} Ti:{p.get('titanium', 0)}  "
        f"Plants:{p.get('plants', 0)} Energy:{p.get('energy', 0)} Heat:{p.get('heat', 0)}",
    ]

    # Conditional one-line advisories (only when actionable)
    heat_now, plants_now = p.get("heat", 0), p.get("plants", 0)
    if heat_now >= 8 and temp < 8:
        lines.append(f">> {heat_now} heat (≥8), temp not maxed — 'Convert 8 heat' = free +1 TR. Do it before passing.")
    if plants_now >= 8 and oxygen < 14:
        lines.append(f">> {plants_now} plants (≥8), O₂ not maxed — 'Convert 8 plants' = greenery +1 TR +1 VP.")
    elif plants_now >= 8 and oxygen >= 14:
        lines.append(f">> {plants_now} plants — O₂ maxed, but each greenery still scores +1 VP if land is free.")
    if temp >= 8:
        lines.append("⚠ Temperature maxed (8°C) — do NOT Convert Heat.")
    if oceans >= 9:
        lines.append("⚠ All 9 oceans placed.")

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

    # Opponents
    for i, opp in enumerate(opponents, 1):
        opp_prod = {k: v for k, v in opp.get("production", {}).items() if v}
        opp_tags = {k: v for k, v in opp.get("tags", {}).items() if v}
        nm = opp.get("name", f"Opponent{i}")
        vp = f" VP:{opp['victoryPoints']}" if opp.get("victoryPoints") is not None else ""
        hs = f"  hand:{opp['handSize']}" if opp.get("handSize") is not None else ""
        lines.append(f"{nm}: TR:{opp.get('terraformRating', 20)}{vp}  MC:{opp.get('megacredits', 0)}{hs}  prod:{opp_prod}  tags:{opp_tags}")

    lines.extend(compute_milestone_status(state))

    n_funded = len(aw) if aw else 0
    if n_funded >= 3:
        lines.append(f"Awards ({n_funded}/3 funded — FUNDING PHASE OVER).")
    else:
        standings = compute_award_standings(state)
        if standings:
            lines.append("Unfunded award standings (fund only if 1st or close 2nd):")
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
    title = _format_message(waiting_for.get("title")).strip() or "Select action"
    lines += ["", f"Decision: {title}", "Options:"]

    card_names_in_decision = _extract_card_names(waiting_for)
    is_hand_decision = _is_card_decision_about_hand(card_names_in_decision, hand_cards)
    for opt in options:
        idx = opt["index"] + 1
        title2 = str(opt["title"])
        desc = _get_card_desc_for_option(opt)
        sp_cost = standard_project_cost(
            title2.split(":", 1)[1].strip() if title2.lower().startswith("standard projects:") else title2
        )
        cost_tag = ""
        if sp_cost is not None:
            cost_tag = (f"  [{sp_cost} MC — NOT AFFORDABLE, you have {mc} MC]" if sp_cost > mc
                        else f"  [{sp_cost} MC; you have {mc} MC]")
        if wf_type == "card" and is_hand_decision and card_names_in_decision:
            lines.append(f"  {idx}. {title2[:70]} (see hand above){cost_tag}")
        elif desc:
            lines.append(f"  {idx}. {desc}{cost_tag}")
        else:
            lines.append(f"  {idx}. {title2[:70]}{cost_tag}")

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

def capture_tactical(text: str) -> str | None:
    m = re.search(r"TACTICAL:\s*(.*?)(?=\n\s*CHOICE:|\n\s*PAYMENT:|\Z)", text, re.IGNORECASE | re.DOTALL)
    if m:
        tactical = m.group(1).strip()
        return tactical or None
    return None


def parse_action_response(text: str, options: list[dict], waiting_for: dict, player_id: str,
                          player: dict | None = None) -> tuple[dict, dict]:
    m = re.search(r"CHOICE:\s*(\d+)", text)
    if not m:
        logger.warning("No CHOICE line (player=%s) — defaulting to option 1. Response: %.300s", player_id, text)
    chosen = int(m.group(1)) - 1 if m else 0
    chosen = max(0, min(chosen, len(options) - 1))
    option = options[chosen]
    logger.info("Action choice player=%s: %d. %s", player_id, chosen + 1, option["title"])
    response = index_to_response(waiting_for, option["path"])

    wf_type = waiting_for.get("type", "")
    p = player or {}

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
            "<150-250 words: engine type (PRIMARY plan), priority tags, milestone/award targets,",
            " key synergies, pace plan. End with a BACKUP PLAN: one sentence naming an alternative",
            " engine/scoring path to pivot to if the primary stalls.>",
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
        m = re.search(r"CORPORATION:\s*(.+)", text)
        if m:
            raw = m.group(1).strip().rstrip(".")
            for c in corps:
                if c.lower() in raw.lower() or raw.lower() in c.lower():
                    chosen_corp = c
                    break

        bought: list[str] = []
        m = re.search(r"BUY_CARDS:\s*(.+?)(?:\nPRELUDE|\nCEO|\nSTRATEGY|\Z)", text, re.DOTALL | re.IGNORECASE)
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
            m2 = re.search(r"PRELUDE_CARDS:\s*(.+?)(?:\nCEO|\nSTRATEGY|\Z)", text, re.DOTALL | re.IGNORECASE)
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
            m3 = re.search(r"CEO_CARD:\s*(.+?)(?:\nSTRATEGY|\Z)", text, re.DOTALL | re.IGNORECASE)
            if m3 and ceos:
                raw_ceo = m3.group(1).strip().rstrip(".")
                for c in ceos:
                    if c.lower() in raw_ceo.lower() or raw_ceo.lower() in c.lower():
                        ceo_chosen = c
                        break
            if not ceo_chosen and ceos:
                ceo_chosen = ceos[0]

        m4 = re.search(r"STRATEGY:\s*(.*)", text, re.DOTALL)
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
        m = re.search(r"CHOICE:\s*(\d+)", text)
        idx = max(0, min((int(m.group(1)) - 1 if m else 0), len(options) - 1))
        strategy = prior_strategy or "Play balanced."
        m2 = re.search(r"STRATEGY_UPDATE:\s*(.*)", text, re.DOTALL)
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
        + "\nYour strategy notes from last generation (ALL you remember — state is supplied fresh each turn):\n"
        "----- PRIOR STRATEGY -----\n"
        f"{prior_strategy or '(none yet — first strategy update)'}\n"
        "--------------------------\n\n"
        "Rewrite your strategy for this generation in ~150-200 words:\n"
        "1. STANDING: your VP/TR vs each opponent — ahead, level, or behind?\n"
        "2. ENGINE (PRIMARY): your path to victory and how you score each gen.\n"
        "3. MILESTONE TARGET: pick a still-available milestone from the status above (never one "
        "marked ✗). If you ALREADY meet one (✓), claim it as your FIRST action this generation.\n"
        "4. AWARD TARGET: only one you can win 1st/close-2nd; else 'none'.\n"
        "5. NEXT-GEN PRIORITY: concrete ordered actions. Convert spare heat/plants FIRST.\n"
        "6. BACKUP PLAN: one-sentence alternative engine/scoring path.\n"
        "7. SWITCH DECISION: 'Keep primary' or 'Switch: <reason>'. If >20 VP behind with few "
        "generations left, you MUST switch to an aggressive catch-up plan.\n"
        "DEFERRAL CHECK: if a goal you wrote last gen is still undone (same city count, milestone "
        "still unclaimed, card still in hand), name it and execute it as your FIRST action — do "
        "not write the same plan again without acting on it."
    )
