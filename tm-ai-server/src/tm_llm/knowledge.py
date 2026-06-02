"""Game-knowledge database and context formatting for the LLM prompts.

Provides:
  CARD_DB                      — dict[name, entry] loaded from data/card_db.json
  format_card_context(names)   — compact (1 line/card) descriptions for prompt injection
  format_config_context(game)  — board + expansions + variants + this game's milestones/awards
  format_board_layout(spaces)  — static board layout (bonuses/oceans/volcanic) for the system prompt
"""
from __future__ import annotations
import json

from .config import CARD_DB_PATH

# ---------------------------------------------------------------------------
# Card database
# ---------------------------------------------------------------------------

CARD_DB: dict[str, dict] = {}
try:
    with open(CARD_DB_PATH) as f:
        CARD_DB = json.load(f)
except Exception:
    pass  # server still works without card descriptions

# Cap a single card's description length in the prompt. Full multi-sentence text every
# turn is the main per-turn token sink; one trimmed line keeps the decision-relevant gist.
_MAX_DESC_CHARS = 140


def _trim_desc(desc: str) -> str:
    desc = " ".join(desc.split())
    if len(desc) <= _MAX_DESC_CHARS:
        return desc
    return desc[:_MAX_DESC_CHARS].rstrip() + "…"


def _vp_str(vp) -> str:
    if vp is None:
        return ""
    if isinstance(vp, (int, float)):
        return f" [{int(vp)} VP]"
    if isinstance(vp, dict):
        return " [VP/resource]"
    return " [VP]"


def format_card_context(card_names: list, header: str = "", max_cards: int = 30) -> str:
    """Return a compact one-line-per-card block for the listed cards (from CARD_DB)."""
    lines = []
    shown = 0
    for name in card_names:
        name = name if isinstance(name, str) else name.get("name", "")
        if shown >= max_cards:
            lines.append(f"  … and {len(card_names) - max_cards} more cards")
            break
        entry = CARD_DB.get(name)
        if not entry:
            lines.append(f"  {name}")
            shown += 1
            continue
        cost = entry.get("cost")
        cost_str = f"{cost} MC" if cost else "free"
        tags = entry.get("tags") or []
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        desc = _trim_desc(entry.get("description", ""))
        vp = _vp_str(entry.get("victoryPoints"))
        lines.append(f"  {name} ({cost_str}{tag_str}): {desc}{vp}")
        shown += 1
    if not lines:
        return ""
    prefix = f"{header}\n" if header else ""
    return prefix + "\n".join(lines)


# ---------------------------------------------------------------------------
# Board descriptions — special tiles and strategic notes only
# ---------------------------------------------------------------------------

BOARD_INFO: dict[str, dict] = {
    "tharsis": {
        "display": "Tharsis",
        "special_tiles": [
            "Noctis City — pre-placed city on the western plateau; counts toward Mayor milestone "
            "and adjacency scoring. Cannot place another city adjacent to it.",
        ],
        "notes": (
            "Classic, well-balanced board. Noctis City gives you a free adjacency "
            "target — placing greenery next to it scores VP without needing another city."
        ),
    },
    "hellas": {
        "display": "Hellas (Southern Hemisphere)",
        "special_tiles": [
            "Hellas Ocean — special space on the −6°C row; placing an ocean tile here costs "
            "only 6 MC (vs 18 MC for Aquifer standard project), raises an ocean +1 TR, "
            "and grants you +6 heat. Excellent early value.",
            "South Pole — bottom-left space with up to +3 heat placement bonuses.",
            "No starting pre-placed tile (unlike Tharsis).",
        ],
        "notes": (
            "Hellas favours energy/heat engines (Energizer milestone), Jovian strategies "
            "(Rim Settler milestone), and space-tag decks (Space Baron award). "
            "The cheap Hellas Ocean enables fast TR gain early game."
        ),
    },
    "elysium": {
        "display": "Elysium (Eastern Hemisphere)",
        "special_tiles": [
            "Elysium Space (top-right) — cluster of bonus spaces giving up to 8 mixed resources.",
            "Ascraeus Mons / Pavonis Mons — marked special spaces with extra production bonuses.",
        ],
        "notes": (
            "Elysium rewards versatile engines (Generalist milestone), event chains "
            "(Legend milestone), and expensive high-impact cards (Celebrity award). "
            "Estate Dealer makes ocean adjacency especially valuable."
        ),
    },
    "arabia terra": {
        "display": "Arabia Terra",
        "special_tiles": [
            "Arsia Mons — grants extra plant resources on placement.",
            "Multiple ocean-only spaces clustered together in the center.",
        ],
        "notes": "Fan board with rich ocean placement options; favours plant and animal engines.",
    },
    "vastitas borealis": {
        "display": "Vastitas Borealis",
        "special_tiles": ["Outer ring spaces have elevated placement bonuses."],
        "notes": "Fan board focused on resource diversity and production engines.",
    },
    "t. cimmeria": {
        "display": "Terra Cimmeria",
        "special_tiles": ["Multiple mountain (restricted) spaces; bonus resources on surrounding hexes."],
        "notes": "Fan board; Gambler/Warmonger milestones/awards reward event-heavy strategies.",
    },
    "utopia planitia": {
        "display": "Utopia Planitia",
        "special_tiles": [],
        "notes": "Balanced fan board; Metropolist award makes city building rewarding.",
    },
    "amazonis p.": {
        "display": "Amazonis Planitia",
        "special_tiles": ["Amazonis zones with extra placement bonuses."],
        "notes": "Fan board; Minimalist milestone rewards playing cards quickly then going lean.",
    },
}


# ---------------------------------------------------------------------------
# Expansion descriptions
# ---------------------------------------------------------------------------

EXPANSION_INFO: dict[str, str] = {
    "venus": (
        "Venus Next — adds a Venus parameter track (0%→30%, raised in +2% steps, 15 steps); "
        "each raise = +1 TR. New Venus-tag cards and corporations. "
        "Board tiles: Dawn City, Luna Metropolis, Maxwell Base, Stratopolis."
    ),
    "colonies": (
        "Colonies — Colony tiles (Moon, Ganymede, Titan, Callisto, Enceladus, etc.). "
        "Each generation you can trade one colony for resources (titanium from Titan, "
        "cards from Ganymede, microbes from Enceladus…). "
        "Build colony markers to improve trade bonuses. Trade fleets are limited — timing matters."
    ),
    "prelude": (
        "Prelude — each player plays 2 Prelude cards before generation 1, giving a strong "
        "engine head-start (production boosts, free cards, TR increases, or resources). "
        "Prelude choice dramatically shapes opening strategy."
    ),
    "prelude2": "Prelude 2 — second set of Prelude cards, same mechanic as original Prelude.",
    "turmoil": (
        "Turmoil — Politics track with five Parties (Mars First, Scientists, Greens, Unity, Reds). "
        "Dominant party at generation end applies a Global Event. "
        "Delegates placed each generation influence party control. "
        "Ruling party bonus affects all players; Chairman position gives extra TR each generation."
    ),
    "moon": (
        "The Moon — mini Moon board with Colony Rate / Mining Rate / Logistics Rate tracks. "
        "Cards place tiles on the Moon and raise these tracks (+1 TR each). "
        "New milestones (One Giant Step, Lunarchitect) and Moon-tag cards."
    ),
    "pathfinders": (
        "Pathfinders — Data and Preservation tags; planetary tracks for bonuses "
        "across the solar system. New starting conditions."
    ),
    "underworld": (
        "Underworld — underground excavation for resources and artifacts. "
        "Corruption tokens; Risktaker and Tunneler milestones."
    ),
    "ares": (
        "Ares — Hazard tiles (dust storms, erosion) that slow but reward mitigation. "
        "Improved adjacency bonuses. Networker milestone, Entrepreneur/Rugged awards."
    ),
    "corpEra": (
        "Corporate Era — players start at 0 production (vs 1 in Base). "
        "More cards and corporations. Strong production ramp-up is essential."
    ),
}


# ---------------------------------------------------------------------------
# Game variant descriptions
# ---------------------------------------------------------------------------

GAME_VARIANT_DESCRIPTIONS: dict[str, str] = {
    "draftVariant": (
        "Research Phase Draft — instead of drawing 4 cards and keeping any, players pass cards "
        "around the table. You see 4 cards but keep only 1 before passing. CARD DENIAL is "
        "possible — withhold cards that synergise with an opponent's engine."
    ),
    "initialDraftVariant": (
        "Initial Cards Draft — the 10 starting project cards are drafted rather than dealt. "
        "You can deny opponent key synergy cards; your initial hand is more curated."
    ),
    "preludeDraftVariant": "Prelude Draft — prelude cards are drafted; block strong engine preludes from opponents.",
    "ceosDraftVariant": "CEO Draft — CEO cards are drafted; pick the CEO matching your engine, deny powerful ones.",
    "twoCorpsVariant": (
        "Two Corporations Variant — each player starts with 2 corporations (plays both). "
        "Doubled starting resources and combined abilities — look for complementary synergy."
    ),
    "solarPhaseOption": (
        "World Government Terraforming (Solar Phase) — at the start of each generation one player "
        "(rotating) acts as World Government and raises one global parameter for free (no TR). "
        "SIGNIFICANTLY speeds up the game — expect 2-3 fewer generations. Accelerate early; slow "
        "starts are punished. As World Government, raise whichever parameter benefits you most."
    ),
    "soloTR": (
        "Solo Mode (TR 63 Victory) — must reach Terraform Rating 63 by game end. "
        "Maximise TR gain above all else; card VP matters much less."
    ),
    "randomMA": (
        "Randomized Milestones & Awards — the 5 milestones and 5 awards are randomly selected. "
        "Study the actual list below; target 1-2 milestones and compete for 1-2 awards."
    ),
    "modularMA": (
        "Modular Milestones & Awards — milestones/awards drawn from the expanded modular pool. "
        "See the actual list below."
    ),
    "requiresVenusTrackCompletion": (
        "Venus Must Be Completed — the game does not end until Venus reaches max. "
        "Invest in Venus cards; the game extends, giving engines more time."
    ),
    "requiresMoonTrackCompletion": (
        "Moon Tracks Must Be Completed — all three Moon tracks must be maxed before game ends. "
        "Moon investment is mandatory; plan Moon tile placement early."
    ),
    "politicalAgendasExtension:Random": (
        "Political Agendas (Random) — Turmoil party bonuses/policies randomly assigned and fixed. "
        "Study the fixed policies; some heavily favour certain strategies."
    ),
    "politicalAgendasExtension:Chairman": (
        "Political Agendas (Chairman) — the Chairman chooses party bonuses/policies each generation. "
        "Being Chairman is very powerful — compete for it."
    ),
    "removeNegativeGlobalEventsOption": (
        "No Negative Global Events — Turmoil Global Events only have neutral/positive effects. "
        "Less variance; global events are less threatening."
    ),
    "altVenusBoard": "Alt Venus Board — an alternative Venus parameter arrangement. Standard Venus principles apply.",
}


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def format_config_context(game: dict) -> str:
    """Return a comprehensive setup block for the cached system prompt.

    Includes board (special tiles + notes), active expansions with descriptions, game
    variant settings, and this game's actual milestones/awards (may differ if randomised).
    """
    board_key = game.get("boardName", "tharsis")
    expansions = game.get("expansions") or []
    variants = game.get("gameVariants") or {}
    milestones = game.get("availableMilestones") or []
    awards = game.get("availableAwards") or []

    board_info = BOARD_INFO.get(str(board_key).lower()) or BOARD_INFO.get("tharsis", {})

    lines = ["=== GAME CONFIGURATION ==="]

    board_display = board_info.get("display", str(board_key))
    lines.append(f"Board: {board_display}")
    special = board_info.get("special_tiles") or []
    if special:
        lines.append("Special spaces:")
        for t in special:
            lines.append(f"  • {t}")
    note = board_info.get("notes", "")
    if note:
        lines.append(f"Board note: {note}")

    if expansions:
        lines.append(f"\nExpansions active: {', '.join(expansions)}")
        for e in expansions:
            desc = EXPANSION_INFO.get(e)
            if desc:
                lines.append(f"  [{e}] {desc}")

    if variants:
        lines.append("\nGame variants / rule changes:")
        for key, val in variants.items():
            lookup_key = f"{key}:{val}" if key == "politicalAgendasExtension" else key
            desc = GAME_VARIANT_DESCRIPTIONS.get(lookup_key) or GAME_VARIANT_DESCRIPTIONS.get(key)
            lines.append(f"  [{key}] {desc}" if desc else f"  [{key}] = {val}")

    if milestones:
        lines.append("\nMilestones available (5 VP to claim; costs 8 MC; max 3 per game):")
        for m in milestones:
            lines.append(f"  • {m.get('name', '?')}: {m.get('description', '')}")
        lines.append("  Tip: claim as soon as you meet the requirement — being blocked loses 5 VP.")

    if awards:
        lines.append("\nAwards available (5 VP 1st / 2 VP 2nd; fund costs 8/14/20 MC; max 3 per game):")
        for a in awards:
            lines.append(f"  • {a.get('name', '?')}: {a.get('description', '?')}")
        lines.append(
            "  Tip: fund only an award you are already winning, before opponents fund it. "
            "Multiple awards can be won by the same player."
        )

    lines.append("=== END GAME CONFIGURATION ===")
    return "\n".join(lines)


def format_board_layout(board_spaces: list[dict]) -> str:
    """Static board layout for the system prompt — ocean / volcanic / bonus-land spaces.

    This is the unchanging map (placement bonuses, ocean-reserved spaces). The LIVE tile
    state and adjacency are rendered separately each turn by board.py.
    """
    if not board_spaces:
        return ""

    BONUS_ABBREV = {
        "steel": "St", "titanium": "Ti", "plant": "Pl", "card": "Cd",
        "heat": "He", "MC": "MC", "ocean": "Oc", "animal": "An",
        "microbe": "Mi", "energy": "En", "data": "Da", "science": "Sc",
        "energy production": "EP", "temperature": "Tp",
    }

    ocean_spaces: list[str] = []
    volcanic_spaces: list[str] = []
    bonus_land: list[str] = []

    for s in board_spaces:
        sid = s.get("id", "?")
        x, y = s.get("x", 0), s.get("y", 0)
        stype = s.get("t", "land")
        bonuses: list[str] = s.get("b") or []
        bonus_str = "+".join(BONUS_ABBREV.get(b, b) for b in bonuses) if bonuses else ""

        if stype in ("ocean", "cove"):
            ocean_spaces.append(f"  hex-{sid}({x},{y}):{bonus_str or '—'}")
        elif s.get("v"):
            tag = f" [{bonus_str}]" if bonus_str else ""
            volcanic_spaces.append(f"  hex-{sid}({x},{y}){tag}")
        elif bonuses:
            bonus_land.append(f"  hex-{sid}({x},{y}): {bonus_str}")

    lines = ["=== BOARD LAYOUT (static) ===",
             "Space IDs: hex-NN where NN is the ID shown during tile placement.",
             "Position (x,y): x=column (0=leftmost in row), y=row (0=top).",
             "Greenery MUST be placed adjacent to your own tile if possible.",
             "City CANNOT be adjacent to another city.",
             ""]

    if ocean_spaces:
        lines.append(f"Ocean-only spaces ({len(ocean_spaces)} total) — hex-ID(x,y):placement_bonuses:")
        row_size = 6
        for i in range(0, len(ocean_spaces), row_size):
            lines.append("  " + "  ".join(ocean_spaces[i:i + row_size]).replace("  hex-", " hex-"))
        lines.append("  → Placing ocean gives +1 TR and +2 MC to each adjacent tile owner.")

    if volcanic_spaces:
        lines.append("\nVolcanic spaces — targeted by volcanic-event cards (Lava Flows etc.):")
        lines.append("  " + ",  ".join(volcanic_spaces))

    if bonus_land:
        lines.append("\nLand spaces with placement bonuses — hex-ID(x,y): bonuses:")
        lines.extend(bonus_land)

    lines.append("\nBonus abbreviations: St=steel, Ti=titanium, Pl=plant, Cd=card, He=heat, MC=MC.")
    lines.append("=== END BOARD LAYOUT ===")
    return "\n".join(lines)
