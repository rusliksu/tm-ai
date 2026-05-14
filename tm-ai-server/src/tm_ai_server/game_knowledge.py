"""
Game knowledge database for the LLM player.

Provides:
  CARD_DB         — dict[card_name, card_entry] loaded from data/card_db.json
  format_card_context(names)          — formatted card descriptions for prompt injection
  format_game_context(board, exps)    — board + expansion context for system prompt
"""

from __future__ import annotations
import json
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Card database — loaded from extract_card_db.ts output
# ---------------------------------------------------------------------------

_DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "card_db.json"

CARD_DB: dict[str, dict] = {}
try:
    with open(_DB_PATH) as f:
        CARD_DB = json.load(f)
except Exception:
    pass  # server still works without card descriptions


def _vp_str(vp) -> str:
    if vp is None:
        return ""
    if isinstance(vp, (int, float)):
        return f" [{int(vp)} VP]"
    if isinstance(vp, dict):
        per = vp.get("per")
        if per:
            return f" [VP/resource]"
    return f" [VP]"


def format_card_context(card_names: list[str], header: str = "", max_cards: int = 30) -> str:
    """Return a compact description block for the listed cards (from CARD_DB)."""
    lines = []
    shown = 0
    for name in card_names:
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
        desc = entry.get("description", "")
        vp = _vp_str(entry.get("victoryPoints"))
        lines.append(f"  {name} ({cost_str}{tag_str}): {desc}{vp}")
        shown += 1
    if not lines:
        return ""
    prefix = f"{header}\n" if header else ""
    return prefix + "\n".join(lines)


# ---------------------------------------------------------------------------
# Board descriptions
# ---------------------------------------------------------------------------

BOARD_INFO: dict[str, dict] = {
    "tharsis": {
        "display": "Tharsis",
        "special_tiles": [
            "Noctis City — pre-placed city on the western edge; counts toward Mayor milestone.",
        ],
        "milestones": [
            "Terraformer (TR ≥ 35)",
            "Mayor (own ≥ 3 city tiles)",
            "Gardener (own ≥ 3 greenery tiles)",
            "Builder (≥ 8 building tags in play)",
            "Planner (≥ 16 cards in hand)",
        ],
        "awards": [
            "Landlord (most tiles on board)",
            "Scientist (most science tags)",
            "Banker (highest MC production)",
            "Thermalist (most heat cubes)",
            "Miner (most steel + titanium cubes)",
        ],
        "notes": "Classic board. Noctis City is already placed — useful for city adjacency scoring.",
    },
    "hellas": {
        "display": "Hellas (Southern Hemisphere)",
        "special_tiles": [
            "Hellas Ocean — special space at −6°C row; costs 6 MC to place an ocean tile here "
            "(counts as raising an ocean +1 TR); also grants you +6 heat.",
            "South Pole — space with extra heat placement bonuses (up to +3 heat).",
        ],
        "milestones": [
            "Diversifier (8 different tag types in play)",
            "Tactician (4+ cards with requirements)",
            "Polar Explorer (3+ tiles on the bottom two rows)",
            "Energizer (energy production ≥ 6)",
            "Rim Settler (3+ Jovian tags)",
        ],
        "awards": [
            "Cultivator (most greenery tiles)",
            "Magnate (most automated cards played)",
            "Space Baron (most space tags, excluding event cards)",
            "Excentric (most resources on cards)",
            "Contractor (most building tags including events)",
        ],
        "notes": (
            "Hellas rewards energy production (Energizer milestone), Jovian tags (Rim Settler), "
            "and space strategies (Space Baron award). The Hellas Ocean space is a great early "
            "ocean placement — the 6 MC cost is well below the Aquifer standard project (18 MC)."
        ),
    },
    "elysium": {
        "display": "Elysium (Eastern Hemisphere)",
        "special_tiles": [
            "Elysium Space — top-right area with high placement bonuses (up to 8 resources).",
        ],
        "milestones": [
            "Generalist (raised all 6 production tracks by at least 1)",
            "Specialist (one production track ≥ 10)",
            "Ecologist (4+ plant, animal, or microbe tags)",
            "Tycoon (15+ project cards with ≥ 1 tag each)",
            "Legend (5+ event cards played)",
        ],
        "awards": [
            "Celebrity (12+ cards with cost ≥ 20 MC)",
            "Industrialist (most steel + energy resources on cards)",
            "Desert Settler (most tiles in the bottom 3 rows)",
            "Estate Dealer (most tiles adjacent to ocean tiles)",
            "Benefactor (TR ≥ 40 at end of game)",
        ],
        "notes": (
            "Elysium rewards versatile engines (Generalist), event chains (Legend milestone, "
            "event-based corporations), and expensive high-impact cards (Celebrity award). "
            "Estate Dealer makes ocean adjacency extra valuable."
        ),
    },
    "arabia terra": {
        "display": "Arabia Terra",
        "special_tiles": [
            "Arsia Mons — gives extra plant resources on placement.",
            "Multiple ocean-only spaces clustered together.",
        ],
        "milestones": [
            "Economizer (energy production ≥ 3 AND heat production ≥ 3)",
            "Pioneer (2+ colony tiles, if Colonies expansion active)",
            "Land Specialist (6+ non-ocean tiles placed)",
            "Martian (5+ Mars tags in play)",
            "Terran (5+ Earth tags in play)",
        ],
        "awards": [
            "Cosmic Settler (most colony markers, or tiles in bottom rows)",
            "Botanist (most plant resources on cards)",
            "Promoter (most cards with special resource types)",
            "Zoologist (most animal resources)",
            "Manufacturer (most steel + titanium resources on cards)",
        ],
        "notes": "Fan-designed board. Rich in ocean spots; favours plant and animal engines.",
    },
    "vastitas borealis": {
        "display": "Vastitas Borealis",
        "special_tiles": [
            "Restricted zones with high placement bonuses on the outer edges.",
        ],
        "milestones": [
            "Electrician (energy production ≥ 4)",
            "Smith (produce both steel and titanium ≥ 2 each)",
            "Tradesman (3+ different resource types on cards)",
            "Irrigator (own ≥ 3 ocean tiles)",
            "Capitalist (MC production ≥ 15)",
        ],
        "awards": [
            "Forecaster (most tags including wild)",
            "Edgedancer (most tiles in the outer ring)",
            "Visionary (most science tags)",
            "Naturalist (most plant resources)",
            "Voyager (most Jovian tags + space tags)",
        ],
        "notes": "Fan board focused on resource diversity and production engines.",
    },
    "t. cimmeria": {
        "display": "Terra Cimmeria",
        "special_tiles": [
            "Multiple mountain spaces (restricted); bonus resources on surrounding areas.",
        ],
        "milestones": [
            "Collector (8+ resource markers total on cards)",
            "Firestarter (4+ temperature increases contributed)",
            "Terra Pioneer (own tile in north + south regions)",
            "Spacefarer (4+ Jovian or space tags)",
            "Gambler (4+ event cards played)",
        ],
        "awards": [
            "Biologist (most microbe + animal + plant tags)",
            "Incorporator (most blue active cards)",
            "Politician (highest TR at game end — Turmoil style)",
            "Urbanist (most city tiles)",
            "Warmonger (most event cards)",
        ],
        "notes": "Fan board with diverse engine paths; Gambler/Warmonger reward event-heavy strategies.",
    },
    "utopia planitia": {
        "display": "Utopia Planitia",
        "special_tiles": [],
        "milestones": [
            "Land Specialist (6+ non-ocean tiles placed)",
            "Pioneer (2+ colony tiles)",
            "Tradesman (3+ different resource types on cards)",
            "Smith (produce steel ≥ 2 AND titanium ≥ 2)",
            "Researcher (5+ science tags)",
        ],
        "awards": [
            "Edgedancer (most tiles in outer ring)",
            "Investor (highest MC production)",
            "Botanist (most plant resources on cards)",
            "Incorporator (most blue active cards)",
            "Metropolist (most city tiles)",
        ],
        "notes": "Balanced fan board; Metropolist award makes city building rewarding.",
    },
    "vastitas borealis nova": {
        "display": "Vastitas Borealis Nova",
        "special_tiles": [],
        "milestones": [
            "Agronomist (5+ plant production)",
            "Spacefarer (V. Spacefarer — 4+ space tags)",
            "Geologist (3+ tiles on special spaces)",
            "Engineer (3+ blue cards in play)",
            "Farmer (5+ plant resources)",
        ],
        "awards": [
            "Traveller (most tiles on the board)",
            "Landscaper (most greenery tiles)",
            "Highlander (tiles in the north + south regions)",
            "Promoter (most special resource cards)",
            "Blacksmith (most steel + titanium resources)",
        ],
        "notes": "Fan board variant; rewards plant production and blue-card engines.",
    },
    "terra cimmeria nova": {
        "display": "Terra Cimmeria Nova",
        "special_tiles": [],
        "milestones": [
            "Planetologist (4+ different tag types)",
            "Architect (4+ blue active cards)",
            "Coastguard (3+ ocean tiles on coast spaces)",
            "Forester (C. Forester — 4+ greenery tiles)",
            "Fundraiser (MC production ≥ 8)",
        ],
        "awards": [
            "Electrician (most energy production)",
            "Founder (highest TR at end)",
            "Mogul (most unique tags including wild)",
            "Zoologist (most animal resources)",
            "Forecaster (most tags including wild)",
        ],
        "notes": "Fan board variant.",
    },
    "amazonis p.": {
        "display": "Amazonis Planitia",
        "special_tiles": [
            "Amazonis Planitia marked zones — extra placement bonuses.",
        ],
        "milestones": [
            "Colonizer (2+ colony tiles)",
            "Forester (4+ greenery tiles)",
            "Minimalist (low hand size at milestone claim — 3 or fewer)",
            "Terran (5+ Earth tags)",
            "Tropicalist (4+ plant production + plant resources)",
        ],
        "awards": [
            "Curator (most unique tag types)",
            "Engineer (most blue active cards)",
            "Promoter (most special resource cards)",
            "Tourist (most VP on played cards)",
            "Zoologist (most animal resources)",
        ],
        "notes": "Fan board; Minimalist milestone rewards playing cards quickly then going lean.",
    },
    "Hollandia": {
        "display": "Hollandia",
        "special_tiles": [],
        "milestones": [],
        "awards": [],
        "notes": "Community board with custom milestone/award rules.",
    },
}


def format_game_context(board_name: str, expansions: list[str]) -> str:
    """Return a prompt block describing the active board and expansions."""
    board_key = board_name.lower().strip()
    info = BOARD_INFO.get(board_key) or BOARD_INFO.get("tharsis")
    assert info is not None

    lines = [
        f"=== GAME CONFIGURATION ===",
        f"Board: {info['display']}",
    ]
    if info["special_tiles"]:
        lines.append("Special tiles:")
        for t in info["special_tiles"]:
            lines.append(f"  • {t}")
    if info["milestones"]:
        lines.append(f"Milestones: {' | '.join(info['milestones'])}")
    if info["awards"]:
        lines.append(f"Awards: {' | '.join(info['awards'])}")
    if info["notes"]:
        lines.append(f"Note: {info['notes']}")

    if expansions:
        exp_descs = [EXPANSION_INFO.get(e, e) for e in expansions]
        lines.append(f"Active expansions: {', '.join(expansions)}")
        for e, d in zip(expansions, exp_descs):
            lines.append(f"  {e}: {d}")

    lines.append("=== END GAME CONFIGURATION ===")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Expansion descriptions
# ---------------------------------------------------------------------------

EXPANSION_INFO: dict[str, str] = {
    "venus": (
        "Venus Next — adds the Venus parameter track (−20° to +10°, 49 steps); "
        "raises give +1 TR. Adds Venus cards and corporations. "
        "Hoverlord milestone (7+ Venus tags). Venuphile award (most Venus tags). "
        "Special board tiles: Dawn City, Luna Metropolis, Maxwell Base, Stratopolis."
    ),
    "colonies": (
        "Colonies — adds Colony tiles (Moon, Ganymede, Titan, Callisto, etc.). "
        "Players build colony markers and trade for resources every generation. "
        "Titan colony gives titanium. Ganymede gives cards. Enceladus gives microbes. "
        "Trade fleets are limited — timing your trade matters."
    ),
    "prelude": (
        "Prelude — each player plays 2 Prelude cards before generation 1, "
        "giving a strong engine head-start. Preludes can give production boosts, "
        "free cards, TR increases, or resources. Prelude choice dramatically shapes "
        "the opening strategy."
    ),
    "prelude2": (
        "Prelude 2 — second set of Prelude cards, same mechanic as Prelude."
    ),
    "turmoil": (
        "Turmoil — adds a Politics track with five Parties (Mars First, Scientists, "
        "Greens, Unity, Reds). Dominant party at generation end applies a Global Event "
        "affecting all players. Delegates placed each generation influence party control. "
        "Chairman bonus: −1 TR for non-ruling party players. Careful delegate management "
        "can give ongoing TR and MC benefits."
    ),
    "moon": (
        "The Moon — adds a Moon mini-board with three tracks: Colony Rate, Mining Rate, "
        "Road Network. Cards place tiles on the Moon and raise these tracks. "
        "Lunarchitect milestone (3+ Moon tiles). One Giant Step milestone."
    ),
    "pathfinders": (
        "Pathfinders — adds Data and Preservation tags. Planetary track bonuses "
        "for each planet in the solar system. New starting conditions and corporations."
    ),
    "underworld": (
        "Underworld — adds underground excavation mechanic. Players dig for resources "
        "and artifacts on special underground spaces. Adds corruption tokens and "
        "new Risktaker/Tunneler milestones."
    ),
    "ares": (
        "Ares — adds Hazard tiles (dust storms, erosion) that slow terraforming "
        "but give adjacency bonuses when mitigated. Networker milestone, "
        "Entrepreneur/Rugged awards."
    ),
    "corpera": (
        "Corporate Era — players start at 0 production for all resources "
        "(instead of 1 for base game). More project cards and corporations. "
        "Stronger production ramp-up is essential."
    ),
}
