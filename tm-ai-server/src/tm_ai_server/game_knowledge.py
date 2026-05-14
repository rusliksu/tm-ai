"""
Game knowledge database for the LLM player.

Provides:
  CARD_DB                     — dict[name, entry] loaded from data/card_db.json
  format_card_context(names)  — formatted card descriptions for prompt injection
  format_config_context(game) — comprehensive setup block: board + expansions +
                                 game variants + actual milestones/awards for this game
"""

from __future__ import annotations
import json
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
        return " [VP/resource]"
    return " [VP]"


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
# Board descriptions — special tiles and strategic notes only
# (Milestones/awards come from the live game state, not hardcoded here)
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
        "special_tiles": [
            "Outer ring spaces have elevated placement bonuses.",
        ],
        "notes": "Fan board focused on resource diversity and production engines.",
    },
    "t. cimmeria": {
        "display": "Terra Cimmeria",
        "special_tiles": [
            "Multiple mountain (restricted) spaces; bonus resources on surrounding hexes.",
        ],
        "notes": "Fan board; Gambler/Warmonger milestones/awards reward event-heavy strategies.",
    },
    "utopia planitia": {
        "display": "Utopia Planitia",
        "special_tiles": [],
        "notes": "Balanced fan board; Metropolist award makes city building rewarding.",
    },
    "vastitas borealis nova": {
        "display": "Vastitas Borealis Nova",
        "special_tiles": [],
        "notes": "Fan board variant; rewards plant production and blue-card engines.",
    },
    "terra cimmeria nova": {
        "display": "Terra Cimmeria Nova",
        "special_tiles": [],
        "notes": "Fan board variant.",
    },
    "amazonis p.": {
        "display": "Amazonis Planitia",
        "special_tiles": [
            "Amazonis zones with extra placement bonuses.",
        ],
        "notes": "Fan board; Minimalist milestone rewards playing cards quickly then going lean.",
    },
    "Hollandia": {
        "display": "Hollandia",
        "special_tiles": [],
        "notes": "Community board with custom rules.",
    },
}


# ---------------------------------------------------------------------------
# Expansion descriptions
# ---------------------------------------------------------------------------

EXPANSION_INFO: dict[str, str] = {
    "venus": (
        "Venus Next — adds a Venus parameter track (−20° to +10°, 49 steps); each raise = +1 TR. "
        "New Venus-tag cards and corporations. "
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
    "prelude2": (
        "Prelude 2 — second set of Prelude cards, same mechanic as original Prelude."
    ),
    "turmoil": (
        "Turmoil — Politics track with five Parties (Mars First, Scientists, Greens, Unity, Reds). "
        "Dominant party at generation end applies a Global Event. "
        "Delegates placed each generation influence party control. "
        "Ruling party bonus affects all players; Chairman position gives extra TR each generation."
    ),
    "moon": (
        "The Moon — mini Moon board with Colony Rate / Mining Rate / Road Network tracks. "
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
    "corpera": (
        "Corporate Era — players start at 0 production (vs 1 in Base). "
        "More cards and corporations. Strong production ramp-up is essential."
    ),
}


# ---------------------------------------------------------------------------
# Game variant descriptions — all configuration options with strategic impact
# ---------------------------------------------------------------------------

GAME_VARIANT_DESCRIPTIONS: dict[str, str] = {
    "draftVariant": (
        "Research Phase Draft — instead of drawing 4 cards and keeping any, players pass cards "
        "around the table (like a card draft). You see all 4 cards but keep only 1 before passing. "
        "Strategic impact: CARD DENIAL is possible — withhold cards that synergise with an "
        "opponent's engine even if they aren't your best pick."
    ),
    "initialDraftVariant": (
        "Initial Cards Draft — the 10 starting project cards are drafted rather than dealt directly. "
        "You pass cards around and pick sequentially. "
        "Strategic impact: you can deny opponent key synergy cards; your initial hand is more curated."
    ),
    "preludeDraftVariant": (
        "Prelude Draft — prelude cards are drafted. "
        "Strategic impact: you can block strong engine-boosting preludes from opponents."
    ),
    "ceosDraftVariant": (
        "CEO Draft — CEO cards are drafted. "
        "Strategic impact: pick the CEO that best matches your engine; deny powerful ones."
    ),
    "twoCorpsVariant": (
        "Two Corporations Variant — each player starts with 2 corporations (plays both). "
        "Strategic impact: doubled starting resources and combined corporation abilities. "
        "Synergy between the two corps is crucial — look for complementary abilities."
    ),
    "solarPhaseOption": (
        "World Government Terraforming (Solar Phase) — at the start of each generation, "
        "one player (rotating) acts as World Government and raises one global parameter for free "
        "(temperature +2°C, oxygen +1%, place ocean, or raise Venus). No TR gained. "
        "Strategic impact: SIGNIFICANTLY speeds up the game — expect 2-3 fewer generations than normal. "
        "Accelerate your engine early; slow starts are punished heavily. "
        "When it's your turn as World Government, raise whichever parameter benefits your strategy most."
    ),
    "soloTR": (
        "Solo Mode (TR 63 Victory) — must reach Terraform Rating 63 by game end to win. "
        "Standard VP scoring is replaced by a binary win/lose condition. "
        "Strategic impact: maximise TR gain above all else; card VP matters much less."
    ),
    "randomMA": (
        "Randomized Milestones & Awards — the 5 milestones and 5 awards are randomly selected "
        "rather than board-specific. The actual milestones and awards are listed below. "
        "Study them carefully — your engine should target at least 1-2 milestones and compete "
        "for 1-2 awards."
    ),
    "modularMA": (
        "Modular Milestones & Awards — milestones/awards drawn from the expanded modular pool "
        "(broader variety including fan-designed ones). See the actual list below."
    ),
    "requiresVenusTrackCompletion": (
        "Venus Must Be Completed — the game does not end until Venus reaches +10°C (max). "
        "Strategic impact: invest in Venus cards even if the track isn't your primary focus; "
        "the game extends until Venus is done, giving more time for engines to develop."
    ),
    "requiresMoonTrackCompletion": (
        "Moon Tracks Must Be Completed — all three Moon tracks must be maxed before game ends. "
        "Strategic impact: Moon investment is mandatory; plan Moon tile placement early."
    ),
    "politicalAgendasExtension:Random": (
        "Political Agendas (Random) — Turmoil party bonuses and policies are randomly assigned "
        "at the start of the game and remain fixed. "
        "Strategic impact: study the fixed policies; some may heavily favour certain strategies "
        "(e.g. Kelvinists policy giving +2 MC for heat production is very powerful)."
    ),
    "politicalAgendasExtension:Chairman": (
        "Political Agendas (Chairman) — the Chairman chooses party bonuses/policies each generation. "
        "Strategic impact: being Chairman is very powerful — compete for Chairman position."
    ),
    "removeNegativeGlobalEventsOption": (
        "No Negative Global Events — Turmoil Global Events only have neutral or positive effects "
        "(negative effects are removed). "
        "Strategic impact: less variance; global events are less threatening to your plans."
    ),
    "altVenusBoard": (
        "Alt Venus Board — an alternative Venus parameter arrangement. "
        "Standard strategic principles still apply for Venus-tag engines."
    ),
}


# ---------------------------------------------------------------------------
# Main format functions
# ---------------------------------------------------------------------------

def format_config_context(game: dict) -> str:
    """Return a comprehensive setup block for the initial LLM system prompt.

    Includes: board (special tiles + notes), active expansions with descriptions,
    game variant settings, and the ACTUAL milestones/awards for this specific game
    (which may differ from board defaults if randomised).
    """
    board_key = game.get("boardName", "tharsis")
    expansions = game.get("expansions") or []
    variants = game.get("gameVariants") or {}
    milestones = game.get("availableMilestones") or []
    awards = game.get("availableAwards") or []

    board_info = BOARD_INFO.get(str(board_key).lower()) or BOARD_INFO.get("tharsis", {})

    lines = ["=== GAME CONFIGURATION ==="]

    # --- Board ---
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

    # --- Active expansions ---
    if expansions:
        lines.append(f"\nExpansions active: {', '.join(expansions)}")
        for e in expansions:
            desc = EXPANSION_INFO.get(e)
            if desc:
                lines.append(f"  [{e}] {desc}")

    # --- Game variants ---
    if variants:
        lines.append("\nGame variants / rule changes:")
        for key, val in variants.items():
            # Build a lookup key — for politicalAgendasExtension include the value
            if key == "politicalAgendasExtension":
                lookup_key = f"{key}:{val}"
            else:
                lookup_key = key
            desc = GAME_VARIANT_DESCRIPTIONS.get(lookup_key) or GAME_VARIANT_DESCRIPTIONS.get(key)
            if desc:
                lines.append(f"  [{key}] {desc}")
            else:
                lines.append(f"  [{key}] = {val}")

    # --- Milestones ---
    if milestones:
        lines.append("\nMilestones available (5 VP to claim; costs 8 MC; max 3 per game):")
        for m in milestones:
            name = m.get("name", "?")
            desc = m.get("description", "")
            lines.append(f"  • {name}: {desc}")
        lines.append(
            "  Tip: claim early if you meet the requirement — being blocked costs 0 MC "
            "but losing 5 VP is enormous."
        )

    # --- Awards ---
    if awards:
        lines.append("\nAwards available (5 VP 1st / 2 VP 2nd; fund costs 8/14/20 MC; max 3 per game):")
        for a in awards:
            name = a.get("name", "?")
            desc = a.get("description", "?")
            lines.append(f"  • {name}: {desc}")
        lines.append(
            "  Tip: fund an award you are already winning, before opponents can fund it. "
            "Funding late is wasteful (20 MC for 3rd). "
            "Multiple awards can be won by the same player."
        )

    lines.append("=== END GAME CONFIGURATION ===")
    return "\n".join(lines)


def format_board_layout(board_spaces: list[dict]) -> str:
    """Generate a compact board layout for the initial LLM system prompt.

    board_spaces is the boardSpaces array from the AI request state — each entry has:
      id, x, y, t (spaceType), b (bonus names list), v (volcanic bool, optional),
      tile (placed tile type, optional), pc (playerColor, optional).
    """
    if not board_spaces:
        return ""

    BONUS_ABBREV = {
        "steel": "St", "titanium": "Ti", "plant": "Pl", "card": "Cd",
        "heat": "He", "MC": "MC", "ocean": "Oc", "animal": "An",
        "microbe": "Mi", "energy": "En", "data": "Da", "science": "Sc",
        "energy production": "EP", "temperature": "Tp",
    }

    # Separate spaces by type
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
            tag = f"{bonus_str}" if bonus_str else "—"
            ocean_spaces.append(f"  hex-{sid}({x},{y}):{tag}")
        elif s.get("v"):
            tag = f" [{bonus_str}]" if bonus_str else ""
            volcanic_spaces.append(f"  hex-{sid}({x},{y}){tag}")
        elif bonuses:
            bonus_land.append(f"  hex-{sid}({x},{y}): {bonus_str}")

    lines = ["=== BOARD LAYOUT ===",
             "Space IDs: hex-NN where NN is the ID shown during tile placement.",
             "Position (x,y): x=column (0=leftmost in row), y=row (0=top).",
             "Hex adjacency: spaces are adjacent if they share an edge (differ by at most 1 in",
             "  x and y, following the offset hex grid pattern).",
             "Greenery MUST be placed adjacent to your own tile if possible.",
             "City CANNOT be adjacent to another city.",
             ""]

    if ocean_spaces:
        lines.append(f"Ocean-only spaces ({len(ocean_spaces)} total) — format: hex-ID(x,y):placement_bonuses:")
        # Show in rows for readability
        row_size = 6
        for i in range(0, len(ocean_spaces), row_size):
            lines.append("  " + "  ".join(ocean_spaces[i:i + row_size]).replace("  hex-", " hex-"))
        lines.append("  → Placing ocean gives +1 TR and +2 MC to each adjacent tile owner.")

    if volcanic_spaces:
        lines.append(f"\nVolcanic spaces — targeted by volcanic-event cards (Lava Flows etc.):")
        lines.append("  " + ",  ".join(volcanic_spaces))

    if bonus_land:
        lines.append(f"\nLand spaces with placement bonuses — format: hex-ID(x,y): bonuses:")
        for entry in bonus_land:
            lines.append(entry)

    lines.append("\nBonus abbreviations: St=steel, Ti=titanium, Pl=plant, Cd=card, He=heat, MC=MC.")
    lines.append("=== END BOARD LAYOUT ===")
    return "\n".join(lines)


def format_game_context(board_name: str, expansions: list[str]) -> str:
    """Legacy/fallback: board + expansions only (no milestones/awards/variants).

    Prefer format_config_context(game_state) when the full game dict is available.
    """
    board_key = str(board_name).lower()
    board_info = BOARD_INFO.get(board_key) or BOARD_INFO.get("tharsis", {})
    lines = [
        "=== GAME CONFIGURATION ===",
        f"Board: {board_info.get('display', board_name)}",
    ]
    special = board_info.get("special_tiles") or []
    for t in special:
        lines.append(f"  • {t}")
    note = board_info.get("notes", "")
    if note:
        lines.append(f"Note: {note}")
    if expansions:
        lines.append(f"Expansions: {', '.join(expansions)}")
        for e in expansions:
            desc = EXPANSION_INFO.get(e)
            if desc:
                lines.append(f"  [{e}] {desc}")
    lines.append("=== END GAME CONFIGURATION ===")
    return "\n".join(lines)
