import os

PHASES = ["action", "research", "drafting", "production", "solar"]
BOARDS = ["tharsis", "hellas", "elysium", "amazonisPlanitia", "vastitas"]
EXPANSION_FLAGS = [
    "corpEra", "venus", "colonies", "prelude", "prelude2",
    "turmoil", "community", "ares", "moon", "pathfinders", "ceo",
    "starwars", "underworld",
]
TAG_TYPES = [
    "science", "building", "space", "power", "earth", "jovian",
    "venus", "plant", "microbe", "animal", "city", "event", "wild",
]

# Special card resources (CardResource enum strings from TM server).
CARD_RESOURCE_TYPES = [
    "Animal",   # base / venus
    "Microbe",  # base
    "Science",  # base
    "Floater",  # venus
    "Asteroid", # venus
    "Fighter",  # venus
]

# Normalisation caps derived from historical logs (ceil to nearest round number).
RESOURCE_CAPS = {
    "megacredits":    130,
    "steel":           20,
    "titanium":        30,
    "plants":          40,
    "energy":          20,
    "heat":            70,
    "terraformRating": 80,
}
# Production is encoded as (value + 5) / (cap + 5) to handle the -5 minimum for MC.
PRODUCTION_CAPS = {
    "megacredits": 60,  # raw range -5..+60
    "steel":       10,
    "titanium":    10,
    "plants":      20,
    "energy":      20,
    "heat":        30,
}
CARD_RESOURCE_CAPS = {
    "Animal":   20,
    "Microbe":  20,
    "Science":  10,
    "Floater":  20,
    "Asteroid":  5,
    "Fighter":   5,
}

# ---------------------------------------------------------------------------
# STATE_DIM breakdown
# ---------------------------------------------------------------------------
#  Global        : generation(1) + temperature(1) + oxygen(1) + oceans(1) + phase_onehot(5) =  9
#  Player (self) : 7 resources + 6 production + 13 tags + 1 handSize
#                  + 6 card_resources + 1 played_count + 3 board_tiles                      = 37
#  Opponent (×1) : 7 resources + 6 production + 13 tags
#                  + 6 card_resources + 1 played_count + 3 board_tiles                      = 36
#                  (no handSize — hidden info)
#  Milestones/   : ms_self(1) + ms_total(1) + aw_self(1) + aw_total(1)                     =  4
#  Awards
#  Config        : player_count(1) + 5 board_onehot + 13 expansion_flags                   = 19
#  Total                                                                                    = 105

_SELF_DIMS = (
    len(RESOURCE_CAPS)          # 7 resources (incl. TR)
    + len(PRODUCTION_CAPS)      # 6 production
    + len(TAG_TYPES)            # 13 tags
    + 1                         # handSize
    + len(CARD_RESOURCE_TYPES)  # 6 card resources
    + 1                         # played_card_count
    + 3                         # board tiles (greenery, city, special)
)  # = 37

_OPP_DIMS = (
    len(RESOURCE_CAPS)          # 7
    + len(PRODUCTION_CAPS)      # 6
    + len(TAG_TYPES)            # 13
    + len(CARD_RESOURCE_TYPES)  # 6
    + 1                         # played_card_count
    + 3                         # board tiles
)  # = 36

STATE_DIM = (
    4 + len(PHASES)                            # global (9)
    + _SELF_DIMS                               # active player (37)
    + _OPP_DIMS                                # one opponent slot (36)
    + 4                                        # milestones + awards (4)
    + 1 + len(BOARDS) + len(EXPANSION_FLAGS)   # game config (19)
)  # = 105

ACTION_SPACE_SIZE = 128
HIDDEN_SIZES = [512, 512, 512]

# Training
LEARNING_RATE = 3e-4
BATCH_SIZE = 64
MAX_EPOCHS = 50
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
N_STEPS = 2048

# Server
PORT = int(os.getenv("PORT", "8000"))
MODEL_PATH = os.getenv("MODEL_PATH", "models/checkpoint_latest.pt")
TM_SERVER_URL = os.getenv("TM_SERVER_URL", "http://localhost:8080")
