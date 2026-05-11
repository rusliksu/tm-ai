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

# Input feature dimension (essential features):
#   global:  generation(1) + temperature(1) + oxygen(1) + oceans(1) + phase_onehot(5) = 9
#   player:  7 resources + 6 production + 13 tags + 1 handSize = 27
#   config:  1 player_count + 5 board_onehot + 13 expansion_flags = 19
STATE_DIM = (
    4 + len(PHASES)                          # global
    + 7 + 6 + len(TAG_TYPES) + 1            # player
    + 1 + len(BOARDS) + len(EXPANSION_FLAGS) # game config
)  # = 55

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
