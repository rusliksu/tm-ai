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

# Per-card resource vocabulary: card display name → resource type string.
# 199 entries covering all cards that can hold resources.
CARD_RESOURCE_VOCAB: dict[str, str] = {
    "Adhai High Orbit Constructions": "Orbital",
    "Aerial Mappers": "Floater",
    "Aerobraked Ammonia Asteroid": "Microbe",
    "Aeron Genomics": "Animal",
    "Air Raid": "Floater",
    "Air-Scrapping Expedition": "Floater",
    "Airliners": "Floater",
    "Ancient Shipyards": "Resource cube",
    "Anthozoa": "Animal",
    "Ants": "Microbe",
    "Applied Science": "Science",
    "Arborist Collective": "Activist",
    "Arklight": "Animal",
    "Asteroid Deflection System": "Asteroid",
    "Asteroid Hollowing": "Asteroid",
    "Asteroid Rights": "Asteroid",
    "AstroDrill": "Asteroid",
    "Atmo Collectors": "Floater",
    "Atmoscoop": "Floater",
    "Atmospheric Enhancers": "Floater",
    "Aurorai": "Data",
    "Bactoviral Research": "Microbe",
    "Bio Printing Facility": "Animal",
    "Bio-Fertilizer Facility": "Microbe",
    "Bio-Sol": "Microbe",
    "Biobatteries": "Microbe",
    "Bioengineering Enclosure": "Animal",
    "Birds": "Animal",
    "Board of Directors": "Director",
    "Botanical Experience": "Data",
    "Breeding Farms": "Animal",
    "Carbon Nanosystems": "Graphene",
    "Cassini Station": "Floater",
    "Celestic": "Floater",
    "Clone Troopers (II)": "Clone Trooper",
    "Cloud City (V)": "Floater",
    "Cloud Tourism": "Floater",
    "Cloud Vortex Outpost": "Floater",
    "Collegium Copernicus": "Data",
    "Comet Aiming": "Asteroid",
    "Communication Center": "Data",
    "Controlled Bloom": "Microbe",
    "Copernicus Tower": "Science",
    "Crashlanding": "Data",
    "Cryptocurrency": "Data",
    "Cyanobacteria": "Microbe",
    "Darkside Incubation Plant": "Microbe",
    "Darkside Observatory": "Science",
    "Data Leak": "Data",
    "Decomposers": "Microbe",
    "Demetron Labs": "Data",
    "Designed Organisms": "Microbe",
    "Desperate Measures": "Resource cube",
    "Deuterium Export": "Floater",
    "Directed Impactors": "Asteroid",
    "Dirigibles": "Floater",
    "Early Expedition": "Data",
    "EcoTec": "Microbe",
    "Ecological Survey": "Animal",
    "Ecological Zone": "Animal",
    "Ecological Zone:ares": "Animal",
    "Ecology Research": "Animal",
    "Economic Espionage": "Data",
    "Eos Chasma National Park": "Animal",
    "Export Convoy": "Microbe",
    "Extractor Balloons": "Floater",
    "Extreme-Cold Fungus": "Microbe",
    "Extremophiles": "Microbe",
    "Fish": "Animal",
    "Floater Leasing": "Floater",
    "Floater Prototypes": "Floater",
    "Floater Technology": "Floater",
    "Floater-Urbanism": "Venusian Habitat",
    "Floating Habs": "Floater",
    "Floating Refinery": "Floater",
    "Floating Trade Hub": "Floater",
    "Forced Precipitation": "Floater",
    "Forest Moon (VI)": "Animal",
    "Freyja Biodomes": "Microbe",
    "GHG Producing Bacteria": "Microbe",
    "GHG Shipment": "Floater",
    "Hecate Speditions": "Supply Chain",
    "Henkei Genetics": "Microbe",
    "Herbivores": "Animal",
    "Hospitals": "Disease",
    "Hydrogen to Venus": "Floater",
    "Hyperspace Drive Prototype": "Fighter",
    "Icy Impactors": "Asteroid",
    "Imported Hydrogen": "Microbe",
    "Imported Nitrogen": "Microbe",
    "Imported Nutrients": "Microbe",
    "Intragen Sanctuary Headquarters": "Animal",
    "Investigative Journalism": "Journalism",
    "Jet Stream Microscrappers": "Floater",
    "Jovian Lanterns": "Floater",
    "Jupiter Floating Station": "Floater",
    "Keplertec": "Fighter",
    "Kuiper Cooperative": "Asteroid",
    "Large Convoy": "Animal",
    "Livestock": "Animal",
    "Local Heat Trapping": "Animal",
    "Local Shading": "Floater",
    "Luna Archives": "Science",
    "Lunar Observation Post": "Data",
    "Main Belt Asteroids": "Asteroid",
    "Martian Culture": "Data",
    "Martian Express": "Ware",
    "Martian Nature Wonders": "Data",
    "Martian Repository": "Data",
    "Martian Zoo": "Animal",
    "Meat Industry": "Animal",
    "Micro-Geodesics": "Microbe",
    "Mind Set Mars": "Agenda",
    "Mining Market Insider": "Data",
    "Mohole Lake": "Microbe",
    "Nanotech Industries": "Science",
    "Neptunian Power Consultants": "Hydroelectric resource",
    "Nitrite Reducing Bacteria": "Microbe",
    "Nitrogen from Titan": "Floater",
    "Nobel Labs": "Microbe",
    "Ocean Sanctuary": "Animal",
    "Olympus Conference": "Science",
    "Oumuamua Type Object Survey": "Data",
    "Penguins": "Animal",
    "Personal Spacecruiser": "Fighter",
    "Pets": "Animal",
    "Pharmacy Union": "Disease",
    "Physics Complex": "Science",
    "Pollinators": "Animal",
    "Predators": "Animal",
    "Pride of the Earth Arkship": "Science",
    "Pristar": "Preservation",
    "Private Military Contractor": "Fighter",
    "Processor Factory": "Data",
    "Protected Habitats": "Animal",
    "Psychrophiles": "Microbe",
    "Quill": "Floater",
    "Recyclon": "Microbe",
    "Red Spot Observatory": "Floater",
    "Refugee Camps": "Camp",
    "Regolith Eaters": "Microbe",
    "Research & Development Hub": "Data",
    "Rey ... Skywalker?! (IX)": "Resource cube",
    "Robin Haulings": "Floater",
    "Rotator Impacts": "Asteroid",
    "Rust Eating Bacteria": "Microbe",
    "Saturn Surfing": "Floater",
    "Search For Life": "Science",
    "Search for Life Underground": "Science",
    "Secret Labs": "Microbe",
    "Security Fleet": "Fighter",
    "Small Animals": "Animal",
    "Soil Enrichment": "Microbe",
    "Soil Export": "Floater",
    "SolBank": "Data",
    "Solar Storm": "Data",
    "Solarpedia": "Data",
    "Soylent Seedling Systems": "Seed",
    "Space Debris Cleaning Operation": "Data",
    "Space Privateers": "Fighter",
    "Space Wargames": "Fighter",
    "Spire": "Science",
    "Splice": "Microbe",
    "Stem Field Subsidies": "Data",
    "Stormcraft Incorporated": "Floater",
    "Stratopolis": "Floater",
    "Stratospheric Birds": "Animal",
    "Stratospheric Expedition": "Floater",
    "Sub-zero Salt Fish": "Animal",
    "Sulphur-Eating Bacteria": "Microbe",
    "Symbiotic Fungus": "Microbe",
    "Tardigrades": "Microbe",
    "Terraforming Robots": "Specialized Robot",
    "The Archaic Foundation Institute": "Resource cube",
    "The Darkside of The Moon Syndicate": "Syndicate Fleet",
    "Thermophiles": "Microbe",
    "Think Tank": "Data",
    "Thiolava Vents": "Microbe",
    "Titan Air-scrapping": "Floater",
    "Titan Floating Launch-pad": "Floater",
    "Titan Manufacturing Colony": "Tool",
    "Titan Shuttles": "Floater",
    "Topsoil Contract": "Microbe",
    "Underground Habitat": "Animal",
    "Urban Decomposers": "Microbe",
    "Valuable Gases": "Floater",
    "Valuable Gases:Pathfinders": "Floater",
    "Venera Base": "Floater",
    "Venus Shuttles": "Floater",
    "Venus Soils": "Microbe",
    "Venusian Animals": "Animal",
    "Venusian Insects": "Microbe",
    "Venusian Plants": "Microbe",
    "Vermin": "Animal",
    "Viral Enhancers": "Microbe",
    "Virus": "Animal",
    "Weather Balloons": "Floater",
    "Whales": "Animal",
    "Will": "Animal",
}

# Card resource resource count cap (most cards rarely exceed 20 in practice).
CARD_RESOURCE_CAP = 20

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

# ---------------------------------------------------------------------------
# STATE_DIM breakdown
# ---------------------------------------------------------------------------
#  Global        : generation(1) + temperature(1) + oxygen(1) + oceans(1) + phase_onehot(5)     =   9
#  Player (self) : 7 resources + 6 production + 13 tags + 1 handSize
#                  + 199 card_resources + 1 played_count + 3 board_tiles                        = 230
#  Opponent (×1) : 7 resources + 6 production + 13 tags + 1 handSize
#                  + 199 card_resources + 1 played_count + 3 board_tiles                        = 230
#  Milestones/   : ms_self(1) + ms_total(1) + aw_self(1) + aw_total(1)                         =   4
#  Awards
#  Config        : player_count(1) + 5 board_onehot + 13 expansion_flags                       =  19
#  Total                                                                                        = 492

_PLAYER_DIMS = (
    len(RESOURCE_CAPS)            # 7 resources (incl. TR)
    + len(PRODUCTION_CAPS)        # 6 production
    + len(TAG_TYPES)              # 13 tags
    + 1                           # handSize
    + len(CARD_RESOURCE_VOCAB)    # 199 per-card resource slots
    + 1                           # played_card_count
    + 3                           # board tiles (greenery, city, special)
)  # = 230

STATE_DIM = (
    4 + len(PHASES)                            # global (9)
    + _PLAYER_DIMS                             # active player (230)
    + _PLAYER_DIMS                             # one opponent slot (230)
    + 4                                        # milestones + awards (4)
    + 1 + len(BOARDS) + len(EXPANSION_FLAGS)   # game config (19)
)  # = 492

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
