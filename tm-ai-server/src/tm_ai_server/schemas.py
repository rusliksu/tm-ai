from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class GameContext(BaseModel):
    id: str
    phase: str
    generation: int
    oxygen: int
    temperature: int
    oceanCount: int
    boardName: str = "tharsis"
    expansions: List[str] = []


class PlayerProduction(BaseModel):
    megacredits: int
    steel: int
    titanium: int
    plants: int
    heat: int
    energy: int


class PlayerContext(BaseModel):
    id: str
    name: str
    color: str
    terraformRating: int
    megacredits: int
    steel: int
    titanium: int
    plants: int
    energy: int
    heat: int
    handSize: int
    production: PlayerProduction
    tags: Dict[str, int] = {}
    isAI: bool = False
    playedCards: List[str] = []
    playedCardCount: int = 0
    corporations: List[str] = []
    cardResources: Dict[str, Any] = {}
    boardTiles: Dict[str, int] = {}


class MoveRequestState(BaseModel):
    game: GameContext
    player: PlayerContext
    waitingFor: Optional[Dict[str, Any]] = None
    opponents: List[Dict[str, Any]] = []
    milestones: List[Dict[str, Any]] = []
    awards: List[Dict[str, Any]] = []
    board: List[Dict[str, Any]] = []


class LegalAction(BaseModel):
    action_id: str
    type: str
    title: str
    payload: Optional[Dict[str, Any]] = None


class Metadata(BaseModel):
    schema_version: int = 1


class MoveRequest(BaseModel):
    game_id: str
    player_id: str
    state: MoveRequestState
    legal_actions: List[LegalAction]
    metadata: Metadata


class MoveDebug(BaseModel):
    policy_logits: Optional[List[float]] = None
    value_estimate: Optional[float] = None


class MoveResponse(BaseModel):
    input_response: Dict[str, Any]
    debug: Optional[MoveDebug] = None


class HealthResponse(BaseModel):
    status: str = "ok"


class VersionResponse(BaseModel):
    model_version: str
    git_commit: str
    config: Dict[str, Any]
