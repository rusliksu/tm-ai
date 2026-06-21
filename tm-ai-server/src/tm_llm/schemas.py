"""Pydantic models mirroring the TM server's camelCase AI request/response payloads."""
from __future__ import annotations
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
    availableMilestones: List[Dict[str, str]] = []
    availableAwards: List[Dict[str, str]] = []
    gameVariants: Dict[str, Any] = {}
    recentLog: List[str] = []


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
    cardsInHand: List[str] = []
    victoryPoints: Optional[int] = None


class MoveRequestState(BaseModel):
    game: GameContext
    player: PlayerContext
    waitingFor: Optional[Dict[str, Any]] = None
    opponents: List[Dict[str, Any]] = []
    milestones: List[Dict[str, Any]] = []
    awards: List[Dict[str, Any]] = []
    board: List[Dict[str, Any]] = []
    boardSpaces: List[Dict[str, Any]] = []


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
    last_error: Optional[str] = None


class MoveResponse(BaseModel):
    input_response: Dict[str, Any]
    debug: Optional[Dict[str, Any]] = None


class PlayerRegisterRequest(BaseModel):
    player_id: str
    game_id: str
    model: Optional[str] = None  # None → use OPENROUTER_MODEL default


class PlayerRegisterResponse(BaseModel):
    ok: bool
    player_id: str
    model: str


class HealthResponse(BaseModel):
    status: str = "ok"


class VersionResponse(BaseModel):
    model_version: str
    git_commit: str
    config: Dict[str, Any]
