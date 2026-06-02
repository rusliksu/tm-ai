"""FastAPI app for the LLM AI server.

Endpoints:
  GET  /health            — liveness
  GET  /version           — git commit + config
  POST /move              — pick a move for the active AI player
  POST /player/register   — assign a model to a player before the game starts
  POST /game-done         — log the per-player token/cost summary for a finished game
"""
from __future__ import annotations
import logging
import subprocess
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

_app_logger = logging.getLogger("tm_llm")
_app_logger.setLevel(logging.INFO)
if not _app_logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s:     %(name)s - %(message)s"))
    _app_logger.addHandler(_h)
    _app_logger.propagate = False

from . import config, registry
from .engine import select_action_llm
from .openrouter import ensure_client
from .schemas import (
    HealthResponse, MoveRequest, MoveResponse,
    PlayerRegisterRequest, PlayerRegisterResponse, VersionResponse,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if config.OPENROUTER_API_KEY:
        ensure_client()
        logger.info("OpenRouter client ready (default model: %s)", config.OPENROUTER_MODEL)
    else:
        logger.warning("OPENROUTER_API_KEY is not set — /move calls will fail")
    registry.prune_stale_state()
    try:
        yield
    finally:
        registry.save_all_active_players()


app = FastAPI(title="TM LLM AI Server", version="0.2.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse()


@app.get("/version", response_model=VersionResponse)
async def version():
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_commit = "unknown"
    return VersionResponse(
        model_version="0.2.0",
        git_commit=git_commit,
        config={"default_model": config.OPENROUTER_MODEL},
    )


@app.post("/move", response_model=MoveResponse)
async def move(request: MoveRequest):
    waiting_for = request.state.waitingFor
    if waiting_for is None:
        raise HTTPException(status_code=400, detail="state.waitingFor is required")

    state = request.state.model_dump()
    input_response, debug = select_action_llm(
        state, waiting_for,
        game_id=request.game_id,
        player_id=request.player_id,
        last_error=request.last_error,
    )
    return MoveResponse(input_response=input_response, debug=debug or None)


@app.post("/player/register", response_model=PlayerRegisterResponse)
async def player_register(body: PlayerRegisterRequest):
    """Register an AI player with a specific model before the game starts (once per player)."""
    player = registry.register_player(body.player_id, body.game_id, body.model)
    return PlayerRegisterResponse(ok=True, player_id=player.player_id, model=player.model)


@app.post("/game-done")
async def game_done(body: dict):
    """Log the final token summary for a completed game and clean up its state files."""
    game_id = body.get("game_id", "")
    if not game_id:
        raise HTTPException(status_code=400, detail="game_id required")
    registry.log_game_token_summary(game_id)
    return {"ok": True, "game_id": game_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=config.PORT)
