import logging
import subprocess
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

_app_logger = logging.getLogger("tm_ai_server")
_app_logger.setLevel(logging.INFO)
if not _app_logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s:     %(name)s - %(message)s"))
    _app_logger.addHandler(_h)
    _app_logger.propagate = False

from .config import STATE_DIM, HIDDEN_SIZES, ACTION_SPACE_SIZE, PORT
from .inference import select_action, select_advice
from .llm_player import validate_llm_config, log_game_token_summary, register_player
from .schemas import (
    AdviceRequest, AdviceResponse, HealthResponse, MoveDebug,
    MoveRequest, MoveResponse, PlayerRegisterRequest, PlayerRegisterResponse, VersionResponse,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_llm_config()
    yield


app = FastAPI(title="TM AI Server", version="0.1.0", lifespan=lifespan)


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
        model_version="0.1.0",
        git_commit=git_commit,
        config={"state_dim": STATE_DIM, "hidden_sizes": HIDDEN_SIZES, "action_space_size": ACTION_SPACE_SIZE},
    )


@app.post("/move", response_model=MoveResponse)
async def move(request: MoveRequest):
    waiting_for = request.state.waitingFor
    if waiting_for is None:
        raise HTTPException(status_code=400, detail="state.waitingFor is required")

    game_spec = None
    if request.legal_actions:
        payload = request.legal_actions[0].payload or {}
        game_spec = payload.get("game_spec")

    state = request.state.model_dump()

    input_response, debug_info = select_action(
        state, waiting_for, game_spec,
        game_id=request.game_id,
        player_id=request.player_id,
        last_error=request.last_error,
    )
    debug = MoveDebug(**debug_info) if debug_info else None
    return MoveResponse(input_response=input_response, debug=debug)


@app.post("/advise", response_model=AdviceResponse)
async def advise(request: AdviceRequest):
    waiting_for = request.state.waitingFor
    if waiting_for is None:
        raise HTTPException(status_code=400, detail="state.waitingFor is required")

    state = request.state.model_dump()
    advice_text, recommendation = select_advice(
        state, waiting_for, request.game_id, request.player_id,
        user_question=request.user_question,
    )
    return AdviceResponse(advice_text=advice_text, recommendation=recommendation)


@app.post("/player/register", response_model=PlayerRegisterResponse)
async def player_register(body: PlayerRegisterRequest):
    """Register an AI player with a specific model before the game starts.

    Called by the TM server or play script at game creation, once per AI player:
      curl -X POST http://localhost:8000/player/register \\
        -H 'Content-Type: application/json' \\
        -d '{"player_id":"p1abc","game_id":"g123","model":"anthropic/claude-opus-4-7"}'
    """
    player = register_player(body.player_id, body.game_id, body.model)
    return PlayerRegisterResponse(ok=True, player_id=player.player_id, model=player.model)


@app.post("/game-done")
async def game_done(body: dict):
    """Log the final token summary for a completed game.

    Called by the TM server after game end, or manually via curl:
      curl -X POST http://localhost:8000/game-done -H 'Content-Type: application/json' -d '{"game_id":"gXXX"}'
    """
    game_id = body.get("game_id", "")
    if not game_id:
        raise HTTPException(status_code=400, detail="game_id required")
    log_game_token_summary(game_id)
    return {"ok": True, "game_id": game_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
