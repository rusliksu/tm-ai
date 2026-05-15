import logging
import subprocess
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
from .schemas import (
    AdviceRequest, AdviceResponse, HealthResponse, MoveDebug,
    MoveRequest, MoveResponse, VersionResponse,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="TM AI Server", version="0.1.0")


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

    input_response, debug_info = select_action(state, waiting_for, game_spec,
                                               last_error=request.last_error)
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
