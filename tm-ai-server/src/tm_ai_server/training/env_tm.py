"""
Phase 2: Gymnasium environment wrapping the TM server over HTTP.

The TM server exposes:
  POST /api/ai/new-game  → {game_id, player_id, state, waitingFor, game_spec}
  POST /api/ai/step      → {input_response} → {player_id, state, waitingFor, done, result}

Each step represents one player decision; the model plays all positions (both AI players).
"""

from __future__ import annotations
import logging
from typing import Any

import time

import gymnasium as gym
import numpy as np
import requests

from ..config import ACTION_SPACE_SIZE, STATE_DIM, TM_SERVER_URL
from ..encoding import build_mask, encode_state, flatten_options, index_to_response

logger = logging.getLogger(__name__)

DEFAULT_GAME_CONFIG = {
    "boardName": "tharsis",
}


class TerraformingMarsEnv(gym.Env):
    """
    Gymnasium environment for TM self-play training (Phase 2).

    Observation: float32 vector of shape (STATE_DIM,)
    Action:      Discrete(ACTION_SPACE_SIZE) — index into flattened options list
    Reward:      relative VP at game end; 0 at intermediate steps

    The model plays as all players alternately. After each step, the
    observation switches to the perspective of whoever is next to act.
    """

    metadata = {"render_modes": []}

    def __init__(self, server_url: str = TM_SERVER_URL, game_config: dict | None = None,
                 log_dir: str | None = None):
        super().__init__()
        self.server_url = server_url.rstrip("/")
        self.game_config = {**DEFAULT_GAME_CONFIG, **(game_config or {})}
        self.log_dir = log_dir
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(ACTION_SPACE_SIZE)

        self._game_id: str | None = None
        self._player_id: str | None = None
        self._waiting_for: dict | None = None
        self._game_spec: dict | None = None
        self._state: dict | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        payload = dict(self.game_config)
        if self.log_dir:
            payload["logDir"] = self.log_dir
        for attempt in range(5):
            try:
                resp = requests.post(
                    f"{self.server_url}/api/ai/new-game",
                    json=payload,
                    timeout=60,
                )
                resp.raise_for_status()
                break
            except requests.exceptions.ReadTimeout:
                wait = 5 * (attempt + 1)
                logger.warning("reset() timeout (attempt %d/5); retrying in %ds", attempt + 1, wait)
                time.sleep(wait)
        else:
            raise RuntimeError("TM server unresponsive after 5 reset() attempts")
        data = resp.json()

        self._game_id = data["game_id"]
        self._player_id = data["player_id"]
        self._waiting_for = data["waitingFor"]
        self._game_spec = data.get("game_spec")
        self._state = data["state"]

        obs = encode_state(self._state, None)
        return obs, {"game_id": self._game_id, "player_id": self._player_id}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._waiting_for is None or self._game_id is None:
            raise RuntimeError("Call reset() before step()")

        options = flatten_options(self._waiting_for)
        if not options:
            # No selectable options (shouldn't happen after encoding fixes, but guard anyway)
            logger.warning("Empty options for game %s wf_type=%s; skipping step",
                           self._game_id, self._waiting_for.get("type"))
            return np.zeros(STATE_DIM, dtype=np.float32), 0.0, True, False, {"error": "empty_options"}
        option_index = options[min(action, len(options) - 1)]["index"]
        input_response = index_to_response(self._waiting_for, option_index)

        def _do_step(ir: dict) -> requests.Response:
            for attempt in range(4):
                try:
                    return requests.post(
                        f"{self.server_url}/api/ai/step",
                        json={"game_id": self._game_id, "player_id": self._player_id, "input_response": ir},
                        timeout=60,
                    )
                except requests.exceptions.ReadTimeout:
                    wait = 5 * (attempt + 1)
                    logger.warning("step() timeout (attempt %d/4); retrying in %ds", attempt + 1, wait)
                    time.sleep(wait)
            raise RuntimeError("TM server unresponsive after 4 step() attempts")

        resp = _do_step(input_response)
        if resp.status_code == 400:
            # Invalid move — retry with each option from last to first until one succeeds
            succeeded = False
            for fallback_opt in reversed(options):
                fb_resp = _do_step(index_to_response(self._waiting_for, fallback_opt["index"]))
                if fb_resp.status_code == 200:
                    resp = fb_resp
                    succeeded = True
                    break
            if not succeeded:
                import json as _json
                _errs = []
                for _opt in list(reversed(options))[:6]:
                    _ir = index_to_response(self._waiting_for, _opt["index"])
                    _r = _do_step(_ir)
                    _errs.append(f"idx={_opt['index']} {_r.status_code} {_r.json().get('error','?')[:50]!r} {_json.dumps(_ir)[:70]}")
                logger.warning("All fallback failed game=%s type=%r title=%r\n  %s",
                    self._game_id, self._waiting_for.get("type"),
                    self._waiting_for.get("title", ""), "\n  ".join(_errs))
                self._state = None
                self._waiting_for = None
                self._player_id = None
                return np.zeros(STATE_DIM, dtype=np.float32), 0.0, True, False, {"error": "all_actions_failed"}
        resp.raise_for_status()
        data = resp.json()

        done = data.get("done", False)
        result = data.get("result") or {}

        if done:
            # Return zero obs on game end; reset() will provide the next obs
            obs = np.zeros(STATE_DIM, dtype=np.float32)
            reward = self._compute_reward(result)
            self._state = None
            self._waiting_for = None
            self._player_id = None
        else:
            self._state = data["state"]
            self._waiting_for = data.get("waitingFor")
            self._player_id = data["player_id"]
            obs = encode_state(self._state, None)
            reward = 0.0

        info = {"player_id": self._player_id, "result": result, "game_id": self._game_id}
        return obs, reward, done, False, info

    def action_masks(self) -> np.ndarray:
        """Return valid-action mask for current step (used by MaskablePPO)."""
        if self._waiting_for is None:
            return np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        options = flatten_options(self._waiting_for)
        return build_mask(len(options))

    def _compute_reward(self, result: dict) -> float:
        player_results = result.get("playerResults", [])
        if not player_results:
            return 0.0
        vps = [r.get("vp_total", 0) for r in player_results]
        # The last _player_id before done was the one who triggered game end
        my_vp = next(
            (r.get("vp_total", 0) for r in player_results if r.get("playerId") == self._player_id),
            sum(vps) / len(vps) if vps else 0,
        )
        other_vps = [r.get("vp_total", 0) for r in player_results if r.get("playerId") != self._player_id]
        mean_other = sum(other_vps) / len(other_vps) if other_vps else 0
        reference_vp = max(vps) if vps else 1
        return (my_vp - mean_other) / max(reference_vp, 1)
