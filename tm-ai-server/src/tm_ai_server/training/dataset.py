"""
Dataset loader for Plan-B training logs (one JSONL file per game).

JSONL file format (one JSON object per line):
  {"type":"meta",  "game_id":..., "game_spec":{...}, "players":[...]}
  {"type":"turn",  "step":0, "playerId":..., "generation":3, "phase":"action",
                   "state":{...}, "waitingFor":{...}, "input_response":{...},
                   "is_human":true, "timestamp":...}
  ...
  {"type":"result","endGeneration":14, "playerResults":[{...},...]}

All three record types must be present; games missing meta or result are skipped.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ..config import ACTION_SPACE_SIZE, STATE_DIM
from ..encoding import build_mask, encode_state, flatten_options, response_to_index

logger = logging.getLogger(__name__)


def rank_reward(rank: int, num_players: int) -> float:
    """Normalised reward in [0, 1]: 1st place = 1.0, last place = 0.0."""
    if num_players <= 1:
        return 1.0
    return (num_players - rank) / (num_players - 1)


class TMDataset(Dataset):
    """
    PyTorch dataset of (state_vec, mask, action_index, reward) tuples.

    Only turns where is_human=True are included by default (supervised pre-training).
    Set human_only=False to include AI turns as well.
    """

    def __init__(self, data_dir: str, human_only: bool = True):
        self.samples: list[dict] = []
        self._load(Path(data_dir), human_only)
        logger.info("Loaded %d training samples from %s", len(self.samples), data_dir)

    def _load(self, data_dir: Path, human_only: bool) -> None:
        for path in sorted(data_dir.glob("*.jsonl")):
            try:
                self._load_game(path, human_only)
            except Exception as e:
                logger.warning("Skipping %s: %s", path.name, e)

    def _load_game(self, path: Path, human_only: bool) -> None:
        meta: dict | None = None
        result: dict | None = None
        turns: list[dict] = []

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                t = record.get("type")
                if t == "meta":
                    meta = record
                elif t == "turn":
                    turns.append(record)
                elif t == "result":
                    result = record

        if meta is None or result is None:
            return

        game_spec = meta.get("game_spec", {})
        player_results = {r["playerId"]: r for r in result.get("playerResults", [])}
        num_players = game_spec.get("player_count", 1)

        for turn in turns:
            if human_only and not turn.get("is_human", False):
                continue

            waiting_for = turn.get("waitingFor")
            input_response = turn.get("input_response")
            state = turn.get("state", {})
            player_id = turn.get("playerId", "")

            if not waiting_for or not input_response:
                continue

            action_idx = response_to_index(waiting_for, input_response)
            if action_idx is None:
                continue

            options = flatten_options(waiting_for)
            if action_idx >= len(options):
                continue

            state_vec = encode_state(state, None)
            mask = build_mask(len(options))

            player_result = player_results.get(player_id, {})
            rank = player_result.get("rank", num_players)
            reward = rank_reward(rank, num_players)

            self.samples.append({
                "state": state_vec,
                "mask": mask,
                "action": action_idx,
                "reward": float(reward),
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        return (
            torch.tensor(s["state"], dtype=torch.float32),
            torch.tensor(s["mask"], dtype=torch.bool),
            torch.tensor(s["action"], dtype=torch.long),
            torch.tensor(s["reward"], dtype=torch.float32),
        )
