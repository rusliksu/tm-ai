"""
Phase 2: PPO self-play training using Stable-Baselines3 + sb3-contrib.

All training artifacts are preserved — nothing is deleted or overwritten:
  logs/selfplay/<run_id>/              — per-game JSONL (Plan B, via TM server)
  logs/selfplay/<run_id>/metrics.jsonl — win rates, losses, avg game length, ELO
  logs/selfplay/<run_id>/manifest.json — run config written at start
  models/selfplay/<run_id>/checkpoint_<N>.pt — checkpoint every --checkpoint-interval games
  models/selfplay/<run_id>/checkpoint_best.pt  — best win-rate checkpoint
  models/selfplay/<run_id>/checkpoint_latest.pt

Usage:
    uv run python -m tm_ai_server.training.train_ppo \\
        --checkpoint models/checkpoint_best.pt \\
        --output-dir models \\
        --log-dir logs/selfplay \\
        --total-steps 5_000_000 \\
        --checkpoint-interval 100
"""

from __future__ import annotations
import argparse
import json
import logging
import math
import os
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import (
    ACTION_SPACE_SIZE, CLIP_RANGE, GAE_LAMBDA, GAMMA, HIDDEN_SIZES,
    LEARNING_RATE, N_STEPS, STATE_DIM, TM_SERVER_URL,
)
from ..model import PolicyValueNet
from .env_tm import TerraformingMarsEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class SelfPlayCallback:
    """
    SB3-compatible callback that tracks games, saves checkpoints,
    and writes metrics after each checkpoint interval.
    """

    def __init__(
        self,
        run_dir: Path,
        model_dir: Path,
        checkpoint_interval: int,
        model_ref: Any,
    ):
        self.run_dir = run_dir
        self.model_dir = model_dir
        self.checkpoint_interval = checkpoint_interval
        self.model_ref = model_ref

        self.games_completed = 0
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.best_mean_reward = -math.inf
        self._ep_reward = 0.0
        self._ep_len = 0
        self._metrics_path = run_dir / "metrics.jsonl"
        # Per-game VP tracking (filled on done steps from result info)
        self._winner_vps: list[float] = []
        self._mean_vps: list[float] = []
        self._end_generations: list[int] = []

    def on_step(self, reward: float, done: bool, info: dict | None = None) -> None:
        self._ep_reward += reward
        self._ep_len += 1
        if done:
            self.games_completed += 1
            self.episode_rewards.append(self._ep_reward)
            self.episode_lengths.append(self._ep_len)
            self._ep_reward = 0.0
            self._ep_len = 0
            # Extract VP scores and generation from game result
            result = (info or {}).get("result") or {}
            player_results = result.get("playerResults", [])
            if player_results:
                vps = [r.get("vp_total", 0) for r in player_results]
                self._winner_vps.append(float(max(vps)))
                self._mean_vps.append(float(sum(vps) / len(vps)))
            if result.get("endGeneration"):
                self._end_generations.append(int(result["endGeneration"]))
            if self.games_completed % self.checkpoint_interval == 0:
                self._checkpoint()

    def _checkpoint(self) -> None:
        n = self.games_completed
        recent_rewards = self.episode_rewards[-self.checkpoint_interval:]
        recent_lengths = self.episode_lengths[-self.checkpoint_interval:]
        recent_winner_vps = self._winner_vps[-self.checkpoint_interval:]
        recent_mean_vps = self._mean_vps[-self.checkpoint_interval:]
        recent_gens = self._end_generations[-self.checkpoint_interval:]
        mean_reward = float(np.mean(recent_rewards)) if recent_rewards else 0.0
        mean_length = float(np.mean(recent_lengths)) if recent_lengths else 0.0
        win_rate = float(np.mean([r > 0 for r in recent_rewards])) if recent_rewards else 0.0
        mean_winner_vp = float(np.mean(recent_winner_vps)) if recent_winner_vps else 0.0
        mean_vp = float(np.mean(recent_mean_vps)) if recent_mean_vps else 0.0
        mean_generation = float(np.mean(recent_gens)) if recent_gens else 0.0

        # Write custom scalars to TensorBoard
        tb_logger = getattr(self.model_ref, "logger", None)
        if tb_logger is not None:
            tb_logger.record("selfplay/win_rate", win_rate)
            tb_logger.record("selfplay/mean_reward", mean_reward)
            tb_logger.record("selfplay/winner_vp", mean_winner_vp)
            tb_logger.record("selfplay/mean_vp", mean_vp)
            tb_logger.record("selfplay/mean_game_length", mean_length)
            tb_logger.record("selfplay/mean_generation", mean_generation)
            tb_logger.record("selfplay/games_completed", float(n))
            tb_logger.dump(step=n)

        # Periodic numbered checkpoint
        ckpt_path = self.model_dir / f"checkpoint_{n}.pt"
        self.model_ref.save(str(ckpt_path))
        logger.info("Saved checkpoint at game %d → %s (win_rate=%.3f, winner_vp=%.1f)",
                    n, ckpt_path, win_rate, mean_winner_vp)

        # Always update latest
        self.model_ref.save(str(self.model_dir / "checkpoint_latest"))

        # Update best if improved
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self.model_ref.save(str(self.model_dir / "checkpoint_best"))
            logger.info("New best checkpoint (mean_reward=%.3f)", mean_reward)

        # Append metrics
        record = {
            "games": n,
            "win_rate": win_rate,
            "mean_reward": mean_reward,
            "mean_game_length": mean_length,
            "winner_vp": mean_winner_vp,
            "mean_vp": mean_vp,
            "mean_generation": mean_generation,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(self._metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        logger.info("Metrics: %s", record)


def train(
    checkpoint_path: str | None,
    output_dir: str,
    log_dir: str,
    total_steps: int,
    checkpoint_interval: int,
) -> None:
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
    except ImportError:
        logger.error("Install sb3-contrib: uv add sb3-contrib")
        return

    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    run_dir = Path(log_dir) / run_id
    model_dir = Path(output_dir) / "selfplay" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Run ID: %s", run_id)
    logger.info("Game logs → %s", run_dir)
    logger.info("Checkpoints → %s", model_dir)

    # Write manifest
    manifest = {
        "run_id": run_id,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_checkpoint": checkpoint_path,
        "total_steps": total_steps,
        "checkpoint_interval_games": checkpoint_interval,
        "hyperparameters": {
            "learning_rate": LEARNING_RATE,
            "n_steps": N_STEPS,
            "gamma": GAMMA,
            "gae_lambda": GAE_LAMBDA,
            "clip_range": CLIP_RANGE,
        },
        "model_config": {
            "state_dim": STATE_DIM,
            "hidden_sizes": HIDDEN_SIZES,
            "action_space_size": ACTION_SPACE_SIZE,
        },
        "tm_server_url": TM_SERVER_URL,
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    def make_env():
        env = TerraformingMarsEnv(server_url=TM_SERVER_URL, log_dir=str(run_dir))
        return ActionMasker(env, lambda e: e.action_masks())

    env = make_env()

    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        verbose=1,
        tensorboard_log=str(model_dir / "tensorboard"),
    )

    if checkpoint_path and Path(checkpoint_path).exists():
        _load_supervised_weights(model, checkpoint_path)
        logger.info("Loaded supervised weights from %s", checkpoint_path)

    cb = SelfPlayCallback(
        run_dir=run_dir,
        model_dir=model_dir,
        checkpoint_interval=checkpoint_interval,
        model_ref=model,
    )

    # Wrap learn() with per-step callback tracking
    _train_with_callback(model, env, total_steps, cb)

    model.save(str(model_dir / "checkpoint_latest"))
    logger.info("Training complete. Run artifacts in %s and %s", run_dir, model_dir)


def _train_with_callback(model: Any, env: Any, total_steps: int, cb: SelfPlayCallback) -> None:
    """Run PPO training, feeding done/reward signals to the callback for per-game tracking."""
    try:
        from stable_baselines3.common.callbacks import BaseCallback

        class _GameCallback(BaseCallback):
            def __init__(self, outer: SelfPlayCallback):
                super().__init__(verbose=0)
                self._outer = outer

            def _on_step(self) -> bool:
                rewards = self.locals.get("rewards", [])
                dones = self.locals.get("dones", [])
                infos = self.locals.get("infos", [{}] * len(rewards))
                for r, d, info in zip(rewards, dones, infos):
                    self._outer.on_step(float(r), bool(d), info)
                return True

        model.learn(total_timesteps=total_steps, callback=_GameCallback(cb), progress_bar=True)
    except Exception as e:
        logger.error("Training error: %s", e)
        raise


def _load_supervised_weights(ppo_model: Any, checkpoint_path: str) -> None:
    """Transfer backbone weights from supervised PolicyValueNet to SB3 MlpPolicy."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("config", {})
    src = PolicyValueNet(
        state_dim=cfg.get("state_dim", STATE_DIM),
        hidden_sizes=cfg.get("hidden_sizes", HIDDEN_SIZES),
        action_space_size=cfg.get("action_space_size", ACTION_SPACE_SIZE),
    )
    src.load_state_dict(checkpoint["model_state_dict"])
    try:
        policy_net = ppo_model.policy.mlp_extractor.policy_net
        value_net = ppo_model.policy.mlp_extractor.value_net
        src_layers = [m for m in src.backbone if isinstance(m, torch.nn.Linear)]
        dst_policy = [m for m in policy_net if isinstance(m, torch.nn.Linear)]
        dst_value = [m for m in value_net if isinstance(m, torch.nn.Linear)]
        n = 0
        for sl, dl, dv in zip(src_layers, dst_policy, dst_value):
            if sl.weight.shape == dl.weight.shape:
                dl.weight.data.copy_(sl.weight.data)
                dl.bias.data.copy_(sl.bias.data)
            if sl.weight.shape == dv.weight.shape:
                dv.weight.data.copy_(sl.weight.data)
                dv.bias.data.copy_(sl.bias.data)
            n += 1
        logger.info("Transferred %d shared MLP layers", n)
    except Exception as e:
        logger.warning("Weight transfer failed (SB3 API mismatch?): %s", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO self-play training for Terraforming Mars AI")
    parser.add_argument("--checkpoint", default="models/checkpoint_best.pt",
                        help="Supervised checkpoint to initialise from")
    parser.add_argument("--output-dir", default="models",
                        help="Directory for model checkpoints")
    parser.add_argument("--log-dir", default="logs/selfplay",
                        help="Base directory for per-run game logs")
    parser.add_argument("--total-steps", type=int, default=5_000_000,
                        help="Total PPO timesteps")
    parser.add_argument("--checkpoint-interval", type=int, default=100,
                        help="Save checkpoint every N completed games")
    args = parser.parse_args()
    train(
        args.checkpoint,
        args.output_dir,
        args.log_dir,
        args.total_steps,
        args.checkpoint_interval,
    )
