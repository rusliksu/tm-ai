"""
Phase 2: PPO self-play training using Stable-Baselines3.

Requires:
  - TM server with /api/ai/new-game and /api/ai/step endpoints (Phase 2 work)
  - A supervised-learning checkpoint to initialise from (Phase 1 output)
  - pip install stable-baselines3 sb3-contrib  (sb3-contrib for MaskablePPO)

Usage:
    uv run python -m tm_ai_server.training.train_ppo \\
        --checkpoint models/checkpoint_best.pt \\
        --output-dir models \\
        --total-steps 10_000_000
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path

import torch

from ..config import (
    ACTION_SPACE_SIZE, CLIP_RANGE, GAE_LAMBDA, GAMMA, HIDDEN_SIZES,
    LEARNING_RATE, N_STEPS, STATE_DIM, TM_SERVER_URL,
)
from ..model import PolicyValueNet
from .env_tm import TerraformingMarsEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def train(checkpoint_path: str | None, output_dir: str, total_steps: int) -> None:
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        from stable_baselines3.common.env_util import make_vec_env
    except ImportError:
        logger.error("Install sb3-contrib: uv add sb3-contrib")
        return

    def make_env():
        env = TerraformingMarsEnv(server_url=TM_SERVER_URL)
        return ActionMasker(env, lambda e: e.action_masks())

    env = make_vec_env(make_env, n_envs=1)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        verbose=1,
        tensorboard_log=str(output_path / "tensorboard"),
    )

    if checkpoint_path and Path(checkpoint_path).exists():
        _load_supervised_weights(model, checkpoint_path)
        logger.info("Initialised PPO policy from supervised checkpoint %s", checkpoint_path)

    model.learn(total_timesteps=total_steps, progress_bar=True)
    model.save(str(output_path / "ppo_latest"))
    logger.info("PPO training complete. Model saved to %s", output_path / "ppo_latest")


def _load_supervised_weights(ppo_model: Any, checkpoint_path: str) -> None:
    """Transfer shared MLP weights from supervised PolicyValueNet to SB3 MlpPolicy."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("config", {})
    src = PolicyValueNet(
        state_dim=cfg.get("state_dim", STATE_DIM),
        hidden_sizes=cfg.get("hidden_sizes", HIDDEN_SIZES),
        action_space_size=cfg.get("action_space_size", ACTION_SPACE_SIZE),
    )
    src.load_state_dict(checkpoint["model_state_dict"])
    # SB3 MlpPolicy internals differ — attempt best-effort weight copy
    try:
        policy_net = ppo_model.policy.mlp_extractor.policy_net
        value_net = ppo_model.policy.mlp_extractor.value_net
        src_layers = [m for m in src.backbone if isinstance(m, torch.nn.Linear)]
        dst_policy = [m for m in policy_net if isinstance(m, torch.nn.Linear)]
        dst_value = [m for m in value_net if isinstance(m, torch.nn.Linear)]
        for sl, dl, dv in zip(src_layers, dst_policy, dst_value):
            dl.weight.data.copy_(sl.weight.data)
            dl.bias.data.copy_(sl.bias.data)
            dv.weight.data.copy_(sl.weight.data)
            dv.bias.data.copy_(sl.bias.data)
        logger.info("Transferred %d shared MLP layers", len(src_layers))
    except Exception as e:
        logger.warning("Weight transfer failed (SB3 API mismatch?): %s", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--total-steps", type=int, default=10_000_000)
    args = parser.parse_args()
    train(args.checkpoint, args.output_dir, args.total_steps)
