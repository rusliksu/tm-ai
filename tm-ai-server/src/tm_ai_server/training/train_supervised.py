"""
Phase 1: Supervised learning from human game logs.

Usage:
    uv run python -m tm_ai_server.training.train_supervised \\
        --data-dir logs/training \\
        --output-dir models \\
        --epochs 50
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from ..config import (
    ACTION_SPACE_SIZE, BATCH_SIZE, HIDDEN_SIZES, LEARNING_RATE, MAX_EPOCHS, STATE_DIM
)
from ..model import PolicyValueNet
from .dataset import TMDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def train(data_dir: str, output_dir: str, epochs: int = MAX_EPOCHS) -> None:
    dataset = TMDataset(data_dir, human_only=True)
    if len(dataset) == 0:
        logger.error("No training samples found in %s", data_dir)
        return

    n_val = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on %s, %d samples (%d train / %d val)", device, len(dataset), n_train, n_val)

    model = PolicyValueNet(STATE_DIM, HIDDEN_SIZES, ACTION_SPACE_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_policy_loss = train_value_loss = 0.0
        for states, masks, actions, rewards in train_loader:
            states, masks, actions, rewards = (
                states.to(device), masks.to(device), actions.to(device), rewards.to(device)
            )
            logits, values = model(states, masks)
            p_loss = policy_loss_fn(logits, actions)
            v_loss = value_loss_fn(values, rewards)
            loss = p_loss + 0.5 * v_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_policy_loss += p_loss.item()
            train_value_loss += v_loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for states, masks, actions, rewards in val_loader:
                states, masks, actions, rewards = (
                    states.to(device), masks.to(device), actions.to(device), rewards.to(device)
                )
                logits, values = model(states, masks)
                val_loss += policy_loss_fn(logits, actions).item()

        n_train_batches = max(1, len(train_loader))
        n_val_batches = max(1, len(val_loader))
        logger.info(
            "Epoch %d/%d  policy_loss=%.4f  value_loss=%.4f  val_loss=%.4f",
            epoch, epochs,
            train_policy_loss / n_train_batches,
            train_value_loss / n_train_batches,
            val_loss / n_val_batches,
        )

        avg_val = val_loss / n_val_batches
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            _save(model, output_path / "checkpoint_best.pt")

    _save(model, output_path / "checkpoint_latest.pt")
    logger.info("Training complete. Best val loss: %.4f", best_val_loss)


def _save(model: PolicyValueNet, path: Path) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "state_dim": STATE_DIM,
                "hidden_sizes": HIDDEN_SIZES,
                "action_space_size": ACTION_SPACE_SIZE,
            },
        },
        path,
    )
    logger.info("Saved checkpoint to %s", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="logs/training")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    args = parser.parse_args()
    train(args.data_dir, args.output_dir, args.epochs)
