"""
Model loading and action selection.

Falls back to random-valid-action selection when no checkpoint is found.
"""

from __future__ import annotations
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch

from .config import STATE_DIM, HIDDEN_SIZES, ACTION_SPACE_SIZE, MODEL_PATH
from .encoding import encode_state, flatten_options, build_mask, index_to_response
from .model import PolicyValueNet

logger = logging.getLogger(__name__)

_model: PolicyValueNet | None = None
_model_path: str | None = None


def load_model(path: str | None = None) -> PolicyValueNet | None:
    """Load model from checkpoint. Returns None if path does not exist."""
    global _model, _model_path
    path = path or MODEL_PATH
    if not Path(path).exists():
        logger.info("No model checkpoint found at %s — using random policy", path)
        return None
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        cfg = checkpoint.get("config", {})
        model = PolicyValueNet(
            state_dim=cfg.get("state_dim", STATE_DIM),
            hidden_sizes=cfg.get("hidden_sizes", HIDDEN_SIZES),
            action_space_size=cfg.get("action_space_size", ACTION_SPACE_SIZE),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        _model = model
        _model_path = path
        logger.info("Loaded model from %s", path)
        return model
    except Exception as e:
        logger.error("Failed to load model from %s: %s", path, e)
        return None


def get_model() -> PolicyValueNet | None:
    """Return the cached model, loading it on first call."""
    global _model
    if _model is None:
        _model = load_model()
    return _model


def select_action(
    state: dict,
    waiting_for: dict,
    game_spec: dict | None = None,
    model: PolicyValueNet | None = None,
) -> tuple[dict, dict]:
    """
    Given the game state and the current decision node, return:
      - input_response: valid InputResponse dict for the TM server
      - debug: dict with policy_logits and value_estimate (may be empty)
    """
    if os.getenv("USE_LLM", "false").lower() == "true":
        from .llm_player import select_action_llm
        return select_action_llm(state, waiting_for)

    options = flatten_options(waiting_for)
    if not options:
        logger.warning("No options found in waitingFor node type=%s", waiting_for.get("type"))
        return {"type": "option"}, {}

    mask = build_mask(len(options))
    state_vec = encode_state(state, game_spec)

    m = model or get_model()

    if m is None:
        # Random policy
        valid_indices = [i for i, v in enumerate(mask) if v]
        chosen_idx = random.choice(valid_indices)
        return index_to_response(waiting_for, options[chosen_idx]["index"]), {}

    # Model-based policy
    x = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
    m_tensor = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)
    with torch.no_grad():
        logits, value = m(x, m_tensor)
    probs = torch.softmax(logits.squeeze(0), dim=-1)
    chosen_slot = int(torch.argmax(probs).item())
    chosen_option = options[min(chosen_slot, len(options) - 1)]
    input_response = index_to_response(waiting_for, chosen_option["index"])
    debug = {
        "policy_logits": logits.squeeze(0).tolist(),
        "value_estimate": float(value.item()),
    }
    return input_response, debug
