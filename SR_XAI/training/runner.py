from __future__ import annotations

from typing import Any

from SR_XAI.training.train import train
from SR_XAI.utilities.config import apply_overrides, load_config


def train_from_config(config_path: str | None = None, **overrides: Any):
    """Run training from YAML, with optional keyword overrides."""
    config = load_config(config_path) if config_path else {}
    params = apply_overrides(config, overrides)
    return train(**params)

