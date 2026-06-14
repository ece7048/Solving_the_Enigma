from __future__ import annotations

from typing import Any

from SR_XAI.inference.test import test
from SR_XAI.utilities.config import apply_overrides, load_config


def inference_from_config(config_path: str | None = None, **overrides: Any):
    """Run inference from YAML, with optional keyword overrides."""
    config = load_config(config_path) if config_path else {}
    params = apply_overrides(config, overrides)
    return test(**params)

