from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file into a plain dictionary."""
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyYAML is required to read configuration files. Install with `pip install -e .`.") from exc

    path = Path(config_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    return data


def apply_overrides(config: dict[str, Any], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a copy of `config` with non-None override values applied."""
    merged = dict(config)
    for key, value in (overrides or {}).items():
        if value is not None:
            merged[key] = value
    return merged


def parse_set_overrides(values: list[str] | None) -> dict[str, Any]:
    """Parse CLI overrides in KEY=VALUE form using Python literal coercion."""
    parsed: dict[str, Any] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"Override must use KEY=VALUE format: {item}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Override key cannot be empty: {item}")
        try:
            parsed[key] = ast.literal_eval(raw_value)
        except (SyntaxError, ValueError):
            parsed[key] = raw_value
    return parsed
