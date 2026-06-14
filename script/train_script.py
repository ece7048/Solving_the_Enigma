from __future__ import annotations

from pathlib import Path

from SR_XAI.cli import main


if __name__ == "__main__":
    config_path = Path(__file__).resolve().parents[1] / "configs" / "train_config.yaml"
    main(["train", "--config", str(config_path)])
