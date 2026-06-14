from __future__ import annotations

from pathlib import Path

from SR_XAI.cli import main


if __name__ == "__main__":
    config_path = Path(__file__).resolve().parents[1] / "configs" / "inference_config.yaml"
    main(["infer", "--config", str(config_path)])
