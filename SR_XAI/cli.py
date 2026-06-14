from __future__ import annotations

import argparse

from SR_XAI.utilities.config import parse_set_overrides


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SR_XAI training or inference.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run training.")
    train_parser.add_argument("--config", required=True, help="Path to a YAML training configuration.")
    train_parser.add_argument("--set", action="append", default=[], help="Override a config value, e.g. --set lr=1e-4.")

    infer_parser = subparsers.add_parser("infer", help="Run inference.")
    infer_parser.add_argument("--config", required=True, help="Path to a YAML inference configuration.")
    infer_parser.add_argument("--set", action="append", default=[], help="Override a config value, e.g. --set bz=1.")

    return parser


def main(argv: list[str] | None = None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    overrides = parse_set_overrides(args.set)

    if args.command == "train":
        from SR_XAI.training.runner import train_from_config

        return train_from_config(args.config, **overrides)
    if args.command == "infer":
        from SR_XAI.inference.runner import inference_from_config

        return inference_from_config(args.config, **overrides)
    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

