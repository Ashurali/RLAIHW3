"""Train DQN on Atari Pong (Task 1 -- value-based baseline and ablations).

Examples:
    python pong/train_pong_dqn.py --config configs/P1.yaml --seed 0
    python pong/train_pong_dqn.py --config configs/P2_targetoff.yaml --seed 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repo root importable so "common" resolves when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_baselines3 import DQN  # noqa: E402

from common.train_core import run_training  # noqa: E402
from common.utils import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Train DQN on Atari Pong.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None,
                    help="override config seed (use 0/1/2 for >=3-seed curves)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    assert cfg["algo"] == "dqn", f"expected algo: dqn, got {cfg['algo']}"
    run_training(cfg, DQN)


if __name__ == "__main__":
    main()
