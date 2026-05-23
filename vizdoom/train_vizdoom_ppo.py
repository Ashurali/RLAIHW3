"""Train PPO on a VizDoom scenario (Task 2 -- policy-based primary).

Examples:
    python vizdoom/train_vizdoom_ppo.py --config configs/V0_basic.yaml --seed 0
    python vizdoom/train_vizdoom_ppo.py --config configs/V1_defendcenter.yaml --seed 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repo root importable so "common" resolves when run as a script.
# (This folder is intentionally NOT a package, so it cannot shadow the
# installed "vizdoom" pip package that common.vizdoom_wrappers imports.)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_baselines3 import PPO  # noqa: E402

from common.train_core import run_training  # noqa: E402
from common.utils import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Train PPO on VizDoom.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None,
                    help="override config seed (use 0/1/2 for >=3-seed curves)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    assert cfg["algo"] == "ppo", f"expected algo: ppo, got {cfg['algo']}"
    run_training(cfg, PPO)


if __name__ == "__main__":
    main()
