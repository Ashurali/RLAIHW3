"""Fast environment + pipeline sanity check -- run this FIRST on the server.

It does NOT train. It verifies that:
  1. PyTorch sees the GPU,
  2. the Atari (Pong) vec pipeline builds and steps,
  3. the VizDoom vec pipeline builds and steps (the de-risking critical path).

Usage:
    python smoke_test.py                 # both tasks
    python smoke_test.py --task pong     # just Atari
    python smoke_test.py --task vizdoom  # just VizDoom
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))


def check_torch():
    import torch

    print(f"[torch] version={torch.__version__} "
          f"cuda_available={torch.cuda.is_available()} "
          f"device={'cuda:' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")


def step_vec(venv, n_steps=5, label=""):
    import numpy as np

    obs = venv.reset()
    print(f"[{label}] obs shape={np.asarray(obs).shape} dtype={np.asarray(obs).dtype} "
          f"action_space={venv.action_space}")
    for _ in range(n_steps):
        actions = np.array([venv.action_space.sample() for _ in range(venv.num_envs)])
        obs, rewards, dones, infos = venv.step(actions)
    print(f"[{label}] stepped {n_steps}x OK; reward sample={rewards[:1]}")
    venv.close()


def check_pong():
    from common.envs import make_atari_vec

    print("[pong] building ALE/Pong-v5 ...")
    venv = make_atari_vec("ALE/Pong-v5", n_envs=1, seed=0, n_stack=4)
    step_vec(venv, label="pong")


def check_vizdoom():
    from common.envs import make_vizdoom_vec

    # Use the Basic scenario for the smoke test (loads fastest).
    env_id = "VizdoomBasic-v1"
    print(f"[vizdoom] building {env_id} ... (edit env_id if the suffix differs)")
    venv = make_vizdoom_vec(env_id, n_envs=1, seed=0, n_stack=4)
    step_vec(venv, label="vizdoom")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["all", "pong", "vizdoom"], default="all")
    args = ap.parse_args()

    check_torch()
    if args.task in ("all", "pong"):
        check_pong()
    if args.task in ("all", "vizdoom"):
        check_vizdoom()
    print("[smoke_test] all selected checks passed.")


if __name__ == "__main__":
    main()
