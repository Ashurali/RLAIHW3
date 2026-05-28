"""Configuration loading, seeding and run-directory helpers."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import yaml

# common/ lives one level below the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"


def load_config(path) -> dict:
    """Load a YAML experiment config into a plain dict."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for required in ("exp_id", "task", "algo", "env_id", "total_timesteps"):
        if required not in cfg:
            raise ValueError(f"Config {path} is missing required key '{required}'.")
    return cfg


def _linear_schedule(initial_value: float):
    """SB3 schedule: linear from ``initial_value`` (progress=1) down to 0.

    SB3 invokes the callable with ``progress_remaining`` going 1.0 -> 0.0 over
    training. This is the standard schedule used by the SB3-zoo / Schulman
    PPO-Atari recipe: linear LR-decay and clip-range-decay both critical for
    Pong convergence.
    """
    def schedule(progress_remaining: float) -> float:
        return float(progress_remaining) * float(initial_value)
    schedule.__name__ = f"lin_{initial_value:g}"
    return schedule


def resolve_schedules(kwargs: dict) -> dict:
    """Convert ``lin_<float>`` strings in algo_kwargs into SB3 schedule callables.

    Lets configs say ``learning_rate: lin_2.5e-4`` and have SB3 receive a
    proper linear-decay callable. Returns a NEW dict (does not mutate input).
    """
    out = dict(kwargs)
    for k, v in list(out.items()):
        if isinstance(v, str) and v.startswith("lin_"):
            try:
                init = float(v[len("lin_"):])
            except ValueError:
                continue
            out[k] = _linear_schedule(init)
    return out


def set_global_seeds(seed: int) -> None:
    """Seed Python, NumPy and (when present) PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        # torch is only required on the training server, not for tooling.
        pass


def configure_torch_perf() -> None:
    """Enable throughput-oriented GPU settings (safe on any CUDA GPU).

    cuDNN autotuning picks the fastest conv algorithms for our fixed 84x84
    inputs; TF32 matmuls use the Ampere/Ada tensor cores. Both trade a little
    numerical determinism for speed, which is fine for RL throughput.
    """
    try:
        import torch

        torch.backends.cudnn.benchmark = True
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
    except ImportError:
        pass


def get_run_dir(exp_id: str, seed: int, create: bool = True) -> Path:
    """Return ``results/<exp_id>_s<seed>/``; create it unless told otherwise."""
    run_dir = RESULTS_DIR / f"{exp_id}_s{seed}"
    if create:
        run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config_copy(cfg: dict, run_dir: Path) -> None:
    """Persist the exact config used for a run next to its outputs."""
    with open(Path(run_dir) / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
