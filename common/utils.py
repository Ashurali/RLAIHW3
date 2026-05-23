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
