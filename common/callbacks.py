"""Callback assembly: periodic deterministic evaluation + checkpointing."""
from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback


def make_callbacks(cfg, run_dir: Path, eval_env):
    """Build an ``EvalCallback`` (best model + eval log) and a checkpointer.

    Frequencies in the config are expressed in *environment steps*. SB3 counts
    callback ticks per environment, so we divide by ``n_envs`` to keep the
    wall-clock cadence stable regardless of how many parallel envs a run uses.
    """
    n_envs = max(int(cfg.get("n_envs", 1)), 1)
    eval_freq = max(int(cfg.get("eval_freq", 100_000)) // n_envs, 1)
    ckpt_freq = max(int(cfg.get("checkpoint_freq", 250_000)) // n_envs, 1)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(Path(run_dir) / "best_model"),
        log_path=str(run_dir),
        eval_freq=eval_freq,
        n_eval_episodes=int(cfg.get("eval_episodes", 10)),
        deterministic=True,
        render=False,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=ckpt_freq,
        save_path=str(Path(run_dir) / "checkpoints"),
        name_prefix=cfg["exp_id"],
    )
    return [eval_cb, ckpt_cb]
