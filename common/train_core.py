"""Config-driven training loop shared by all four (task, algo) entry points."""
from __future__ import annotations

import shutil
from pathlib import Path

from stable_baselines3.common.logger import configure

from common.callbacks import make_callbacks
from common.envs import build_env_fn
from common.eval_utils import evaluate_and_record
from common.plotting import plot_curve
from common.utils import (
    configure_torch_perf,
    get_run_dir,
    resolve_schedules,
    save_config_copy,
    set_global_seeds,
)


def run_training(cfg, model_cls):
    """Train one run end-to-end and emit the standard artifact set.

    Under ``results/<exp_id>_s<seed>/`` this produces: ``config.yaml``,
    ``metrics.csv``, ``curve.png``, ``eval.json``, ``gameplay.gif``,
    ``model.zip``, plus ``best_model/`` and ``checkpoints/``.
    """
    seed = int(cfg.get("seed", 0))
    set_global_seeds(seed)
    configure_torch_perf()

    run_dir = get_run_dir(cfg["exp_id"], seed)
    save_config_copy(cfg, run_dir)

    env_fn = build_env_fn(cfg)
    n_envs = int(cfg.get("n_envs", 1))
    train_env = env_fn(seed=seed, n_envs=n_envs, eval_mode=False)
    eval_env = env_fn(seed=seed + 1000, n_envs=1, eval_mode=True)

    # Translate `lin_<float>` strings (e.g. `learning_rate: lin_2.5e-4`) into
    # SB3 linear-decay callables. Required for the literature-recipe PPO config.
    algo_kwargs = resolve_schedules(cfg.get("algo_kwargs", {}) or {})

    model = model_cls(
        cfg.get("policy", "CnnPolicy"),
        train_env,
        seed=seed,
        device=cfg.get("device", "cuda"),
        verbose=1,
        tensorboard_log=str(run_dir / "tb"),
        **algo_kwargs,
    )
    # Mirror stdout into a CSV (-> progress.csv) and TensorBoard event files.
    model.set_logger(configure(str(run_dir), ["stdout", "csv", "tensorboard"]))

    callbacks = make_callbacks(cfg, run_dir, eval_env)
    model.learn(
        total_timesteps=int(cfg["total_timesteps"]),
        callback=callbacks,
        progress_bar=True,
    )

    model.save(str(run_dir / "model"))
    train_env.close()
    eval_env.close()

    # The plan's per-run deliverable is metrics.csv; SB3 emits progress.csv.
    progress = run_dir / "progress.csv"
    if progress.exists():
        shutil.copyfile(progress, run_dir / "metrics.csv")

    evaluate_and_record(
        model, env_fn, cfg, run_dir, seed, record_gif=cfg.get("record_gif", True)
    )
    try:
        plot_curve(run_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"[train_core] curve plotting skipped: {exc}")

    print(f"[train_core] done -> {run_dir}")
    return run_dir
