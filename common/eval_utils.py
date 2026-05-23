"""Final evaluation and gameplay-GIF recording.

Shared by the training core (so every run auto-produces ``eval.json`` and
``gameplay.gif``) and by the standalone ``eval_record.py`` entry points (so a
finished model can be re-evaluated / re-recorded without retraining).
"""
from __future__ import annotations

import json
from pathlib import Path

import imageio
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy


def _base_envs(venv):
    """Drill through VecEnvWrappers (e.g. VecFrameStack) to the env list."""
    while hasattr(venv, "venv"):
        venv = venv.venv
    return getattr(venv, "envs", None)


def evaluate_and_record(
    model,
    env_fn,
    cfg,
    run_dir: Path,
    seed: int,
    record_gif: bool = True,
    gif_episodes: int = 1,
    max_gif_steps: int = 3000,
):
    """Evaluate the policy and optionally record a gameplay GIF.

    Writes ``eval.json`` (mean/std reward over ``cfg['eval_episodes']``) and,
    when ``record_gif``, ``gameplay.gif``. Returns the stats dict.
    """
    run_dir = Path(run_dir)

    eval_env = env_fn(seed=seed + 10_000, n_envs=1, eval_mode=True)
    mean_r, std_r = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=int(cfg.get("eval_episodes", 20)),
        deterministic=True,
    )
    eval_env.close()

    stats = {
        "exp_id": cfg["exp_id"],
        "seed": seed,
        "n_eval_episodes": int(cfg.get("eval_episodes", 20)),
        "mean_reward": float(mean_r),
        "std_reward": float(std_r),
    }
    with open(run_dir / "eval.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[eval] {cfg['exp_id']} seed={seed}: "
          f"{mean_r:.2f} +/- {std_r:.2f} over {stats['n_eval_episodes']} eps")

    if record_gif:
        # Rendering is best-effort: a recording failure must never fail a run.
        try:
            _record_gif(model, env_fn, seed, run_dir, gif_episodes, max_gif_steps)
        except Exception as exc:  # noqa: BLE001
            print(f"[eval] GIF recording skipped: {exc}")

    return stats


def _record_gif(model, env_fn, seed, run_dir, gif_episodes, max_gif_steps):
    render_env = env_fn(
        seed=seed + 20_000, n_envs=1, eval_mode=True, render_mode="rgb_array"
    )
    base = _base_envs(render_env)
    frames = []
    obs = render_env.reset()
    episodes, steps = 0, 0
    while episodes < gif_episodes and steps < max_gif_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = render_env.step(action)
        if base is not None:
            frame = base[0].render()
            if frame is not None:
                frames.append(np.asarray(frame))
        steps += 1
        if bool(dones[0]):
            episodes += 1
    render_env.close()

    if frames:
        imageio.mimsave(run_dir / "gameplay.gif", frames, fps=30)
        print(f"[eval] saved {len(frames)} frames -> {run_dir / 'gameplay.gif'}")


def run_eval_cli():
    """CLI shared by ``pong/eval_record.py`` and ``vizdoom/eval_record.py``."""
    import argparse

    from stable_baselines3 import DQN, PPO

    from common.envs import build_env_fn
    from common.utils import get_run_dir, load_config

    algos = {"dqn": DQN, "ppo": PPO}

    ap = argparse.ArgumentParser(description="Re-evaluate and record a trained run.")
    ap.add_argument("--config", required=True, help="path to the run's YAML config")
    ap.add_argument("--seed", type=int, default=None, help="override config seed")
    ap.add_argument("--no-gif", action="store_true", help="skip GIF recording")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    seed = int(cfg.get("seed", 0))

    run_dir = get_run_dir(cfg["exp_id"], seed, create=False)
    model = algos[cfg["algo"]].load(str(run_dir / "model"))
    env_fn = build_env_fn(cfg)
    evaluate_and_record(
        model, env_fn, cfg, run_dir, seed, record_gif=not args.no_gif
    )
