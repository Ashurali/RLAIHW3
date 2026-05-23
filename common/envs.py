"""Environment factories for both tasks, with a unified image front-end.

Both Pong and VizDoom are turned into a ``VecFrameStack``'d image ``VecEnv`` so
the exact same ``CnnPolicy`` code trains on each. :func:`build_env_fn` returns a
closure ``env_fn(seed, n_envs, eval_mode, render_mode)`` that the training core
uses for the training, evaluation and rendering environments alike.
"""
from __future__ import annotations

from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
)

from common.vizdoom_wrappers import VizDoomScreenWrapper


def make_atari_vec(env_id, n_envs, seed, n_stack=4, render_mode=None):
    """Atari pipeline: AtariWrapper (grayscale/resize/skip) + frame stack."""
    env_kwargs = {"render_mode": render_mode} if render_mode else None
    venv = make_atari_env(env_id, n_envs=n_envs, seed=seed, env_kwargs=env_kwargs)
    return VecFrameStack(venv, n_stack=n_stack)


def make_vizdoom_vec(
    env_id,
    n_envs,
    seed,
    n_stack=4,
    frame_skip=4,
    obs_shape=(84, 84),
    grayscale=True,
    env_kwargs=None,
    render_mode=None,
):
    """VizDoom pipeline mirroring Atari: screen -> 84x84 (+gray) + frame stack."""
    kwargs = dict(env_kwargs or {})
    kwargs.setdefault("frame_skip", frame_skip)
    if render_mode:
        kwargs["render_mode"] = render_mode

    # SubprocVecEnv parallelises env stepping (the CPU bottleneck) for PPO;
    # a single env (DQN / evaluation) stays in-process for simplicity.
    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    venv = make_vec_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=kwargs,
        wrapper_class=VizDoomScreenWrapper,
        wrapper_kwargs={"shape": tuple(obs_shape), "grayscale": grayscale},
        vec_env_cls=vec_cls,
    )
    return VecFrameStack(venv, n_stack=n_stack)


def build_env_fn(cfg):
    """Return ``env_fn(seed, n_envs, eval_mode, render_mode)`` for ``cfg['task']``."""
    task = cfg["task"]

    if task == "pong":
        def env_fn(seed, n_envs, eval_mode=False, render_mode=None):
            return make_atari_vec(
                cfg["env_id"],
                n_envs=1 if eval_mode else n_envs,
                seed=seed,
                n_stack=cfg.get("n_stack", 4),
                render_mode=render_mode,
            )

        return env_fn

    if task == "vizdoom":
        def env_fn(seed, n_envs, eval_mode=False, render_mode=None):
            return make_vizdoom_vec(
                cfg["env_id"],
                n_envs=1 if eval_mode else n_envs,
                seed=seed,
                n_stack=cfg.get("n_stack", 4),
                frame_skip=cfg.get("frame_skip", 4),
                obs_shape=cfg.get("obs_shape", (84, 84)),
                grayscale=cfg.get("grayscale", True),
                env_kwargs=cfg.get("env_kwargs"),
                render_mode=render_mode,
            )

        return env_fn

    raise ValueError(f"Unknown task '{task}' (expected 'pong' or 'vizdoom').")
