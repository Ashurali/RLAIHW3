# Appendix B — Program code

The full source tree is reproduced below, grouped by role for top-down reading.
The same files are also available at <https://github.com/Ashurali/RLAIHW3>
(branch `main`); this appendix is a self-contained, frozen snapshot for grading.

Every source file is auto-extracted from the repository by
`tools/build_appendix.py`, so what you read here is exactly what trained the
models behind the results in §3. The code is documented inline — each module
opens with a docstring explaining its role, and non-obvious decisions
(schedule resolver, frame-stack wrapper, CSV-corruption fallback, GPU perf
flags) carry comments at point-of-use.


\newpage

## B.1  Common engine (algorithm-agnostic training & evaluation stack)

### `common/utils.py` (101 lines)

```python
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
```

### `common/envs.py` (95 lines)

```python
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

import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

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
```

### `common/vizdoom_wrappers.py` (63 lines)

```python
"""VizDoom -> SB3 image pipeline.

The Farama Gymnasium wrapper for ViZDoom returns a *Dict* observation: a
``screen`` image plus a ``gamevariables`` vector. SB3's ``CnnPolicy`` expects a
single image tensor, so this wrapper drops the game variables and converts the
screen to the same 84x84 (optionally grayscale) uint8 format used for Atari.

With a subsequent ``VecFrameStack(4)`` this reproduces the classic Atari DQN/PPO
front-end, giving Pong and VizDoom an identical observation pipeline so the
value-based vs policy-based comparison is not confounded by preprocessing.
"""
from __future__ import annotations

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Importing the wrapper module registers all "Vizdoom*-v*" Gymnasium ids as a
# side effect. Requires vizdoom >= 1.2 (Gymnasium support).
import vizdoom.gymnasium_wrapper  # noqa: F401


class VizDoomScreenWrapper(gym.ObservationWrapper):
    """Extract the screen buffer, resize it and optionally grayscale it.

    Output observation: a uint8 array of shape ``(H, W, C)`` (channel-last,
    ``C == 1`` when grayscale else ``3``). Channel-last is what SB3's
    ``VecTransposeImage`` expects before it auto-converts to channel-first.
    """

    def __init__(self, env, shape=(84, 84), grayscale: bool = True):
        super().__init__(env)
        self.shape = tuple(shape)
        self.grayscale = grayscale
        channels = 1 if grayscale else 3
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.shape[0], self.shape[1], channels),
            dtype=np.uint8,
        )

    def observation(self, obs):
        # The Dict obs carries the frame under "screen"; tolerate a bare array.
        screen = obs["screen"] if isinstance(obs, dict) else obs
        screen = np.asarray(screen)

        # Normalize to channel-last (H, W, C); some screen formats are (C, H, W).
        if screen.ndim == 3 and screen.shape[0] in (1, 3) and screen.shape[2] not in (1, 3):
            screen = np.transpose(screen, (1, 2, 0))

        if self.grayscale:
            screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)

        # cv2.resize takes (width, height); self.shape is (height, width).
        screen = cv2.resize(
            screen, (self.shape[1], self.shape[0]), interpolation=cv2.INTER_AREA
        )

        if self.grayscale:
            screen = screen[:, :, None]
        return screen.astype(np.uint8)
```

### `common/callbacks.py` (34 lines)

```python
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
```

### `common/eval_utils.py` (124 lines)

```python
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
```

### `common/plotting.py` (133 lines)

```python
"""Plot per-run learning curves and aggregate seeds into report figures.

Per the plan, every figure in the report must be built here from data under
``results/`` only -- never hand-edited. Outputs land in ``report_assets/``.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe on the training server
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common.utils import REPO_ROOT, RESULTS_DIR  # noqa: E402

X_COL = "time/total_timesteps"
Y_COL = "rollout/ep_rew_mean"
ASSETS_DIR = REPO_ROOT / "report_assets"


def _load_metrics(run_dir):
    """Return (timesteps, ep_rew_mean) arrays from a run's metrics CSV.

    Coerces both columns to numeric and drops non-numeric rows. This handles
    edge cases like concurrent-write corruption where a column might contain
    string garbage instead of floats; we keep whatever valid rows survive.
    Raises ValueError if no usable rows remain.
    """
    run_dir = Path(run_dir)
    path = run_dir / "metrics.csv"
    if not path.exists():
        path = run_dir / "progress.csv"
    df = pd.read_csv(path)
    df = df[[X_COL, Y_COL]].copy()
    df[X_COL] = pd.to_numeric(df[X_COL], errors="coerce")
    df[Y_COL] = pd.to_numeric(df[Y_COL], errors="coerce")
    df = df.dropna()
    if df.empty:
        raise ValueError(f"No usable rows in {path}")
    return df[X_COL].to_numpy(dtype=float), df[Y_COL].to_numpy(dtype=float)


def plot_curve(run_dir):
    """Single-run training curve -> ``<run_dir>/curve.png``."""
    x, y = _load_metrics(run_dir)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, color="tab:blue")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode reward (mean)")
    ax.set_title(Path(run_dir).name)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = Path(run_dir) / "curve.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def aggregate_seeds(exp_id, n_points=200):
    """Mean +/- std curve across ``results/<exp_id>_s*/`` runs.

    Curves are interpolated onto a shared timestep grid (the overlap of all
    seeds) so they can be averaged even when logging cadence differs. Runs
    with unparseable metrics CSVs (e.g. concurrent-write corruption) are
    skipped with a printed note; the aggregate proceeds across surviving
    seeds. Raises FileNotFoundError if NO seeds are usable.
    """
    run_dirs = sorted(glob.glob(str(RESULTS_DIR / f"{exp_id}_s*")))
    if not run_dirs:
        raise FileNotFoundError(f"No runs found for exp_id '{exp_id}'.")
    curves = []
    for d in run_dirs:
        try:
            curves.append(_load_metrics(d))
        except (ValueError, KeyError) as exc:
            print(f"  [aggregate_seeds] skip {Path(d).name}: {exc}")
    if not curves:
        raise FileNotFoundError(f"No usable metrics CSVs for exp_id '{exp_id}'.")
    x_min = max(c[0].min() for c in curves)
    x_max = min(c[0].max() for c in curves)
    grid = np.linspace(x_min, x_max, n_points)
    stacked = np.stack([np.interp(grid, x, y) for x, y in curves])
    return grid, stacked.mean(axis=0), stacked.std(axis=0)


def plot_comparison(exp_ids, labels=None, title="", out_name="comparison.png"):
    """Overlay mean +/- std curves for several experiments -> report_assets/."""
    labels = labels or exp_ids
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for exp_id, label in zip(exp_ids, labels):
        grid, mean, std = aggregate_seeds(exp_id)
        ax.plot(grid, mean, label=label)
        ax.fill_between(grid, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode reward (mean)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    out = ASSETS_DIR / out_name
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description="Plot RL training curves.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("curve", help="single-run curve from a run directory")
    pc.add_argument("run_dir")

    cmp = sub.add_parser("compare", help="overlay several experiments (seeds avg)")
    cmp.add_argument("--ids", nargs="+", required=True)
    cmp.add_argument("--labels", nargs="+")
    cmp.add_argument("--title", default="")
    cmp.add_argument("--out", default="comparison.png")

    args = parser.parse_args()
    if args.cmd == "curve":
        print(plot_curve(args.run_dir))
    elif args.cmd == "compare":
        print(plot_comparison(args.ids, args.labels, args.title, args.out))


if __name__ == "__main__":
    main()
```

### `common/train_core.py` (82 lines)

```python
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
```


\newpage

## B.2  Per-task training & evaluation entry points

### `pong/train_pong_dqn.py` (37 lines)

```python
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
```

### `pong/train_pong_ppo.py` (36 lines)

```python
"""Train PPO on Atari Pong (Task 1 -- policy-based arm of the P5 comparison).

Example:
    python pong/train_pong_ppo.py --config configs/P5_ppo_pong.yaml --seed 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repo root importable so "common" resolves when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_baselines3 import PPO  # noqa: E402

from common.train_core import run_training  # noqa: E402
from common.utils import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Train PPO on Atari Pong.")
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
```

### `vizdoom/train_vizdoom_dqn.py` (39 lines)

```python
"""Train DQN on a VizDoom scenario (Task 2 -- value-based arm of V5).

DQN needs a Discrete action space, so keep the default action config
(max_buttons_pressed = 1) for this run rather than the MultiDiscrete variant.

Example:
    python vizdoom/train_vizdoom_dqn.py --config configs/V5_dqn_defendcenter.yaml --seed 0
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
    ap = argparse.ArgumentParser(description="Train DQN on VizDoom.")
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
```

### `vizdoom/train_vizdoom_ppo.py` (39 lines)

```python
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
```

### `pong/eval_record.py` (17 lines)

```python
"""Re-evaluate a trained Pong model and (re)record its gameplay GIF.

Example:
    python pong/eval_record.py --config configs/P1.yaml --seed 0
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable so "common" resolves when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.eval_utils import run_eval_cli  # noqa: E402

if __name__ == "__main__":
    run_eval_cli()
```

### `vizdoom/eval_record.py` (17 lines)

```python
"""Re-evaluate a trained VizDoom model and (re)record its gameplay GIF.

Example:
    python vizdoom/eval_record.py --config configs/V1_defendcenter.yaml --seed 0
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable so "common" resolves when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.eval_utils import run_eval_cli  # noqa: E402

if __name__ == "__main__":
    run_eval_cli()
```


\newpage

## B.3  Experiment configs (one YAML per (task, algorithm, ablation))

### `configs/P1.yaml` (34 lines)

```yaml
# P1 - Pong DQN baseline (Tier A).
# Reference run. Also serves as the reused arm for several ablations:
#   P2 = "target net ON", P3 = "default epsilon", P4 = "large buffer".
# Run >=3 seeds for the report:  --seed 0 / 1 / 2
exp_id: P1
task: pong
algo: dqn
env_id: ALE/Pong-v5
policy: CnnPolicy
n_envs: 1
n_stack: 4
seed: 0
total_timesteps: 7000000      # ~3.8 h on the 4090; long enough to plateau (~+20).
                              # 2M left it undertrained (curve still rising at -3).
eval_freq: 100000
eval_episodes: 20
checkpoint_freq: 500000
device: cuda
record_gif: true
algo_kwargs:
  learning_rate: 1.0e-4
  buffer_size: 500000         # "large" buffer (~28GB RAM, frame-stacked uint8;
                              # pre-allocated, not dynamic). Fits the shared server
                              # (~42GB free). Raise to 1000000 only if dedicated.
                              # P4 uses a small buffer for the ablation.
  learning_starts: 100000
  batch_size: 32
  gamma: 0.99
  train_freq: 4
  gradient_steps: 1
  target_update_interval: 1000   # lagged target network ON (P2 turns this off)
  exploration_fraction: 0.1      # default epsilon schedule (P3 varies this)
  exploration_initial_eps: 1.0
  exploration_final_eps: 0.01
```

### `configs/P2_targetoff.yaml` (31 lines)

```yaml
# P2 - DQN with the lagged target network effectively OFF (Tier A).
# Aspect: stability. Setting target_update_interval = 1 syncs the target net to
# the online net every gradient step, removing the lag that tames the
# moving-target problem. Compare against P1 (target_update_interval = 1000).
# Everything else is identical to P1.
exp_id: P2_targetoff
task: pong
algo: dqn
env_id: ALE/Pong-v5
policy: CnnPolicy
n_envs: 1
n_stack: 4
seed: 0
total_timesteps: 7000000      # match P1 (extended from 2M)
eval_freq: 100000
eval_episodes: 20
checkpoint_freq: 500000
device: cuda
record_gif: true
algo_kwargs:
  learning_rate: 1.0e-4
  buffer_size: 500000         # match P1 (~28GB; sized for shared server RAM)
  learning_starts: 100000
  batch_size: 32
  gamma: 0.99
  train_freq: 4
  gradient_steps: 1
  target_update_interval: 1       # <-- target net lag removed (the ablation)
  exploration_fraction: 0.1
  exploration_initial_eps: 1.0
  exploration_final_eps: 0.01
```

### `configs/P3_epsfast.yaml` (30 lines)

```yaml
# P3a - DQN with FAST epsilon decay (Tier A).
# Aspect: exploration vs exploitation. Epsilon reaches its floor very early
# (exploration_fraction = 0.02), so the agent commits to exploitation quickly.
# Compare against P1 (0.1, default) and P3_epsslow (0.5, slow).
exp_id: P3_epsfast
task: pong
algo: dqn
env_id: ALE/Pong-v5
policy: CnnPolicy
n_envs: 1
n_stack: 4
seed: 0
total_timesteps: 7000000      # match P1 (extended from 2M)
eval_freq: 100000
eval_episodes: 20
checkpoint_freq: 500000
device: cuda
record_gif: true
algo_kwargs:
  learning_rate: 1.0e-4
  buffer_size: 500000         # match P1 (~28GB; sized for shared server RAM)
  learning_starts: 100000
  batch_size: 32
  gamma: 0.99
  train_freq: 4
  gradient_steps: 1
  target_update_interval: 1000
  exploration_fraction: 0.02      # <-- fast decay (the ablation)
  exploration_initial_eps: 1.0
  exploration_final_eps: 0.01
```

### `configs/P3_epsslow.yaml` (30 lines)

```yaml
# P3b - DQN with SLOW epsilon decay (Tier A).
# Aspect: exploration vs exploitation. Epsilon decays over half of training
# (exploration_fraction = 0.5), keeping the agent exploratory for much longer.
# Compare against P1 (0.1, default) and P3_epsfast (0.02, fast).
exp_id: P3_epsslow
task: pong
algo: dqn
env_id: ALE/Pong-v5
policy: CnnPolicy
n_envs: 1
n_stack: 4
seed: 0
total_timesteps: 7000000      # match P1 (extended from 2M)
eval_freq: 100000
eval_episodes: 20
checkpoint_freq: 500000
device: cuda
record_gif: true
algo_kwargs:
  learning_rate: 1.0e-4
  buffer_size: 500000         # match P1 (~28GB; sized for shared server RAM)
  learning_starts: 100000
  batch_size: 32
  gamma: 0.99
  train_freq: 4
  gradient_steps: 1
  target_update_interval: 1000
  exploration_fraction: 0.5       # <-- slow decay (the ablation)
  exploration_initial_eps: 1.0
  exploration_final_eps: 0.01
```

### `configs/P4_buffersmall.yaml` (30 lines)

```yaml
# P4 - DQN with a SMALL replay buffer (Tier B).
# Aspect: memory / sample correlation. A small buffer keeps only very recent,
# highly correlated transitions, weakening the i.i.d. assumption behind
# experience replay. Compare against P1 (buffer_size = 200000).
exp_id: P4_buffersmall
task: pong
algo: dqn
env_id: ALE/Pong-v5
policy: CnnPolicy
n_envs: 1
n_stack: 4
seed: 0
total_timesteps: 2000000      # T2 ablation budget; compare vs P1's 2M snapshot from metrics.csv
eval_freq: 100000
eval_episodes: 20
checkpoint_freq: 500000
device: cuda
record_gif: true
algo_kwargs:
  learning_rate: 1.0e-4
  buffer_size: 20000          # <-- small buffer (the ablation)
  learning_starts: 10000      # must be <= buffer_size
  batch_size: 32
  gamma: 0.99
  train_freq: 4
  gradient_steps: 1
  target_update_interval: 1000
  exploration_fraction: 0.1
  exploration_initial_eps: 1.0
  exploration_final_eps: 0.01
```

### `configs/P5_ppo_pong.yaml` (29 lines)

```yaml
# P5 - PPO on Pong (Tier B): policy-based arm of the algorithm-family thread.
# Aspect: algorithm family (value-based DQN vs policy-based PPO on the SAME task).
# Pair with P1 in the report. PPO needs many more frames than DQN on Atari, so
# total_timesteps is large; reduce it if compute/time is short.
exp_id: P5_ppo_pong
task: pong
algo: ppo
env_id: ALE/Pong-v5
policy: CnnPolicy
n_envs: 16                     # SubprocVecEnv parallel rollouts (~16 CPU threads)
n_stack: 4
seed: 0
total_timesteps: 7000000      # equalized with P1 (DQN) for a fair T1 comparison
                              # (same env-step budget; Pong converges < 7M for PPO)
eval_freq: 250000
eval_episodes: 20
checkpoint_freq: 1000000
device: cuda
record_gif: true
algo_kwargs:
  learning_rate: 2.5e-4
  n_steps: 128                # rollout length per env (16 * 128 = 2048 / update)
  batch_size: 512
  n_epochs: 4
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.1
  ent_coef: 0.01
  vf_coef: 0.5
```

### `configs/P5b_ppo_zoo.yaml` (37 lines)

```yaml
# P5b - PPO on Pong with the SB3-zoo / Schulman-2017 Atari recipe.
# Sanity check for T1: our original P5 used constant LR / clip_range and lost
# (-6.58 at 7M, far below literature). Literature PPO on Pong reaches +18-21
# at ~7-10M with LINEAR decay on both learning_rate and clip_range. This run
# applies the literature recipe to isolate whether T1's "DQN beats PPO" surface
# result was a hyperparameter artifact.
#
# Differences vs P5 (the originally-published numbers):
#   learning_rate:  2.5e-4  ->  lin_2.5e-4   (linear decay to 0)
#   clip_range:     0.1     ->  lin_0.1      (linear decay to 0)
#   batch_size:     512     ->  256          (zoo default; smaller minibatches)
# Everything else (n_envs, n_steps, n_epochs, gamma, ent_coef, vf_coef) matches
# P5 so the only thing being isolated is the LR / clip / batch-size recipe.
exp_id: P5b_ppo_zoo
task: pong
algo: ppo
env_id: ALE/Pong-v5
policy: CnnPolicy
n_envs: 16
n_stack: 4
seed: 0
total_timesteps: 7000000      # same budget as P5 (fair T1-side comparison)
eval_freq: 250000
eval_episodes: 20
checkpoint_freq: 1000000
device: cuda
record_gif: true
algo_kwargs:
  learning_rate: lin_2.5e-4   # SB3 linear schedule: 2.5e-4 -> 0
  n_steps: 128
  batch_size: 256             # zoo default
  n_epochs: 4
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: lin_0.1         # SB3 linear schedule: 0.1 -> 0
  ent_coef: 0.01
  vf_coef: 0.5
```

### `configs/V1_defendcenter.yaml` (32 lines)

```yaml
# V1 - VizDoom Defend-the-Center, PPO (Tier A): the 3D headline baseline.
# Reference for Task 2. Also the reused arm for:
#   V2 = "Discrete actions", V4 = "frame stack 4", V5 = "PPO".
# Run >=3 seeds:  --seed 0 / 1 / 2
# NOTE: confirm env id on first run (VizdoomDefendCenter-v0 expected).
exp_id: V1_defendcenter
task: vizdoom
algo: ppo
env_id: VizdoomDefendCenter-v1
policy: CnnPolicy
n_envs: 16
n_stack: 4
frame_skip: 4
obs_shape: [84, 84]
grayscale: true
seed: 0
total_timesteps: 2000000      # ~30-90 min on an RTX 4090
eval_freq: 100000
eval_episodes: 20
checkpoint_freq: 500000
device: cuda
record_gif: true
algo_kwargs:
  learning_rate: 2.5e-4
  n_steps: 128
  batch_size: 512
  n_epochs: 4
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.01
  vf_coef: 0.5
```

### `configs/V2_multibinary.yaml` (35 lines)

```yaml
# V2 - VizDoom Defend-the-Center, PPO, MULTI-button action space (Tier A).
# Aspect: action-space design (unique to VizDoom). The Farama wrapper exposes a
# Discrete space of single-button presses by default (max_buttons_pressed = 1).
# Setting max_buttons_pressed = 0 lets the agent press any combination of
# buttons at once -> a MultiDiscrete ("multi-binary"-style) action space.
# Compare against V1 (Discrete). PPO supports MultiDiscrete; DQN would not.
exp_id: V2_multibinary
task: vizdoom
algo: ppo
env_id: VizdoomDefendCenter-v1
policy: CnnPolicy
n_envs: 16
n_stack: 4
frame_skip: 4
obs_shape: [84, 84]
grayscale: true
env_kwargs:
  max_buttons_pressed: 0      # <-- multi-button (MultiDiscrete) action space
seed: 0
total_timesteps: 2000000
eval_freq: 100000
eval_episodes: 20
checkpoint_freq: 500000
device: cuda
record_gif: true
algo_kwargs:
  learning_rate: 2.5e-4
  n_steps: 128
  batch_size: 512
  n_epochs: 4
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.01
  vf_coef: 0.5
```

### `configs/V3_healthgathering.yaml` (31 lines)

```yaml
# V3 - VizDoom Health Gathering, PPO (Tier B): hardest scenario in the ladder.
# Aspect: task complexity. Used in the difficulty-scaling thread together with
# V0 (Basic) and V1 (Defend-Center): final reward vs scenario difficulty.
# NOTE: confirm env id on first run (VizdoomHealthGathering-v0 expected).
exp_id: V3_healthgathering
task: vizdoom
algo: ppo
env_id: VizdoomHealthGathering-v1
policy: CnnPolicy
n_envs: 16
n_stack: 4
frame_skip: 4
obs_shape: [84, 84]
grayscale: true
seed: 0
total_timesteps: 3000000      # harder exploration -> larger budget
eval_freq: 150000
eval_episodes: 20
checkpoint_freq: 500000
device: cuda
record_gif: true
algo_kwargs:
  learning_rate: 2.5e-4
  n_steps: 128
  batch_size: 512
  n_epochs: 4
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.01
  vf_coef: 0.5
```

### `configs/V4_stack1.yaml` (31 lines)

```yaml
# V4 - VizDoom Defend-the-Center, PPO, NO frame stacking (Tier B).
# Aspect: partial observability. A single frame (n_stack = 1) hides motion and
# direction, which matters more in a 3D first-person view than in 2D Pong.
# Compare against V1 (n_stack = 4).
exp_id: V4_stack1
task: vizdoom
algo: ppo
env_id: VizdoomDefendCenter-v1
policy: CnnPolicy
n_envs: 16
n_stack: 1                    # <-- no temporal context (the ablation)
frame_skip: 4
obs_shape: [84, 84]
grayscale: true
seed: 0
total_timesteps: 2000000
eval_freq: 100000
eval_episodes: 20
checkpoint_freq: 500000
device: cuda
record_gif: true
algo_kwargs:
  learning_rate: 2.5e-4
  n_steps: 128
  batch_size: 512
  n_epochs: 4
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.01
  vf_coef: 0.5
```

### `configs/V5_dqn_defendcenter.yaml` (32 lines)

```yaml
# V5 - VizDoom Defend-the-Center, DQN (Tier B): value-based arm, mirror of P5.
# Aspect: algorithm family (DQN vs PPO on the SAME 3D task). Pair with V1 in the
# report. Keeps the default Discrete action space (DQN requires it).
exp_id: V5_dqn_defendcenter
task: vizdoom
algo: dqn
env_id: VizdoomDefendCenter-v1
policy: CnnPolicy
n_envs: 1
n_stack: 4
frame_skip: 4
obs_shape: [84, 84]
grayscale: true
seed: 0
total_timesteps: 2000000
eval_freq: 100000
eval_episodes: 20
checkpoint_freq: 500000
device: cuda
record_gif: true
algo_kwargs:
  learning_rate: 1.0e-4
  buffer_size: 500000         # large buffer (~28GB; sized for shared server RAM)
  learning_starts: 50000
  batch_size: 32
  gamma: 0.99
  train_freq: 4
  gradient_steps: 1
  target_update_interval: 1000
  exploration_fraction: 0.2
  exploration_initial_eps: 1.0
  exploration_final_eps: 0.05
```


\newpage

## B.4  Analysis pipeline (figures, summary tables, this appendix)

### `tools/build_report_assets.py` (285 lines)

```python
"""Build the report's grouped T1-T5 figures + numbered summary tables.

Reads ``results/*/eval.json`` and ``results/*/metrics.csv`` after a fetch, and
writes everything the report needs into ``report_assets/``:

  T1_algo_family.png    Pong DQN vs PPO  +  Defend-Center PPO vs DQN  (subplots)
  T2_dqn_components.png Pong DQN ablations (baseline, target-off, eps, buffer)
  T3_framestack.png     VizDoom Defend-Center stack-4 vs stack-1
  T4_actionspace.png    VizDoom Defend-Center Discrete vs MultiDiscrete
  T5_difficulty.png     Final reward by VizDoom scenario (DC, HG)
  summary.md            Auto-filled tables for RESULTS.md / the report

Experiments without any results are skipped gracefully (the script just prints
which ones it couldn't build). Run after ``pwsh deploy/fetch.ps1 -Lite``:

    python tools/build_report_assets.py
"""
from __future__ import annotations

import glob
import json
import statistics
import sys
from pathlib import Path

# This repo's path contains CJK characters; force UTF-8 stdout so our prints
# don't crash on the Windows cp1252 console.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Make ``common`` importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.plotting import aggregate_seeds  # noqa: E402
from common.utils import REPO_ROOT, RESULTS_DIR  # noqa: E402

ASSETS = REPO_ROOT / "report_assets"
ASSETS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def eval_summary(exp_id: str):
    """(n_seeds, mean, std-across-seed-means, [per-seed mean rewards])."""
    rewards = []
    for f in sorted(glob.glob(str(RESULTS_DIR / f"{exp_id}_s*" / "eval.json"))):
        with open(f, encoding="utf-8") as fh:
            rewards.append(json.load(fh)["mean_reward"])
    if not rewards:
        return 0, float("nan"), float("nan"), []
    mean = statistics.mean(rewards)
    std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    return len(rewards), mean, std, rewards


def fmt(stats):
    n, m, s, _ = stats
    return "—" if n == 0 else f"{m:.2f} ± {s:.2f} (n={n})"


def overlay_curves(exp_ids, labels, ax, title):
    """Plot mean±std learning curves for several experiments on one axis."""
    plotted_any = False
    for exp_id, label in zip(exp_ids, labels):
        try:
            x, mean, std = aggregate_seeds(exp_id)
            ax.plot(x, mean, label=label)
            ax.fill_between(x, mean - std, mean + std, alpha=0.2)
            plotted_any = True
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {exp_id}: {exc}")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode reward (mean ± std across seeds)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if plotted_any:
        ax.legend()
    return plotted_any


# ---------------------------------------------------------------------------
# Per-thread figure builders
# ---------------------------------------------------------------------------
def build_T1():
    """T1a now overlays THREE Pong curves: DQN baseline, original PPO, and the
    SB3-zoo / Schulman 2017 literature-recipe PPO (linear LR + clip decay).
    The zoo recipe is the sanity-check for whether the DQN > PPO inversion at
    7 M is a hyperparameter artifact (it isn't — both PPO variants lose to DQN).
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    overlay_curves(
        ["P1", "P5_ppo_pong", "P5b_ppo_zoo"],
        ["DQN (P1)", "PPO original (P5)", "PPO zoo recipe (P5b)"],
        axes[0],
        "T1a — Pong: DQN vs PPO (two hyperparam recipes)",
    )
    overlay_curves(
        ["V1_defendcenter", "V5_dqn_defendcenter"],
        ["PPO (V1)", "DQN (V5)"],
        axes[1],
        "T1b — Defend-Center: policy-based vs value-based",
    )
    fig.tight_layout()
    out = ASSETS / "T1_algo_family.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def build_T2():
    fig, ax = plt.subplots(figsize=(8.5, 5))
    overlay_curves(
        ["P1", "P2_targetoff", "P3_epsfast", "P3_epsslow", "P4_buffersmall"],
        ["Baseline (P1)", "Target net OFF (P2)", "ε fast decay (P3a)",
         "ε slow decay (P3b)", "Small buffer (P4)"],
        ax,
        "T2 — DQN components on Pong (2 M ablation budget)",
    )
    # Clip to the 2 M ablation budget. P1 was additionally extended to 7 M for
    # T1, and one P2 seed completed 7 M, but T2 is anchored at 2 M for a clean
    # 3-seed comparison across all ablations.
    ax.set_xlim(0, 2.05e6)
    fig.tight_layout()
    out = ASSETS / "T2_dqn_components.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def build_T3():
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    overlay_curves(
        ["V1_defendcenter", "V4_stack1"],
        ["Stack 4 (V1)", "Stack 1 (V4)"],
        ax,
        "T3 — Frame stacking on VizDoom Defend-Center",
    )
    fig.tight_layout()
    out = ASSETS / "T3_framestack.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def build_T4():
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    overlay_curves(
        ["V1_defendcenter", "V2_multibinary"],
        ["Discrete (V1)", "MultiDiscrete (V2)"],
        ax,
        "T4 — Action-space design on VizDoom Defend-Center",
    )
    fig.tight_layout()
    out = ASSETS / "T4_actionspace.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def build_T5():
    """Final-reward bar chart across difficulty (DC, HG). Different scenarios
    have different reward scales, so a bar chart of final scores is clearer
    than overlaying learning curves on a shared y-axis. Basic (V0) is dropped.
    """
    exps = ["V1_defendcenter", "V3_healthgathering"]
    labels = ["Defend-Center (V1)", "Health-Gathering (V3)"]
    means, stds, ns = [], [], []
    for e in exps:
        n, m, s, _ = eval_summary(e)
        means.append(m if n else 0)
        stds.append(s if n else 0)
        ns.append(n)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(labels, means, yerr=stds, capsize=6,
                  color=["tab:blue", "tab:orange"])
    ax.set_ylabel("Final eval reward (mean of seed means)")
    ax.set_title("T5 — Difficulty ladder (final reward by scenario)")
    ax.grid(alpha=0.3, axis="y")
    for bar, m, n in zip(bars, means, ns):
        label = f"{m:.1f} (n={n})" if n else "no data"
        ax.text(bar.get_x() + bar.get_width() / 2,
                m + (0.02 * max(abs(np.array(means)).max(), 1)),
                label, ha="center", va="bottom")
    fig.tight_layout()
    out = ASSETS / "T5_difficulty.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------
def build_summary_md():
    p1 = eval_summary("P1")
    p2 = eval_summary("P2_targetoff")
    p3a = eval_summary("P3_epsfast")
    p3b = eval_summary("P3_epsslow")
    p4 = eval_summary("P4_buffersmall")
    p5 = eval_summary("P5_ppo_pong")
    p5b = eval_summary("P5b_ppo_zoo")
    v1 = eval_summary("V1_defendcenter")
    v2 = eval_summary("V2_multibinary")
    v3 = eval_summary("V3_healthgathering")
    v4 = eval_summary("V4_stack1")
    v5 = eval_summary("V5_dqn_defendcenter")

    L = []
    L.append("# Results summary (auto-generated)\n")
    L.append("Numbers are `mean ± std` of per-seed eval means. `n` = seeds used.\n")

    L.append("\n## T1 — Algorithm family (DQN vs PPO, equal env-step budget)\n")
    L.append("| Task | DQN | PPO |")
    L.append("|---|---|---|")
    L.append(f"| Pong (7M, original PPO recipe) | {fmt(p1)} | {fmt(p5)} |")
    L.append(f"| Pong (7M, SB3-zoo PPO recipe) | {fmt(p1)} | {fmt(p5b)} |")
    L.append(f"| Defend-Center (2M each) | {fmt(v5)} | {fmt(v1)} |")

    L.append("\n## T2 — DQN components on Pong\n")
    L.append("| Variant | Reward |")
    L.append("|---|---|")
    L.append(f"| Baseline (P1) | {fmt(p1)} |")
    L.append(f"| Target net OFF (P2) | {fmt(p2)} |")
    L.append(f"| ε fast (P3a) | {fmt(p3a)} |")
    L.append(f"| ε slow (P3b) | {fmt(p3b)} |")
    L.append(f"| Small buffer (P4) | {fmt(p4)} |")

    L.append("\n## T3 — Partial observability (frame stacking, Defend-Center)\n")
    L.append("| Stack | Reward |")
    L.append("|---|---|")
    L.append(f"| 4 (V1) | {fmt(v1)} |")
    L.append(f"| 1 (V4) | {fmt(v4)} |")

    L.append("\n## T4 — Action-space design (Defend-Center)\n")
    L.append("| Action space | Reward |")
    L.append("|---|---|")
    L.append(f"| Discrete (V1) | {fmt(v1)} |")
    L.append(f"| MultiDiscrete (V2) | {fmt(v2)} |")

    L.append("\n## T5 — Difficulty ladder (Basic dropped)\n")
    L.append("| Scenario | Reward |")
    L.append("|---|---|")
    L.append(f"| Defend-Center (V1) | {fmt(v1)} |")
    L.append(f"| Health-Gathering (V3) | {fmt(v3)} |")

    out = ASSETS / "summary.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


def main():
    builders = [
        ("T1 algo family", build_T1),
        ("T2 DQN components", build_T2),
        ("T3 frame stack", build_T3),
        ("T4 action space", build_T4),
        ("T5 difficulty", build_T5),
        ("summary.md", build_summary_md),
    ]
    for name, fn in builders:
        try:
            out = fn()
            # Print a repo-relative path so the CJK-laden absolute path doesn't
            # leak into the console output.
            try:
                rel = Path(out).relative_to(REPO_ROOT)
            except ValueError:
                rel = Path(out).name
            print(f"[ok]    {name:20s} -> {rel}")
        except Exception as exc:  # noqa: BLE001
            print(f"[skip]  {name:20s} ({exc})")


if __name__ == "__main__":
    main()
```

### `tools/build_appendix.py` (164 lines)

```python
"""Auto-generate report/APPENDIX_CODE.md from the source tree.

The course requirement is: *include your program code as an appendix (not
counting toward the 10-page limit)*. Rather than hand-paste files into the
report -- which goes stale the moment any source file is edited -- this
script walks a curated list of paths and emits one Markdown file with the
content of each, wrapped in fenced code blocks with the right language tag.

Usage::

    python tools/build_appendix.py [output_path]
    # defaults to report/APPENDIX_CODE.md if no path is given

That generated file is then appended to ``report/REPORT.md`` (or included at
pandoc time) so the PDF carries the full, current source code as Appendix B.
We write directly to a UTF-8 file (not stdout) because the Windows console's
cp1252 codec cannot encode characters like the ``ε`` literal in our code.

Grouping is by role so a reader can navigate top-down:

  1. Common engine       (the algorithm-agnostic training/eval stack)
  2. Per-task entry      (train_*.py and eval_record.py for each task/algo)
  3. Experiment configs  (one YAML per experiment - the experiment design)
  4. Analysis pipeline   (tools/* -- figures, summary tables, this appendix)
  5. Deployment helpers  (selected SSH/scp scripts for the no-git server)

Files NOT included on purpose:
  - common/__init__.py            empty package marker
  - smoke_test.py                 dev-time only, not part of experiments
  - deploy/*.ps1                  Windows-specific workflow plumbing
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def lang_for(path: Path) -> str:
    """Best-effort fenced-code-block language tag for the file's extension."""
    suf = path.suffix.lower()
    return {
        ".py": "python",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".sh": "bash",
        ".ps1": "powershell",
        ".md": "markdown",
        ".json": "json",
    }.get(suf, "text")


# Order matters: this is the order the appendix presents the project in.
GROUPS = [
    (
        "B.1  Common engine (algorithm-agnostic training & evaluation stack)",
        [
            "common/utils.py",
            "common/envs.py",
            "common/vizdoom_wrappers.py",
            "common/callbacks.py",
            "common/eval_utils.py",
            "common/plotting.py",
            "common/train_core.py",
        ],
    ),
    (
        "B.2  Per-task training & evaluation entry points",
        [
            "pong/train_pong_dqn.py",
            "pong/train_pong_ppo.py",
            "vizdoom/train_vizdoom_dqn.py",
            "vizdoom/train_vizdoom_ppo.py",
            "pong/eval_record.py",
            "vizdoom/eval_record.py",
        ],
    ),
    (
        "B.3  Experiment configs (one YAML per (task, algorithm, ablation))",
        [
            "configs/P1.yaml",
            "configs/P2_targetoff.yaml",
            "configs/P3_epsfast.yaml",
            "configs/P3_epsslow.yaml",
            "configs/P4_buffersmall.yaml",
            "configs/P5_ppo_pong.yaml",
            "configs/P5b_ppo_zoo.yaml",
            "configs/V1_defendcenter.yaml",
            "configs/V2_multibinary.yaml",
            "configs/V3_healthgathering.yaml",
            "configs/V4_stack1.yaml",
            "configs/V5_dqn_defendcenter.yaml",
        ],
    ),
    (
        "B.4  Analysis pipeline (figures, summary tables, this appendix)",
        [
            "tools/build_report_assets.py",
            "tools/build_appendix.py",
        ],
    ),
    (
        "B.5  Deployment helpers (selected SSH/scp scripts)",
        [
            "deploy/_activate.sh",
            "deploy/remote_setup.sh",
            "deploy/remote_train.sh",
            "deploy/remote_queue.sh",
            "deploy/remote_queue_round3.sh",
        ],
    ),
]


def emit_header(out):
    out.write(
        """# Appendix B — Program code

The full source tree is reproduced below, grouped by role for top-down reading.
The same files are also available at <https://github.com/Ashurali/RLAIHW3>
(branch `main`); this appendix is a self-contained, frozen snapshot for grading.

Every source file is auto-extracted from the repository by
`tools/build_appendix.py`, so what you read here is exactly what trained the
models behind the results in §3. The code is documented inline — each module
opens with a docstring explaining its role, and non-obvious decisions
(schedule resolver, frame-stack wrapper, CSV-corruption fallback, GPU perf
flags) carry comments at point-of-use.

"""
    )


def emit_file(out, rel_path: str):
    path = REPO / rel_path
    if not path.exists():
        out.write(f"### `{rel_path}`\n\n*(file missing at appendix-build time — skipped)*\n\n")
        return
    content = path.read_text(encoding="utf-8")
    # Strip trailing whitespace on lines; keep blank lines.
    content = "\n".join(line.rstrip() for line in content.splitlines())
    n_lines = len(content.splitlines())
    lang = lang_for(path)
    out.write(f"### `{rel_path}` ({n_lines} lines)\n\n")
    out.write(f"```{lang}\n{content}\n```\n\n")


def main():
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "report" / "APPENDIX_CODE.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        emit_header(out)
        for title, files in GROUPS:
            out.write(f"\n\\newpage\n\n## {title}\n\n")
            for rel in files:
                emit_file(out, rel)
    # Lightweight summary to stdout (ASCII only, safe for Windows console).
    n_lines = sum(1 for _ in out_path.open(encoding="utf-8"))
    print(f"wrote {out_path.relative_to(REPO)} ({n_lines} lines)")


if __name__ == "__main__":
    main()
```


\newpage

## B.5  Deployment helpers (selected SSH/scp scripts)

### `deploy/_activate.sh` (68 lines)

```bash
#!/usr/bin/env bash
# Sourced by the remote deploy scripts to create/activate the training env.
# Supports two backends so it works with or without sudo:
#   ENV_KIND=venv   -> python -m venv  (needs python3.x-venv / ensurepip)
#   ENV_KIND=conda  -> conda env       (no system packages / sudo needed)
#
# Reads (with defaults) from the environment, passed by deploy/remote.ps1:
#   ENV_KIND          conda | venv                 (default: conda)
#   VENV              conda env name OR venv dir    (default: hw3 / REMOTE_VENV)
#   PY_VERSION        python for the conda env      (default: 3.12)
#   PY_BIN            interpreter for venv creation (auto-detect if empty)
#   REMOTE_CONDA_BASE conda base dir override       (auto-detect if empty)

VENV="${VENV:-${REMOTE_VENV:-hw3}}"
ENV_KIND="${ENV_KIND:-conda}"
PY_VERSION="${PY_VERSION:-3.12}"
PY_BIN="${PY_BIN:-}"

_ensure_conda() {
  command -v conda >/dev/null 2>&1 && return 0
  local c
  for c in "${REMOTE_CONDA_BASE:-}" "$HOME/miniconda3" "$HOME/anaconda3" \
           "$HOME/miniforge3" "$HOME/mambaforge" /opt/conda /opt/anaconda3; do
    if [ -n "$c" ] && [ -f "$c/etc/profile.d/conda.sh" ]; then
      # shellcheck disable=SC1091
      source "$c/etc/profile.d/conda.sh"
      return 0
    fi
  done
  echo "ERROR: cannot locate conda. Set REMOTE_CONDA_BASE in deploy/server.env" >&2
  echo "       to the output of 'conda info --base' on the server." >&2
  return 1
}

_pick_py() {
  local p
  for p in python3.12 python3.11 python3.10 python3; do
    command -v "$p" >/dev/null 2>&1 && { echo "$p"; return 0; }
  done
  return 1
}

# Activate an already-created environment into the current shell.
activate_env() {
  if [ "$ENV_KIND" = "conda" ]; then
    _ensure_conda || return 1
    conda activate "$VENV"
  else
    # shellcheck disable=SC1091
    conda activate "$VENV"
  fi
}

# Create the environment, then activate it.
create_env() {
  if [ "$ENV_KIND" = "conda" ]; then
    _ensure_conda || return 1
    echo "conda env: $VENV (python=$PY_VERSION) under $(conda info --base)"
    conda create -y -n "$VENV" "python=$PY_VERSION"
    conda activate "$VENV"
  else
    local py="${PY_BIN:-$(_pick_py)}"
    echo "venv interpreter: $py = $($py --version 2>&1)"
    "$py" -m venv "$VENV"
    # shellcheck disable=SC1091
    conda activate $VENV
  fi
}
```

### `deploy/remote_setup.sh` (55 lines)

```bash
#!/usr/bin/env bash
# One-time server setup, DETACHED (survives SSH logout). Probes the hardware,
# creates the conda env, installs requirements, and verifies CUDA.
#
# Usage (via deploy/remote.ps1 -Action setup, or directly on the server):
#   REMOTE_VENV=hw3 bash deploy/remote_setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."                      # repo root on the server
VENV="${REMOTE_VENV:-hw3}"
ENV_KIND="${ENV_KIND:-conda}"
export REMOTE_VENV ENV_KIND PY_VERSION PY_BIN REMOTE_CONDA_BASE
mkdir -p logs

run_setup() {
  echo "=== server probe ($(date '+%F %T')) ==="
  if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; else echo "nvidia-smi not found"; fi
  echo "CPUs(threads): $(nproc 2>/dev/null || echo '?')"
  echo "RAM: $(free -h 2>/dev/null | awk '/Mem:/{print $2}' || echo '?')"
  echo "Python(default): $(python3 --version 2>&1)"
  echo "Available interpreters:"
  for p in python3.12 python3.11 python3.10; do
    command -v "$p" >/dev/null 2>&1 && echo "  - $p = $($p --version 2>&1)"
  done

  # Create the env via deploy/_activate.sh (conda or venv per ENV_KIND). conda
  # needs no sudo / system packages, which is why it's the default on the lab box.
  echo "=== creating environment ($ENV_KIND -> $VENV) ==="
  source deploy/_activate.sh
  set +u; create_env; set -u
  python -m pip install --upgrade pip wheel

  echo "=== installing requirements (CUDA torch via extra-index-url) ==="
  pip install -r requirements.txt

  echo "=== verifying torch sees the GPU ==="
  python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0), "| count:", torch.cuda.device_count())
    print("cuda runtime:", torch.version.cuda)
else:
    print("WARNING: CUDA not available -- training will fall back to CPU.")
PY
  echo "SETUP_DONE"
}

# Re-exec a detached worker so the install survives SSH logout.
if [ "${1:-}" = "--worker" ]; then run_setup; exit 0; fi

log="logs/setup.log"
nohup bash "$0" --worker > "$log" 2>&1 < /dev/null &
echo $! > logs/setup.pid
echo "STARTED setup  pid=$(cat logs/setup.pid)"
echo "  log: $log    (wait for SETUP_DONE:  tail -f $log)"
```

### `deploy/remote_train.sh` (36 lines)

```bash
#!/usr/bin/env bash
# Launch ONE training run DETACHED on the server (survives SSH logout).
# Output goes to logs/<exp>_s<seed>.log so progress is trackable with tail -f.
#
# Usage (run from the repo root on the server, or via deploy/remote.ps1):
#   REMOTE_VENV=hw3 bash deploy/remote_train.sh configs/P1.yaml 0
set -euo pipefail
cd "$(dirname "$0")/.."                      # repo root on the server

CONFIG="${1:?usage: remote_train.sh <config.yaml> [seed]}"
SEED="${2:-0}"
VENV="${REMOTE_VENV:-hw3}"
mkdir -p logs

name="$(basename "$CONFIG" .yaml)"
task="$(awk '/^task:/{print $2; exit}' "$CONFIG")"
algo="$(awk '/^algo:/{print $2; exit}' "$CONFIG")"
case "${task}_${algo}" in
  pong_dqn)    script="pong/train_pong_dqn.py" ;;
  pong_ppo)    script="pong/train_pong_ppo.py" ;;
  vizdoom_ppo) script="vizdoom/train_vizdoom_ppo.py" ;;
  vizdoom_dqn) script="vizdoom/train_vizdoom_dqn.py" ;;
  *) echo "ERROR: cannot map task='$task' algo='$algo' from $CONFIG" >&2; exit 1 ;;
esac

log="logs/${name}_s${SEED}.log"
pidf="logs/${name}_s${SEED}.pid"

# nohup + redirect + </dev/null detaches the job so it keeps running after the
# SSH session closes. -u keeps Python output unbuffered for live tailing.
nohup bash -c "source deploy/_activate.sh && activate_env && exec python -u '$script' --config '$CONFIG' --seed '$SEED'" \
  > "$log" 2>&1 < /dev/null &
echo $! > "$pidf"

echo "STARTED $name seed=$SEED  pid=$(cat "$pidf")  script=$script"
echo "  log: $log    (track:  tail -f $log)"
```

### `deploy/remote_queue.sh` (67 lines)

```bash
#!/usr/bin/env bash
# Run a LIST of experiments SEQUENTIALLY, detached (survives SSH logout).
# One GPU -> run jobs back-to-back rather than thrashing in parallel.
# Master log: logs/queue_<timestamp>.log ; per-run: logs/<exp>_s<seed>.log
#
# Usage (via deploy/remote.ps1 -Action queue, or directly):
#   REMOTE_VENV=hw3 bash deploy/remote_queue.sh
set -euo pipefail
cd "$(dirname "$0")/.."                      # repo root on the server
VENV="${REMOTE_VENV:-hw3}"
mkdir -p logs

# --- EDIT THIS QUEUE -------------------------------------------------------
# Each entry is "<config_basename> <seed>". Tier A x3 seeds shown by default;
# uncomment Tier B lines to queue the full scope.
LIST=(
  # Round 3: catch-up run on 2026-05-29 to fill the T1b gap.
  # ONLY V5 (DQN on Defend-Center), 3 seeds, 2M steps. Everything else is done.
  "V5_dqn_defendcenter 0" "V5_dqn_defendcenter 1" "V5_dqn_defendcenter 2"
)
# ---------------------------------------------------------------------------

map_script() {
  local cfg="$1" task algo
  task="$(awk '/^task:/{print $2; exit}' "$cfg")"
  algo="$(awk '/^algo:/{print $2; exit}' "$cfg")"
  case "${task}_${algo}" in
    pong_dqn)    echo "pong/train_pong_dqn.py" ;;
    pong_ppo)    echo "pong/train_pong_ppo.py" ;;
    vizdoom_ppo) echo "vizdoom/train_vizdoom_ppo.py" ;;
    vizdoom_dqn) echo "vizdoom/train_vizdoom_dqn.py" ;;
    *) return 1 ;;
  esac
}

run_queue() {
  source deploy/_activate.sh
  activate_env
  local n=${#LIST[@]} i=0
  for entry in "${LIST[@]}"; do
    i=$((i + 1))
    # shellcheck disable=SC2086
    set -- $entry
    local name="$1" seed="$2" cfg="configs/$1.yaml" script rlog
    if [ ! -f "$cfg" ]; then echo "[$i/$n] SKIP $name: $cfg not found"; continue; fi
    if ! script="$(map_script "$cfg")"; then echo "[$i/$n] SKIP $name: cannot map task/algo"; continue; fi
    rlog="logs/${name}_s${seed}.log"
    echo "=== [$i/$n] $(date '+%F %T') START $name seed=$seed -> $rlog ==="
    if python -u "$script" --config "$cfg" --seed "$seed" > "$rlog" 2>&1; then
      echo "=== [$i/$n] $(date '+%F %T') DONE  $name seed=$seed ==="
    else
      echo "=== [$i/$n] $(date '+%F %T') FAIL  $name seed=$seed (see $rlog) ==="
    fi
  done
  echo "QUEUE_DONE"
}

# Re-exec a detached worker so the whole queue survives SSH logout.
if [ "${1:-}" = "--worker" ]; then run_queue; exit 0; fi

ts="$(date +%Y%m%d_%H%M%S)"
qlog="logs/queue_${ts}.log"
nohup bash "$0" --worker > "$qlog" 2>&1 < /dev/null &
echo $! > logs/queue.pid
echo "STARTED queue (${#LIST[@]} runs)  pid=$(cat logs/queue.pid)"
echo "  master log: $qlog    (track:  tail -f $qlog)"
echo "  per-run logs: logs/<exp>_s<seed>.log"
```

### `deploy/remote_queue_round3.sh` (63 lines)

```bash
#!/usr/bin/env bash
# Round 3 catch-up queue: runs AFTER V5 finishes.
#   - P4_buffersmall × 3 seeds at 2M (T2 small-buffer gap)
#   - P5b_ppo_zoo × 3 seeds at 7M (T1 PPO literature-recipe sanity check)
#
# Same nohup / detached pattern as remote_queue.sh. Launched by the V5 watcher
# when the round-2 queue master exits.
set -euo pipefail
cd "$(dirname "$0")/.."                      # repo root on the server
VENV="${REMOTE_VENV:-hw3}"
mkdir -p logs

LIST=(
  # Cheap small-buffer ablation first (2M, single-env DQN, ~30-60min/seed).
  "P4_buffersmall 0" "P4_buffersmall 1" "P4_buffersmall 2"
  # Literature-recipe PPO last (7M, 16 envs, ~2h/seed at idle GPU).
  "P5b_ppo_zoo 0" "P5b_ppo_zoo 1" "P5b_ppo_zoo 2"
)

map_script() {
  local cfg="$1" task algo
  task="$(awk '/^task:/{print $2; exit}' "$cfg")"
  algo="$(awk '/^algo:/{print $2; exit}' "$cfg")"
  case "${task}_${algo}" in
    pong_dqn)    echo "pong/train_pong_dqn.py" ;;
    pong_ppo)    echo "pong/train_pong_ppo.py" ;;
    vizdoom_ppo) echo "vizdoom/train_vizdoom_ppo.py" ;;
    vizdoom_dqn) echo "vizdoom/train_vizdoom_dqn.py" ;;
    *) return 1 ;;
  esac
}

run_queue() {
  source deploy/_activate.sh
  activate_env
  local n=${#LIST[@]} i=0
  for entry in "${LIST[@]}"; do
    i=$((i + 1))
    # shellcheck disable=SC2086
    set -- $entry
    local name="$1" seed="$2" cfg="configs/$1.yaml" script rlog
    if [ ! -f "$cfg" ]; then echo "[$i/$n] SKIP $name: $cfg not found"; continue; fi
    if ! script="$(map_script "$cfg")"; then echo "[$i/$n] SKIP $name: cannot map task/algo"; continue; fi
    rlog="logs/${name}_s${seed}.log"
    echo "=== [$i/$n] $(date '+%F %T') START $name seed=$seed -> $rlog ==="
    if python -u "$script" --config "$cfg" --seed "$seed" > "$rlog" 2>&1; then
      echo "=== [$i/$n] $(date '+%F %T') DONE  $name seed=$seed ==="
    else
      echo "=== [$i/$n] $(date '+%F %T') FAIL  $name seed=$seed (see $rlog) ==="
    fi
  done
  echo "QUEUE_DONE"
}

if [ "${1:-}" = "--worker" ]; then run_queue; exit 0; fi

ts="$(date +%Y%m%d_%H%M%S)"
qlog="logs/queue_round3_${ts}.log"
nohup bash "$0" --worker > "$qlog" 2>&1 < /dev/null &
echo $! > logs/queue_round3.pid
echo "STARTED round3 queue (${#LIST[@]} runs)  pid=$(cat logs/queue_round3.pid)"
echo "  master log: $qlog    (track:  tail -f $qlog)"
echo "  per-run logs: logs/<exp>_s<seed>.log"
```

