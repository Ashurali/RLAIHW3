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
