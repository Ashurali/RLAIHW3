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
