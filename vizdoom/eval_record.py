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
