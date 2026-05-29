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
