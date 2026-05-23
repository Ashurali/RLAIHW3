# RESULTS — aggregated, report-ready tables

Built from `results/` only (via `common/plotting.py` and `eval.json`). These
tables and the figures in `report_assets/` are the report's §3 raw material,
organized by the five threads (T1–T5) from the plan, not by experiment ID.

> All cells `_TBD_` until runs complete. Do not hand-enter numbers.

## T1 — Algorithm family (DQN vs PPO, both tasks)
Figure: `report_assets/T1_algo_family.png` (P1 vs P5, V1 vs V5).

| Task | DQN final reward | PPO final reward | Faster to learn | 🧑 Takeaway |
|---|---|---|---|---|
| Pong | _TBD (P1)_ | _TBD (P5)_ | _TBD_ | _TBD_ |
| VizDoom Defend-Center | _TBD (V5)_ | _TBD (V1)_ | _TBD_ | _TBD_ |

## T2 — DQN components (target net, ε, buffer)
Figure: `report_assets/T2_dqn_components.png` (P1/P2/P3/P4 curves).

| Variant | Config | Final reward | vs baseline P1 | 🧑 Takeaway |
|---|---|---|---|---|
| Baseline | P1 | _TBD_ | — | _TBD_ |
| Target net off | P2_targetoff | _TBD_ | _TBD_ | _TBD_ |
| ε fast decay | P3_epsfast | _TBD_ | _TBD_ | _TBD_ |
| ε slow decay | P3_epsslow | _TBD_ | _TBD_ | _TBD_ |
| Small buffer | P4_buffersmall | _TBD_ | _TBD_ | _TBD_ |

## T3 — Partial observability (frame stacking)
Figure: `report_assets/T3_framestack.png` (V1 stack-4 vs V4 stack-1).

| Env | Stack 4 | Stack 1 | Δ | 🧑 Takeaway |
|---|---|---|---|---|
| VizDoom Defend-Center | _TBD (V1)_ | _TBD (V4)_ | _TBD_ | _TBD_ |

## T4 — Action-space design
Figure: `report_assets/T4_actionspace.png` (V1 Discrete vs V2 MultiDiscrete).

| Action space | Config | Final reward | 🧑 Takeaway |
|---|---|---|---|
| Discrete (single button) | V1 | _TBD_ | _TBD_ |
| MultiDiscrete (combinations) | V2_multibinary | _TBD_ | _TBD_ |

## T5 — Task complexity (difficulty ladder)
Figure: `report_assets/T5_difficulty.png` (final reward vs scenario difficulty).

| Scenario | Config | Final reward | 🧑 Takeaway |
|---|---|---|---|
| Basic | V0 | _TBD_ | _TBD_ |
| Defend-Center | V1 | _TBD_ | _TBD_ |
| Health Gathering | V3 | _TBD_ | _TBD_ |

## Examples (S3 qualitative)
- Pong baseline gameplay: `results/P1_s0/gameplay.gif`
- VizDoom baseline gameplay: `results/V1_defendcenter_s0/gameplay.gif`
