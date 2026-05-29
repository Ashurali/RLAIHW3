# Results summary (auto-generated)

Numbers are `mean ± std` of per-seed eval means. `n` = seeds used.


## T1 — Algorithm family (DQN vs PPO, equal env-step budget)

| Task | DQN | PPO |
|---|---|---|
| Pong (7M, original PPO recipe) | 4.77 ± 6.32 (n=3) | -6.58 ± 1.86 (n=3) |
| Pong (7M, SB3-zoo PPO recipe) | 4.77 ± 6.32 (n=3) | -6.07 ± 3.13 (n=3) |
| Defend-Center (2M each) | 8.85 ± 1.19 (n=3) | 9.37 ± 0.54 (n=3) |

## T2 — DQN components on Pong

| Variant | Reward |
|---|---|
| Baseline (P1) | 4.77 ± 6.32 (n=3) |
| Target net OFF (P2) | -6.92 ± 9.04 (n=3) |
| ε fast (P3a) | -7.73 ± 3.17 (n=3) |
| ε slow (P3b) | -7.90 ± 4.78 (n=3) |
| Small buffer (P4) | -5.03 ± 2.50 (n=3) |

## T3 — Partial observability (frame stacking, Defend-Center)

| Stack | Reward |
|---|---|
| 4 (V1) | 9.37 ± 0.54 (n=3) |
| 1 (V4) | 11.82 ± 0.88 (n=3) |

## T4 — Action-space design (Defend-Center)

| Action space | Reward |
|---|---|
| Discrete (V1) | 9.37 ± 0.54 (n=3) |
| MultiDiscrete (V2) | 6.92 ± 0.94 (n=3) |

## T5 — Difficulty ladder (Basic dropped)

| Scenario | Reward |
|---|---|
| Defend-Center (V1) | 9.37 ± 0.54 (n=3) |
| Health-Gathering (V3) | 321.33 ± 6.44 (n=3) |
