# EXPERIMENTS — run journal

One row per training run (per seed, results aggregate later). Fill in
`result` and the 🧑 `takeaway` as runs land. **Numbers come only from
`results/<id>/eval.json` and `metrics.csv` — never typed by hand.**

Tier A guarantees a complete report; Tier B is full scope, queued after A.

| ID | Tier | Date | Task / Env | Algo | Key hyperparams | Seeds | Result (mean ± std) | 🧑 Takeaway (the "message") |
|---|---|---|---|---|---|---|---|---|
| P1 | A | | Pong `ALE/Pong-v5` | DQN | baseline; buffer 200k; tgt 1000; eps_frac 0.1; 2M steps | 0,1,2 | _TBD_ | _TBD_ |
| P2_targetoff | A | | Pong | DQN | target_update_interval=1 (no lag) | 0,1,2 | _TBD_ | _TBD_ |
| P3_epsfast | A | | Pong | DQN | exploration_fraction=0.02 | 0,1,2 | _TBD_ | _TBD_ |
| P3_epsslow | A | | Pong | DQN | exploration_fraction=0.5 | 0,1,2 | _TBD_ | _TBD_ |
| P4_buffersmall | B | | Pong | DQN | buffer_size=20k | 0,1,2 | _TBD_ | _TBD_ |
| P5_ppo_pong | B | | Pong | PPO | 8 envs; 10M steps | 0,1,2 | _TBD_ | _TBD_ |
| V0_basic | A | | VizDoom Basic | PPO | sanity; 8 envs; 300k steps | 0,1,2 | _TBD_ | _TBD_ |
| V1_defendcenter | A | | VizDoom Defend-Center | PPO | baseline; stack 4; Discrete; 2M | 0,1,2 | _TBD_ | _TBD_ |
| V2_multibinary | A | | VizDoom Defend-Center | PPO | max_buttons_pressed=0 (MultiDiscrete) | 0,1,2 | _TBD_ | _TBD_ |
| V3_healthgathering | B | | VizDoom Health Gathering | PPO | 3M steps | 0,1,2 | _TBD_ | _TBD_ |
| V4_stack1 | B | | VizDoom Defend-Center | PPO | n_stack=1 | 0,1,2 | _TBD_ | _TBD_ |
| V5_dqn_defendcenter | B | | VizDoom Defend-Center | DQN | buffer 200k; 2M steps | 0,1,2 | _TBD_ | _TBD_ |

## Reuse map (do NOT retrain)
- **P2** target-ON arm = **P1**.
- **P3** default-eps arm = **P1**.
- **P4** large-buffer arm = **P1**.
- **P5** DQN arm = **P1**.
- **V2** Discrete arm = **V1**.
- **V4** stack-4 arm = **V1**.
- **V5** PPO arm = **V1**.
- **V3** difficulty ladder reuses **V0** (Basic) + **V1** (Defend-Center).
