# AI HW3 — Reinforcement Learning: Plan of Work (v4 — full scope)

**Course:** Artificial Intelligence, NYCU Spring 2026 (Prof. Tsaipei Wang / 王才沛)
**Due:** 2026-05-29 (Fri). Late ≤ 5 days at −10%/day. Target: on time.
**Deliverable:** PDF report ≤ 10 pages single-spaced (code appendix excluded) + program code.
**Student ID MUST appear in the filename AND on page 1.**

---

## 0. Decision summary
- **Two tasks (both deep RL, image-based):**
  - **Task 1 — Atari Pong** (`ALE/Pong-v5`), primary **DQN** (value-based).
  - **Task 2 — VizDoom** (`VizdoomBasic-v1` sanity, `VizdoomDefendCenter-v1` headline), primary **PPO** (policy-based).
- **Full scope:** all 11 experiments (§6) are in. Tiering protects the report if time runs short.
- **Report spine:** value-based vs policy-based, on a 2D fully-observable arcade game vs a 3D partially-observable FPS.
- **Compute:** RTX 4090 (CUDA), i7-13700K (16C/24T), 128 GB RAM. Native PyTorch + SB3. No DirectML, no Colab.
- **The real limits now are the 10-page cap and human time — NOT compute.** §10 organizes the report by *aspect* so 11 experiments fit.

---

## 1. HOW TO USE THIS PLAN WITH CLAUDE CODE  *(read first)*
This is coursework; it must reflect **Michael's** decisions and analysis. Claude Code builds scaffolding and runs experiments; the intellectual content is Michael's.
1. **Build the repo (§5) FIRST**, commit, before any training.
2. **Work in tiers (§6) and STOP at every "🧑 INPUT" checkpoint (§8).** Do not auto-run the whole project.
3. **Logging discipline — every run produces, no exceptions:** `configs/<id>.yaml`, `results/<id>/metrics.csv`, `results/<id>/curve.png`, `results/<id>/eval.json`, `results/<id>/gameplay.gif`, and one appended row in `EXPERIMENTS.md`.
4. **Reuse runs — do NOT retrain (see §6 "reuses").** Several Tier-B experiments are comparisons built on Tier-A baselines.
5. **`EXPERIMENTS.md` + `RESULTS.md` are the report's raw material.** Keep current.
6. **Code is graded (appendix).** Clear, commented, organized from the start.
7. **Maintain `REFERENCES.md`** incl. AI-tool usage (required by brief).
8. **Never fabricate numbers.** Tables/plots come only from `results/`.

---

## 2. ASSIGNMENT REQUIREMENTS CHECKLIST  *(nothing may be skipped)*
| # | Requirement (from brief) | Where | Owner |
|---|---|---|---|
| R1 | Set up Gymnasium envs per docs | §4 + smoke tests | Claude Code |
| R2 | ≥2 tasks, ≥1 Atari, other any env | Pong + VizDoom | ✔ |
| R3 | Any RL algorithm | DQN + PPO | ✔ |
| R4 | References for resources/code reused | `REFERENCES.md`→§5 | both |
| R5 | Experiments on aspects affecting performance | §6 (each row names its aspect) | CC runs, 🧑 frames |
| R6 | Meaningful experiments > difficulty | comparisons, each w/ takeaway | 🧑 |
| S1 | Research question + motivation, plain text | report §1 | 🧑 **writes** |
| S2 | Methods + refs (libs, code, **AI tools**) | report §2 | both |
| S3 | Experiments: results + **examples** | report §3 (tables+curves+GIFs) | both |
| S4 | **Discussion** | report §4 | 🧑 **writes** |
| S5 | References | report §5 | both |
| S6 | Code appendix, separate page, organized+commented | appendix←repo | Claude Code |
| N1 | Submit via E3; late policy | submission | 🧑 |
| N2 | Student ID in filename + page 1 | cover | 🧑 |
| N3 | Methods readable WITHOUT code | §2 style | both |
| N4 | Results + "what message?" in text | takeaway per result | 🧑 **writes** |
| N5 | Tables (NOT screenshots) + charts | `report_assets/` | CC makes, 🧑 approves |
| N6 | ≥12 pt body, ≥10 pt tables/figures | template | both |

---

## 3. Compute strategy
- `device="cuda"`. Bottleneck is env stepping (CPU): PPO `n_envs=8–16` (`SubprocVecEnv`); DQN large replay buffer (128 GB RAM).
- Wall-clock per run: Pong DQN ~1–2 h; VizDoom Defend-Center PPO ~30–90 min; Basic ~minutes. ~5 extra Tier-B runs ≈ 5–10 h, queued in background during the presentation/writing.
- Fresh CUDA venv, separate from HW2 DirectML env. `CheckpointCallback`+`EvalCallback`; TensorBoard+CSV.

---

## 4. Environment setup
```bash
python -m venv hw3 && conda activate hw3         # Windows: hw3\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install gymnasium "gymnasium[atari]"
pip install stable-baselines3[extra]
pip install vizdoom
pip install rl_zoo3                                    # OPTIONAL: tuned Atari hyperparameters
```
Smoke test both envs. **VizDoom plumbing (critical path, do FIRST):** obs is a Dict (`screen`+`gamevariables`); `common/vizdoom_wrappers.py` returns `screen` channel-first → SB3 `CnnPolicy` + `VecFrameStack(4)`, mirroring the Atari pipeline.

---

## 5. Structured project layout (report-ready)
```
hw3/
  README.md  requirements.txt
  EXPERIMENTS.md   # journal: id | date | task | hyperparams | result | 🧑 takeaway
  RESULTS.md       # aggregated report-ready tables
  REFERENCES.md    # libs, repos, tutorials, AI tools (→ §5)
  configs/         # one yaml per experiment
  common/  vizdoom_wrappers.py  callbacks.py  plotting.py
  pong/    train_pong_dqn.py  train_pong_ppo.py  eval_record.py
  vizdoom/ train_vizdoom_ppo.py train_vizdoom_dqn.py eval_record.py
  results/<id>/    # metrics.csv curve.png eval.json config.yaml gameplay.gif
  report_assets/   # FINAL grouped figures + tables (built by plotting.py)
  report/          # draft + cover (student ID)
```
**Rule:** report figures/tables come ONLY from `report_assets/`, built ONLY from `results/`.

---

## 6. Experiment matrix (full scope, tiered)
≥3 seeds per curve (mean ± std). **Tier A = finish first (guarantees a complete report). Tier B = full scope, queued after A.** "reuses" = do not retrain.

### Task 1 — Pong, DQN
| ID | Tier | Experiment | Aspect / question | Reuses |
|---|---|---|---|---|
| P1 | A | Baseline DQN | Reference: does it learn to win, how fast? | — |
| P2 | A | Target network ON vs OFF | Stability: does the lagged target tame the moving-target problem? | P1 = ON |
| P3 | A | ε-decay fast vs slow | Exploration vs exploitation | P1 = default |
| P4 | B | Replay buffer small vs large | Memory / sample correlation | P1 = large |
| P5 | B | DQN vs PPO on Pong | Algorithm family | P1 (DQN) + new PPO-Pong run |

### Task 2 — VizDoom, PPO
| ID | Tier | Experiment | Aspect / question | Reuses |
|---|---|---|---|---|
| V0 | A | Basic sanity run | Pipeline validation (3D wrapper works) | — |
| V1 | A | Defend-the-Center baseline | Reference for the 3D task | — |
| V2 | A | Discrete vs MultiBinary actions | Action-space design (unique to VizDoom) | V1 = Discrete |
| V3 | B | Difficulty scaling (Basic→Defend-Center→Health Gathering) | Task complexity | V0 + V1 + new HealthGathering run |
| V4 | B | Frame stack 1 vs 4 | Partial observability (does temporal context matter more in 3D?) | V1 = stack 4 |
| V5 | B | PPO vs DQN on Defend-the-Center | Algorithm family (mirror of P5) | V1 (PPO) + new DQN-DefendCenter run |

**New training runs total:** Tier A = P1, P2, P3, V0, V1, V2 (+ seeds). Tier B adds only ≈5: P4, PPO-Pong, HealthGathering-PPO, DefendCenter-stack1-PPO, DQN-DefendCenter.

---

## 7. Report threads (organize §3 around THESE, not by experiment ID — this is how 11 experiments fit in 10 pages)
- **T1 Algorithm family** — P5 + V5 → one cross-task comparison table (DQN vs PPO on both games).
- **T2 DQN components** — P1/P2/P3/P4 → one multi-curve figure + one table (target net, ε, buffer).
- **T3 Partial observability** — V4, contrasted with Pong's frame-stack dependence → one figure.
- **T4 Action-space design** — V2 → one figure.
- **T5 Task complexity** — V3 → one figure (final reward vs scenario difficulty).
- Baselines (P1, V0, V1) anchor the section with two gameplay GIFs/frame strips as the "examples" (S3).

---

## 8. 🧑 Human-in-the-loop checkpoints (Michael's input — NOT automated)
- **C1** Task/algorithm sign-off. (Done.)
- **C2** Hyperparameters before each run — review/edit config (lr, steps, buffer, n_envs, ε).
- **C3** After P1/V1 baselines — inspect curves before launching ablations.
- **C4** Takeaway per result — Michael writes the one-sentence "what message" (N4). Graded substance.
- **C5** Research question + motivation (§1) and Discussion (§4) — Michael writes in his own words; Claude Code only structures/proofreads.
- **C6** Final formatting + submission — verify ≤10 pp, fonts, student ID (filename + page 1), upload to E3.

---

## 9. Schedule (today = Sat 5/23, due Fri 5/29)
| Day | Tasks |
|---|---|
| **Sat 5/23** | Repo structure. venv + installs. Smoke tests. **VizDoom wrapper (de-risk).** 🧑C2 → launch P1; run V0. |
| **Sun 5/24** | 🧑C3 on P1. Queue P2, P3. 🧑C2 → launch V1. 🧑C5: draft research question + motivation. |
| **Mon 5/25** | Presentation prep priority. Queue V2 + P4 + PPO-Pong (Tier B) in background. Pong results → `RESULTS.md`. |
| **Tue 5/26** | Deliver presentation. Queue remaining Tier B (DQN-DefendCenter, HealthGathering, DefendCenter-stack1). 🧑C4: write takeaways as runs land. |
| **Wed 5/27** | Build grouped `report_assets/` (T1–T5). Write experiments section by thread. |
| **Thu 5/28** | 🧑C5: write discussion. References. Code appendix. Full draft. Format check (N3,N5,N6); cut to ≤10 pp. |
| **Fri 5/29** | 🧑C6: polish, student-ID checks, submit on E3. |

---

## 10. Report skeleton (→ checklist §2; experiments grouped per §7)
1. Research question + motivation (S1, plain text). 🧑
2. Methods (S2): DQN — Q-learning + target net + replay + ε-greedy; PPO — clipped objective, advantage, parallel rollouts. Readable without code (N3). Refs to libs/code/AI tools.
3. Experiments (S3): organized by threads T1–T5; formatted tables (N5) + grouped learning curves + 2 gameplay examples. Each result + 🧑 takeaway (N4).
4. Discussion (S4): synthesize across T1–T5; limitations. 🧑
5. References (S5).
6. Appendix — code (S6): new page, organized, commented.

---

## 11. Risk register
| Risk | Mitigation |
|---|---|
| 11 experiments overflow 10 pages | Group by thread (§7); combined figures/tables; methods stay concise; code in appendix. Tier A alone is a complete report. |
| Time short before 5/29 | Tier A guarantees a finished report; Tier B drops gracefully (runs queue in background while you present/write). |
| VizDoom plumbing stalls | Do first; if > ½ day, fall back to Cliff Walking (Q-learning vs SARSA) for Task 2. |
| VizDoom "is it Gymnasium?" | Farama-maintained Gymnasium wrapper → defensible; state in §2, optionally confirm with Prof. |
| Run interrupted | Checkpoint + resume. |

---

## 12. References to collect (→ `REFERENCES.md` → §5)
Gymnasium (Farama); Stable-Baselines3 (Raffin et al. 2021); ALE (Bellemare et al. 2013; Machado et al. 2018); ViZDoom (Kempka et al. 2016 + Farama docs); DQN (Mnih et al. 2015); PPO (Schulman et al. 2017); Sutton & Barto; reused code/tutorials; AI tools used.
