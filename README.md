# AI HW3 — Reinforcement Learning (Pong DQN + VizDoom PPO)

Value-based vs policy-based deep RL on a 2D fully-observable arcade game
(**Atari Pong**, DQN) and a 3D partially-observable FPS (**VizDoom**, PPO).
See [`plan.md`](plan.md) for the full plan; this README is the runbook.

Both tasks share one image pipeline: 84×84 (grayscale) frames + `VecFrameStack(4)`
feeding an SB3 `CnnPolicy`, so the DQN-vs-PPO comparison is not confounded by
preprocessing.

## Project layout
```
common/      shared engine (config, env factories, training loop, eval, plotting)
  vizdoom_wrappers.py   Dict-obs -> 84x84 image wrapper (the 3D critical path)
  envs.py               make_atari_vec / make_vizdoom_vec / build_env_fn
  train_core.py         run_training(cfg, model_cls): one run -> all artifacts
  callbacks.py          EvalCallback + CheckpointCallback
  eval_utils.py         final eval + gameplay GIF; shared eval_record CLI
  plotting.py           per-run curves + multi-seed aggregate figures
pong/        train_pong_dqn.py  train_pong_ppo.py  eval_record.py
vizdoom/     train_vizdoom_ppo.py  train_vizdoom_dqn.py  eval_record.py
configs/     one YAML per experiment (P1..P5, V0..V5)
results/     <exp_id>_s<seed>/  metrics.csv curve.png eval.json config.yaml gameplay.gif
report_assets/  final grouped figures (built by plotting.py)
EXPERIMENTS.md  RESULTS.md  REFERENCES.md   report journal / tables / refs
smoke_test.py   env sanity check — run FIRST on the server
```

## Setup (on the GPU server)
```bash
python -m venv hw3
source hw3/bin/activate          # Windows: hw3\Scripts\activate
pip install -r requirements.txt
```
`requirements.txt` pulls CUDA 12.4 PyTorch wheels via `--extra-index-url`. If pip
resolves a CPU build, install torch first:
`pip install torch --index-url https://download.pytorch.org/whl/cu124`.
Atari ROMs ship with `ale-py` (no AutoROM step). VizDoom needs its system build
deps on Linux (e.g. `sudo apt install build-essential libsdl2-dev` if a wheel
isn't available for your platform).

## Step 0 — smoke test (do this before any training)
```bash
python smoke_test.py            # checks GPU + builds/steps Pong and VizDoom
```
This de-risks the VizDoom plumbing. If `VizdoomBasic-v0` is not found, list the
registered ids and update `env_id` in the configs (the suffix is occasionally
`-v1`):
```bash
python -c "import vizdoom.gymnasium_wrapper, gymnasium as gym; print([e for e in gym.envs.registry if 'Vizdoom' in e])"
```

## Training
Each run writes everything under `results/<exp_id>_s<seed>/`:
`config.yaml`, `metrics.csv`, `curve.png`, `eval.json`, `gameplay.gif`,
`model.zip`, plus `best_model/` and `checkpoints/`.

```bash
# Pong DQN baseline, 3 seeds (the report needs >=3 seeds per curve)
python pong/train_pong_dqn.py --config configs/P1.yaml --seed 0
python pong/train_pong_dqn.py --config configs/P1.yaml --seed 1
python pong/train_pong_dqn.py --config configs/P1.yaml --seed 2

# VizDoom PPO sanity, then the 3D baseline
python vizdoom/train_vizdoom_ppo.py --config configs/V0_basic.yaml --seed 0
python vizdoom/train_vizdoom_ppo.py --config configs/V1_defendcenter.yaml --seed 0
```
The `--seed` flag overrides the config so multi-seed runs reuse one config.

### Tier A first (a complete report), then Tier B
- **Tier A:** `P1`, `P2_targetoff`, `P3_epsfast`, `P3_epsslow`, `V0_basic`,
  `V1_defendcenter`, `V2_multibinary`.
- **Tier B:** `P4_buffersmall`, `P5_ppo_pong`, `V3_healthgathering`,
  `V4_stack1`, `V5_dqn_defendcenter`.

Do **not** retrain reused arms (see the reuse map in `EXPERIMENTS.md`):
e.g. P2's "target-on" arm *is* P1; V2's "Discrete" arm *is* V1.

## Re-evaluate / re-record a finished run
```bash
python pong/eval_record.py    --config configs/P1.yaml --seed 0
python vizdoom/eval_record.py --config configs/V1_defendcenter.yaml --seed 0
```

## Monitoring & plots
```bash
tensorboard --logdir results            # live curves
python -m common.plotting curve results/P1_s0          # one run -> curve.png
python -m common.plotting compare --ids P1 P2_targetoff P3_epsfast P3_epsslow \
    --title "DQN components (Pong)" --out T2_dqn_components.png   # -> report_assets/
```
`compare` averages across seeds (`results/<id>_s*/`) with a mean ± std band.

## Notes / things to confirm on the first run
- **VizDoom env ids:** configs default to the `-v0` suffix; verify and edit if needed.
- **VizDoom action space (V2):** `env_kwargs.max_buttons_pressed: 0` requests a
  MultiDiscrete (multi-button) space vs V1's default Discrete. Confirm the
  wrapper honors it on your VizDoom version.
- **PPO budgets** (`P5` 10M, `V3` 3M) are generous — trim `total_timesteps` if
  compute/time is tight; Tier A alone is a complete report.
- **GPU:** training uses `device: cuda`; set it to `cpu` in a config only for a
  quick local check.
