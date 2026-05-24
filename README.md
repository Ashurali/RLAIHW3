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

## Remote workflow (no git on the server — scp/ssh only)
The `deploy/` folder drives everything over SSH. Training runs **detached**
(`nohup`) on the server, so you can close the SSH session / shut the laptop and
it keeps going; progress is written to `logs/` and tracked with `tail`.

**One-time:** copy `deploy/server.env.example` to `deploy/server.env` and fill in
your host / user / port / remote path (`server.env` is gitignored).

```powershell
# 1) push code to the server (scp; excludes results/venv/git, fixes LF)
pwsh deploy/upload.ps1
# 2) create venv + install deps + verify CUDA (detached; wait for SETUP_DONE)
pwsh deploy/remote.ps1 -Action setup
pwsh deploy/remote.ps1 -Action tail -Log setup
# 3) sanity-check the envs on the GPU box
pwsh deploy/remote.ps1 -Action smoke
# 4) launch training in the background
pwsh deploy/remote.ps1 -Action train -Config P1 -Seed 0     # one run
pwsh deploy/remote.ps1 -Action queue                        # whole Tier-A queue
# 5) watch / manage (laptop can disconnect at any time)
pwsh deploy/remote.ps1 -Action status
pwsh deploy/remote.ps1 -Action tail -Log P1_s0
pwsh deploy/remote.ps1 -Action stop -Log P1_s0
# 6) pull results back to the laptop
pwsh deploy/fetch.ps1 -Lite    # metrics/curve/eval/config/gif + logs (small)
pwsh deploy/fetch.ps1          # full (includes model .zip)
```
Needs the Windows OpenSSH client + `tar` (both ship with Windows 11). Edit the
queue list in `deploy/remote_queue.sh` to add Tier B. Fetched `logs/` and full
`results/` are gitignored.

## Hardware utilization
Tuned for the measured server (**RTX 4090 24 GB, i7-13700K = 24 threads,
125 GB RAM but shared — ~40 GB typically free, Ubuntu 24.04, CUDA-13 driver**):
- **PPO** (Pong/VizDoom): `n_envs: 16` parallel `SubprocVecEnv` rollouts +
  `batch_size: 512`, to keep the GPU fed while the CPU steps environments.
- **DQN** (Pong/VizDoom): `buffer_size: 500_000` (~28 GB RAM with frame-stacked
  uint8 obs; pre-allocated, not dynamic) — fits the *shared* ~42 GB free. Raise
  toward 1,000,000 only if the box is dedicated.
- **Python:** the venv is built from `python3.12` (not the conda base 3.13) for
  the widest torch/vizdoom wheel coverage; the CUDA-13 driver runs cu124 wheels.
- `common/utils.configure_torch_perf()` turns on cuDNN autotuning + TF32 tensor
  cores for the fixed 84×84 CNN inputs.

The `setup` step prints `nvidia-smi` / `nproc` / `free` at the top of
`logs/setup.log`. The box is **shared**, so check free RAM/GPU before launching
and scale `n_envs` / `buffer_size` down in the YAML if others are running jobs.

## Manual setup (alternative to the deploy scripts)
```bash
# python -m venv hw3
conda activate hw3        # Windows: hw3\Scripts\activate
pip install -r requirements.txt
# verify the GPU is visible before training:
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
`requirements.txt` pulls CUDA 12.4 PyTorch wheels via `--extra-index-url`. Match
the CUDA build to the server driver: run `nvidia-smi` (top-right shows the max
CUDA version) and if it's older than 12.4, install the matching wheels instead,
e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu121`.
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
