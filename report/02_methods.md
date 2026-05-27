# 2. Methods

This section describes the two tasks, the shared image-processing pipeline, the
DQN and PPO algorithms, the software stack, and the evaluation protocol. It is
intended to be read without referring to the code; exact hyperparameter values
live in the per-experiment YAML files under `configs/`.

## 2.1 Tasks and environments

Two image-based environments are used, chosen to contrast a 2D fully-observable
arcade game with a 3D partially-observable first-person shooter:

- **Atari Pong** (`ALE/Pong-v5`, Arcade Learning Environment [Bellemare et al.,
  2013; Machado et al., 2018]). A two-paddle game whose reward is the score
  margin, bounded in `[-21, +21]`. Sticky actions (action-repeat probability
  0.25) inject the stochasticity that makes pure trajectory memorisation
  insufficient, so an agent must learn a robust policy rather than a fixed
  sequence.
- **VizDoom** [Kempka et al., 2016] via the Farama Gymnasium wrapper. Two
  scenarios are used in the report: **Defend-the-Center** (the headline 3D
  task) and **Health-Gathering** (a harder scenario used for the difficulty
  comparison). Both expose a first-person RGB frame plus a small vector of game
  variables; only the frame is fed to the agent (see §2.2).

Both environments are wrapped to produce the same observation shape so that the
same convolutional policy network can be reused across tasks.

## 2.2 Shared image pipeline

For Atari, the standard `AtariWrapper` from Stable-Baselines3 is applied: no-op
reset, frame-skip 4, grayscale conversion, resize to 84×84, reward clipping to
`{-1, 0, +1}`, and episodic life resetting where applicable. For VizDoom, a
custom `VizDoomScreenWrapper` mirrors the same shape: it discards the game
variables, grayscales the screen, and resizes to 84×84 uint8.

Both pipelines are then wrapped with `VecFrameStack(4)`, so each observation
the policy sees is a stack of the four most recent frames (84×84×4). This
provides the temporal context that single frames lack — most importantly the
ball velocity in Pong and projectile/enemy motion in VizDoom — and is the
standard recipe behind every result quoted from the Atari literature.

The neural network is the **Nature-CNN** architecture introduced with the
original DQN [Mnih et al., 2015]: three convolutional layers (32–64–64 channels
with 8×8/4×4/3×3 kernels and strides 4/2/1) followed by a 512-unit MLP head and
the algorithm-specific output layer.

## 2.3 DQN (value-based)

DQN [Mnih et al., 2015] approximates the optimal action-value function
`Q*(s,a)` with a neural network `Q_θ(s,a)`. Three components stabilise training
of this off-policy bootstrapped target:

- **Experience replay.** Every observed transition `(s, a, r, s', done)` is
  stored in a fixed-size buffer `D`. Each gradient update samples a minibatch
  from `D`, decorrelating consecutive samples and approximating an
  i.i.d.-from-the-stationary-distribution assumption.
- **Target network.** A *lagged* copy `Q_{θ⁻}` provides the bootstrap target
  `y_t = r_t + γ · max_{a'} Q_{θ⁻}(s_{t+1}, a')`. Without it, the target moves
  every gradient step and training is prone to divergence (the "moving target"
  problem). `θ⁻` is hard-copied from `θ` every `target_update_interval` steps.
- **ε-greedy exploration.** The behaviour policy selects a uniformly random
  action with probability `ε` (linearly annealed from 1 down to 0.01 over a
  fraction of training) and the greedy action otherwise.

The loss is the squared TD error
`L(θ) = E_{(s,a,r,s')~D}[(y - Q_θ(s,a))^2]`, optimised with Adam. The discount
factor is `γ = 0.99`. Concrete settings for the Pong baseline (P1): replay
buffer 500 k, learning starts after 100 k transitions, batch size 32,
`train_freq = 4` (one gradient step every four environment steps),
target-network sync every 1 000 steps, ε annealed over the first 10 % of
training to a final 0.01.

## 2.4 PPO (policy-based)

PPO [Schulman et al., 2017] directly optimises a stochastic policy `π_θ(a|s)`
in actor–critic form, alongside a value baseline `V_φ(s)`. It collects
on-policy rollouts in parallel from many environments, then performs several
epochs of minibatch updates on each rollout using the clipped surrogate
objective

`L^CLIP(θ) = E_t [ min( r_t(θ) · A_t,  clip(r_t(θ), 1−ε, 1+ε) · A_t ) ]`,

where `r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)` is the importance ratio
between the new and old policies. The `clip` prevents updates that move the
policy too far from the data-generating distribution, keeping the on-policy
approximation valid for several reuse epochs.

Advantages `A_t` are estimated with **Generalised Advantage Estimation** (GAE,
`λ = 0.95`), which interpolates between the high-variance Monte-Carlo and
high-bias one-step TD estimators. The full training loss adds a value-function
regression term and an entropy bonus to maintain exploration:
`L = L^CLIP − c_v · L^V + c_e · H[π_θ]`.

Rollouts are collected from `n_envs = 16` parallel environments
(`SubprocVecEnv`) of length `n_steps = 128`, giving 2 048 transitions per
update; each rollout is shuffled into minibatches of 512 for `n_epochs = 4`
passes. Other settings: `γ = 0.99`, clip range `ε = 0.1` (Pong) / `0.2`
(VizDoom), entropy coefficient `0.01`, value coefficient `0.5`, Adam optimiser
with learning rate `2.5 × 10⁻⁴`.

## 2.5 Software, hardware, and AI-tool disclosure

- **Stable-Baselines3** [Raffin et al., 2021] provides the DQN and PPO
  implementations and the vectorised-environment plumbing. The `CnnPolicy`
  selects the Nature-CNN feature extractor described in §2.2.
- **Gymnasium** [Farama Foundation] is the environment API. The
  Atari side uses `ale-py` (which bundles the ROMs); the VizDoom side uses the
  `vizdoom.gymnasium_wrapper` registrations.
- **PyTorch** with CUDA 12.4 runs the network forward/backward on the GPU.
- Training was performed on a shared lab workstation (RTX 4090 24 GB,
  i7-13700K = 24 threads, Ubuntu 24.04), inside a Conda environment with
  Python 3.12. cuDNN autotuning and TF32 matmuls were enabled for throughput.

**AI-tool disclosure (course requirement).** *Claude Code* (Anthropic) was
used to scaffold the repository, implement the training/evaluation/plotting
code, build the SSH/scp deployment pipeline, and draft this Methods section
for the author to review and edit. All experimental design, hyperparameter
choices, result interpretation and the discussion in §4 are the author's own.

## 2.6 Experiment protocol

Each (configuration, seed) pair is trained from scratch with a fixed seed. The
report uses **≥ 3 seeds per curve**; learning curves show the mean and the
±1 standard-deviation band across seeds, computed by interpolating each seed
onto a shared timestep grid and aggregating.

At the end of training, every run is evaluated for 20 deterministic episodes
(no exploration noise) with a fresh seed; the mean and standard deviation of
those 20 episode returns are reported in `eval.json`. During training,
`EvalCallback` runs a 10-episode deterministic evaluation every 100 k steps and
keeps the best-performing checkpoint, which guards against late-training
instability (more relevant for DQN than for PPO).

Both Pong DQN and Pong PPO were trained for **7 M environment steps** so the
DQN-vs-PPO comparison is at an equal budget. The VizDoom Defend-Center
DQN-vs-PPO comparison is at **2 M steps** each. All hyperparameters for every
experiment are versioned in `configs/<id>.yaml`.

---

*References cited in this section are listed in §5; the full list is also in
`REFERENCES.md` in the repository.*
