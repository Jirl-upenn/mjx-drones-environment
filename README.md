# MJ-drones-gym

**MuJoCo-based multi-drone Gymnasium environments for single and multi-agent reinforcement learning of quadcopter control.**

High-fidelity quadcopter simulation with GPU-vectorized environments, Dryden wind turbulence, domain randomization, obstacle generation, and curriculum learning — all built on [MuJoCo](https://mujoco.org/).

## Features

- **MuJoCo physics** — faster and more accurate than PyBullet
- **Gymnasium API** — drop-in compatible with stable-baselines3, CleanRL, etc.
- **Multi-drone support** — N arbitrary drones with inter-drone effects
- **Aerodynamic effects** — ground effect, drag, downwash (individually toggleable)
- **Multiple action types** — RPM, normalized thrust, attitude (CTBR-style), velocity, PID waypoint
- **Multiple observation types** — kinematics (state vector), RGB camera
- **PID controllers** — tuned cascaded position/attitude PID (PIDControl + DSLPIDControl)
- **PettingZoo multi-agent** — parallel environment wrapper for MARL
- **GPU-vectorized MJX training** — `MJXVectorAviary`/`MultiVectorAviary` run thousands of parallel JAX/MJX envs with motor-lag modeling and per-episode domain randomization; see [Vectorized Training Interface](#vectorized-training-interface)

## Installation

```bash
git clone <this-repo>
cd multi_drone_mujoco/
pip install -e .          # core
pip install -e ".[all]"   # RL, MARL, visualization extras
```

### Requirements
- Python ≥ 3.8
- MuJoCo ≥ 3.0
- Gymnasium ≥ 0.29
- NumPy ≥ 1.21

For vectorized GPU training, additionally install:
```bash
pip install mujoco-mjx
pip install 'jax[cuda12]'
```


## Environments

All task-specific behavior (observations, reward, reset, termination) is
supplied by a `CPUTaskPlugin` passed to the generic `TaskAviary` — see
`multi_drone_mujoco/envs/plugins.py` for the interface and
`multi_drone_mujoco/envs/example_plugins.py` for reference implementations.
Bring your own plugin from any repo to define a new task without touching
this package.

| Plugin | Obs Dim | Action | Description |
|---|---|---|---|
| `SimpleHoverPlugin` | 12 | 4 (normalized RPM) | Hover at a fixed target height |
| `SimpleRacePlugin` | 21×N | 4×N | Gate racing with lap timing |
| `MultiAgentAviary` | per-agent | per-agent | PettingZoo parallel wrapper |

## Action Interface

A policy never commands per-motor RPM directly except under `ActionType.RPM`
on the CPU path. Everywhere else, an action passes through a **mixer**
(`multi_drone_mujoco/utils/mixer.py`) that converts 4-dimensional outputs
into 4 per-motor RPMs.

**RPM channel semantics differ by entry point** — this is the one thing worth
memorizing before debugging a "policy acts insane" report:

| Entry point | `action_type` | Semantics |
|---|---|---|
| `BaseAviary` (CPU) | `ActionType.RPM` | Raw absolute RPM per motor, clipped to `[0, MAX_RPM]` — **not normalized** |
| `BaseAviary` (CPU) | `ActionType.ONE_D_RPM` | One normalized scalar in `[-1, 1]`, same RPM to all 4 motors, via `mix_rpm_action` |
| `MJXVectorAviary` / `MultiVectorAviary` | `action_type="rpm"` | 4 independent normalized channels in `[-1, 1]`, each motor mapped through `mix_rpm_action` |
| `BaseAviary` / MJX aviaries | `ActionType.ATTITUDE` / `action_type="attitude"` | `[thrust, roll, pitch, yaw_rate]`, each in `[-1, 1]`, via `mix_attitude_rpm` |

`mix_rpm_action` maps a normalized channel to RPM in two linear segments
rather than one line across `[-1, 1]`, because `hover_rpm` isn't the
midpoint of `[0, MAX_RPM]`:

```
action <= 0:  RPM = (action + 1) * hover_rpm      # -1 -> 0 RPM,        0 -> hover
action  > 0:  RPM = hover_rpm + (MAX_RPM - hover_rpm) * action   #  0 -> hover, +1 -> MAX_RPM
```

so `action = 0` reproduces an exact hover on every channel regardless of
where `hover_rpm` actually sits in the motor's range.

For `ATTITUDE`, the collective channel reuses that same curve; roll/pitch/yaw_rate
are **fixed-scale per-motor RPM differentials**, not a closed-loop rate
controller — there's no feedback from measured angular velocity, so a
`roll` command is a direct torque-ish nudge rather than a literal desired
body rate:

```python
scale = differential_frac * hover_rpm       # differential_frac = 0.02 nominal
rpm = [
    collective + roll*scale - pitch*scale - yaw_rate*scale,
    collective - roll*scale - pitch*scale + yaw_rate*scale,
    collective - roll*scale + pitch*scale - yaw_rate*scale,
    collective + roll*scale + pitch*scale + yaw_rate*scale,
]  # clipped to [0, MAX_RPM]
```

Once per-motor RPM is known, both backends apply the same physical model —
`forces = kf * rpm**2` per motor, `total_thrust = sum(forces)` along the
body +z axis, and X-configuration differential-thrust torques about x/y with
a reaction torque about z from `km`. The result is injected as a single
external wrench (`xfrc_applied`: world-frame force + torque on the drone's
free-floating body) rather than by simulating individual propeller
actuators/joints inside MuJoCo — the quadrotor is one rigid body with a
`<freejoint>`, and everything about "how a motor spins up" happens outside
MuJoCo, upstream of that wrench.

The MJX vectorized path adds one thing the CPU `RPM`/`ATTITUDE` paths don't
model at all: **first-order motor lag**. Commanded RPM (the mixer's output)
doesn't apply instantly — it's integrated toward the actual per-motor RPM at
the *physics* rate (`sim_freq`), not the control rate, since a real motor's
spin-up/spin-down happens continuously between control ticks:

```
alpha = sim_dt / (motor_tau + sim_dt)
motor_rpm = alpha * commanded_rpm + (1 - alpha) * motor_rpm   # every physics substep
```

`motor_tau=0.15s` is the default (matches the Crazyflie figure reported in
*Learning to Fly in Seconds*); pass `motor_tau=0.0` for instant, zero-order-hold
actuation — the formula degrades to `motor_rpm := commanded_rpm` with no
separate branch. The lagged `motor_rpm` persists across physics substeps and
across control steps within an episode, and only resets to `hover_rpm` (not
zero) at episode reset, so an episode doesn't open with an artificial
spin-up ramp through zero thrust.

## Core Physics

Both the CPU `BaseAviary` and the MJX vectorized aviaries model the
quadrotor as a single free-floating rigid body (`<freejoint>`, diagonal
inertia, no separate propeller joints) and integrate it with MuJoCo's
**RK4** explicit integrator (`<option integrator="RK4">`) — there is no
`Physics.DYN`-equivalent mode on the vectorized path; RK4-through-MuJoCo is
the only physics vectorized training ever sees.

Every environment step is decomposed into `sim_steps_per_ctrl =
sim_freq // ctrl_freq` physics substeps, e.g. the MJX default of `sim_freq=240`,
`ctrl_freq=48` runs 5 substeps per policy decision. The mixed per-motor RPM
command (and, on the MJX path, the motor-lag state) is recomputed every
substep and reapplied as a fresh `xfrc_applied` wrench before each
`mj_step`/`mjx.step` call — forces are not held as an actuator with its own
internal state inside MuJoCo, they're pure external force injection recomputed
from scratch each substep.

`BaseAviary` additionally exposes `Physics.DYN`: an explicit, hand-rolled
semi-implicit Euler integrator (`_dynamics`/`_integrateQ` in
`envs/base_aviary.py`) that bypasses `mj_step` entirely — ported from
gym-pybullet-drones' reference integration, kept as a CPU-only sanity/debug
path rather than something training relies on. The remaining CPU-only
`Physics.*` modes layer optional aerodynamic extras (ground effect, drag,
downwash — all hand-coded force models applied on top of `xfrc_applied`,
not MuJoCo's own fluid dynamics) onto the base RK4 path:

| Mode | Description |
|---|---|
| `Physics.MJC` | Pure MuJoCo (force injection via `xfrc_applied`) |
| `Physics.DYN` | Explicit dynamics (semi-implicit Euler, bypasses `mj_step`) |
| `Physics.MJC_GND` | MuJoCo + ground effect |
| `Physics.MJC_DRAG` | MuJoCo + aerodynamic drag |
| `Physics.MJC_DW` | MuJoCo + downwash |
| `Physics.MJC_GND_DRAG_DW` | MuJoCo + all aerodynamic effects |

**None of the aerodynamic extras exist on the MJX vectorized path** — it's
intentionally the bare rigid-body + thrust/torque + optional motor-lag
model, so it can `vmap`/`jit` cleanly across tens of thousands of envs. A
task plugin that needs ground effect, drag, or downwash during vectorized
training has to add those forces itself inside `TaskPlugin.step`.

## Vectorized Training Interface

`multi_drone_mujoco/vectorized/` is a GPU-native reimplementation of the
physics/step loop above using MuJoCo MJX (JAX backend), for training at
thousands-of-parallel-envs scale. It is not installed by default —
`pip install mujoco-mjx` and `pip install 'jax[cuda12]'` (or plain `jax` for
CPU) are required only if you actually use this path; everything else in the
package works without them.

### Two aviary classes

| Class | Controlled bodies | `action` shape | `reward`/`done` shape | Use when |
|---|---|---|---|---|
| `MJXVectorAviary` | **Only body 1** — `_BODY_ID` is hardcoded. Any extra `num_drones` bodies in the XML are geometry only (e.g. gates), not force-controlled agents. | `(num_envs, 4)` | `reward: (num_envs,)`, `done: (num_envs,)` | Single-agent tasks (hover, single-drone racing) |
| `MultiVectorAviary` | All `num_drones` bodies, each independently force-controlled | `(num_envs, num_drones, 4)` | `reward: (num_envs, num_drones)` — one reward per agent, `done: (num_envs,)` — one *joint* termination | Multi-agent tasks where each drone trains its own policy |

`MultiVectorAviary` is a separate class rather than a generalization of
`MJXVectorAviary` specifically to avoid regressing the single-agent
hover/race training paths that depend on the latter's exact behavior — see
`vectorized/multi_aviary.py`'s module docstring. Both share the same
`MJXState` container and `TaskPlugin` interface; a multi-agent plugin's
`step`/`get_obs` just operate on stacked per-agent arrays (`obs_dim` stays
**per-agent**, not multiplied by `num_drones`).

### Bring-your-own-plugin, JAX-native

Exactly like the CPU side's `CPUTaskPlugin`, all task logic (observations,
reward, reset, termination) lives outside this package in a `TaskPlugin`
subclass (`vectorized/plugins.py`) — adding a task never means editing
`MJXVectorAviary`/`MultiVectorAviary`. The difference is that every
`TaskPlugin` method operates on a **single** environment and must be
pure, JIT-traceable JAX code (no Python-level branching on traced values, no
side effects) — the aviary classes apply `jax.vmap` over `num_envs`
externally, so a plugin author never writes batching logic themselves.

Required methods: `obs_dim`, `init_task_state(num_envs)`,
`reset_task_state(data, rng, hint, old_task_state)`, `step(data, action,
task_state, motor_rpm, phys_params) -> (new_task_state, obs, reward,
extra_done)`, `get_obs(...)`. Optional hooks (`termination_checks` +
`termination_names`, `task_metrics` + `task_metric_names`,
`extra_worldbody_xml`, `camera_config`) are detected via `getattr`/`hasattr`
at construction time — a plugin that skips them costs nothing at trace time,
there's no branch for "hook not implemented".

### `MJXState` is a plain JAX pytree

```python
class MJXState(NamedTuple):
    mjx_data: Any        # mjx.Data, batched over num_envs
    step_count: Any
    done: Any
    info: Dict[str, Any]
    task_state: Any      # whatever NamedTuple the plugin defines
    phys_params: Any     # PhysParams, batched — see domain randomization below
    motor_rpm: Any        # (num_envs, 4) or (num_envs, num_drones, 4) lagged RPM
```

Because `task_state` is itself a pytree, `jax.jit`/`vmap`/`tree_map` traverse
it automatically without the aviary needing to know its concrete type —
this, plus the plugin interface above, is what lets a brand-new task be
"just a plugin" with zero changes to this module.

### Functional step loop, not `gym.Env`

The interface is PureJaxRL-style: `obs, state = env.reset(keys)` and `obs,
state, reward, done, info = env.step(keys, state, action)` — state is
threaded through explicitly rather than mutated on `self`, so the entire
rollout loop (including `env.step` itself) can be wrapped in `jax.jit` /
`lax.scan` for on-device training with no Python-loop overhead per step.
`action` is always clipped to `[-1, 1]` before mixing, regardless of
`action_type`.

**Auto-reset happens inside `step`.** Environments that terminate this step
are reset immediately and returned in the *same* batch — `final_obs`/
`final_state` select between the freshly-reset and the just-stepped values
per-env via the `done` mask (`jnp.where`). This keeps every array in the
batch at a fixed shape across the whole loop, which JAX's tracing requires;
there's no separate "call `.reset()` when an episode ends" step like classic
Gym. The cost is that `obs` on a `done` step is already the *next* episode's
start observation — so `info["true_final_obs"]` (and
`info["true_final_critic_obs"]`, which also folds in
`get_critic_obs()`'s privileged features) is captured **before** that
overwrite, specifically so a value function can bootstrap from the state a
timeout/termination actually landed in rather than from an unrelated fresh
episode's start state.

### Domain randomization and motor dynamics

`domain_rand_fn: (rng, nominal: PhysParams) -> PhysParams`, if supplied, runs
once per env per episode reset (inside vmap, independent of `reset_fn` —
`reset_fn` decides *where* the drone spawns, `domain_rand_fn` decides *what
its dynamics are* for that episode) and is held fixed for the whole episode.
It only randomizes the mixer/control-layer coefficients carried in
`PhysParams` (`kf`, `km`, `arm_length`, `max_rpm`, `hover_rpm`,
`differential_frac`, `motor_tau`) — **not** mass or inertia, which are baked
into the compiled `mjx.Model` at XML-build time rather than being plain JAX
values this module owns; randomizing those would need a batched
`mjx.Model` *and* `mjx.Data`, out of scope here. `MultiVectorAviary` samples
`PhysParams` independently **per drone**, not just per env, matching real
unit-to-unit manufacturing variance between physically separate vehicles.

### Optional logging decomposition

If a plugin's reward function exposes `compute_terms`/`term_names`, or the
plugin itself exposes `termination_names`/`termination_checks`,
`task_metric_names`/`task_metrics`, or `task_episode_metric_names`, the
aviary automatically populates matching keys in `info` each step (reward
term breakdown, per-cause termination flags, task metrics) for W&B-style
logging — otherwise these are simply absent, no zero-filled arrays leaking
into `info`. A generic `bounding_box=(hx, hy, hz)` workspace box is also
available independent of the plugin; on `MultiVectorAviary` it's a **joint**
termination (any one agent leaving the box ends the episode for the whole
group, same semantics as a lap-limit finish).

### Gymnasium bridge

`MJXVecEnvGymWrapper` adapts `MJXVectorAviary` to a numpy-based Gymnasium
`VectorEnv` for SB3 and similar — at the cost of a GPU→CPU transfer every
step. For actual training throughput, drive the JAX-native `reset`/`step`
interface directly so nothing leaves the device.

```python
import jax
from multi_drone_mujoco.vectorized import MJXVectorAviary
from hover.plugin import HoverPlugin  # task plugins live outside this repo

env = MJXVectorAviary(num_envs=4096, plugin=HoverPlugin(), action_type="attitude")
rng = jax.random.PRNGKey(0)
obs, state = env.reset(jax.random.split(rng, env.num_envs))

action = jax.numpy.zeros((env.num_envs, env.act_dim))
obs, state, reward, done, info = env.step(
    jax.random.split(rng, env.num_envs), state, action
)
```

## Tests

```bash
pytest multi_drone_mujoco/tests/ -v
```

## Project Structure

```
multi_drone_mujoco/
├── envs/
│   ├── base_aviary.py          # Core physics engine + Gymnasium env
│   ├── plugins.py              # CPUTaskPlugin interface (bring your own task)
│   ├── task_aviary.py          # Generic TaskAviary — all task logic via a plugin
│   ├── example_plugins.py      # SimpleHoverPlugin / SimpleRacePlugin reference plugins
│   └── multi_agent_aviary.py   # PettingZoo wrapper
├── control/
│   ├── pid_control.py          # Cascaded PID controller
│   └── dsl_pid_control.py      # Enhanced PID with anti-windup
├── vectorized/
│   ├── __init__.py             # MJXVectorAviary — single-agent GPU-vectorized aviary
│   ├── multi_aviary.py         # MultiVectorAviary — N-agent GPU-vectorized aviary
│   └── plugins.py              # TaskPlugin interface (JAX-native, bring your own task)
├── utils/
│   ├── enums.py                # DroneModel, Physics, ActionType, etc.
│   ├── mixer.py                # RPM/attitude action mixers, shared by CPU + MJX paths
│   └── logger.py               # CSV logging + matplotlib plotting
├── examples/
│   ├── pid.py                  # PID control demos
│   ├── downwash.py             # Downwash visualization
│   ├── learn.py                # SB3 PPO training
│   └── play.py                 # Trained model playback
├── tests/
│   ├── test_envs.py            # Environment tests
│   ├── test_control.py         # Controller tests
│   ├── test_multi_agent.py     # MARL tests
│   └── test_features.py        # Wrappers + MJX vectorized import/instantiation tests
└── setup.py
```

## Differences from gym-pybullet-drones

| | gym-pybullet-drones | gym-mujoco-drones |
|---|---|---|
| Physics | PyBullet | MuJoCo (faster, more accurate) |
| Rendering | PyBullet GUI | MuJoCo viewer / offscreen RGB |
| Firmware SITL | Betaflight, CF firmware | — |
| Task environments | 2 (Hover, MultiHover) | 7 (+ velocity, waypoint, formation, race) |
| Multi-agent | Custom | PettingZoo standard |

## Citation

If you use this work, please cite:

```bibtex
@misc{tayal2026mujocodronesgym,
  title={MuJoCo-Drones-Gym: A GPU-Accelerated Multi-Drone Simulator for Control and Reinforcement Learning}, 
      author={Manan Tayal},
      year={2026},
      eprint={2606.08039},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2606.08039}, 
}
```

## Acknowledgements

- [gym-pybullet-drones](https://github.com/learnsyslab/gym-pybullet-drones) — inspiration for the environment API and task design
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) — Bitcraze Crazyflie 2.x MJCF model
- [Bitcraze](https://www.bitcraze.io/) — Crazyflie 2.x hardware platform and firmware parameters

## License

MIT
