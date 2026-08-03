# MJ-drones-gym

MuJoCo-based multi-drone simulation for single- and multi-agent quadcopter RL.

This branch utilizes the more sophisticated system identification for Crazyflie 2.1 Brushless from [How to Model Your Crazyflie Brushless](https://arxiv.org/pdf/2603.05944)

## Installation

```bash
git clone <this-repo>
cd multi_drone_mujoco/
pip install -e ".[all]"
```

Requires Python ≥ 3.8, MuJoCo ≥ 3.0

## Core Interface

- **`MJXVectorAviary`:** (`multi_drone_mujoco/vectorized/`): parallel MJX (JAX-backend MuJoCo) envs on GPU, one controlled drone per environment.
- **`MultiVectorAviary`:** same engine with `num_drones` per env
- **`MJXState`:** manages both per-env physics state `mjx.Data, phys_params, motor_rpm` and task state as defined by the custom plugin `NamedTuple`.

## Task Interface

Tasks are specified by `vectorized.plugins.TaskPlugin`, which exposes the following single-env methods / attributes (vectorized using `jax.vmap` in `MJXVectorAviary / MultiVectorAviary`):
1. **State:**  a `NamedTuple` for the task's per-env state (mirrors `task_state` in `MJXState`).
2. **`obs_dim / init_task_state(num_envs):`** batched (leading dim `num_envs`) initial state.
3. **`reset_task_state(data, rng, hint, old_task_state)`**: the single-env state right after a reset; `hint` carries anything your custom reset function chose (e.g. spawn racing gate index), `old_task_state` lets persistent fields survive across episodes.
4. **`step(data, action, task_state, motor_rpm, phys_params)`**: returns `(new_task_state, obs, reward, extra_done)` — `extra_done` is task-specific termination beyond a step-count timeout.
5. **`get_obs(data, task_state, motor_rpm, phys_params)`**.
6. Optional hooks, all `getattr`/`hasattr`-detected at construction (skipping one costs nothing at trace time): `termination_checks`/`termination_names`, `task_metrics`/`task_metric_names` for W&B logging; `extra_worldbody_xml`/`default_camera_mode`/`camera_config` for rendering.

For a fleshed-out reference — reward/observation/reset decomposition, domain randomization, rendering hooks, all built on top of this exact interface — see [`mjc_dronetests`](../mjc_dronetests)'s `envspecs.plugin_base.ComposedTaskPlugin` and its `race`/`ma_race` task packages, which are what actually train policies against this engine.

## Physics

**Actions** are `(..., 4)` array in `[-1, 1]`, mixed into per-motor RPM via one of the following `action_type` interfaces:
- **`"rpm"`** → `mix_rpm_action` — 4 independent channels for each motor
- **`"attitude"`** → `mix_attitude_rpm` — `[thrust, roll, pitch, yaw_rate]`. `thrust` reuses the `"rpm"` hover-centered curve; `roll`/`pitch`/`yaw_rate` are fixed-scale per-motor RPM differentials (gain = `differential_frac`) with **no closed-loop rate feedback** — a `roll` command is a direct differential-thrust nudge, not a tracked body rate.

Both are normalized such that 0-actions cause the drone to enter `hover_rpm`.

**Quadrotor Parameters:** fields of `PhysParams` sampled once per env per episode reset by `domain_rand_fn: (rng, nominal: PhysParams) -> PhysParams`. Ranges and the actual randomization call are left to upstream projects.

- **`kf`**, **`km` (float):** thrust / reaction-torque coefficients.
- **`arm_length` (float, m):** motor-to-center distance; sets torque lever arm.
- **`max_rpm`**, **`hover_rpm` (float):** RPM ceiling / weight-balancing RPM; both mixers' hover-centered curves are built from these.
- **`differential_frac` (float):** `"attitude"` mixer's roll/pitch/yaw_rate gain; unused for RPM actions
- **`motor_tau` (float, s):** first-order motor lag time constant. Commanded RPM is integrated toward actual RPM every physics substep (not every control step); `motor_tau=0` (this module's default) is instant zero-order-hold. Sampled as an absolute range by `mjc_dronetests`, not scaled around nominal, since 0 is a degenerate "disabled" value.
