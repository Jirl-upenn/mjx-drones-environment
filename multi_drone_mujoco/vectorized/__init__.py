"""GPU-Vectorized Drone Environment using MuJoCo MJX (JAX backend).

Enables thousands of parallel drone simulations on GPU for massively
accelerated RL training. Compatible with JAX-based RL libraries
(e.g., PureJaxRL, Brax-style training loops).

Requirements: pip install mujoco-mjx jax[cuda12]

Architecture
------------
- Compiles the MuJoCo drone model to XLA via MJX
- Vectorizes across N_envs using jax.vmap
- All physics + reward + reset logic runs on GPU
- Returns batched observations/rewards as JAX arrays
- Task-specific state is encapsulated in TaskPlugin subclasses;
  adding a new task only requires a new plugin — no changes to
  MJXVectorAviary or MJXState.

Usage
-----
>>> from multi_drone_mujoco.vectorized.mjx_aviary import MJXVectorAviary
>>> env = MJXVectorAviary(num_envs=4096, num_drones=1, task="hover")
>>> state = env.reset(jax.random.PRNGKey(0))
>>> state, obs, reward, done = env.step(state, action)
"""

from __future__ import annotations

import os
from pathlib import Path
import gymnasium as gym
from typing import Any, Dict, NamedTuple, Optional, Tuple

import numpy as np

from .plugins import TaskPlugin
from multi_drone_mujoco.utils.mixer import mix_attitude_rpm, mix_rpm_action

# Lazy imports for JAX/MJX (only required at runtime)
_JAX_AVAILABLE = False
_MJX_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    from jax import random, vmap, jit, lax
    _JAX_AVAILABLE = True
except ImportError:
    pass

try:
    import mujoco
    from mujoco import mjx
    _MJX_AVAILABLE = True
except (ImportError, AttributeError):
    try:
        import mujoco
        import mujoco.mjx as mjx
        _MJX_AVAILABLE = True
    except (ImportError, AttributeError):
        pass


def _check_deps():
    if not _JAX_AVAILABLE:
        raise ImportError(
            "JAX is required for GPU-vectorized environments.\n"
            "Install with: pip install 'jax[cuda12]' (GPU) or pip install jax (CPU)"
        )
    if not _MJX_AVAILABLE:
        raise ImportError(
            "MuJoCo MJX is required for GPU-vectorized environments.\n"
            "Install with: pip install mujoco-mjx"
        )


# ---------------------------------------------------------------------------
# Shared state container
# ---------------------------------------------------------------------------

class MJXState(NamedTuple):
    """State container for vectorized simulation.

    ``task_state`` holds a task-specific NamedTuple (e.g. HoverTaskState or
    RaceTaskState).  It is a JAX pytree, so jit/vmap/tree_map all handle it
    transparently without needing to know its concrete type.
    """
    mjx_data: Any           # mjx.Data (batched over num_envs)
    step_count: Any         # jnp.ndarray (num_envs,) int32
    done: Any               # jnp.ndarray (num_envs,) bool
    info: Dict[str, Any]    # additional info
    task_state: Any         # HoverTaskState | RaceTaskState | custom


# ---------------------------------------------------------------------------
# Aviary
# ---------------------------------------------------------------------------

_BODY_ID = 1  # drone body index in the MJX model

class MJXVectorAviary:
    """GPU-vectorized multi-drone aviary using MuJoCo MJX.

    Runs N_envs independent simulations in parallel on GPU using JAX's
    vmap over MJX compiled physics. All compute stays on device.

    Task-specific logic (observations, rewards, state bookkeeping) is
    delegated to a ``TaskPlugin`` instance so that adding new tasks does
    not require modifying this class.

    Parameters
    ----------
    num_envs : int
        Number of parallel environments (typical: 1024-65536).
    num_drones : int
        Number of drones per environment.
    task : str
        Task name; used only for XML generation and default episode length.
    plugin : TaskPlugin
        Task plugin that supplies obs, reward, and per-env state logic.
    sim_freq : int
        Physics simulation frequency (Hz).
    ctrl_freq : int
        Control frequency (Hz).
    episode_length : int
        Maximum episode steps.
    backend : str
        JAX backend ("gpu", "cpu"). Auto-detected if None.
    reset_fn : callable or None
        Custom JAX reset function ``(data, rng) -> data`` called inside vmap.
        When None a small position-noise reset is used for hover; the plugin
        should supply a reset_fn for tasks that need something different.

    Example
    -------
    >>> from hover.plugin import HoverPlugin
    >>> plugin = HoverPlugin()
    >>> env = MJXVectorAviary(num_envs=4096, task="hover", plugin=plugin)
    >>> rng = jax.random.PRNGKey(0)
    >>> state = env.reset(rng)
    >>> action = jnp.zeros((4096, 4))
    >>> state, obs, reward, done, info = env.step(state, action)
    """

    def __init__(
        self,
        num_envs: int = 4096,
        num_drones: int = 1,
        task: str = "hover",
        plugin: Optional[TaskPlugin] = None,
        sim_freq: int = 240,
        ctrl_freq: int = 48,
        episode_length: int = 500,
        backend: Optional[str] = None,
        reset_fn=None,
        action_type: str = "rpm",
        bounding_box: Optional[Tuple[float, float, float]] = None,
    ):
        _check_deps()
        if plugin is None:
            raise ValueError(
                "plugin is required. Instantiate a TaskPlugin subclass and pass it as "
                "plugin=... (e.g. from hover.plugin import HoverPlugin; plugin=HoverPlugin())."
            )

        self.num_envs = num_envs
        self.num_drones = num_drones
        self.task = task
        self.sim_freq = sim_freq
        self.ctrl_freq = ctrl_freq
        self.sim_steps_per_ctrl = sim_freq // ctrl_freq

        if action_type not in ("rpm", "attitude"):
            raise ValueError(f"action_type must be 'rpm' or 'attitude', got {action_type!r}")
        self._action_type = action_type
        self._custom_reset_fn = reset_fn
        self._bounding_box = bounding_box

        # Physical constants (Crazyflie 2.x)
        self.mass = 0.027
        self.gravity = 9.81
        self.kf = 3.16e-10
        self.km = 7.94e-12
        self.arm_length = 0.0397
        self.max_rpm = 21714.0
        self.hover_rpm = np.sqrt((self.mass * self.gravity) / (4 * self.kf))

        # Gate xyz positions — needed only for XML marker generation.
        # Retrieved from the plugin so there's one source of truth.
        self._gates_xyz_np = getattr(plugin, "gate_xyz_np", None)

        # Build MuJoCo model
        self._xml = self._generate_xml(plugin)
        self._mj_model = mujoco.MjModel.from_xml_string(self._xml)
        self._mjx_model = mjx.put_model(self._mj_model)

        if task == "race" and episode_length == 500:
            self.episode_length = 1440
        else:
            self.episode_length = episode_length

        self._task: TaskPlugin = plugin
        self.obs_dim = self._task.obs_dim
        self.act_dim = 4 * num_drones

        # Reward term decomposition (optional — enabled when reward fn exposes compute_terms)
        _reward_fn = getattr(plugin, '_reward_fn', None)
        self._compute_terms_fn = getattr(_reward_fn, 'compute_terms', None)
        self._num_terms: int = len(getattr(_reward_fn, 'term_names', []))

        # Named termination conditions (optional — enabled when plugin exposes termination_names)
        self._num_terminations: int = len(getattr(plugin, 'termination_names', []))

        # Task metric decomposition (optional — enabled when plugin exposes task_metric_names)
        self._num_task_metrics: int = len(getattr(plugin, 'task_metric_names', []))

        # Per-episode task metrics (optional — enabled when plugin exposes task_episode_metric_names)
        self._num_episode_task_metrics: int = len(getattr(plugin, 'task_episode_metric_names', []))

        # Bounding box adds one extra termination flag.
        self._num_bbox_terms: int = 1 if bounding_box is not None else 0

        # Death cost: read from reward fn; grace terminations skip the penalty.
        _reward_fn = getattr(plugin, '_reward_fn', None)
        self._death_cost: float = getattr(_reward_fn, 'death_cost', 0.0)
        _grace_names = {"timeout", "lap_limit"}
        self._grace_indices = [
            i for i, n in enumerate(self.termination_names) if n in _grace_names
        ]

        # JIT-compile core functions
        self._step_fn  = jit(vmap(self._single_step))
        self._reset_fn = jit(vmap(self._single_reset))

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return (self.obs_dim,)

    @property
    def action_shape(self) -> Tuple[int, ...]:
        return (self.act_dim,)

    @property
    def termination_names(self) -> list:
        """Ordered termination condition names matching the last axis of info['terminations']."""
        plugin_names = list(getattr(self._task, 'termination_names', []))
        names = ["timeout"] + plugin_names
        if self._bounding_box is not None:
            names += ["out_of_bounds"]
        return names

    class _Space:
        """Minimal space stand-in that exposes .shape for obs/action dims."""
        def __init__(self, shape: tuple):
            self.shape = shape

    def observation_space(self, params=None) -> "_Space":
        return self._Space((self.obs_dim,))

    def action_space(self, params=None) -> "_Space":
        return self._Space((self.act_dim,))

    def reset(self, keys: Any, params=None) -> Tuple[Any, MJXState]:
        """Reset all environments.  PureJaxRL interface: keys has shape (num_envs, 2).

        Returns
        -------
        (obs, state) : ((num_envs, obs_dim), MJXState)
        """
        rng = keys[0]
        rngs = random.split(rng, self.num_envs)
        mjx_data = mjx.put_data(self._mj_model, mujoco.MjData(self._mj_model))
        batched_data = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (self.num_envs, *x.shape)).copy(),
            mjx_data,
        )
        state = MJXState(
            mjx_data=batched_data,
            step_count=jnp.zeros(self.num_envs, dtype=jnp.int32),
            done=jnp.zeros(self.num_envs, dtype=jnp.bool_),
            info={},
            task_state=self._task.init_task_state(self.num_envs),
        )
        state = self._reset_fn(state, rngs)
        obs = vmap(self._task.get_obs)(state.mjx_data, state.task_state)
        return obs, state

    def step(self, keys: Any, state: MJXState, action: Any, params=None) -> Tuple[Any, MJXState, Any, Any, Dict]:
        """Step all environments with auto-reset for done envs.

        PureJaxRL interface: keys has shape (num_envs, 2); used as per-env
        reset seeds for environments that terminate this step.

        Returns
        -------
        (obs, state, reward, done, info)
        """
        action = jnp.clip(action, -1.0, 1.0)
        next_state, obs, reward, done = self._step_fn(state, action)
        info = next_state.info

        if self._death_cost != 0.0:
            terminations = info["terminations"]
            is_grace = jnp.zeros(self.num_envs, dtype=jnp.bool_)
            for idx in self._grace_indices:
                is_grace = is_grace | terminations[:, idx]
            is_crash = done & ~is_grace
            reward = jnp.where(is_crash, reward + self._death_cost, reward)

        reset_state = self._reset_fn(next_state, keys)
        reset_obs = vmap(self._task.get_obs)(reset_state.mjx_data, reset_state.task_state)

        def _select(r, n):
            if not hasattr(r, "ndim") or r.ndim == 0:
                return r
            mask = done.reshape((done.shape[0],) + (1,) * (r.ndim - 1))
            return jnp.where(mask, r, n)

        final_state = jax.tree_util.tree_map(_select, reset_state, next_state)
        final_obs = jnp.where(done[:, None], reset_obs, obs)

        return final_obs, final_state, reward, done, info

    def get_obs(self, state: MJXState) -> Any:
        """Extract observations from a batched state."""
        return vmap(self._task.get_obs)(state.mjx_data, state.task_state)

    def _single_step(self, state: MJXState, action: Any) -> Tuple[MJXState, Any, Any, Any]:
        """Step a single environment (vmapped over batch)."""
        if self._action_type == "attitude":
            rpm = mix_attitude_rpm(jnp, action, self.hover_rpm, self.max_rpm)
        else:  # "rpm"
            rpm = mix_rpm_action(jnp, action, self.hover_rpm, self.max_rpm)

        # Compute forces from RPM
        forces = self.kf * rpm ** 2
        total_thrust = jnp.sum(forces)

        data = state.mjx_data
        xmat = data.xmat[_BODY_ID].reshape(3, 3)
        thrust_world = xmat @ jnp.array([0.0, 0.0, total_thrust])

        L, s2 = self.arm_length, jnp.sqrt(2.0)
        tau_x = (forces[0] + forces[1] - forces[2] - forces[3]) * L / s2
        tau_y = (-forces[0] + forces[1] + forces[2] - forces[3]) * L / s2
        tau_z = (-forces[0] + forces[1] - forces[2] + forces[3]) * self.km / self.kf
        torque_world = xmat @ jnp.array([tau_x, tau_y, tau_z])

        xfrc = jnp.zeros_like(data.xfrc_applied)
        xfrc = xfrc.at[_BODY_ID, :3].set(thrust_world)
        xfrc = xfrc.at[_BODY_ID, 3:].set(torque_world)
        data = data.replace(xfrc_applied=xfrc)

        def _physics_step(d, _):
            return mjx.step(self._mjx_model, d), None

        data, _ = lax.scan(_physics_step, data, None, length=self.sim_steps_per_ctrl)

        step_count = state.step_count + 1
        new_task_state, obs, reward, extra_done = self._task.step(data, action, state.task_state)
        timeout = step_count >= self.episode_length

        # Named termination checks — plugin-defined, resolved at trace time.
        if self._num_terminations > 0:
            plugin_terms = self._task.termination_checks(data, new_task_state)
            extra_done = extra_done | jnp.any(plugin_terms)
        else:
            plugin_terms = jnp.zeros(0, dtype=jnp.bool_)

        # Generic workspace bounding box — resolved at trace time.
        if self._bounding_box is not None:
            pos = data.xpos[_BODY_ID]
            hx, hy, hz = self._bounding_box
            out_of_bounds = (
                (jnp.abs(pos[0]) > hx)
                | (jnp.abs(pos[1]) > hy)
                | (pos[2] > hz)
            )
            extra_done = extra_done | out_of_bounds
            bbox_terms = jnp.array([out_of_bounds], dtype=jnp.bool_)
        else:
            bbox_terms = jnp.zeros(0, dtype=jnp.bool_)

        done = timeout | extra_done

        step_info = {}
        if self._num_terms > 0:
            step_info["reward_terms"] = self._compute_terms_fn(data, action, state.task_state)
        if self._num_task_metrics > 0:
            step_info["task_metrics"] = self._task.task_metrics(new_task_state)
        if self._num_episode_task_metrics > 0:
            step_info["episode_task_metrics"] = self._task.task_episode_metrics(new_task_state)
        step_info["terminations"] = jnp.concatenate(
            [jnp.array([timeout], dtype=jnp.bool_), plugin_terms, bbox_terms]
        )

        new_state = MJXState(
            mjx_data=data,
            step_count=step_count,
            done=done,
            info=step_info,
            task_state=new_task_state,
        )
        return new_state, obs, reward, done

    def _single_reset(self, state: MJXState, rng: Any) -> MJXState:
        """Reset a single environment (vmapped over batch)."""
        data = mjx.put_data(self._mj_model, mujoco.MjData(self._mj_model))
        reset_hint = None
        if self._custom_reset_fn is not None:
            result = self._custom_reset_fn(data, rng, state.task_state)
            if isinstance(result, tuple):
                data, reset_hint = result
            else:
                data = result
        elif self.task == "race":
            qpos = data.qpos.at[:3].set(jnp.array([0.0, 0.0, 0.5]))
            qpos = qpos.at[3:7].set(jnp.array([1.0, 0.0, 0.0, 0.0]))
            data = data.replace(qpos=qpos, qvel=jnp.zeros_like(data.qvel))
        else:
            rng, noise_rng = jax.random.split(rng)
            pos_noise = random.normal(noise_rng, shape=(3,)) * 0.02
            qpos = data.qpos.at[0:3].add(pos_noise)
            data = data.replace(qpos=qpos)

        # Propagate qpos → xpos/xmat/cvel so the initial observation and
        # reset_task_state see the correct post-reset kinematic state.
        data = mjx.forward(self._mjx_model, data)

        rng, task_rng = jax.random.split(rng)
        task_state = self._task.reset_task_state(data, task_rng, reset_hint, state.task_state)

        reset_info = {}
        if self._num_terms > 0:
            reset_info["reward_terms"] = jnp.zeros(self._num_terms)
        if self._num_task_metrics > 0:
            reset_info["task_metrics"] = jnp.zeros(self._num_task_metrics)
        if self._num_episode_task_metrics > 0:
            reset_info["episode_task_metrics"] = jnp.zeros(self._num_episode_task_metrics)
        reset_info["terminations"] = jnp.zeros(1 + self._num_terminations + self._num_bbox_terms, dtype=jnp.bool_)
        return MJXState(
            mjx_data=data,
            step_count=jnp.int32(0),
            done=jnp.bool_(False),
            info=reset_info,
            task_state=task_state,
        )

    def _generate_xml(self, plugin) -> str:
        """Generate minimal MuJoCo XML for MJX compilation."""
        mass = self.mass
        ixx, iyy, izz = 1.4e-5, 1.4e-5, 2.17e-5

        # Arm tip offset along each axis (45° quadrotor X-frame layout).
        a = 0.064

        drones = ""
        for d in range(self.num_drones):
            x_offset = d * 0.5
            drones += f"""
    <body name="drone{d}" pos="{x_offset} 0 0.115">
      <freejoint name="drone{d}_joint"/>
      <inertial pos="0 0 0" mass="{mass}" diaginertia="{ixx} {iyy} {izz}"/>
      <geom type="cylinder" size="0.04 0.007" rgba="1 1 1 1" contype="1" conaffinity="1"/>
      <geom type="capsule" fromto="0 0 0  {a:.3f}  {a:.3f} 0" size="0.006" rgba="1 1 1 1" contype="0" conaffinity="0"/>
      <geom type="capsule" fromto="0 0 0 -{a:.3f}  {a:.3f} 0" size="0.006" rgba="1 1 1 1" contype="0" conaffinity="0"/>
      <geom type="capsule" fromto="0 0 0  {a:.3f} -{a:.3f} 0" size="0.006" rgba="1 1 1 1" contype="0" conaffinity="0"/>
      <geom type="capsule" fromto="0 0 0 -{a:.3f} -{a:.3f} 0" size="0.006" rgba="1 1 1 1" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos=" {a:.3f}  {a:.3f} 0.008" size="0.025 0.003" rgba="1 1 1 1" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="-{a:.3f}  {a:.3f} 0.008" size="0.025 0.003" rgba="1 1 1 1" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos=" {a:.3f} -{a:.3f} 0.008" size="0.025 0.003" rgba="1 1 1 1" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="-{a:.3f} -{a:.3f} 0.008" size="0.025 0.003" rgba="1 1 1 1" contype="0" conaffinity="0"/>
      <site name="drone{d}_center" pos="0 0 0"/>
    </body>"""

        extra_worldbody = plugin.extra_worldbody_xml()

        return f"""<mujoco model="mjx_aviary">
  <option integrator="RK4" timestep="{1.0/self.sim_freq}" gravity="0 0 -{self.gravity}"/>
  <compiler autolimits="true"/>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.55 0.72 0.95" rgb2="0.25 0.45 0.80" width="512" height="3072"/>
  </asset>

  <visual>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.5 0.5 0.5" specular="0.2 0.2 0.2"/>
  </visual>

  <worldbody>
    <light pos="0 0 5" dir="0 0 -1" diffuse="1.0 1.0 0.95" specular="0.6 0.6 0.5" directional="true"/>
    <light pos="4 2 4" dir="-1 -0.5 -1" diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1" directional="true"/>
    <light pos="-2 4 3" dir="0.5 -1 -1" diffuse="0.35 0.35 0.35" directional="true"/>
    <geom name="floor" size="2 2 0.05" type="plane" rgba="0.15 0.30 0.65 1" contype="1" conaffinity="1"/>
{drones}{extra_worldbody}
  </worldbody>
</mujoco>"""


# ---------------------------------------------------------------------------
# Gymnasium VectorEnv wrapper (unchanged)
# ---------------------------------------------------------------------------

class MJXVecEnvGymWrapper:
    """Gymnasium VectorEnv-compatible wrapper around MJXVectorAviary.

    Bridges the JAX-native interface to numpy-based Gymnasium API
    for compatibility with SB3 and other standard RL libraries.

    Note: This involves GPU→CPU transfers each step. For maximum
    performance, use the JAX-native interface directly.
    """

    def __init__(self, **kwargs):
        _check_deps()
        self._env = MJXVectorAviary(**kwargs)
        self._state = None
        self.num_envs = self._env.num_envs
        self.single_observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=self._env.observation_shape, dtype=np.float32
        )
        self.single_action_space = gym.spaces.Box(
            -1.0, 1.0, shape=self._env.action_shape, dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf,
            shape=(self.num_envs, *self._env.observation_shape),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            -1.0, 1.0,
            shape=(self.num_envs, *self._env.action_shape),
            dtype=np.float32,
        )

    def reset(self, seed: int = 0):
        rng = random.PRNGKey(seed)
        self._state = self._env.reset(rng)
        obs = self._env.get_obs(self._state)
        return np.asarray(obs)

    def step(self, action: np.ndarray):
        action_jax = jnp.array(action)
        self._state, obs, reward, done, info = self._env.step(self._state, action_jax)
        return (
            np.asarray(obs),
            np.asarray(reward),
            np.asarray(done),
            info,
        )
