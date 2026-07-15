"""GPU-vectorized N-drone aviary for multi-agent tasks (MJX/JAX backend).

Separate from ``MJXVectorAviary`` (vectorized/__init__.py) rather than a
generalization of it: that class hardcodes ``_BODY_ID = 1`` and only ever
applies motor forces to a single body, so every extra drone in its XML is
inert. Bolting N-drone control onto it in place would risk regressing the
single-agent hover/race training paths that depend on it today. Instead this
module reimplements just the physics-stepping half (force/torque application,
looped over ``num_drones``) while reusing the same ``MJXState`` container and
the same ``TaskPlugin`` interface — a multi-agent plugin's ``step``/``get_obs``
simply operate on stacked per-agent arrays instead of a single agent's.

Contract for a plugin used with MultiVectorAviary (vs. single-agent TaskPlugin):
  - ``obs_dim`` is the PER-AGENT observation dimension (not multiplied by
    num_drones).
  - ``step(data, action, task_state)``: ``action`` has shape
    ``(num_drones, act_dim_per_drone)``; returns ``obs`` shaped
    ``(num_drones, obs_dim)`` and ``reward`` shaped ``(num_drones,)`` — one
    reward per agent, since each agent trains its own policy. ``extra_done``
    stays a single scalar: episode termination is a joint, not per-agent,
    condition (e.g. "all agents crashed or someone finished").
  - ``termination_checks``/``task_metrics``/``task_episode_metrics`` return
    flat 1-D arrays as usual; when a metric is naturally per-agent (e.g.
    win rate), the plugin flattens it into per-agent-named scalar entries
    (matching how CPU obs functions already concatenate one block per drone)
    rather than adding another array axis, so W&B logging code doesn't need
    to know about the agent dimension at all.

Nothing here hardcodes 2 drones — ``num_drones`` is a free parameter.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from . import MJXState, PhysParams, _INERTIA_DIAG
from .plugins import TaskPlugin
from multi_drone_mujoco.utils.mixer import (
    mix_attitude_rpm, mix_rpm_action, mix_attitude_pid_rpm,
    KP_OMEGA_RP, KI_OMEGA_RP, KD_OMEGA_RP, KP_OMEGA_Y, KI_OMEGA_Y, KD_OMEGA_Y,
)

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


class MultiVectorAviary:
    """GPU-vectorized N-drone-per-env aviary using MuJoCo MJX.

    Same batching model as ``MJXVectorAviary`` (vmap over ``num_envs``), but
    every drone body in the model actually receives its own motor forces —
    ``action`` carries a leading agent axis, and the plugin is responsible
    for turning per-agent physics into per-agent obs/reward.

    Parameters
    ----------
    num_envs : int
    num_drones : int
        Number of independently-controlled drones per env. No task-side code
        in this class assumes any particular value.
    plugin : TaskPlugin
        Must satisfy the multi-agent contract described in the module
        docstring (obs_dim is per-agent; step() takes/returns stacked
        per-agent arrays).
    reset_fn : callable
        Required (unlike MJXVectorAviary, there is no sensible generic
        default spawn for N drones) — ``(data, rng, task_state) -> mjx.Data``
        or ``(mjx.Data, hint)``, called inside vmap.
    action_type : {"rpm", "attitude", "attitude_pid"}
        Same per-drone mixer as the single-agent aviary; applied independently
        to each drone's own 4-dim action slice. See vectorized/__init__.py's
        MJXVectorAviary docstring for what each mode does.
    bounding_box : (hx, hy, hz) or None
        Generic workspace bounding box, mirroring MJXVectorAviary's own —
        episode terminates if ANY agent's position exits ±hx, ±hy, or [*, hz]
        (joint termination, same semantics as the plugin's own extra_done
        conditions like a lap limit: one agent diverging arbitrarily far off
        the track ends the episode for the whole group rather than letting
        the rest keep racing around it). None disables the check entirely.
    """

    def __init__(
        self,
        num_envs: int = 4096,
        num_drones: int = 2,
        task: str = "ma_race",
        plugin: Optional[TaskPlugin] = None,
        sim_freq: int = 240,
        ctrl_freq: int = 48,
        episode_length: int = 1440,
        reset_fn=None,
        action_type: str = "rpm",
        bounding_box: Optional[Tuple[float, float, float]] = None,
        mass: float = 0.027,
        kf: float = 3.16e-10,
        km: float = 7.94e-12,
        arm_length: float = 0.0397,
        max_rpm: float = 21714.0,
        domain_rand_fn=None,
        motor_tau: float = 0.0,
    ):
        _check_deps()
        if plugin is None:
            raise ValueError("plugin is required — instantiate a multi-agent TaskPlugin and pass it as plugin=...")
        if reset_fn is None:
            raise ValueError(
                "reset_fn is required for MultiVectorAviary — there is no generic "
                "default spawn for N independently-controlled drones."
            )
        if num_drones < 1:
            raise ValueError(f"num_drones must be >= 1, got {num_drones}")

        self.num_envs = num_envs
        self.num_drones = num_drones
        self._body_ids = jnp.arange(1, num_drones + 1)  # drone{d} -> body 1+d, see _body_id
        self.task = task
        self.sim_freq = sim_freq
        self.ctrl_freq = ctrl_freq
        self.sim_steps_per_ctrl = sim_freq // ctrl_freq
        self.episode_length = episode_length

        if action_type not in ("rpm", "attitude", "attitude_pid"):
            raise ValueError(
                f"action_type must be 'rpm', 'attitude', or 'attitude_pid', got {action_type!r}"
            )
        self._action_type = action_type
        self._custom_reset_fn = reset_fn
        self._bounding_box = bounding_box
        self._num_bbox_terms: int = 1 if bounding_box is not None else 0
        self._domain_rand_fn = domain_rand_fn
        self._motor_tau = motor_tau

        # Physical constants — Crazyflie 2.x by default; pass mass/kf/km/
        # arm_length/max_rpm to simulate a different drone platform. See
        # vectorized/__init__.py's MJXVectorAviary for the identical
        # single-agent version of everything below.
        self.mass = mass
        self.gravity = 9.81
        self.kf = kf
        self.km = km
        self.arm_length = arm_length
        self.max_rpm = max_rpm
        self.hover_rpm = np.sqrt((self.mass * self.gravity) / (4 * self.kf))
        self.differential_frac = 0.02  # nominal mix_attitude_rpm gain — see PhysParams
        # attitude_pid's rate-PID needs inertia @ angular_acceleration =
        # commanded moment — see vectorized/__init__.py's _INERTIA_DIAG.
        self._inertia_diag = jnp.array(_INERTIA_DIAG)
        self.act_dim_per_drone = 4
        # Nominal per-drone dynamics — every drone uses this fixed value
        # unless domain_rand_fn overrides it (independently per drone, not
        # just per env — see _single_reset: each drone gets its own
        # domain_rand_fn draw, matching real manufacturing/unit-to-unit
        # variance between physically separate vehicles).
        self._nominal_phys = PhysParams(
            kf=self.kf, km=self.km, arm_length=self.arm_length,
            max_rpm=self.max_rpm, hover_rpm=self.hover_rpm,
            differential_frac=self.differential_frac, motor_tau=self._motor_tau,
            # attitude_pid rate-PID gains — see PhysParams docstring.
            kp_omega_rp=KP_OMEGA_RP, ki_omega_rp=KI_OMEGA_RP, kd_omega_rp=KD_OMEGA_RP,
            kp_omega_y=KP_OMEGA_Y, ki_omega_y=KI_OMEGA_Y, kd_omega_y=KD_OMEGA_Y,
        )

        self._gates_xyz_np = getattr(plugin, "gate_xyz_np", None)

        self._xml = self._generate_xml(plugin)
        self._mj_model = mujoco.MjModel.from_xml_string(self._xml)
        self._mjx_model = mjx.put_model(self._mj_model)

        self._task: TaskPlugin = plugin
        self.obs_dim = self._task.obs_dim  # per-agent
        self.act_dim = self.act_dim_per_drone  # per-agent
        self.privileged_dim = getattr(plugin, "privileged_dim", 0)  # per-agent, 0 if none

        self._num_terminations: int = len(getattr(plugin, 'termination_names', []))
        self._num_terms: int = len(getattr(getattr(plugin, '_reward_fn', None), 'term_names', []))
        self._compute_terms_fn = getattr(getattr(plugin, '_reward_fn', None), 'compute_terms', None)
        self._num_task_metrics: int = len(getattr(plugin, 'task_metric_names', []))
        self._num_episode_task_metrics: int = len(getattr(plugin, 'task_episode_metric_names', []))

        self._step_fn = jit(vmap(self._single_step))
        self._reset_fn = jit(vmap(self._single_reset))

    def _body_id(self, agent_idx: int) -> int:
        """Body index for drone ``agent_idx`` in the MJX model — world body is 0,
        drone bodies follow in order (matches _generate_xml below)."""
        return 1 + agent_idx

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return (self.num_drones, self.obs_dim)

    @property
    def action_shape(self) -> Tuple[int, ...]:
        return (self.num_drones, self.act_dim)

    @property
    def termination_names(self) -> list:
        names = ["timeout"] + list(getattr(self._task, 'termination_names', []))
        if self._bounding_box is not None:
            names += ["out_of_bounds"]
        return names

    class _Space:
        def __init__(self, shape: tuple):
            self.shape = shape

    def observation_space(self, params=None) -> "_Space":
        return self._Space((self.num_drones, self.obs_dim))

    def action_space(self, params=None) -> "_Space":
        return self._Space((self.num_drones, self.act_dim))

    def reset(self, keys: Any, params=None) -> Tuple[Any, MJXState]:
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
            # Placeholders only — _reset_fn (_single_reset) below returns a
            # complete new MJXState, including its own freshly-sampled
            # phys_params/motor_rpm, so these values are never actually read.
            phys_params=jax.tree_util.tree_map(
                lambda x: jnp.broadcast_to(x, (self.num_envs, self.num_drones)), self._nominal_phys
            ),
            motor_rpm=jnp.broadcast_to(self.hover_rpm, (self.num_envs, self.num_drones, 4)),
            pid_integral=jnp.zeros((self.num_envs, self.num_drones, 3)),
            pid_prev_omega=jnp.zeros((self.num_envs, self.num_drones, 3)),
        )
        state = self._reset_fn(state, rngs)
        obs = vmap(self._task.get_obs)(state.mjx_data, state.task_state)
        return obs, state

    def step(self, keys: Any, state: MJXState, action: Any, params=None) -> Tuple[Any, MJXState, Any, Any, Dict]:
        """action: (num_envs, num_drones, act_dim). reward: (num_envs, num_drones).
        done: (num_envs,) — shared/joint termination, plugin-determined."""
        action = jnp.clip(action, -1.0, 1.0)
        next_state, obs, reward, done = self._step_fn(state, action)
        # Shallow-copied, not mutated in place: next_state.info is about to
        # be fed into self._reset_fn/tree_map below (to build the carried
        # final_state), so it must keep its original key set — the extra
        # "true_final_obs" key only belongs on the dict this method returns.
        # See vectorized/__init__.py's MJXVectorAviary.step for the full
        # rationale (bootstrapping a value estimate from a timeout-ended
        # episode's real end state, not the unrelated reset state).
        info = dict(next_state.info)
        info["true_final_obs"] = obs
        # Same rationale, for the asymmetric/centralized critic's own input
        # (own obs ++ privileged features) — computed from next_state (the
        # true, pre-reset state) rather than the post-reset final_state
        # get_critic_obs() would otherwise be called on.
        info["true_final_critic_obs"] = jnp.concatenate(
            [obs, self.get_critic_obs(next_state)], axis=-1
        )

        reset_state = self._reset_fn(next_state, keys)
        reset_obs = vmap(self._task.get_obs)(reset_state.mjx_data, reset_state.task_state)

        def _select(r, n):
            if not hasattr(r, "ndim") or r.ndim == 0:
                return r
            mask = done.reshape((done.shape[0],) + (1,) * (r.ndim - 1))
            return jnp.where(mask, r, n)

        final_state = jax.tree_util.tree_map(_select, reset_state, next_state)
        final_obs = jnp.where(done[:, None, None], reset_obs, obs)

        return final_obs, final_state, reward, done, info

    def get_obs(self, state: MJXState) -> Any:
        return vmap(self._task.get_obs)(state.mjx_data, state.task_state)

    def get_critic_obs(self, state: MJXState) -> Any:
        """Critic-only privileged features, shape (num_envs, num_drones,
        privileged_dim) — 0-width (empty) unless the plugin's obs_fn defines
        some (see envspecs.observations.ObservationFunction.privileged).
        Callers wanting a centralized/asymmetric critic concatenate this with
        get_obs()'s regular actor observation themselves; MultiVectorAviary
        doesn't do it automatically since most callers (pure IPPO, no
        privileged info) don't need it at all."""
        if not hasattr(self._task, "get_privileged_obs"):
            return jnp.zeros((state.step_count.shape[0], self.num_drones, 0))
        return vmap(self._task.get_privileged_obs)(state.mjx_data, state.task_state)

    def _single_step(self, state: MJXState, action: Any) -> Tuple[MJXState, Any, Any, Any]:
        """action: (num_drones, act_dim_per_drone) for a single env (vmapped over batch)."""
        data = state.mjx_data
        body_ids = self._body_ids
        phys = state.phys_params  # each field shape (num_drones,) — see PhysParams

        if self._action_type == "attitude_pid":
            # Computed fresh every physics substep inside _physics_step below
            # (needs each drone's current measured angular velocity) rather
            # than once here — see mix_attitude_pid_rpm's docstring.
            rpm_cmd = None
        elif self._action_type == "attitude":
            rpm_cmd = vmap(mix_attitude_rpm, in_axes=(None, 0, 0, 0, 0))(
                jnp, action, phys.hover_rpm, phys.max_rpm, phys.differential_frac)
        else:
            # phys.hover_rpm/max_rpm are (num_drones,) — broadcast against
            # action's (num_drones, 4) via an explicit trailing axis (unlike
            # the old shared-scalar case, which broadcast for free).
            rpm_cmd = mix_rpm_action(jnp, action, phys.hover_rpm[:, None], phys.max_rpm[:, None])

        L, s2 = phys.arm_length, jnp.sqrt(2.0)
        sim_dt = 1.0 / self.sim_freq
        # alpha=1.0 (no lag) exactly when phys.motor_tau=0 — see
        # vectorized/__init__.py's MJXVectorAviary._single_step for the full
        # rationale (same formula, per-drone here instead of per-env).
        alpha = sim_dt / (phys.motor_tau + sim_dt)  # (num_drones,)

        def _physics_step(carry, _):
            d, motor_rpm, pid_integral, prev_omega = carry
            xmat = d.xmat[body_ids].reshape(self.num_drones, 3, 3)

            if self._action_type == "attitude_pid":
                # Per-drone measured body-frame angular velocity: MJX's
                # freejoint qvel angular component is world-frame — rotate
                # with xmat^T (per drone), see vectorized/__init__.py's
                # single-agent version for the same rotation the other way
                # (xmat @ v_body -> v_world) just below.
                omega_world = d.qvel.reshape(self.num_drones, 6)[:, 3:6]  # (num_drones, 3)
                omega_body = jnp.einsum("nji,nj->ni", xmat, omega_world)
                step_rpm_cmd, pid_integral, prev_omega = vmap(
                    mix_attitude_pid_rpm,
                    in_axes=(None, 0, 0, 0, 0, None, None, 0, 0, 0, 0, None, None,
                             0, 0, 0, 0, 0, 0),
                )(
                    jnp, action, omega_body, pid_integral, prev_omega,
                    self.mass, self.gravity, phys.kf, phys.km, phys.arm_length,
                    phys.max_rpm, self._inertia_diag, sim_dt,
                    phys.kp_omega_rp, phys.ki_omega_rp, phys.kd_omega_rp,
                    phys.kp_omega_y, phys.ki_omega_y, phys.kd_omega_y,
                )
            else:
                step_rpm_cmd = rpm_cmd

            motor_rpm = alpha[:, None] * step_rpm_cmd + (1.0 - alpha[:, None]) * motor_rpm  # (num_drones, 4)

            forces = phys.kf[:, None] * motor_rpm ** 2        # (num_drones, 4)
            total_thrust = jnp.sum(forces, axis=-1)           # (num_drones,)

            thrust_world = xmat[:, :, 2] * total_thrust[:, None]  # xmat @ [0,0,T] == T * z-column

            tau_x = (forces[:, 0] + forces[:, 1] - forces[:, 2] - forces[:, 3]) * L / s2
            tau_y = (-forces[:, 0] + forces[:, 1] + forces[:, 2] - forces[:, 3]) * L / s2
            tau_z = (-forces[:, 0] + forces[:, 1] - forces[:, 2] + forces[:, 3]) * phys.km / phys.kf
            torque_body = jnp.stack([tau_x, tau_y, tau_z], axis=-1)        # (num_drones, 3)
            torque_world = jnp.einsum("nij,nj->ni", xmat, torque_body)

            xfrc = jnp.zeros_like(d.xfrc_applied)
            xfrc = xfrc.at[body_ids, :3].set(thrust_world)
            xfrc = xfrc.at[body_ids, 3:].set(torque_world)
            d = d.replace(xfrc_applied=xfrc)

            d = mjx.step(self._mjx_model, d)
            return (d, motor_rpm, pid_integral, prev_omega), None

        (data, motor_rpm, pid_integral, prev_omega), _ = lax.scan(
            _physics_step,
            (data, state.motor_rpm, state.pid_integral, state.pid_prev_omega),
            None, length=self.sim_steps_per_ctrl,
        )

        step_count = state.step_count + 1
        new_task_state, obs, reward, extra_done = self._task.step(data, action, state.task_state)
        timeout = step_count >= self.episode_length

        if self._num_terminations > 0:
            plugin_terms = self._task.termination_checks(data, new_task_state)
        else:
            plugin_terms = jnp.zeros(0, dtype=jnp.bool_)

        # Generic workspace bounding box — resolved at trace time, mirrors
        # MJXVectorAviary's single-agent version. Joint termination: ANY
        # agent out of bounds ends the episode for the whole group (same
        # semantics as a lap-limit finish), not just the offending agent.
        if self._bounding_box is not None:
            positions = data.xpos[body_ids]  # (num_drones, 3)
            hx, hy, hz = self._bounding_box
            out_of_bounds_per_agent = (
                (jnp.abs(positions[:, 0]) > hx)
                | (jnp.abs(positions[:, 1]) > hy)
                | (positions[:, 2] > hz)
            )
            out_of_bounds = jnp.any(out_of_bounds_per_agent)
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
            # Dynamics are fixed for the episode's duration — only resampled
            # at reset (_single_reset), never mid-episode.
            phys_params=phys,
            # The lagged motor state carries forward continuously (that's
            # the whole point of modeling lag) — not reset here.
            motor_rpm=motor_rpm,
            # Same continuity rationale as motor_rpm — unchanged pass-through
            # when action_type != "attitude_pid".
            pid_integral=pid_integral,
            pid_prev_omega=prev_omega,
        )
        return new_state, obs, reward, done

    def _single_reset(self, state: MJXState, rng: Any) -> MJXState:
        data = mjx.put_data(self._mj_model, mujoco.MjData(self._mj_model))
        # Split off reset_fn's own key rather than reusing the same `rng`
        # for phys_rng/task_rng below — see vectorized/__init__.py's
        # MJXVectorAviary._single_reset for why reusing it would correlate
        # rather than decorrelate them.
        rng, reset_fn_rng = jax.random.split(rng)
        result = self._custom_reset_fn(data, reset_fn_rng, state.task_state)
        reset_hint = None
        if isinstance(result, tuple):
            data, reset_hint = result
        else:
            data = result

        data = mjx.forward(self._mjx_model, data)

        rng, task_rng = jax.random.split(rng)
        task_state = self._task.reset_task_state(data, task_rng, reset_hint, state.task_state)

        # Domain randomization — independent per DRONE, not just per env
        # (real manufacturing/unit-to-unit variance is between physically
        # separate vehicles, not shared by every drone in the same race).
        # Reuses the exact same domain_rand_fn(rng, nominal)->PhysParams
        # contract as the single-agent MJXVectorAviary unchanged — an outer
        # vmap over one rng key per drone is all that's needed to turn a
        # single-drone sampler into a per-drone one.
        rng, phys_rng = jax.random.split(rng)
        if self._domain_rand_fn is not None:
            drone_rngs = jax.random.split(phys_rng, self.num_drones)
            phys_params = vmap(self._domain_rand_fn, in_axes=(0, None))(drone_rngs, self._nominal_phys)
        else:
            phys_params = jax.tree_util.tree_map(
                lambda x: jnp.broadcast_to(x, (self.num_drones,)), self._nominal_phys
            )

        reset_info = {}
        if self._num_terms > 0:
            reset_info["reward_terms"] = jnp.zeros((self.num_drones, self._num_terms))
        if self._num_task_metrics > 0:
            # (num_drones, num_task_metrics) — matches reward_terms' own
            # per-agent axis just above; task_metrics is per-agent here
            # (e.g. ma_race/loggers.py's lap_completed), unlike the
            # single-agent aviary where it's a flat (num_names,) vector.
            reset_info["task_metrics"] = jnp.zeros((self.num_drones, self._num_task_metrics))
        if self._num_episode_task_metrics > 0:
            reset_info["episode_task_metrics"] = jnp.zeros(self._num_episode_task_metrics)
        reset_info["terminations"] = jnp.zeros(
            1 + self._num_terminations + self._num_bbox_terms, dtype=jnp.bool_
        )
        return MJXState(
            mjx_data=data,
            step_count=jnp.int32(0),
            done=jnp.bool_(False),
            info=reset_info,
            task_state=task_state,
            phys_params=phys_params,
            motor_rpm=jnp.broadcast_to(phys_params.hover_rpm[:, None], (self.num_drones, 4)),
            # Rate-PID state resets to 0 every episode, independently per
            # drone — see vectorized/__init__.py's MJXVectorAviary._single_reset
            # for the full rationale (matches AgileFlight's own per-env reset).
            pid_integral=jnp.zeros((self.num_drones, 3)),
            pid_prev_omega=jnp.zeros((self.num_drones, 3)),
        )

    def _generate_xml(self, plugin) -> str:
        """Same per-drone body template as MJXVectorAviary._generate_xml — every
        drone body is now actually force-controlled (see _single_step above),
        so this doesn't need to change beyond looping over num_drones."""
        mass = self.mass
        ixx, iyy, izz = 1.4e-5, 1.4e-5, 2.17e-5
        a = 0.064

        drones = ""
        for d in range(self.num_drones):
            x_offset = d * 0.5
            drones += f"""
    <body name="drone{d}" pos="{x_offset} 0 0.115">
      <freejoint name="drone{d}_joint"/>
      <inertial pos="0 0 0" mass="{mass}" diaginertia="{ixx} {iyy} {izz}"/>
      <geom type="cylinder" size="0.04 0.007" rgba="1 1 1 1" contype="0" conaffinity="0"/>
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

        return f"""<mujoco model="multi_aviary">
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
    <geom name="floor" size="2 2 0.05" type="plane" rgba="0.15 0.30 0.65 1" contype="0" conaffinity="0"/>
{drones}{extra_worldbody}
  </worldbody>
</mujoco>"""
