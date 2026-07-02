"""TaskAviary: generic CPU aviary driven entirely by an injected CPUTaskPlugin.

Mirrors MJXVectorAviary's task-agnostic design on the CPU/gymnasium path —
this class carries zero task-specific knowledge. Anything task-specific
(observations, reward, per-episode reset bookkeeping, termination, task
metrics, extra XML geometry) is delegated to the injected ``plugin``
(multi_drone_mujoco.envs.plugins.CPUTaskPlugin). ``reset_fn`` is injected
separately, alongside the plugin, mirroring MJXVectorAviary's own
plugin + reset_fn split.

Only genuinely task-agnostic behavior lives here directly: episode timeout
(``episode_len_sec``) and workspace bounding-box termination (``fly_box``) —
both configured directly on TaskAviary, exactly like MJXVectorAviary's own
``episode_length``/``bounding_box`` constructor params.
"""

from typing import Optional

import numpy as np
from gymnasium import spaces

from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.envs.plugins import CPUTaskPlugin
from multi_drone_mujoco.utils.enums import DroneModel, Physics, ActionType, ObservationType


class TaskAviary(BaseAviary):
    """Generic drone aviary — all task logic lives in ``plugin``."""

    def __init__(
        self,
        plugin: CPUTaskPlugin,
        reset_fn=None,
        drone_model: DroneModel = DroneModel.CF2X,
        num_drones: int = 1,
        physics: Physics = Physics.MJC,
        sim_freq: int = 240,
        ctrl_freq: int = 48,
        gui: bool = False,
        record: bool = False,
        obstacles: bool = False,
        initial_xyzs=None,
        initial_rpys=None,
        render_mode=None,
        act_type: ActionType = ActionType.RPM,
        episode_len_sec: Optional[float] = None,
        fly_box=None,
    ):
        """
        plugin : CPUTaskPlugin
            Encapsulates observations, reward, per-episode reset bookkeeping,
            termination, task metrics, and any extra worldbody XML.
        reset_fn : ResetFunction-like or None
            ``reset_fn.cpu(env, seed=None, options=None) -> (obs, info)``.
            When provided, it is called instead of the plain physics reset;
            it's still expected to call ``env.reset(...)`` itself to trigger
            the underlying physics reset (matching this project's existing
            ResetFunction.cpu() convention) — TaskAviary detects and breaks
            that re-entrant call automatically.
        episode_len_sec : float or None
            Generic episode timeout; None disables it (plugin.compute_truncated
            can still truncate on its own).
        fly_box : (hx, hy, hz) or None
            Generic workspace bounding box; episode terminates if any drone
            exits ±hx, ±hy, or [0, hz].
        """
        self._plugin = plugin
        self._reset_fn = reset_fn
        self._resetting_via_fn = False
        self._episode_len_sec = episode_len_sec
        self._fly_box = tuple(fly_box) if fly_box is not None else None

        if initial_xyzs is None:
            initial_xyzs = plugin.initial_xyzs(num_drones)
        if initial_rpys is None:
            initial_rpys = plugin.initial_rpys(num_drones)

        super().__init__(
            drone_model=drone_model,
            num_drones=num_drones,
            physics=physics,
            sim_freq=sim_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obstacles=obstacles,
            obs_type=ObservationType.KIN,
            act_type=act_type,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            render_mode=render_mode,
            extra_worldbody_xml=plugin.extra_worldbody_xml(),
        )

    ############################################################################

    def _actionSpace(self):
        """Normalized [-1, 1] per motor, 4 per drone — mapped to RPM internally."""
        n = 4 * self.NUM_DRONES
        return spaces.Box(low=-np.ones(n, dtype=np.float32), high=np.ones(n, dtype=np.float32))

    def _observationSpace(self):
        return spaces.Box(
            low=-np.inf * np.ones(self._plugin.obs_dim, dtype=np.float32),
            high=np.inf * np.ones(self._plugin.obs_dim, dtype=np.float32),
        )

    def _preprocessAction(self, action):
        """Normalized [-1, 1] action -> RPM.

        BaseAviary's own ActionType.RPM branch assumes the input is already
        raw RPM in [0, MAX_RPM]; this project's convention (and every task
        plugin's action space) is normalized [-1, 1] instead, so RPM mode is
        remapped through the same mixer MJX training uses. Other action
        types (ATTITUDE, VEL, PID) already expect normalized input in
        BaseAviary's default implementation, so those fall through unchanged.
        """
        if self.ACT_TYPE == ActionType.RPM:
            action = np.clip(np.array(action).reshape(self.NUM_DRONES, 4), -1.0, 1.0)
            return self._normalizedActionToRPM(action)
        return super()._preprocessAction(action)

    ############################################################################

    def reset(self, seed=None, options=None):
        # reset_fn.cpu() implementations call env.reset(...) themselves to
        # trigger the underlying physics reset (this project's established
        # ResetFunction.cpu() convention) — _resetting_via_fn breaks that
        # re-entrant call into the plain physics-reset path below instead of
        # looping back into reset_fn.cpu() again.
        if self._reset_fn is not None and not self._resetting_via_fn:
            self._resetting_via_fn = True
            try:
                return self._reset_fn.cpu(self, seed=seed, options=options)
            finally:
                self._resetting_via_fn = False

        self._plugin.reset(self, seed=seed, options=options)
        return super().reset(seed=seed, options=options)

    def _postPhysicsStep(self):
        self._plugin.post_physics_step(self)

    def _computeObs(self):
        return self._plugin.compute_obs(self)

    def _computeReward(self):
        return float(self._plugin.compute_reward(self))

    def _computeTerminated(self):
        if self._plugin.compute_terminated(self):
            return True
        if self._fly_box is not None:
            hx, hy, hz = self._fly_box
            for i in range(self.NUM_DRONES):
                p = self.pos[i]
                if abs(p[0]) > hx or abs(p[1]) > hy or p[2] > hz or p[2] < 0.0:
                    return True
        return False

    def _computeTruncated(self):
        if self._plugin.compute_truncated(self):
            return True
        if self._episode_len_sec is not None:
            return self.step_counter / self.SIM_FREQ >= self._episode_len_sec
        return False

    def _computeInfo(self):
        return self._plugin.compute_info(self)
