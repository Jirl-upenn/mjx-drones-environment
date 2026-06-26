"""Hover Aviary: single-drone hover task for RL training.

Task: hover at z=1.0 and remain stable.
Observation: 12-dim [pos(3), rpy(3), vel(3), ang_vel(3)]  (normalized)
Action: 4-dim normalized RPMs [-1, 1]
"""

import numpy as np
from gymnasium import spaces

from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.utils.enums import DroneModel, Physics, ActionType, ObservationType


class HoverAviary(BaseAviary):
    """Single drone hover task — matches gym-pybullet-drones HoverAviary."""

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        physics: Physics = Physics.MJC,
        sim_freq: int = 240,
        ctrl_freq: int = 48,
        gui: bool = False,
        record: bool = False,
        obstacles: bool = False,
        target_height: float = 1.0,
        episode_len_sec: float = 10.0,
        initial_xyzs=None,
        render_mode=None,
        act_type: ActionType = ActionType.RPM,
        fly_box=None,
    ):
        self.TARGET_HEIGHT = target_height
        self.EPISODE_LEN_SEC = episode_len_sec
        self._fly_box = tuple(fly_box) if fly_box is not None else None
        if initial_xyzs is None:
            initial_xyzs = np.array([[0.0, 0.0, 0.1]])

        super().__init__(
            drone_model=drone_model,
            num_drones=1,
            physics=physics,
            sim_freq=sim_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obstacles=obstacles,
            obs_type=ObservationType.KIN,
            act_type=act_type,
            initial_xyzs=initial_xyzs,
            render_mode=render_mode,
        )

    def _actionSpace(self):
        """Normalized [-1, 1] → mapped to RPM internally."""
        return spaces.Box(low=-np.ones(4, dtype=np.float32), high=np.ones(4, dtype=np.float32))

    def _observationSpace(self):
        """12-dim observation: pos, rpy, vel, ang_vel."""
        return spaces.Box(
            low=-np.inf * np.ones(12, dtype=np.float32),
            high=np.inf * np.ones(12, dtype=np.float32),
        )

    def _preprocessAction(self, action):
        """Convert normalized action to RPMs."""
        action = np.clip(np.array(action).flatten(), -1, 1)
        rpms = self._normalizedActionToRPM(action).reshape(1, 4)
        return rpms

    def _computeObs(self):
        """12-dim observation."""
        state = self._getDroneStateVector(0)
        # pos(3), rpy(3), vel(3), ang_vel(3)
        obs = np.hstack([state[0:3], state[7:10], state[10:13], state[13:16]])
        return obs.astype(np.float32)

    def _computeReward(self):
        """Dense reward: penalize distance to target height and attitude."""
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        vel = state[10:13]
        rpy = state[7:10]

        # Reward for being at target height
        height_error = abs(pos[2] - self.TARGET_HEIGHT)
        xy_error = np.linalg.norm(pos[0:2])

        reward = -height_error - 0.1 * xy_error
        reward -= 0.05 * np.linalg.norm(vel)
        reward -= 0.05 * (abs(rpy[0]) + abs(rpy[1]))

        # Bonus for being close
        if height_error < 0.05 and xy_error < 0.05:
            reward += 1.0

        if self._computeTerminated():
            reward -= 100.0

        return float(reward)

    def _computeTerminated(self):
        """Matches MJXAviaryPureJaxAdapter termination conditions exactly."""
        pos = self._getDroneStateVector(0)[0:3]
        if pos[2] < 0.05:  # floor contact (cylinder half-height 0.015m → rests at ~0.015m)
            return True
        if self._fly_box is not None:
            hx, hy, hz = self._fly_box
            if abs(pos[0]) > hx or abs(pos[1]) > hy or pos[2] > hz:
                return True
        return False

    def _computeTruncated(self):
        """Truncate after episode time limit."""
        return self.step_counter / self.SIM_FREQ >= self.EPISODE_LEN_SEC

    def _computeInfo(self):
        return {
            "position": self.pos[0].tolist(),
            "height_error": abs(self.pos[0, 2] - self.TARGET_HEIGHT),
        }
