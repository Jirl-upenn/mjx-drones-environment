"""Minimal example CPUTaskPlugin implementations.

These exist so multi_drone_mujoco's own examples/tests can exercise
TaskAviary end-to-end without depending on any external task repo, and to
serve as a reference for what a task plugin from a separate repo (e.g. a
project-specific HoverPlugin/RacePlugin) looks like. They intentionally
replicate the exact default hover/race logic that used to be hardcoded into
the now-removed HoverAviary/RaceAviary classes.
"""

import numpy as np

from multi_drone_mujoco.envs.plugins import CPUTaskPlugin


class SimpleHoverPlugin(CPUTaskPlugin):
    """Single-drone hover at a fixed target height.

    Observation: 12-dim [pos(3), rpy(3), vel(3), ang_vel(3)].
    Reward: dense, penalizes distance to target height/attitude/velocity.
    """

    obs_dim = 12

    def __init__(self, target_height: float = 1.0):
        self.target_height = target_height

    def initial_xyzs(self, num_drones: int):
        return np.array([[0.0, 0.0, 0.1]])

    def reset(self, env, seed=None, options=None) -> None:
        # Seed once; don't clobber external mutation of env.TARGET_HEIGHT
        # between resets (e.g. a wrapper adjusting difficulty externally).
        if not hasattr(env, "TARGET_HEIGHT"):
            env.TARGET_HEIGHT = self.target_height

    def compute_obs(self, env):
        state = env._getDroneStateVector(0)
        obs = np.hstack([state[0:3], state[7:10], state[10:13], state[13:16]])
        return obs.astype(np.float32)

    def compute_reward(self, env) -> float:
        state = env._getDroneStateVector(0)
        pos, vel, rpy = state[0:3], state[10:13], state[7:10]

        height_error = abs(pos[2] - env.TARGET_HEIGHT)
        xy_error = np.linalg.norm(pos[0:2])

        reward = -height_error - 0.1 * xy_error
        reward -= 0.05 * np.linalg.norm(vel)
        reward -= 0.05 * (abs(rpy[0]) + abs(rpy[1]))

        if height_error < 0.05 and xy_error < 0.05:
            reward += 1.0
        if self.compute_terminated(env):
            reward -= 100.0
        return float(reward)

    def compute_terminated(self, env) -> bool:
        # Floor contact — cylinder half-height 0.015m → rests at ~0.015m.
        return bool(env.pos[0, 2] < 0.05)

    def compute_info(self, env) -> dict:
        return {
            "position": env.pos[0].tolist(),
            "height_error": abs(env.pos[0, 2] - env.TARGET_HEIGHT),
        }


class SimpleRacePlugin(CPUTaskPlugin):
    """Fly through a sequence of gates as fast as possible.

    Observation: per drone, 21-dim [pos(3), rpy(3), vel(3), ang_vel(3),
    gate_pos(3), rel_to_gate(3), gate_after(3)].
    Reward: gate-clearing bonus + speed bonus - distance-to-gate penalty.
    """

    def __init__(self, num_drones: int = 1, gates=None, gate_radius: float = 0.2,
                 max_laps: int = 2):
        self.num_drones = num_drones
        self.GATES = np.array(gates) if gates is not None else np.array([
            [1.0, 0.0, 1.0], [2.0, 1.0, 1.2], [1.0, 2.0, 1.5],
            [0.0, 2.0, 1.3], [-1.0, 1.0, 1.0], [0.0, 0.0, 0.8],
        ])
        self.gate_radius = gate_radius
        self.max_laps = max_laps

    @property
    def obs_dim(self) -> int:
        return 21 * self.num_drones

    def initial_xyzs(self, num_drones: int):
        xyzs = np.array([[0.0, 0.0, 0.5]] * num_drones)
        for i in range(num_drones):
            xyzs[i, 1] = -0.3 * i
        return xyzs

    def reset(self, env, seed=None, options=None) -> None:
        env.GATES = self.GATES
        env.gates_passed = np.zeros((env.NUM_DRONES,), dtype=int)

    def compute_obs(self, env):
        obs_list = []
        for i in range(env.NUM_DRONES):
            state = env._getDroneStateVector(i)
            gate_idx = env.gates_passed[i] % len(env.GATES)
            next_gate = env.GATES[gate_idx]
            gate_after = env.GATES[(gate_idx + 1) % len(env.GATES)]
            rel_gate = next_gate - env.pos[i]
            obs_list.append(np.hstack([
                state[0:3], state[7:10], state[10:13], state[13:16],
                next_gate, rel_gate, gate_after,
            ]))
        return np.concatenate(obs_list).astype(np.float32)

    def post_physics_step(self, env) -> None:
        """Gate-crossing detection — must run once per step regardless of
        which reward/obs computation follows, so gates_passed advances even
        when this plugin's reward isn't the one actually being scored."""
        for i in range(env.NUM_DRONES):
            gate_idx = env.gates_passed[i] % len(env.GATES)
            gate = env.GATES[gate_idx]
            dist = np.linalg.norm(env.pos[i] - gate)
            if dist < self.gate_radius:
                env.gates_passed[i] += 1

    def compute_reward(self, env) -> float:
        total = 0.0
        for i in range(env.NUM_DRONES):
            gate_idx = env.gates_passed[i] % len(env.GATES)
            dist = np.linalg.norm(env.pos[i] - env.GATES[gate_idx])
            total -= dist * 0.05
            if env.pos[i, 2] < 0.1:
                total -= 1.0
        if self.compute_terminated(env):
            total -= 100.0
        return float(total)

    def compute_terminated(self, env) -> bool:
        return bool(np.any(env.pos[:env.NUM_DRONES, 2] < 0.0))

    def compute_truncated(self, env) -> bool:
        return bool(all(gp >= self.max_laps * len(self.GATES) for gp in env.gates_passed))

    def compute_info(self, env) -> dict:
        return {
            "gates_passed": [int(gp) for gp in env.gates_passed],
            "laps_completed": [int(gp // len(self.GATES)) for gp in env.gates_passed],
            "total_gates": len(self.GATES),
        }
