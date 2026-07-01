"""Race Aviary: drone racing through gates.

Task: Complete a racing circuit as fast as possible.
"""

import numpy as np
from gymnasium import spaces

from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.utils.enums import DroneModel, Physics, ActionType, ObservationType


class RaceAviary(BaseAviary):
    """Drone racing task — fly through gates as fast as possible."""

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        num_drones: int = 1,
        physics: Physics = Physics.MJC,
        sim_freq: int = 240,
        ctrl_freq: int = 48,
        gui: bool = False,
        record: bool = False,
        gates=None,
        gate_radius: float = 0.2,
        gate_normals=None,
        gate_side: float = 1.0,
        initial_xyzs=None,
        render_mode=None,
        extra_worldbody_xml: str = "",
        act_type: ActionType = ActionType.RPM,
    ):
        self.EPISODE_LEN_SEC = 30
        self.GATE_RADIUS = gate_radius

        if gates is None:
            # Oval racing circuit
            self.GATES = np.array([
                [1.0, 0.0, 1.0],
                [2.0, 1.0, 1.2],
                [1.0, 2.0, 1.5],
                [0.0, 2.0, 1.3],
                [-1.0, 1.0, 1.0],
                [0.0, 0.0, 0.8],
            ])
        else:
            self.GATES = np.array(gates)

        # Gate geometry for plane-crossing detection (matches MJX training).
        self.GATE_NORMALS = np.array(gate_normals, dtype=np.float32) if gate_normals is not None else None
        self._gate_half   = gate_side * 0.5
        self.gates_passed = None
        self._prev_gate_x = None  # sign of (pos - gate_pos) · normal, previous step

        if initial_xyzs is None:
            initial_xyzs = np.array([[0.0, 0.0, 0.5]] * num_drones)
            for i in range(num_drones):
                initial_xyzs[i, 1] = -0.3 * i

        super().__init__(
            drone_model=drone_model,
            num_drones=num_drones,
            physics=physics,
            sim_freq=sim_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs_type=ObservationType.KIN,
            act_type=act_type,
            initial_xyzs=initial_xyzs,
            render_mode=render_mode,
            extra_worldbody_xml=extra_worldbody_xml,
        )

    def reset(self, seed=None, options=None):
        self.gates_passed = np.zeros((self.NUM_DRONES,), dtype=int)
        if self.GATE_NORMALS is not None:
            self._prev_gate_x = np.ones(self.NUM_DRONES, dtype=np.float32)
        return super().reset(seed=seed, options=options)

    def _actionSpace(self):
        return spaces.Box(
            low=-np.ones(4 * self.NUM_DRONES, dtype=np.float32),
            high=np.ones(4 * self.NUM_DRONES, dtype=np.float32),
        )

    def _observationSpace(self):
        # pos(3) + rpy(3) + vel(3) + angvel(3) + next_gate(3) + rel_gate(3) + gate_after(3) = 21
        return spaces.Box(low=-np.inf, high=np.inf, shape=(21 * self.NUM_DRONES,), dtype=np.float32)

    def _computeObs(self):
        obs_list = []
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            gate_idx = self.gates_passed[i] % len(self.GATES)
            next_gate = self.GATES[gate_idx]
            gate_after = self.GATES[(gate_idx + 1) % len(self.GATES)]
            rel_gate = next_gate - self.pos[i]
            obs_list.append(np.hstack([
                state[0:3], state[7:10], state[10:13], state[13:16],
                next_gate, rel_gate, gate_after,
            ]))
        return np.concatenate(obs_list).astype(np.float32)

    def _computeReward(self):
        total = 0.0
        for i in range(self.NUM_DRONES):
            gate_idx = self.gates_passed[i] % len(self.GATES)
            gate = self.GATES[gate_idx]
            rel = self.pos[i] - gate
            dist = np.linalg.norm(rel)

            if self.GATE_NORMALS is not None:
                # Plane-crossing detection matching MJX plugin:
                #   x: sign change of dot(rel, normal) while within gate frame
                #   y (lateral): |rel[1]*nx - rel[0]*ny|  — cross product z-component;
                #      works because gate normals are horizontal (nz≈0)
                #   z: world z distance (gate z-axis == world z-axis)
                normal = self.GATE_NORMALS[gate_idx]
                x_wrt_gate = float(np.dot(rel, normal))
                y_wrt_gate = abs(float(rel[1] * normal[0] - rel[0] * normal[1]))
                z_wrt_gate = abs(float(rel[2]))
                just_passed = (
                    x_wrt_gate < 0.0
                    and self._prev_gate_x[i] > 0.0
                    and y_wrt_gate < self._gate_half
                    and z_wrt_gate < self._gate_half
                )
                # After crossing, reset x so re-entry from behind doesn't re-trigger.
                self._prev_gate_x[i] = 1.0 if just_passed else x_wrt_gate
            else:
                just_passed = dist < self.GATE_RADIUS

            if just_passed:
                self.gates_passed[i] += 1
                total += 20.0
                total += np.linalg.norm(self.vel[i]) * 2.0

            total -= dist * 0.05

            if self.pos[i, 2] < 0.1:
                total -= 1.0

        if self._computeTerminated():
            total -= 100.0

        return float(total)

    def _computeTerminated(self):
        for i in range(self.NUM_DRONES):
            if self.pos[i, 2] < 0.0:
                return True
            '''
            if abs(self.rpy[i, 0]) > 2.0 or abs(self.rpy[i, 1]) > 2.0:  # More lenient for racing
                return True
            '''
        return False

    def _computeTruncated(self):
        # Complete 2 laps or timeout
        all_done = all(gp >= 2 * len(self.GATES) for gp in self.gates_passed)
        return all_done or self.step_counter / self.SIM_FREQ >= self.EPISODE_LEN_SEC

    def _computeInfo(self):
        return {
            "gates_passed": [int(gp) for gp in self.gates_passed],
            "laps_completed": [int(gp // len(self.GATES)) for gp in self.gates_passed],
            "total_gates": len(self.GATES),
        }
