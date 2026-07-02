"""gym-mujoco-drones: Multi-drone environments for MuJoCo.

A gymnasium-compatible multi-drone simulation environment using MuJoCo physics,
inspired by gym-pybullet-drones but with superior performance, accuracy, and features.

Uses the Bitcraze Crazyflie 2.x model from mujoco_menagerie.
"""

__version__ = "1.0.0"

from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.envs.plugins import CPUTaskPlugin
from multi_drone_mujoco.envs.task_aviary import TaskAviary
from multi_drone_mujoco.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
