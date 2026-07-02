"""Wind Gymnasium wrapper that applies wind forces during simulation.

Wraps a BaseAviary and injects wind disturbance forces at each physics step.
"""

import numpy as np
import gymnasium as gym
from typing import Optional

from multi_drone_mujoco.wrappers.wind import WindField, WindConfig, WindModel


class WindWrapper(gym.Wrapper):
    """Gymnasium wrapper that applies wind disturbance to drone environments.

    Adds wind forces to each drone body via xfrc_applied after the base
    environment's physics step. Compatible with any BaseAviary subclass.

    Example
    -------
    >>> from multi_drone_mujoco.envs.task_aviary import TaskAviary
    >>> from multi_drone_mujoco.envs.example_plugins import SimpleHoverPlugin
    >>> from multi_drone_mujoco.wrappers.wind import WindConfig, WindModel
    >>> from multi_drone_mujoco.wrappers.wind_wrapper import WindWrapper
    >>> config = WindConfig(
    ...     model=WindModel.COMBINED,
    ...     constant_wind=np.array([1.0, 0.0, 0.0]),  # 1 m/s headwind
    ...     turbulence_intensity=1.5,
    ...     gust_intensity=0.01,
    ... )
    >>> env = WindWrapper(TaskAviary(plugin=SimpleHoverPlugin()), config)
    >>> obs, _ = env.reset()
    """

    def __init__(self, env: gym.Env, wind_config: Optional[WindConfig] = None):
        super().__init__(env)
        self.wind_config = wind_config or WindConfig(model=WindModel.DRYDEN)
        self.wind_field = WindField(self.wind_config)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        seed = kwargs.get("seed", None)
        self.wind_field.reset(seed=seed)
        info["wind_config"] = {
            "model": self.wind_config.model.value,
            "constant_wind": self.wind_config.constant_wind.tolist(),
            "turbulence_intensity": self.wind_config.turbulence_intensity,
        }
        return obs, info

    def step(self, action):
        """Step environment and apply wind forces."""
        import mujoco

        # Pre-step: We need to inject wind during the physics sub-stepping.
        # Since we can't hook into the inner loop, we apply wind force BEFORE
        # the environment steps (accumulated over the control timestep).
        base_env = self.env
        # Access the unwrapped base aviary
        while hasattr(base_env, 'env') and not hasattr(base_env, 'data'):
            base_env = base_env.env

        if hasattr(base_env, 'data') and hasattr(base_env, 'NUM_DRONES'):
            # Apply wind force to each drone
            for i in range(base_env.NUM_DRONES):
                body_name = f"drone{i}"
                body_id = mujoco.mj_name2id(
                    base_env.model, mujoco.mjtObj.mjOBJ_BODY, body_name
                )
                force = self.wind_field.get_force(
                    dt=base_env.CTRL_TIMESTEP,
                    position=base_env.pos[i],
                    velocity=base_env.vel[i],
                )
                # Add wind force to whatever forces are applied
                # (will be cleared and re-applied by _physics, so we
                # store it and the step will pick it up via xfrc)
                base_env.data.xfrc_applied[body_id, :3] += force

        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info
