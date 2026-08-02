"""Tests for CPU-side features: wind, obstacles.

Domain randomization and curriculum tests used to live here too but were
removed along with DomainRandomizationWrapper/CurriculumWrapper — that
logic is now handled GPU-side in mjc_dronetests (see
multi_drone_mujoco/wrappers/__init__.py's module docstring)."""

import numpy as np
import pytest

from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.envs.task_aviary import TaskAviary
from multi_drone_mujoco.envs.example_plugins import SimpleHoverPlugin
from multi_drone_mujoco.utils.enums import Physics


def HoverAviary(**kwargs):
    kwargs.setdefault("episode_len_sec", 10.0)
    return TaskAviary(plugin=SimpleHoverPlugin(), **kwargs)
from multi_drone_mujoco.wrappers.wind import WindField, WindConfig, WindModel
from multi_drone_mujoco.wrappers.obstacles import (
    generate_obstacles, obstacles_to_xml, ObstacleConfig, ObstacleType
)

try:
    from multi_drone_mujoco.vectorized import _MJX_AVAILABLE
except ImportError:
    _MJX_AVAILABLE = False


class TestWindModel:
    def test_constant_wind(self):
        config = WindConfig(
            model=WindModel.CONSTANT,
            constant_wind=np.array([2.0, 0, 0]),
        )
        wind = WindField(config)
        wind.reset(seed=0)
        force = wind.get_force(dt=1/240, position=np.array([0, 0, 1.0]),
                               velocity=np.zeros(3))
        assert force[0] > 0  # positive x force from headwind
        assert abs(force[1]) < 1e-10
        assert abs(force[2]) < 1e-10

    def test_dryden_turbulence(self):
        config = WindConfig(model=WindModel.DRYDEN, turbulence_intensity=2.0)
        wind = WindField(config)
        wind.reset(seed=42)
        forces = []
        for _ in range(100):
            f = wind.get_force(1/240, np.array([0, 0, 1.0]), np.zeros(3))
            forces.append(f)
        forces = np.array(forces)
        # Should produce non-zero, varying forces
        assert np.std(forces) > 0
        # Should be bounded (not exploding)
        assert np.max(np.abs(forces)) < 1.0

    def test_gust(self):
        config = WindConfig(model=WindModel.GUST, gust_probability=1.0, gust_intensity=0.01)
        wind = WindField(config)
        wind.reset(seed=0)
        force = wind.get_force(1/240, np.zeros(3), np.zeros(3))
        # With prob=1, first call should trigger a gust
        assert np.linalg.norm(force) > 0

    def test_wind_in_env(self):
        """Test wind integration in BaseAviary."""
        env = BaseAviary(num_drones=1, ctrl_freq=240, sim_freq=240)
        wind_config = WindConfig(
            model=WindModel.CONSTANT,
            constant_wind=np.array([5.0, 0, 0]),
            drag_coefficient=0.01,
        )
        env.set_wind(wind_config)
        env.reset()
        # Step with hover RPM — wind should push drone in +x
        hover = np.full(4, env.HOVER_RPM)
        for _ in range(240):
            env.step(hover)
        # Drone should have moved in x direction due to wind
        assert env.pos[0, 0] > 0.001
        env.close()

    def test_sinusoidal(self):
        config = WindConfig(model=WindModel.SINUSOIDAL, sinusoidal_amplitude=0.01)
        wind = WindField(config)
        wind.reset()
        f1 = wind.get_force(1/240, np.zeros(3), np.zeros(3))
        for _ in range(120):
            wind.get_force(1/240, np.zeros(3), np.zeros(3))
        f2 = wind.get_force(1/240, np.zeros(3), np.zeros(3))
        # Should produce different forces at different times
        assert not np.allclose(f1, f2)


class TestObstacles:
    def test_forest(self):
        config = ObstacleConfig(obstacle_type=ObstacleType.FOREST, num_obstacles=10, seed=42)
        obstacles = generate_obstacles(config)
        assert len(obstacles) > 0
        assert len(obstacles) <= 10

    def test_urban(self):
        config = ObstacleConfig(obstacle_type=ObstacleType.URBAN, num_obstacles=5, seed=0)
        obstacles = generate_obstacles(config)
        assert len(obstacles) > 0

    def test_indoor(self):
        config = ObstacleConfig(obstacle_type=ObstacleType.INDOOR, num_obstacles=10, seed=0)
        obstacles = generate_obstacles(config)
        assert len(obstacles) >= 5  # at least walls

    def test_gates(self):
        config = ObstacleConfig(obstacle_type=ObstacleType.GATES, num_obstacles=4, seed=0)
        obstacles = generate_obstacles(config)
        assert len(obstacles) == 12  # 4 gates * 3 parts each

    def test_xml_generation(self):
        config = ObstacleConfig(obstacle_type=ObstacleType.RANDOM, num_obstacles=3, seed=42)
        obstacles = generate_obstacles(config)
        xml = obstacles_to_xml(obstacles)
        assert "obstacle_0" in xml
        assert 'contype="1"' in xml

    def test_safe_zone(self):
        config = ObstacleConfig(
            obstacle_type=ObstacleType.FOREST,
            num_obstacles=50,
            safe_zone_radius=1.0,
            safe_zone_centers=np.array([[0, 0, 0]]),
            seed=42,
        )
        obstacles = generate_obstacles(config)
        for obs in obstacles:
            dist = np.linalg.norm(obs.position[:2])
            assert dist >= 1.0


class TestMJXVectorized:
    """Tests for GPU-vectorized environment (CPU fallback)."""

    def test_import(self):
        """Module should import without JAX (just check availability flag)."""
        from multi_drone_mujoco.vectorized import _JAX_AVAILABLE, _MJX_AVAILABLE
        # Should not crash on import
        assert isinstance(_JAX_AVAILABLE, bool)
        assert isinstance(_MJX_AVAILABLE, bool)

    @pytest.mark.skipif(not _MJX_AVAILABLE,
                        reason="JAX/MJX not installed")
    def test_instantiation(self):
        from multi_drone_mujoco.vectorized import MJXVectorAviary
        env = MJXVectorAviary(num_envs=4, task="hover")
        assert env.num_envs == 4
