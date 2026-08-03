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


class _TrivialTaskPlugin:
    """Minimal MJXVectorAviary/MultiVectorAviary TaskPlugin for tests that
    only care about physics (motor/thrust/torque), not any real task MDP.
    obs = drone position; reward/extra_done are inert. Imports jax lazily
    (inside methods) so this class can be defined even when JAX isn't
    installed — only actually instantiating/using it needs JAX/MJX."""

    obs_dim = 3
    privileged_dim = 0
    termination_names: list = []
    task_metric_names: list = []
    task_episode_metric_names: list = []

    def init_task_state(self, num_envs):
        import jax.numpy as jnp
        return jnp.zeros((num_envs,))

    def reset_task_state(self, data, rng, hint=None, old_task_state=None):
        import jax.numpy as jnp
        return jnp.zeros(())

    def step(self, data, action, task_state, motor_rpm=None, phys_params=None):
        import jax.numpy as jnp
        return task_state, data.qpos[:3], jnp.float32(0.0), jnp.bool_(False)

    def get_obs(self, data, task_state, motor_rpm=None, phys_params=None):
        return data.qpos[:3]

    def extra_worldbody_xml(self) -> str:
        return ""


class _TrivialMultiTaskPlugin:
    """Minimal MultiVectorAviary TaskPlugin — obs/reward carry a per-drone
    leading axis (obs_dim is PER-agent), step()/get_obs() take no
    motor_rpm/phys_params kwargs, per the multi-agent contract described in
    vectorized/multi_aviary.py's module docstring."""

    obs_dim = 3
    privileged_dim = 0
    termination_names: list = []
    task_metric_names: list = []
    task_episode_metric_names: list = []

    def __init__(self, num_drones: int):
        self.num_drones = num_drones

    def init_task_state(self, num_envs):
        import jax.numpy as jnp
        return jnp.zeros((num_envs,))

    def reset_task_state(self, data, rng, hint=None, old_task_state=None):
        import jax.numpy as jnp
        return jnp.zeros(())

    def step(self, data, action, task_state):
        import jax.numpy as jnp
        return task_state, self.get_obs(data, task_state), jnp.zeros((self.num_drones,)), jnp.bool_(False)

    def get_obs(self, data, task_state):
        return data.qpos.reshape(self.num_drones, 7)[:, :3]

    def extra_worldbody_xml(self) -> str:
        return ""


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
        env = MJXVectorAviary(num_envs=4, task="hover", plugin=_TrivialTaskPlugin())
        assert env.num_envs == 4

    @pytest.mark.skipif(not _MJX_AVAILABLE,
                        reason="JAX/MJX not installed")
    def test_hover_holds_altitude(self):
        """Commanding an exact hover (action=0 under action_type='rpm', which
        mix_rpm_action maps to hover_rpm) should keep altitude within a
        small band over several control steps — the first integration-level
        physics-correctness check for the motor-ODE/cubic-curve/
        reaction-torque port (see motor_physics.py)."""
        import jax
        import jax.numpy as jnp
        from multi_drone_mujoco.vectorized import MJXVectorAviary

        env = MJXVectorAviary(num_envs=8, task="hover", plugin=_TrivialTaskPlugin())
        rng = jax.random.PRNGKey(0)
        keys = jax.random.split(rng, env.num_envs)
        obs, state = env.reset(keys)
        z0 = state.mjx_data.qpos[:, 2]

        action = jnp.zeros((env.num_envs, 4))
        for _ in range(50):
            keys = jax.random.split(keys[0], env.num_envs)
            obs, state, reward, done, info = env.step(keys, state, action)

        z_final = state.mjx_data.qpos[:, 2]
        assert jnp.all(jnp.abs(z_final - z0) < 0.1)
        assert not jnp.any(jnp.isnan(z_final))

    @pytest.mark.skipif(not _MJX_AVAILABLE,
                        reason="JAX/MJX not installed")
    def test_zero_motor_tau_no_nan(self):
        """motor_tau=0 (instant actuation) previously risked a NaN reaction
        torque under the reference's continuous-time formula — regression
        guard for the discrete-rate fix in motor_physics.py."""
        import jax
        import jax.numpy as jnp
        from multi_drone_mujoco.vectorized import MJXVectorAviary

        env = MJXVectorAviary(num_envs=4, task="hover", plugin=_TrivialTaskPlugin(), motor_tau=0.0)
        rng = jax.random.PRNGKey(0)
        keys = jax.random.split(rng, env.num_envs)
        obs, state = env.reset(keys)
        action = jnp.zeros((env.num_envs, 4))
        obs, state, reward, done, info = env.step(keys, state, action)
        assert not jnp.any(jnp.isnan(state.mjx_data.qpos))
        assert not jnp.any(jnp.isnan(state.motor_rpm))

    @pytest.mark.skipif(not _MJX_AVAILABLE,
                        reason="JAX/MJX not installed")
    def test_mass_randomization_changes_dynamics(self):
        """Batched-mjx.Model smoke test for ModelParams/_model_for: envs
        given different masses (via mass_inertia_rand_fn) under identical
        commanded thrust should diverge, with the heavier envs rising less
        — not just "different", but different in the physically correct
        direction, confirming the per-lane tree_replace actually reaches
        the physics (a wiring bug here would silently no-op and every env
        would behave identically regardless of its sampled mass)."""
        import jax
        import jax.numpy as jnp
        from multi_drone_mujoco.vectorized import MJXVectorAviary

        def _mass_inertia_rand_fn(rng, nominal_mass, nominal_inertia_diag):
            is_heavy = jax.random.bernoulli(rng)
            mass = jnp.where(is_heavy, nominal_mass * 3.0, nominal_mass)
            return mass, nominal_inertia_diag

        env = MJXVectorAviary(
            num_envs=16, task="hover", plugin=_TrivialTaskPlugin(),
            mass_inertia_rand_fn=_mass_inertia_rand_fn,
        )
        rng = jax.random.PRNGKey(0)
        keys = jax.random.split(rng, env.num_envs)
        obs, state = env.reset(keys)
        mass = state.model_params.body_mass
        assert jnp.unique(mass).shape[0] == 2   # confirms randomization actually landed two distinct values
        z0 = state.mjx_data.qpos[:, 2]

        # Command well above hover for every env (fixed absolute thrust, not
        # each env's own possibly-different hover point) so heavier envs
        # visibly under-accelerate relative to lighter ones.
        action = jnp.full((env.num_envs, 4), 0.5)
        for _ in range(10):
            keys = jax.random.split(keys[0], env.num_envs)
            obs, state, reward, done, info = env.step(keys, state, action)

        rise = state.mjx_data.qpos[:, 2] - z0
        # Threshold against a known midpoint (2x nominal), not the median —
        # an uneven heavy/light split (e.g. 9 heavy vs 7 light out of 16)
        # can put the median exactly ON the majority value, leaving the
        # minority group's mask all-False.
        threshold = env.mass * 2.0
        heavy_rise = jnp.nanmean(jnp.where(mass > threshold, rise, jnp.nan))
        light_rise = jnp.nanmean(jnp.where(mass <= threshold, rise, jnp.nan))
        assert not jnp.isnan(heavy_rise) and not jnp.isnan(light_rise)
        assert heavy_rise < light_rise

    @pytest.mark.skipif(not _MJX_AVAILABLE,
                        reason="JAX/MJX not installed")
    def test_multi_vector_aviary_instantiates_and_steps(self):
        """MultiVectorAviary had zero test coverage before this — a bare
        instantiate/reset/step smoke test, so a future PhysParams field
        change can't silently desync between this class and
        MJXVectorAviary's independent reimplementation without a test
        noticing."""
        import jax
        import jax.numpy as jnp
        from multi_drone_mujoco.vectorized.multi_aviary import MultiVectorAviary

        def _trivial_reset_fn(data, rng, task_state):
            return data

        num_drones = 2
        env = MultiVectorAviary(
            num_envs=4, num_drones=num_drones, task="ma_race",
            plugin=_TrivialMultiTaskPlugin(num_drones), reset_fn=_trivial_reset_fn,
        )
        rng = jax.random.PRNGKey(0)
        keys = jax.random.split(rng, env.num_envs)
        obs, state = env.reset(keys)
        assert state.phys_params.kf.shape == (env.num_envs, num_drones)

        action = jnp.zeros((env.num_envs, num_drones, 4))
        obs, state, reward, done, info = env.step(keys, state, action)
        assert not jnp.any(jnp.isnan(state.mjx_data.qpos))
        assert reward.shape == (env.num_envs, num_drones)
