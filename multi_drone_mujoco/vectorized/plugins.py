"""Abstract interface for MJXVectorAviary task plugins.

A TaskPlugin encapsulates all task-specific logic so that MJXVectorAviary
remains task-agnostic.  To add a new task: define a NamedTuple for its
per-env state, subclass TaskPlugin, and pass the plugin to MJXVectorAviary.

Every method that takes ``data`` / ``task_state`` arguments operates on a
*single* environment — batching is handled externally via vmap.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple


class TaskPlugin(ABC):
    """Interface that encapsulates all task-specific logic for MJXVectorAviary."""

    @property
    @abstractmethod
    def obs_dim(self) -> int:
        """Dimensionality of observations produced by this task."""

    @abstractmethod
    def init_task_state(self, num_envs: int) -> Any:
        """Return the initial *batched* task state (leading dim = num_envs)."""

    @abstractmethod
    def reset_task_state(self, data: Any, rng: Any, hint: Any = None, old_task_state: Any = None) -> Any:
        """Return the initial task state for a *single* env.  Called inside vmap.

        ``data`` is the post-reset mjx.Data for this env.  ``hint`` carries an
        optional value returned by the custom reset function (e.g. the gate index
        chosen by _PreGate); plugins that don't need it can ignore it.
        ``old_task_state`` is the task state from the previous episode; plugins that
        maintain persistent per-env state (e.g. replay buffers) should copy it here.
        """

    @abstractmethod
    def step(
        self,
        data: Any,
        action: Any,
        task_state: Any,
        motor_rpm: Any = None,
        phys_params: Any = None,
    ) -> Tuple[Any, Any, Any, Any]:
        """Compute one control step for a single env (called inside vmap).

        motor_rpm : the aviary's own actuator state (post-physics-substep,
            (4,) per motor) — None from callers that don't have it (e.g.
            MultiVectorAviary, which doesn't currently thread this through).
            Plugins that don't need it simply ignore the argument.
        phys_params : this env's PhysParams for the episode (kf, km,
            arm_length, max_rpm, hover_rpm, differential_frac, motor_tau) —
            None from callers that don't have it. Ground truth for whatever
            domain_rand_fn sampled this episode; plugins that don't need it
            simply ignore the argument.

        Returns
        -------
        new_task_state : same type as task_state
        obs            : jnp.ndarray (obs_dim,)
        reward         : jnp.float32 scalar
        extra_done     : jnp.bool_ scalar — task-specific termination beyond step count
        """

    @abstractmethod
    def get_obs(
        self, data: Any, task_state: Any, motor_rpm: Any = None, phys_params: Any = None
    ) -> Any:
        """Extract observation for a single env (called inside vmap).

        motor_rpm, phys_params : see step()'s docstring — optional, ignored
            by plugins that don't need them.
        """

    termination_names: list = []

    def termination_checks(self, data, task_state) -> "Any":
        """Return a 1-D bool array of named termination conditions for one env.

        Shape must be (len(termination_names),). Called inside vmap — operates on
        a single env. Default returns empty; override together with termination_names
        to enable per-cause W&B termination logging.
        """
        import jax.numpy as jnp
        return jnp.zeros(0, dtype=jnp.bool_)

    task_metric_names: list = []

    def task_metrics(self, task_state) -> "Any":
        """Return a 1-D array of task-specific metrics for one env at one step.

        Shape must be (len(task_metric_names),). Called inside vmap — operates on
        a single env's task_state, not a batched one. Default returns nothing;
        override together with task_metric_names to enable W&B task metric logging.
        """
        import jax.numpy as jnp
        return jnp.zeros(0)

    def extra_worldbody_xml(self) -> str:
        """Return XML fragment to inject into <worldbody> at model build time.

        Override in task plugins to add task-specific static geometry
        (gates, targets, obstacles, etc.).  The returned string is inserted
        verbatim inside <worldbody> before the drone bodies.
        """
        return ""

    def default_camera_mode(self) -> str:
        """Return the preferred camera mode for eval rendering.

        Override to return a mode string understood by BaseAviary.render()
        (e.g. "overview").  The base default "track" follows the drone.
        """
        return "track"

    def camera_config(self) -> "dict | None":
        """Return camera parameters for the 'overview' render mode, or None.

        Return a dict with keys: lookat ([x,y,z]), distance, azimuth, elevation.
        Used when default_camera_mode() == 'overview'.  None means BaseAviary
        will use its own sensible defaults.
        """
        return None
