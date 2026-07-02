"""Abstract interface for TaskAviary task plugins (CPU/NumPy path).

CPU-side counterpart to ``vectorized.plugins.TaskPlugin``. Where the MJX
plugin is a pure/vmapped set of functions threaded through an explicit
``task_state`` pytree, this operates on a single, stateful ``env``
(a TaskAviary instance) directly — matching BaseAviary's existing
``_compute*`` template-method hooks. Task-specific per-episode state (e.g.
gate-crossing counters) is owned by the plugin instance itself or stashed as
attributes on ``env``, whichever the concrete plugin finds convenient.

To add a new CPU task: subclass CPUTaskPlugin and pass the instance to
TaskAviary(plugin=...). TaskAviary itself has zero task-specific knowledge —
episode timeout and workspace bounding-box termination are the only generic
behaviors it handles itself (config'd directly on TaskAviary, mirroring
MJXVectorAviary's episode_length/bounding_box).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class CPUTaskPlugin(ABC):
    """Interface that encapsulates all task-specific logic for TaskAviary."""

    @property
    @abstractmethod
    def obs_dim(self) -> int:
        """Dimensionality of observations produced by this task."""

    @abstractmethod
    def compute_obs(self, env) -> Any:
        """Return the current observation. Called once per reset/step."""

    @abstractmethod
    def compute_reward(self, env) -> float:
        """Return the current step's reward. Called once per step."""

    def reset(self, env, seed=None, options=None) -> None:
        """Called once per episode, after the physics reset, before the
        first compute_obs()/compute_reward() of the episode.

        Override to (re)initialize any per-episode task state your
        compute_obs/compute_reward/compute_terminated methods need (e.g.
        gate-crossing counters). Default: no-op.
        """
        return None

    def post_physics_step(self, env) -> None:
        """Called once per step, immediately after physics/kinematics update
        but before compute_obs()/compute_reward().

        Override for task bookkeeping that must run exactly once per step
        regardless of which reward/obs function is active (e.g. detecting a
        gate crossing and advancing the target-gate index). Default: no-op.
        """
        return None

    def compute_terminated(self, env) -> bool:
        """Task-specific early-termination check (e.g. floor contact).

        TaskAviary ORs this with its own generic fly-box check when
        ``fly_box`` is configured — no need to duplicate that here.
        """
        return False

    def compute_truncated(self, env) -> bool:
        """Task-specific truncation check beyond the episode timeout (e.g.
        race's "all gates cleared"). TaskAviary ORs this with its own
        generic episode_len_sec timeout — no need to duplicate that here.
        """
        return False

    def compute_info(self, env) -> dict:
        return {}

    def initial_xyzs(self, num_drones: int):
        """Initial spawn positions, shape (num_drones, 3), or None to use
        BaseAviary's default grid layout."""
        return None

    def initial_rpys(self, num_drones: int):
        """Initial spawn orientations, shape (num_drones, 3), or None to use
        BaseAviary's default (all zero)."""
        return None

    def extra_worldbody_xml(self) -> str:
        """Return XML fragment to inject into <worldbody> at model build time.

        Override to add task-specific static geometry (gates, targets,
        obstacles, etc.). Inserted verbatim inside <worldbody> before the
        drone bodies.
        """
        return ""
