"""Action -> per-motor RPM mixers, shared between the MJX (JAX) vectorized
path and the CPU (numpy) reference path.

These used to be duplicated (once in vectorized/__init__.py, once in
envs/base_aviary.py) and silently drifted apart whenever only one side got
updated — a policy trained under one mapping would have its actions
misinterpreted by the other, producing wildly wrong apparent performance with
no error or warning. Keeping the math in exactly one place removes that whole
class of bug: both callers pass in their own array module (`numpy` for the
CPU path, `jax.numpy` for MJX) since the two implementations otherwise need
no backend-specific behavior — `where`/`array`/`clip` share the same API.
"""

from typing import Any


def mix_rpm_action(xp: Any, action: Any, hover_rpm: float, max_rpm: float) -> Any:
    """Direct per-motor normalized RPM command, each channel in [-1, 1].

    action<=0 -> linear from 0 RPM (at -1) to hover_rpm (at 0).
    action>0  -> linear from hover_rpm (at 0) to max_rpm (at +1).

    Two segments (rather than one line through [-1,1]) because hover_rpm
    isn't the midpoint of [0, max_rpm] — this reaches the true floor/ceiling
    on both sides while keeping action=0 an exact hover.
    """
    return xp.where(
        action <= 0.0,
        (action + 1.0) * hover_rpm,
        hover_rpm + (max_rpm - hover_rpm) * action,
    )


def mix_attitude_rpm(xp: Any, action: Any, hover_rpm: float, max_rpm: float,
                      differential_frac: float = 0.02) -> Any:
    """CTBR-ish attitude mixer: action = [thrust_norm, roll, pitch, yaw_rate],
    each in [-1, 1]. Returns per-motor RPM, shape (4,), clipped to [0, max_rpm].

    Collective reuses mix_rpm_action's two-segment mapping: thrust_norm=0
    reproduces hover_rpm exactly, while each side independently spans its full
    physical range (-1 -> 0 RPM, +1 -> max_rpm). roll/pitch/yaw_rate are
    fixed-scale per-motor RPM differentials (not a closed-loop rate controller —
    there's no feedback from measured angular velocity, so these are direct
    torque-ish commands rather than literal desired body rates).
    """
    thrust_norm, roll, pitch, yaw_rate = action[0], action[1], action[2], action[3]
    collective = mix_rpm_action(xp, thrust_norm, hover_rpm, max_rpm)
    scale = differential_frac * hover_rpm
    rpm = xp.array([
        collective + roll * scale - pitch * scale - yaw_rate * scale,
        collective - roll * scale - pitch * scale + yaw_rate * scale,
        collective - roll * scale + pitch * scale - yaw_rate * scale,
        collective + roll * scale + pitch * scale + yaw_rate * scale,
    ])
    return xp.clip(rpm, 0.0, max_rpm)
