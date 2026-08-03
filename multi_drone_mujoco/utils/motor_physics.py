"""Per-motor RPM -> force/torque physics, shared between MJXVectorAviary and
MultiVectorAviary (see vectorized/__init__.py and vectorized/multi_aviary.py).

Ported from a reference Crazyflie sim-to-real motor model (first-order
angular-velocity ODE + empirically-fit cubic thrust/torque curves + a
reaction-torque term for the spinning rotor's own angular-momentum reaction
on the airframe), adapted to stay RPM-native throughout rather than
switching this project's action/state interface to PWM — mixer.py, the
hover_rpm/max_rpm mental model, and every existing obs/reward function stay
untouched; PWM only exists as an internal implementation detail of the
thrust/torque curve evaluation inside this module.

Two deliberate deviations from the reference, both because the reference
never actually exercises the cases that expose them:

1. Reaction torque here is computed from the REALIZED discrete RPM rate
   `(new_rpm - old_rpm) / sim_dt`, not the reference's continuous-time
   formula `a*(target - omega)`. The continuous form divides by motor_tau
   (a stand-in for the reference's own rate constant `a`), which is `inf` at
   motor_tau=0 — a real, currently-supported dmcdrones configuration
   ("instant, zero-order-hold actuation") the reference's own fixed nonzero
   time constant never hits. The discrete-rate form reduces cleanly to
   `(rpm_cmd - old_rpm)/sim_dt` there instead (an ordinary finite number),
   and is arguably more physically apt anyway at whatever (possibly coarser
   than the reference's) sim_dt this project runs at.
2. The reference's motor ODE is parameterized in PWM space with an
   independent steady-state-gain constant `b` (`domega/dt = a*(b*pwm -
   omega)`, and the fitted curves happen to use `b`'s own value, 2900 rad/s,
   as their pwm=omega/b normalization constant). Since dmcdrones already
   commands motors in physical RPM (mixer.py's rpm_cmd is already the
   target steady-state RPM, not a [0,1] duty cycle), the equivalent RPM-space
   ODE has an implicit gain of 1 (`d(rpm)/dt = (rpm_cmd - rpm)/motor_tau`) —
   no separate gain constant is needed; motor_tau alone parameterizes it,
   preserving its existing meaning/units end to end.

Sign convention: motors 0 and 2 spin one way, 1 and 3 the other (the
existing X-configuration yaw-drag pattern `tau_z ~ -m0+m1-m2+m3` already
encodes this). A rotor's inertial reaction torque on the airframe from its
own angular ACCELERATION has the identical geometric handedness as its
aerodynamic drag torque from angular VELOCITY (same physical rotor, same
spin axis) — both oppose the sense of what's increasing (velocity for drag,
the rate-of-change for the inertial reaction) — so the reaction-torque term
uses the exact same `-m0+m1-m2+m3` sign pattern as the existing drag/yaw
term, just driven by `motor_inertia * d(rpm)/dt` per motor instead of
`km * thrust_curve(rpm)`. Locked in by test_motor_wrench_index_ordering_
matches_existing_convention / test_reaction_torque_nonzero_and_signed_on_spinup
in tests/test_motor_physics.py — if this convention is ever found to be
backwards relative to a real vehicle, flip the sign there and here together.
"""

from __future__ import annotations

from typing import Any

# Fixed calibration constant of the empirical cubic fits themselves (the
# omega at which the source data's PWM=1.0), NOT a randomizable physical
# quantity — max_rpm (a real, randomizable field) is independent of this.
_OMEGA_REF_RAD_S = 2900.0
_RPM_TO_RAD_S = 3.14159265358979323846 / 30.0   # rpm * this = rad/s (= 2*pi/60)


def rpm_to_pwm(xp: Any, rpm: Any) -> Any:
    """RPM -> the [0, ~1] normalized command the fitted curves expect."""
    return (rpm * _RPM_TO_RAD_S) / _OMEGA_REF_RAD_S


def pwm_to_thrust(xp: Any, pwm: Any) -> Any:
    """Per-motor thrust (Newtons) at kf=1.0 — cubic fit ported verbatim from
    the reference's pwm_to_thrust. Not monotonic near pwm=0 (a property of
    the empirical fit, not a bug) — only meaningful near/above hover."""
    return -0.23009526 * pwm ** 3 + 0.56176458 * pwm ** 2 - 0.0433191 * pwm


def pwm_to_torque(xp: Any, pwm: Any) -> Any:
    """Per-motor reaction (drag) torque magnitude (N*m) at km=1.0 — cubic
    fit ported verbatim from the reference's pwm_to_torques."""
    return -0.0003396 * pwm ** 3 + 0.00087032 * pwm ** 2 + 0.0002896 * pwm


def motor_physics_step(
    xp: Any,
    motor_rpm: Any,
    rpm_cmd: Any,
    kf: Any,
    km: Any,
    arm_length: Any,
    motor_tau: Any,
    motor_inertia: Any,
    sim_dt: float,
) -> tuple:
    """One physics substep's worth of motor-RPM update + resulting body-frame
    thrust/torque. `phys.*` fields must already be broadcastable against
    motor_rpm's leading dims — MJXVectorAviary passes plain scalars,
    MultiVectorAviary passes e.g. `phys.kf[:, None]` to broadcast against a
    (num_drones, 4) motor_rpm, exactly as it already does for the existing
    quadratic law.

    Returns (new_motor_rpm, thrust_z, tau_x, tau_y, tau_z) in the *body*
    frame — callers still do their own `xmat @ [...]` world-frame rotation
    and xfrc_applied scatter; this function has zero knowledge of
    mjx.Data/body indices, matching mixer.py's existing "pure function, no
    backend/env coupling" convention.
    """
    # First-order RPM ODE, exact-exponential discretization (more accurate
    # than a single-step blend at large sim_dt/motor_tau ratios; reduces to
    # motor_rpm := rpm_cmd exactly at motor_tau=0, matching today's
    # zero-order-hold behavior there).
    ad = xp.exp(-sim_dt / motor_tau)
    new_motor_rpm = ad * motor_rpm + (1.0 - ad) * rpm_cmd

    # Reaction torque: realized discrete angular acceleration (see module
    # docstring point 1) * rotor inertia. motor_inertia=0 exactly zeroes
    # this regardless of motor_tau, same "0 = disabled" convention motor_tau
    # itself already uses.
    domega_dt = (new_motor_rpm - motor_rpm) * _RPM_TO_RAD_S / sim_dt
    reaction_torque = motor_inertia * domega_dt   # (..., 4), Nm per motor

    pwm = rpm_to_pwm(xp, new_motor_rpm)
    forces = kf * pwm_to_thrust(xp, pwm)           # (..., 4), N per motor
    drag_torque = km * pwm_to_torque(xp, pwm)      # (..., 4), Nm per motor

    total_thrust = xp.sum(forces, axis=-1)

    s2 = 1.4142135623730951  # sqrt(2)
    f0, f1, f2, f3 = forces[..., 0], forces[..., 1], forces[..., 2], forces[..., 3]
    tau_x = (f0 + f1 - f2 - f3) * arm_length / s2
    tau_y = (-f0 + f1 + f2 - f3) * arm_length / s2

    # Yaw torque = aerodynamic drag + inertial reaction, same alternating
    # sign pattern for both (see module docstring's sign-convention note).
    d0, d1, d2, d3 = drag_torque[..., 0], drag_torque[..., 1], drag_torque[..., 2], drag_torque[..., 3]
    r0, r1, r2, r3 = reaction_torque[..., 0], reaction_torque[..., 1], reaction_torque[..., 2], reaction_torque[..., 3]
    tau_z = (-d0 + d1 - d2 + d3) + (-r0 + r1 - r2 + r3)

    return new_motor_rpm, total_thrust, tau_x, tau_y, tau_z


def solve_hover_rpm(xp: Any, kf: Any, mass: Any, gravity: Any, max_rpm: Any, iters: int = 50) -> Any:
    """Bisection for the per-motor RPM at which 4 motors' combined thrust
    exactly balances weight — ports the reference's own bisection method
    (not a different numerical approach). Works under either backend:
    `xp=jax.numpy` uses `jax.lax.fori_loop` (jit/vmap-traceable); `xp=numpy`
    uses a plain Python loop, for off-device/test usage.

    kf/mass/max_rpm may be per-env/per-drone arrays (any shape); the search
    bounds and target broadcast against that shape.
    """
    target_thrust = mass * gravity / 4.0
    lo = xp.zeros_like(target_thrust)
    hi = xp.broadcast_to(max_rpm, xp.shape(target_thrust))

    def too_low(rpm):
        return kf * pwm_to_thrust(xp, rpm_to_pwm(xp, rpm)) < target_thrust

    if xp.__name__ == "jax.numpy":
        import jax

        def body(_, carry):
            lo, hi = carry
            mid = (lo + hi) / 2.0
            lo = xp.where(too_low(mid), mid, lo)
            hi = xp.where(too_low(mid), hi, mid)
            return (lo, hi)

        lo, hi = jax.lax.fori_loop(0, iters, body, (lo, hi))
    else:
        for _ in range(iters):
            mid = (lo + hi) / 2.0
            lo = xp.where(too_low(mid), mid, lo)
            hi = xp.where(too_low(mid), hi, mid)

    return (lo + hi) / 2.0
