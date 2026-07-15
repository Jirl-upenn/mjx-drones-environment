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

mix_attitude_pid_rpm below is a separate family (a genuine closed-loop rate
controller, not a fixed-gain differential mixer): it replicates the CTBR
action interface from ~/AgileFlight_MultiAgent's
tasks/race/config/crazyflie/ma_quadcopter_env.py (_pre_physics_step +
_get_moment_from_ctbr + _compute_motor_speeds) as closely as this codebase's
own per-motor force/torque convention allows.
"""

import math
from typing import Any

# AgileFlight_MultiAgent's QuadcopterEnvCfg PID/rate-scale constants
# (ma_quadcopter_env.py lines ~253-266) — reproduced verbatim rather than
# re-tuned, since the point of attitude_pid is to match that reference
# controller's behavior, not to retune it for this model.
_D2R = math.pi / 180.0
BODY_RATE_SCALE_XY = 100.0 * _D2R
BODY_RATE_SCALE_Z = 200.0 * _D2R
KP_OMEGA_RP = 250.0
KI_OMEGA_RP = 500.0
KD_OMEGA_RP = 2.5
I_LIMIT_RP = 33.3
KP_OMEGA_Y = 120.0
KI_OMEGA_Y = 16.70
KD_OMEGA_Y = 0.0
I_LIMIT_Y = 166.7


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


def compute_thrust_command(xp, thrust_norm: Any, mass: float, gravity: float,
                            kf: Any, max_rpm: Any) -> Any:
    """Collective-thrust command in force units (Newtons), for use inside the
    attitude_pid wrench mixer — mirrors mix_rpm_action's two-segment,
    hover-centered shape (thrust_norm=0 is an exact hover, each side
    independently spans its own physical range) but in force rather than RPM
    units, since it needs to combine additively with the rate-PID's commanded
    moment before wrench_to_rpm inverts the pair back to per-motor RPM.

    weight = mass*gravity is exact regardless of domain-randomized kf, since
    hover_rpm is itself defined so that kf*hover_rpm^2*4 == mass*gravity;
    thrust_max scales with kf/max_rpm the same way hover_rpm/max_rpm already
    do elsewhere, so kf-based domain randomization (see PhysParams) flows
    through automatically without a separate TWR mechanism.
    """
    weight = mass * gravity
    thrust_max = 4.0 * kf * max_rpm ** 2
    return xp.where(
        thrust_norm <= 0.0,
        (thrust_norm + 1.0) * weight,
        weight + (thrust_max - weight) * thrust_norm,
    )


def rate_pid_moment(xp, omega_des: Any, omega_meas: Any, pid_integral: Any,
                     prev_omega_meas: Any, dt: float, inertia_diag: Any,
                     kp_rp: Any = KP_OMEGA_RP, ki_rp: Any = KI_OMEGA_RP,
                     kd_rp: Any = KD_OMEGA_RP, kp_y: Any = KP_OMEGA_Y,
                     ki_y: Any = KI_OMEGA_Y, kd_y: Any = KD_OMEGA_Y):
    """Body-rate PID -> commanded body-frame moment, replicating
    AgileFlight_MultiAgent's _get_moment_from_ctbr exactly: proportional +
    anti-windup-clamped integral (separate limits for roll/pitch vs. yaw) +
    derivative-on-measurement (not on error, avoiding derivative kick), then
    moment = inertia @ angular_acceleration.

    omega_meas must be the CURRENT measured body-frame angular velocity (not
    world-frame — see callers for the xmat rotation). pid_integral/
    prev_omega_meas are episode-persistent state (reset to 0 at episode
    reset, like AgileFlight resets _omega_err_integral/_previous_omega_meas
    on env reset) — this function threads them through rather than owning
    them, since the caller (an MJX scan carry) is what actually persists
    state across steps.

    kp_rp/ki_rp/kd_rp/kp_y/ki_y/kd_y default to AgileFlight's own fixed
    values but are ordinary arguments (not baked-in constants) so a
    domain_rand_fn can perturb them per env/drone — see PhysParams'
    kp_omega_rp et al. and envspecs/dynamics.py's *_range fields, mirroring
    AgileFlight's own _init_randomization_ranges. i_limit_rp/i_limit_y and
    the body-rate scales are NOT parameters here: AgileFlight doesn't
    randomize those either (only kp/ki/kd get a range in its
    _init_randomization_ranges), so they stay the fixed I_LIMIT_RP/
    I_LIMIT_Y module constants.

    Returns (moment_body, new_pid_integral, new_prev_omega_meas).
    """
    omega_err = omega_des - omega_meas

    new_integral = pid_integral + omega_err * dt
    limits = xp.array([I_LIMIT_RP, I_LIMIT_RP, I_LIMIT_Y])
    new_integral = xp.clip(new_integral, -limits, limits)

    # First PID update after a reset: prev_omega_meas is still its
    # post-reset zero-init value, which would otherwise produce a spurious
    # derivative kick — treat the derivative as zero instead, exactly like
    # AgileFlight's `torch.where(abs(previous_omega_meas) < 0.0001, ...)`.
    prev_effective = xp.where(xp.abs(prev_omega_meas) < 1e-4, omega_meas, prev_omega_meas)
    omega_meas_dot = (omega_meas - prev_effective) / dt

    kp = xp.array([kp_rp, kp_rp, kp_y])
    ki = xp.array([ki_rp, ki_rp, ki_y])
    kd = xp.array([kd_rp, kd_rp, kd_y])
    omega_dot = kp * omega_err + ki * new_integral - kd * omega_meas_dot

    moment_body = inertia_diag * omega_dot
    return moment_body, new_integral, omega_meas


def wrench_to_rpm(xp, thrust_des: Any, moment_body: Any, kf: Any, km: Any,
                   arm_length: Any, max_rpm: Any) -> Any:
    """Invert this module's own per-motor forward force/torque map (see
    MJXVectorAviary._single_step: T=sum(f), taux=(f0+f1-f2-f3)*L/sqrt2,
    tauy=(-f0+f1+f2-f3)*L/sqrt2, tauz=(-f0+f1-f2+f3)*km/kf) to solve for the
    per-motor forces that realize a desired (thrust, moment) wrench, then
    convert to RPM via f=kf*rpm^2 (sign-preserving, since a moment-heavy
    wrench can call for a negative force from some motors) clipped to
    [0, max_rpm] — a real prop can't spin in reverse, matching AgileFlight's
    own motor_speed_min=0 clamp.

    This is the RPM-domain equivalent of AgileFlight's TM_to_f / k_eta motor
    solve, but built from this module's own rotor-numbering convention
    (rather than copying AgileFlight's arbitrary one) so it stays
    self-consistent with the forward map the surrounding physics step
    already applies — the two must agree for a commanded wrench to actually
    be realized when forces are recombined via arm geometry.
    """
    s2 = xp.sqrt(2.0)
    a = moment_body[0] * s2 / arm_length
    b = moment_body[1] * s2 / arm_length
    c = moment_body[2] * kf / km

    forces = xp.array([
        (thrust_des + a - b - c) / 4.0,
        (thrust_des + a + b + c) / 4.0,
        (thrust_des - a + b - c) / 4.0,
        (thrust_des - a - b + c) / 4.0,
    ])
    rpm = xp.sign(forces) * xp.sqrt(xp.abs(forces) / kf)
    return xp.clip(rpm, 0.0, max_rpm)


def mix_attitude_pid_rpm(xp, action: Any, omega_meas: Any, pid_integral: Any,
                          prev_omega_meas: Any, mass: float, gravity: float,
                          kf: Any, km: Any, arm_length: Any, max_rpm: Any,
                          inertia_diag: Any, dt: float,
                          kp_omega_rp: Any = KP_OMEGA_RP, ki_omega_rp: Any = KI_OMEGA_RP,
                          kd_omega_rp: Any = KD_OMEGA_RP, kp_omega_y: Any = KP_OMEGA_Y,
                          ki_omega_y: Any = KI_OMEGA_Y, kd_omega_y: Any = KD_OMEGA_Y):
    """Full CTBR + inner rate-PID pipeline: action = [thrust_norm,
    roll_rate_cmd, pitch_rate_cmd, yaw_rate_cmd], each in [-1, 1] — same
    slot convention as mix_attitude_rpm, but roll/pitch/yaw_rate are genuine
    closed-loop body-rate commands (scaled by BODY_RATE_SCALE_XY/Z, tracked
    by rate_pid_moment) rather than fixed-gain per-motor RPM differentials.

    Must be called once per PHYSICS substep (dt = 1/sim_freq), not once per
    control step: AgileFlight's own rate-PID loop runs at the physics rate
    (pid_loop_rate_hz == sim_rate_hz == 500 there), since a real rate
    controller reacts to the angular velocity as it evolves continuously
    between control steps, not once at the control boundary. omega_meas must
    already be in the body frame (see callers for the xmat rotation out of
    MJX's world-frame qvel).

    kp_omega_rp/ki_omega_rp/kd_omega_rp/kp_omega_y/ki_omega_y/kd_omega_y
    default to AgileFlight's fixed gains but are ordinary (potentially
    per-env/per-drone, domain-randomized) arguments — see PhysParams and
    rate_pid_moment's docstring.

    Returns (rpm_cmd, new_pid_integral, new_prev_omega_meas).
    """
    thrust_des = compute_thrust_command(xp, action[0], mass, gravity, kf, max_rpm)
    omega_des = xp.array([
        BODY_RATE_SCALE_XY * action[1],
        BODY_RATE_SCALE_XY * action[2],
        BODY_RATE_SCALE_Z * action[3],
    ])
    moment_body, new_integral, new_prev_omega = rate_pid_moment(
        xp, omega_des, omega_meas, pid_integral, prev_omega_meas, dt, inertia_diag,
        kp_rp=kp_omega_rp, ki_rp=ki_omega_rp, kd_rp=kd_omega_rp,
        kp_y=kp_omega_y, ki_y=ki_omega_y, kd_y=kd_omega_y,
    )
    rpm_cmd = wrench_to_rpm(xp, thrust_des, moment_body, kf, km, arm_length, max_rpm)
    return rpm_cmd, new_integral, new_prev_omega
