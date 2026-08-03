"""Tests for utils/motor_physics.py — pure numpy backend, no JAX/MJX
dependency, so these run everywhere (mirrors how utils/mixer.py's own pure
functions are trivially testable without a real env)."""

import numpy as np
import pytest

from multi_drone_mujoco.utils import motor_physics as mp

_SIM_DT = 1.0 / 240.0
_ARM_LENGTH = 0.0499
_MASS = 0.0445
_GRAVITY = 9.81
_MOTOR_INERTIA = 5e-8


def _hover_rpm():
    max_rpm = mp._OMEGA_REF_RAD_S * 60.0 / (2 * np.pi)
    return mp.solve_hover_rpm(np, 1.0, _MASS, _GRAVITY, max_rpm), max_rpm


class TestCubicCurves:
    def test_pwm_curve_monotonic_near_hover(self):
        """Near the hover operating point the fit should be well-behaved
        (increasing thrust with increasing pwm) — the empirical cubic is
        NOT monotonic everywhere (e.g. near pwm=0), which is expected, not
        tested here."""
        hover_rpm, _ = _hover_rpm()
        pwm_lo = mp.rpm_to_pwm(np, hover_rpm * 0.9)
        pwm_hi = mp.rpm_to_pwm(np, hover_rpm * 1.1)
        assert mp.pwm_to_thrust(np, pwm_hi) > mp.pwm_to_thrust(np, pwm_lo)


class TestSolveHoverRpm:
    def test_hover_equilibrium(self):
        hover_rpm, max_rpm = _hover_rpm()
        assert 0.0 < hover_rpm < max_rpm
        pwm = mp.rpm_to_pwm(np, hover_rpm)
        total_thrust = 4.0 * mp.pwm_to_thrust(np, pwm)
        assert total_thrust == pytest.approx(_MASS * _GRAVITY, rel=1e-4)

    def test_scales_with_kf(self):
        """Halving kf (weaker motors) should require a higher hover RPM."""
        max_rpm = mp._OMEGA_REF_RAD_S * 60.0 / (2 * np.pi)
        rpm_full = mp.solve_hover_rpm(np, 1.0, _MASS, _GRAVITY, max_rpm)
        rpm_half = mp.solve_hover_rpm(np, 0.5, _MASS, _GRAVITY, max_rpm)
        assert rpm_half > rpm_full


class TestMotorOde:
    def test_converges_to_command(self):
        for motor_tau in (0.02, 0.15, 0.5):
            rpm = np.array([0.0, 0.0, 0.0, 0.0])
            cmd = np.array([15000.0] * 4)
            for _ in range(3000):
                rpm, *_ = mp.motor_physics_step(
                    np, rpm, cmd, kf=1.0, km=1.0, arm_length=_ARM_LENGTH,
                    motor_tau=np.float64(motor_tau), motor_inertia=_MOTOR_INERTIA, sim_dt=_SIM_DT,
                )
            assert rpm == pytest.approx(cmd, rel=1e-3)

    def test_zero_tau_is_instant_no_nan(self):
        rpm0 = np.array([1000.0, 2000.0, 3000.0, 4000.0])
        cmd = np.array([15000.0] * 4)
        with np.errstate(divide="ignore"):
            new_rpm, thrust, tx, ty, tz = mp.motor_physics_step(
                np, rpm0, cmd, kf=1.0, km=1.0, arm_length=_ARM_LENGTH,
                motor_tau=np.float64(0.0), motor_inertia=_MOTOR_INERTIA, sim_dt=_SIM_DT,
            )
        assert new_rpm == pytest.approx(cmd)
        for val in (new_rpm, thrust, tx, ty, tz):
            assert not np.any(np.isnan(np.atleast_1d(val)))
            assert not np.any(np.isinf(np.atleast_1d(val)))


class TestReactionTorque:
    def test_zero_at_steady_state(self):
        hover_rpm, _ = _hover_rpm()
        rpm = np.full(4, hover_rpm)
        cmd = np.full(4, hover_rpm)
        _, _, _, _, tau_z = mp.motor_physics_step(
            np, rpm, cmd, kf=1.0, km=1.0, arm_length=_ARM_LENGTH,
            motor_tau=np.float64(0.15), motor_inertia=_MOTOR_INERTIA, sim_dt=_SIM_DT,
        )
        assert tau_z == pytest.approx(0.0, abs=1e-9)

    def test_disabled_at_zero_inertia(self):
        rpm = np.array([0.0, 15000.0, 15000.0, 15000.0])
        cmd = np.array([15000.0] * 4)
        _, _, _, _, tau_z_with = mp.motor_physics_step(
            np, rpm, cmd, kf=1.0, km=1.0, arm_length=_ARM_LENGTH,
            motor_tau=np.float64(0.15), motor_inertia=_MOTOR_INERTIA, sim_dt=_SIM_DT,
        )
        _, _, _, _, tau_z_without = mp.motor_physics_step(
            np, rpm, cmd, kf=1.0, km=1.0, arm_length=_ARM_LENGTH,
            motor_tau=np.float64(0.15), motor_inertia=0.0, sim_dt=_SIM_DT,
        )
        assert tau_z_with != pytest.approx(tau_z_without)
        # isolate just the reaction contribution by re-running at rest (no
        # aero drag asymmetry cancels out identically either way, so the
        # difference above already proves inertia=0 changes tau_z; this
        # second check confirms it's exactly zero, not just "smaller").
        rpm_sym = np.full(4, 15000.0)
        _, _, _, _, tau_z_sym_accel = mp.motor_physics_step(
            np, rpm_sym, np.array([16000.0, 16000.0, 16000.0, 16000.0]),
            kf=1.0, km=1.0, arm_length=_ARM_LENGTH,
            motor_tau=np.float64(0.15), motor_inertia=0.0, sim_dt=_SIM_DT,
        )
        assert tau_z_sym_accel == pytest.approx(0.0, abs=1e-9)

    def test_nonzero_and_signed_on_spinup(self):
        """motor 0 accelerating (from rest, toward the other three's
        already-steady RPM) produces a nonzero reaction torque; document the
        sign explicitly so a future change can't silently flip it."""
        rpm = np.array([0.0, 15000.0, 15000.0, 15000.0])
        cmd = np.array([15000.0] * 4)
        _, _, _, _, tau_z = mp.motor_physics_step(
            np, rpm, cmd, kf=1.0, km=1.0, arm_length=_ARM_LENGTH,
            motor_tau=np.float64(0.15), motor_inertia=_MOTOR_INERTIA, sim_dt=_SIM_DT,
        )
        assert tau_z < 0.0   # motor 0 has a "-" sign in the -m0+m1-m2+m3 pattern


class TestMotorWrench:
    def test_symmetric_case_zero_net_torque(self):
        rpm = np.full(4, 15000.0)
        cmd = np.full(4, 15000.0)
        _, thrust, tx, ty, tz = mp.motor_physics_step(
            np, rpm, cmd, kf=1.0, km=1.0, arm_length=_ARM_LENGTH,
            motor_tau=np.float64(0.15), motor_inertia=_MOTOR_INERTIA, sim_dt=_SIM_DT,
        )
        assert tx == pytest.approx(0.0, abs=1e-9)
        assert ty == pytest.approx(0.0, abs=1e-9)
        assert tz == pytest.approx(0.0, abs=1e-9)
        assert thrust > 0.0

    def test_index_ordering_matches_existing_convention(self):
        """One motor (index 0) thrust-boosted relative to the others should
        push tau_x and tau_y in the same direction the pre-existing
        (f0+f1-f2-f3)/(-f0+f1+f2-f3) formula in both aviaries' _physics_step
        already assumes — guards the port from silently transposing axes."""
        rpm = np.array([16000.0, 15000.0, 15000.0, 15000.0])
        cmd = rpm.copy()
        _, _, tx, ty, tz = mp.motor_physics_step(
            np, rpm, cmd, kf=1.0, km=1.0, arm_length=_ARM_LENGTH,
            motor_tau=np.float64(0.15), motor_inertia=_MOTOR_INERTIA, sim_dt=_SIM_DT,
        )
        assert tx > 0.0   # (f0+f1-f2-f3): f0 boosted -> positive
        assert ty < 0.0   # (-f0+f1+f2-f3): f0 boosted -> negative
