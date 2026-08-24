"""Exercise 2: quintic-spline waypoint generation and the PID control law."""

import numpy as np
import pytest
from helpers import call

from exercises.ex2 import generate_quintic_spline_waypoints, pid_control


def quintic_time_scaling(s):
    """The standard quintic time-scaling polynomial: 10s³ − 15s⁴ + 6s⁵."""
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def reference_waypoints(start, end, num_points):
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    f_s = quintic_time_scaling(np.linspace(0.0, 1.0, num_points))
    return start + (end - start) * f_s[:, None]


def reference_pid(history, timestep, Kp, Ki, Kd):
    history = np.asarray(history, dtype=float)
    p_term = history[-1]
    i_term = history.sum(axis=0) * timestep
    if len(history) > 1:
        d_term = (history[-1] - history[-2]) / timestep
    else:
        d_term = np.zeros_like(p_term)
    return Kp * p_term + Ki * i_term + Kd * d_term


START = np.array([0.3, 0.25, 0.25])
END = np.array([0.3, 0.18, 0.31])


# --------------------------------------------------------------------------- #
# generate_quintic_spline_waypoints
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("num_points", [2, 5, 20])
def test_waypoints_shape(num_points):
    waypoints = np.asarray(
        call(generate_quintic_spline_waypoints, START, END, num_points)
    )
    assert waypoints.shape == (num_points, 3)


def test_waypoints_hit_both_endpoints():
    """f(0) = 0 and f(1) = 1, so the spline starts at `start` and ends at `end`."""
    waypoints = np.asarray(call(generate_quintic_spline_waypoints, START, END, 5))
    np.testing.assert_allclose(waypoints[0], START, atol=1e-12)
    np.testing.assert_allclose(waypoints[-1], END, atol=1e-12)


def test_waypoints_match_the_quintic_polynomial():
    for num_points in (2, 3, 5, 11):
        waypoints = np.asarray(
            call(generate_quintic_spline_waypoints, START, END, num_points)
        )
        np.testing.assert_allclose(
            waypoints, reference_waypoints(START, END, num_points), atol=1e-12
        )


def test_waypoints_are_not_linear_interpolation():
    """A linear ramp would pass the endpoint test; the time scaling is what matters."""
    waypoints = np.asarray(call(generate_quintic_spline_waypoints, START, END, 5))
    linear = START + (END - START) * np.linspace(0.0, 1.0, 5)[:, None]
    assert not np.allclose(waypoints, linear, atol=1e-6), (
        "waypoints are evenly spaced — looks like linear interpolation, "
        "not quintic time scaling"
    )


def test_waypoints_are_symmetric_about_the_midpoint():
    """f(s) + f(1−s) = 1, so the spline is point-symmetric around the halfway pose."""
    waypoints = np.asarray(call(generate_quintic_spline_waypoints, START, END, 9))
    np.testing.assert_allclose(waypoints[4], (START + END) / 2, atol=1e-12)
    np.testing.assert_allclose(
        waypoints + waypoints[::-1],
        np.broadcast_to(START + END, waypoints.shape),
        atol=1e-12,
    )


def test_waypoints_progress_monotonically_along_the_segment():
    waypoints = np.asarray(call(generate_quintic_spline_waypoints, START, END, 25))
    direction = END - START
    progress = (waypoints - START) @ direction / (direction @ direction)
    assert np.all(np.diff(progress) > 0), "the spline doubles back on itself"
    np.testing.assert_allclose([progress[0], progress[-1]], [0.0, 1.0], atol=1e-12)


def test_waypoints_ease_in_and_out():
    """Zero velocity at both ends means the end steps are far smaller than the middle."""
    waypoints = np.asarray(call(generate_quintic_spline_waypoints, START, END, 21))
    steps = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    assert steps[0] < 0.25 * steps[len(steps) // 2]
    assert steps[-1] < 0.25 * steps[len(steps) // 2]


def test_waypoints_work_for_joint_space_vectors():
    """The same routine is reused on 6-DoF vectors, not just 3D positions."""
    start = np.zeros(6)
    end = np.array([0.4, -1.2, 0.9, 0.3, -0.2, 0.5])
    waypoints = np.asarray(call(generate_quintic_spline_waypoints, start, end, 7))
    assert waypoints.shape == (7, 6)
    np.testing.assert_allclose(
        waypoints, reference_waypoints(start, end, 7), atol=1e-12
    )


# --------------------------------------------------------------------------- #
# pid_control
# --------------------------------------------------------------------------- #

TIMESTEP = 0.002


def test_pid_shape_matches_the_joint_vector():
    history = np.random.default_rng(0).normal(size=(10, 6))
    signal = np.asarray(call(pid_control, history, TIMESTEP))
    assert signal.shape == (6,)


def test_pid_with_a_single_error_has_no_derivative_term():
    """`scripts/pid_control.py` calls this on a one-row history right after a reset."""
    error = np.array([0.1, -0.2, 0.05, 0.0, 0.3, -0.1])
    history = error.reshape(1, -1)
    signal = np.asarray(call(pid_control, history, TIMESTEP, Kp=150.0, Ki=2.0, Kd=0.01))
    expected = 150.0 * error + 2.0 * error * TIMESTEP
    np.testing.assert_allclose(signal, expected, atol=1e-12)


def test_pid_proportional_term_alone():
    history = np.array([[0.1, -0.4], [0.2, -0.3], [0.05, 0.1]])
    signal = np.asarray(call(pid_control, history, TIMESTEP, Kp=3.0, Ki=0.0, Kd=0.0))
    np.testing.assert_allclose(signal, 3.0 * history[-1], atol=1e-12)


def test_pid_integral_term_accumulates_the_whole_history():
    history = np.array([[0.1, -0.4], [0.2, -0.3], [0.05, 0.1]])
    signal = np.asarray(call(pid_control, history, TIMESTEP, Kp=0.0, Ki=5.0, Kd=0.0))
    np.testing.assert_allclose(signal, 5.0 * history.sum(axis=0) * TIMESTEP, atol=1e-12)


def test_pid_derivative_term_uses_the_last_two_errors():
    history = np.array([[0.1, -0.4], [0.2, -0.3], [0.05, 0.1]])
    signal = np.asarray(call(pid_control, history, TIMESTEP, Kp=0.0, Ki=0.0, Kd=7.0))
    expected = 7.0 * (history[-1] - history[-2]) / TIMESTEP
    np.testing.assert_allclose(signal, expected, atol=1e-9)


@pytest.mark.parametrize("length", [1, 2, 5, 10])
def test_pid_matches_the_reference_formula(length):
    history = np.random.default_rng(length).normal(scale=0.1, size=(length, 6))
    signal = np.asarray(call(pid_control, history, TIMESTEP, Kp=150.0, Ki=1.5, Kd=0.01))
    expected = reference_pid(history, TIMESTEP, Kp=150.0, Ki=1.5, Kd=0.01)
    np.testing.assert_allclose(signal, expected, atol=1e-9)


def test_pid_uses_the_documented_default_gains():
    history = np.random.default_rng(7).normal(scale=0.1, size=(4, 6))
    signal = np.asarray(call(pid_control, history, TIMESTEP))
    expected = reference_pid(history, TIMESTEP, Kp=150.0, Ki=0.0, Kd=0.01)
    np.testing.assert_allclose(signal, expected, atol=1e-9)


def test_pid_drives_the_error_towards_zero():
    """Sanity check on the sign: a positive error must ask for a positive torque."""
    history = np.full((3, 6), 0.05)
    signal = np.asarray(call(pid_control, history, TIMESTEP))
    assert np.all(signal > 0)
    signal = np.asarray(call(pid_control, -history, TIMESTEP))
    assert np.all(signal < 0)


def test_pid_zero_error_gives_zero_control():
    history = np.zeros((5, 6))
    signal = np.asarray(call(pid_control, history, TIMESTEP))
    np.testing.assert_allclose(signal, 0.0, atol=1e-12)
