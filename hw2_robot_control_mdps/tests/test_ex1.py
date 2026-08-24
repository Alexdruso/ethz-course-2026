"""Exercise 1: Lemniscate keypoints and damped-least-squares inverse kinematics."""

import mujoco
import numpy as np
import pytest
from conftest import ee_pos_for
from helpers import call, implemented

from exercises.ex1 import build_keypoints, get_lemniscate_keypoint, ik_track

SITE = "ee_site"


def reference_lemniscate(t, a):
    """y = a·cos(t)/(1+sin²t),  z = a·cos(t)·sin(t)/(1+sin²t)."""
    sin_t, cos_t = np.sin(t), np.cos(t)
    denom = 1.0 + sin_t**2
    return a * cos_t / denom, a * cos_t * sin_t / denom


# --------------------------------------------------------------------------- #
# get_lemniscate_keypoint
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "t, expected",
    [
        (0.0, (0.2, 0.0)),  # right lobe tip
        (np.pi / 2, (0.0, 0.0)),  # crossing point
        (np.pi, (-0.2, 0.0)),  # left lobe tip
        (3 * np.pi / 2, (0.0, 0.0)),  # crossing point again
    ],
)
def test_lemniscate_landmark_points(t, expected):
    y, z = call(get_lemniscate_keypoint, t, a=0.2)
    np.testing.assert_allclose((y, z), expected, atol=1e-12)


def test_lemniscate_matches_the_closed_form():
    for t in np.linspace(0.0, 2 * np.pi, 37):
        y, z = call(get_lemniscate_keypoint, float(t), a=0.25)
        y_ref, z_ref = reference_lemniscate(t, 0.25)
        np.testing.assert_allclose((y, z), (y_ref, z_ref), atol=1e-12)


def test_lemniscate_is_vectorised():
    """The docstring promises float *or* np.ndarray input."""
    t = np.linspace(0.0, 2 * np.pi, 11)
    y, z = call(get_lemniscate_keypoint, t, a=0.3)
    y, z = np.asarray(y), np.asarray(z)
    assert y.shape == t.shape and z.shape == t.shape
    y_ref, z_ref = reference_lemniscate(t, 0.3)
    np.testing.assert_allclose(y, y_ref, atol=1e-12)
    np.testing.assert_allclose(z, z_ref, atol=1e-12)


def test_lemniscate_scales_linearly_with_a():
    """`a` is a pure scale factor, so doubling it doubles both coordinates."""
    t = 0.7
    small = np.asarray(call(get_lemniscate_keypoint, t, a=0.15))
    large = np.asarray(call(get_lemniscate_keypoint, t, a=0.30))
    np.testing.assert_allclose(large, 2.0 * small, atol=1e-12)


def test_lemniscate_default_scale_is_used():
    """Calling without `a` must fall back to the documented default of 0.2."""
    explicit = np.asarray(call(get_lemniscate_keypoint, 1.1, a=0.2))
    implicit = np.asarray(call(get_lemniscate_keypoint, 1.1))
    np.testing.assert_allclose(implicit, explicit, atol=1e-12)


# --------------------------------------------------------------------------- #
# build_keypoints
# --------------------------------------------------------------------------- #


def test_build_keypoints_shape_and_dtype():
    keypoints = np.asarray(call(build_keypoints))
    assert keypoints.shape == (16, 3), "expected one (x, y, z) row per keypoint"
    assert np.issubdtype(keypoints.dtype, np.floating)


def test_build_keypoints_lies_in_a_single_yz_plane():
    keypoints = np.asarray(call(build_keypoints, count=12, x_offset=0.37))
    np.testing.assert_allclose(keypoints[:, 0], 0.37, atol=1e-12)


def test_build_keypoints_matches_the_sampled_curve():
    """Rows must be [x_offset, y, z + z_offset] over t ∈ [0, 2π) — exclusive."""
    count, width, x_offset, z_offset = 16, 0.25, 0.3, 0.25
    keypoints = np.asarray(
        call(
            build_keypoints,
            count=count,
            width=width,
            x_offset=x_offset,
            z_offset=z_offset,
        )
    )
    t = np.linspace(0.0, 2 * np.pi, count, endpoint=False)
    y_ref, z_ref = reference_lemniscate(t, width)
    expected = np.column_stack([np.full(count, x_offset), y_ref, z_ref + z_offset])
    np.testing.assert_allclose(keypoints, expected, atol=1e-12)


def test_build_keypoints_does_not_duplicate_the_endpoint():
    """t = 2π repeats t = 0; `endpoint=False` (or [:-1]) is what keeps the loop clean."""
    count = 8
    keypoints = np.asarray(call(build_keypoints, count=count))
    assert not np.allclose(keypoints[0], keypoints[-1]), (
        "first and last keypoint coincide — the time grid still includes t = 2π"
    )
    # (The curve does pass through its crossing point twice, so repeated rows are
    # fine in general — it is specifically the wrapped endpoint that is a bug.)
    t_inclusive = np.linspace(0.0, 2 * np.pi, count)
    y_wrong, z_wrong = reference_lemniscate(t_inclusive, 0.25)
    wrong = np.column_stack([np.full(count, 0.3), y_wrong, z_wrong + 0.25])
    assert not np.allclose(keypoints, wrong), (
        "the time grid spans [0, 2π] inclusive; it should be [0, 2π)"
    )


def test_build_keypoints_honours_its_arguments():
    keypoints = np.asarray(
        call(build_keypoints, count=9, width=0.1, x_offset=0.42, z_offset=0.05)
    )
    assert keypoints.shape == (9, 3)
    np.testing.assert_allclose(keypoints[:, 0], 0.42, atol=1e-12)
    # The lobe tips sit at y = ±width, and z stays within ±width/(2√2) of the offset.
    np.testing.assert_allclose(np.abs(keypoints[:, 1]).max(), 0.1, atol=1e-9)
    assert np.abs(keypoints[:, 2] - 0.05).max() <= 0.1 / (2 * np.sqrt(2)) + 1e-9


def test_build_keypoints_first_point_is_the_right_lobe_tip():
    keypoints = np.asarray(
        call(build_keypoints, count=16, width=0.25, x_offset=0.3, z_offset=0.25)
    )
    np.testing.assert_allclose(keypoints[0], [0.3, 0.25, 0.25], atol=1e-12)


# --------------------------------------------------------------------------- #
# ik_track
# --------------------------------------------------------------------------- #

# Targets produced by forward kinematics, so they are reachable by construction.
REACHABLE_QPOS = [
    [0.3, -0.8, 0.9, 0.2, 0.0, 0.5],
    [0.0, -1.57, 1.0, 1.0, 0.0, 0.02239],
    [-0.4, -1.0, 1.4, 0.3, 0.2, 0.4],
]


def test_ik_returns_a_full_joint_vector(model, data):
    target = ee_pos_for(model, REACHABLE_QPOS[0])
    qpos = np.asarray(call(ik_track, model, data, SITE, target))
    assert qpos.shape == (model.nq,)
    assert np.all(np.isfinite(qpos)), "IK diverged to NaN/inf"


@pytest.mark.parametrize("seed_qpos", REACHABLE_QPOS)
def test_ik_reaches_reachable_targets(model, data, seed_qpos):
    target = ee_pos_for(model, seed_qpos)
    qpos = call(ik_track, model, data, SITE, target)
    reached = ee_pos_for(model, qpos)
    error = np.linalg.norm(reached - target)
    assert error < 5e-3, f"IK stopped {error:.4f} m from the target {target}"


def test_ik_leaves_the_input_data_untouched(model, data):
    """The scaffolding restores `qpos` on the way out; the caller relies on it."""
    data.qpos[:] = [0.1, -0.9, 0.8, 0.15, -0.1, 0.3]
    mujoco.mj_forward(model, data)
    before = data.qpos.copy()
    target = ee_pos_for(model, REACHABLE_QPOS[2])
    call(ik_track, model, data, SITE, target)
    np.testing.assert_allclose(data.qpos, before, atol=1e-12)


def test_ik_is_deterministic(model, data):
    target = ee_pos_for(model, REACHABLE_QPOS[0])
    first = np.asarray(call(ik_track, model, data, SITE, target))
    second = np.asarray(call(ik_track, model, data, SITE, target))
    np.testing.assert_allclose(first, second, atol=1e-12)


def test_ik_tracks_the_lemniscate_keypoints(model, data):
    """The end-to-end shape of `scripts/inverse_kinematics.py`, minus the viewer."""
    implemented(get_lemniscate_keypoint, build_keypoints, ik_track)
    keypoints = np.asarray(build_keypoints())
    for keypoint in keypoints[::4]:
        target_qpos = ik_track(model, data, SITE, keypoint)
        data.qpos[:] = target_qpos
        mujoco.mj_forward(model, data)
        error = np.linalg.norm(data.site(SITE).xpos - keypoint)
        assert error < 5e-3, f"keypoint {keypoint} missed by {error:.4f} m"
