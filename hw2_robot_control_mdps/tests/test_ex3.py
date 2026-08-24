"""Exercise 3: the RL environment building blocks (reset, action, reward, obs).

`compute_reward` is pinned to the reward described in the handout. The bonus
question invites you to redesign it — once you do, deselect those checks with
`make py-test ARGS='-m "not baseline"'`.
"""

import numpy as np
import pytest
from conftest import rotation
from helpers import assert_same_rotation, call, implemented

from exercises.ex3 import (
    compute_reward,
    get_obs,
    process_action,
    reset_robot,
    reset_target_position,
)
from scripts.utils import quat_conjugate, quat_mul, quat_normalize, rot_mat_to_quat

DEFAULT_QPOS = np.array([0.0, -1.57, 1.0, 1.0, 0.0, 0.02239])
TARGET_RANGES = np.array([[0.2, 0.4], [-0.2, 0.2], [0.1, 0.4]])
JNT_RANGE = np.array(
    [
        [-1.92, 1.92],
        [-3.32, 0.174],
        [-0.174, 3.14],
        [-1.66, 1.66],
        [-2.79, 2.79],
        [-0.174, 1.75],
    ]
)


def reference_obs(qpos, ee_pos_w, ee_rot_w, base_pos_w, base_rot_w, target_pos_w):
    ee_pos_base = base_rot_w.T @ (ee_pos_w - base_pos_w)
    target_pos_base = base_rot_w.T @ (target_pos_w - base_pos_w)
    base_quat = rot_mat_to_quat(base_rot_w)
    ee_quat_base = quat_normalize(
        quat_mul(quat_conjugate(base_quat), rot_mat_to_quat(ee_rot_w))
    )
    return np.concatenate([qpos, ee_pos_base, ee_quat_base, target_pos_base])


# --------------------------------------------------------------------------- #
# reset_robot
# --------------------------------------------------------------------------- #


def test_reset_robot_shape():
    qpos = np.asarray(call(reset_robot, DEFAULT_QPOS))
    assert qpos.shape == DEFAULT_QPOS.shape


def test_reset_robot_stays_within_the_noise_band():
    for _ in range(200):
        qpos = np.asarray(call(reset_robot, DEFAULT_QPOS))
        assert np.all(np.abs(qpos - DEFAULT_QPOS) <= 0.5 + 1e-9), (
            "noise must be uniform on (-0.5, 0.5) around the default pose"
        )


def test_reset_robot_actually_adds_noise():
    qpos = np.asarray(call(reset_robot, DEFAULT_QPOS))
    assert not np.allclose(qpos, DEFAULT_QPOS), (
        "the default pose was returned unchanged"
    )


def test_reset_robot_is_stochastic():
    first = np.asarray(call(reset_robot, DEFAULT_QPOS))
    second = np.asarray(call(reset_robot, DEFAULT_QPOS))
    assert not np.allclose(first, second), "every reset returns the same pose"


def test_reset_robot_noise_is_centred_and_covers_the_band():
    samples = np.array([call(reset_robot, DEFAULT_QPOS) for _ in range(2000)])
    noise = samples - DEFAULT_QPOS
    np.testing.assert_allclose(noise.mean(axis=0), 0.0, atol=0.05)
    assert np.all(noise.min(axis=0) < -0.45), "noise never reaches the lower bound"
    assert np.all(noise.max(axis=0) > 0.45), "noise never reaches the upper bound"


def test_reset_robot_does_not_mutate_its_input():
    default = DEFAULT_QPOS.copy()
    call(reset_robot, default)
    np.testing.assert_allclose(default, DEFAULT_QPOS, atol=1e-12)


# --------------------------------------------------------------------------- #
# reset_target_position
# --------------------------------------------------------------------------- #


def test_reset_target_position_shape():
    target = np.asarray(call(reset_target_position, np.zeros(3)))
    assert target.shape == (3,)


def test_reset_target_position_respects_the_sampling_ranges():
    samples = np.array([call(reset_target_position, np.zeros(3)) for _ in range(2000)])
    assert np.all(samples >= TARGET_RANGES[:, 0] - 1e-9), (
        f"min was {samples.min(axis=0)}"
    )
    assert np.all(samples <= TARGET_RANGES[:, 1] + 1e-9), (
        f"max was {samples.max(axis=0)}"
    )


def test_reset_target_position_covers_the_whole_box():
    samples = np.array([call(reset_target_position, np.zeros(3)) for _ in range(2000)])
    span = TARGET_RANGES[:, 1] - TARGET_RANGES[:, 0]
    assert np.all(samples.min(axis=0) < TARGET_RANGES[:, 0] + 0.05 * span)
    assert np.all(samples.max(axis=0) > TARGET_RANGES[:, 1] - 0.05 * span)


def test_reset_target_position_is_stochastic():
    first = np.asarray(call(reset_target_position, np.zeros(3)))
    second = np.asarray(call(reset_target_position, np.zeros(3)))
    assert not np.allclose(first, second)


def test_reset_target_position_is_anchored_to_the_base():
    """The env writes the result straight into `mocap_pos`, so the base offset counts."""
    base = np.array([1.0, -2.0, 0.5])
    samples = np.array([call(reset_target_position, base) for _ in range(500)])
    offsets = samples - base
    assert np.all(offsets >= TARGET_RANGES[:, 0] - 1e-9) and np.all(
        offsets <= TARGET_RANGES[:, 1] + 1e-9
    ), "the sampled box must be centred on `base_pos`, not on the world origin"


# --------------------------------------------------------------------------- #
# process_action
# --------------------------------------------------------------------------- #


def test_process_action_shape():
    target = np.asarray(call(process_action, np.zeros(6), JNT_RANGE))
    assert target.shape == (6,)


def test_process_action_maps_minus_one_to_the_lower_limit():
    target = np.asarray(call(process_action, -np.ones(6), JNT_RANGE))
    np.testing.assert_allclose(target, JNT_RANGE[:, 0], atol=1e-12)


def test_process_action_maps_plus_one_to_the_upper_limit():
    target = np.asarray(call(process_action, np.ones(6), JNT_RANGE))
    np.testing.assert_allclose(target, JNT_RANGE[:, 1], atol=1e-12)


def test_process_action_maps_zero_to_the_midpoint():
    target = np.asarray(call(process_action, np.zeros(6), JNT_RANGE))
    np.testing.assert_allclose(target, JNT_RANGE.mean(axis=1), atol=1e-12)


@pytest.mark.parametrize("value", [-0.75, -0.25, 0.3, 0.9])
def test_process_action_is_linear_in_between(value):
    action = np.full(6, value)
    target = np.asarray(call(process_action, action, JNT_RANGE))
    low, high = JNT_RANGE[:, 0], JNT_RANGE[:, 1]
    expected = low + (high - low) * (value + 1.0) / 2.0
    np.testing.assert_allclose(target, expected, atol=1e-12)


def test_process_action_uses_the_real_model_limits(model):
    action = np.linspace(-1.0, 1.0, model.nu)
    target = np.asarray(call(process_action, action, model.jnt_range))
    low, high = model.jnt_range[:, 0], model.jnt_range[:, 1]
    assert np.all(target >= low - 1e-9) and np.all(target <= high + 1e-9)
    np.testing.assert_allclose(
        target, low + (high - low) * (action + 1.0) / 2.0, atol=1e-9
    )


def test_process_action_does_not_mutate_its_input():
    action = np.full(6, 0.4)
    original = action.copy()
    call(process_action, action, JNT_RANGE)
    np.testing.assert_allclose(action, original, atol=1e-12)


# --------------------------------------------------------------------------- #
# compute_reward
# --------------------------------------------------------------------------- #


@pytest.mark.baseline
@pytest.mark.parametrize("error", [0.0, 0.001, 0.0049, 0.0051, 0.05, 0.2, 1.0])
def test_compute_reward_matches_the_described_function(error):
    expected = np.exp(-2.0 * error) + (1.0 if error < 0.005 else 0.0)
    assert call(compute_reward, error) == pytest.approx(expected, abs=1e-12)


@pytest.mark.baseline
def test_compute_reward_pays_a_sparse_bonus_near_the_target():
    inside = call(compute_reward, 0.004)
    outside = call(compute_reward, 0.006)
    assert inside - outside == pytest.approx(1.0, abs=1e-2), (
        "expected a +1 sparse bonus once the error drops below 5 mm"
    )


def test_compute_reward_decreases_with_the_tracking_error():
    errors = [0.01, 0.05, 0.1, 0.3, 0.6, 1.0]
    rewards = [call(compute_reward, e) for e in errors]
    assert all(a > b for a, b in zip(rewards, rewards[1:])), (
        "a larger tracking error must never be worth more reward"
    )


def test_compute_reward_returns_a_finite_scalar():
    reward = call(compute_reward, 0.123)
    assert np.isscalar(reward) or np.ndim(reward) == 0
    assert np.isfinite(reward)


# --------------------------------------------------------------------------- #
# get_obs
# --------------------------------------------------------------------------- #

QPOS = np.array([0.1, -1.2, 0.8, 0.3, -0.2, 0.4])
EE_POS_W = np.array([0.35, 0.05, 0.22])
EE_ROT_W = rotation([0.0, 1.0, 0.0], 0.6)
BASE_POS_W = np.array([0.0, 0.0, 0.0])
BASE_ROT_W = rotation([0.0, 0.0, 1.0], np.pi / 2)
TARGET_POS_W = np.array([0.3, -0.1, 0.25])


def observe(**overrides):
    kwargs = {
        "qpos": QPOS,
        "ee_pos_w": EE_POS_W,
        "ee_rot_w": EE_ROT_W,
        "base_pos_w": BASE_POS_W,
        "base_rot_w": BASE_ROT_W,
        "target_pos_w": TARGET_POS_W,
    }
    kwargs.update(overrides)
    return np.asarray(call(get_obs, **kwargs))


def test_obs_length_is_qpos_plus_pos_quat_pos():
    obs = observe()
    assert obs.shape == (len(QPOS) + 3 + 4 + 3,)


def test_obs_starts_with_the_joint_positions():
    np.testing.assert_allclose(observe()[: len(QPOS)], QPOS, atol=1e-12)


def test_obs_matches_the_reference_layout():
    expected = reference_obs(
        QPOS, EE_POS_W, EE_ROT_W, BASE_POS_W, BASE_ROT_W, TARGET_POS_W
    )
    obs = observe()
    np.testing.assert_allclose(obs[:9], expected[:9], atol=1e-9)
    assert_same_rotation(obs[9:13], expected[9:13], "end-effector quaternion mismatch")
    np.testing.assert_allclose(obs[13:], expected[13:], atol=1e-9)


def test_obs_is_the_identity_when_the_base_frame_is_the_world_frame():
    obs = observe(base_pos_w=np.zeros(3), base_rot_w=np.eye(3))
    np.testing.assert_allclose(obs[6:9], EE_POS_W, atol=1e-9)
    np.testing.assert_allclose(obs[13:16], TARGET_POS_W, atol=1e-9)
    assert_same_rotation(obs[9:13], rot_mat_to_quat(EE_ROT_W))


def test_obs_positions_are_expressed_in_the_base_frame():
    base_pos = np.array([0.1, -0.2, 0.05])
    base_rot = rotation([0.0, 0.0, 1.0], 0.9)
    obs = observe(base_pos_w=base_pos, base_rot_w=base_rot)
    np.testing.assert_allclose(obs[6:9], base_rot.T @ (EE_POS_W - base_pos), atol=1e-9)
    np.testing.assert_allclose(
        obs[13:16], base_rot.T @ (TARGET_POS_W - base_pos), atol=1e-9
    )


def test_obs_quaternion_is_normalised():
    quat = observe()[9:13]
    assert np.linalg.norm(quat) == pytest.approx(1.0, abs=1e-9)


def test_obs_quaternion_is_the_relative_rotation():
    obs = observe()
    expected = quat_mul(
        quat_conjugate(rot_mat_to_quat(BASE_ROT_W)), rot_mat_to_quat(EE_ROT_W)
    )
    assert_same_rotation(obs[9:13], quat_normalize(expected))


def test_obs_is_invariant_to_moving_the_whole_robot():
    """The whole point of the base frame: relocating the robot must not change the obs."""
    world_rot = rotation([0.3, -0.5, 0.8], 1.1)
    world_pos = np.array([1.5, -0.7, 0.25])
    baseline = observe()
    moved = observe(
        ee_pos_w=world_rot @ EE_POS_W + world_pos,
        ee_rot_w=world_rot @ EE_ROT_W,
        base_pos_w=world_rot @ BASE_POS_W + world_pos,
        base_rot_w=world_rot @ BASE_ROT_W,
        target_pos_w=world_rot @ TARGET_POS_W + world_pos,
    )
    np.testing.assert_allclose(moved[:9], baseline[:9], atol=1e-8)
    assert_same_rotation(moved[9:13], baseline[9:13], "obs is not frame-invariant")
    np.testing.assert_allclose(moved[13:], baseline[13:], atol=1e-8)


# --------------------------------------------------------------------------- #
# End-to-end: the pieces wired together by SO100TrackEnv
# --------------------------------------------------------------------------- #


@pytest.fixture
def env():
    implemented(
        reset_robot, reset_target_position, process_action, compute_reward, get_obs
    )
    from __init__ import XML_PATH

    from env.so100_tracking_env import SO100TrackEnv

    instance = SO100TrackEnv(xml_path=XML_PATH)
    yield instance
    instance.close()


def test_env_reset_returns_an_observation_in_the_declared_space(env):
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(np.asarray(obs, dtype=np.float64))
    assert isinstance(info, dict)


def test_env_step_returns_finite_rewards(env):
    env.reset(seed=0)
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, _truncated, info = env.step(action)
        assert env.observation_space.contains(np.asarray(obs, dtype=np.float64))
        assert np.isfinite(reward)
        assert np.isfinite(info["ee_tracking_error"])
        assert not terminated


def test_env_truncates_at_the_episode_limit(env):
    env.reset(seed=0)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    for step in range(env.max_episode_length):
        *_, truncated, _ = env.step(action)
        assert truncated == (step == env.max_episode_length - 1)


def test_env_targets_stay_inside_the_sampling_box(env):
    for seed in range(20):
        env.reset(seed=seed)
        target = env.data.mocap_pos[0] - env.data.body("Base").xpos
        assert np.all(target >= TARGET_RANGES[:, 0] - 1e-9)
        assert np.all(target <= TARGET_RANGES[:, 1] + 1e-9)
