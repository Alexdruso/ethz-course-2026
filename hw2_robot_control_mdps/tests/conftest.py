"""Path setup and shared fixtures for the exercise tests.

The handout scripts are run from inside `scripts/`, so they can do
`import __init__` alongside `from exercises... import ...`. pytest collects from
the project root instead, so both directories have to go on `sys.path` here.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
for _path in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import mujoco
from __init__ import TORQUE_CTRL_XML_PATH, XML_PATH


@pytest.fixture(scope="session")
def model():
    """The position-controlled SO-100 model used by ex1 and ex3."""
    return mujoco.MjModel.from_xml_path(str(XML_PATH))


@pytest.fixture(scope="session")
def torque_model():
    """The torque-controlled SO-100 model used by the ex2 PID pipeline."""
    return mujoco.MjModel.from_xml_path(str(TORQUE_CTRL_XML_PATH))


@pytest.fixture
def data(model):
    """A fresh MjData, forwarded once so the site/body poses are populated."""
    mj_data = mujoco.MjData(model)
    mujoco.mj_forward(model, mj_data)
    return mj_data


@pytest.fixture
def rng():
    """Deterministic RNG so failures are reproducible."""
    return np.random.default_rng(20260824)


@pytest.fixture(autouse=True)
def _seed_global_numpy():
    """ex3 samples via `np.random.*`; pin the global seed for reproducibility."""
    np.random.seed(0)


def rotation(axis, angle):
    """Build a 3x3 rotation matrix from an axis/angle, via MuJoCo's quaternions."""
    quat = np.zeros(4)
    mat = np.zeros(9)
    # mju_axisAngle2Quat assumes a unit axis; a non-unit one yields a non-unit
    # quaternion and therefore a non-orthogonal matrix.
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    mujoco.mju_axisAngle2Quat(quat, axis, float(angle))
    mujoco.mju_quat2Mat(mat, quat)
    return mat.reshape(3, 3)


def ee_pos_for(model, qpos):
    """Forward-kinematics helper: where does `ee_site` land for this `qpos`?"""
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = qpos
    mujoco.mj_forward(model, scratch)
    return scratch.site("ee_site").xpos.copy()
