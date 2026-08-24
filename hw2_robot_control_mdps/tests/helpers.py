"""Shared plumbing for the exercise tests.

The exercises ship as stubs, so a test run straight after `make install` would
otherwise be a wall of identical errors. `call` turns "I haven't written this
yet" into a *skip*, which keeps genuine failures (wrong maths, wrong shape)
visible in the report.
"""

import ast
import inspect
import textwrap

import numpy as np
import pytest


def _strip_docstring(body):
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def is_stub(fn):
    """True if `fn` still looks like the unmodified handout scaffolding.

    Three shapes count as unimplemented: an empty body (only a docstring, as in
    `get_obs`), a `raise NotImplementedError`, and the bare `name = ...`
    placeholders that `ik_track` is seeded with.
    """
    try:
        source = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):  # pragma: no cover - builtins have no source
        return False
    try:
        body = _strip_docstring(ast.parse(source).body[0].body)
    except (SyntaxError, AttributeError, IndexError):  # pragma: no cover
        return False

    if all(isinstance(node, ast.Pass) for node in body):
        return True
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.Raise):
            exc = node.exc
            name = getattr(exc, "id", None) or getattr(
                getattr(exc, "func", None), "id", None
            )
            if name == "NotImplementedError":
                return True
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if isinstance(value, ast.Constant) and value.value is Ellipsis:
                return True
    return False


def call(fn, *args, **kwargs):
    """Call an exercise function, skipping the test if it is still unimplemented."""
    where = f"{fn.__module__}.{fn.__name__}"
    if is_stub(fn):
        pytest.skip(f"{where} is still a stub")
    try:
        result = fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"{where} raises NotImplementedError")
    if result is None:
        pytest.skip(f"{where} returned None (no return statement yet?)")
    return result


def implemented(*fns):
    """Skip the current test unless every one of `fns` has been filled in."""
    pending = [f"{fn.__module__}.{fn.__name__}" for fn in fns if is_stub(fn)]
    if pending:
        pytest.skip("depends on unimplemented: " + ", ".join(pending))


def assert_same_rotation(actual_quat, expected_quat, err_msg=""):
    """Compare two wxyz quaternions, tolerating the q / -q double cover."""
    actual = np.asarray(actual_quat, dtype=float)
    expected = np.asarray(expected_quat, dtype=float)
    if np.dot(actual, expected) < 0:
        expected = -expected
    np.testing.assert_allclose(actual, expected, atol=1e-6, err_msg=err_msg)
