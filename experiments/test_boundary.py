import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from boundary.apply import apply_boundary
from boundary.conditions import (
    PERIODIC,
    DIRICHLET,
    NEUMANN,
    OUTFLOW,
)


# ============================================================
# helpers
# ============================================================

def assert_close(a, b, name, atol=1e-12):

    a = np.asarray(a)
    b = np.asarray(b)

    err = np.max(np.abs(a - b))

    print(f"{name}: error = {err:.3e}")

    if err > atol:
        raise AssertionError(
            f"{name} FAILED: error={err:.3e}"
        )


def show(name, result, ng):

    print(f"\n{name}")
    print("-" * 60)
    print("full:")
    print(np.asarray(result))

    print("physical:")
    print(np.asarray(result[ng:-ng]))


# ============================================================
# TEST 1
# Dirichlet / Dirichlet
# ============================================================

def test_dirichlet():

    print("\n" + "=" * 70)
    print("TEST 1: DIRICHLET / DIRICHLET")
    print("=" * 70)

    ng = 3
    n = 8
    dx = 1.0

    u = jnp.linspace(0.1, 0.8, n)

    left_value = 1.25
    right_value = -2.5

    result = apply_boundary(
        u,
        axis=0,
        left_type=DIRICHLET,
        right_type=DIRICHLET,
        left_value=left_value,
        right_value=right_value,
        ng=ng,
        dx=dx,
    )

    show("Dirichlet", result, ng)

    # --------------------------------------------------------
    # left:
    #
    # ghost = 2*g - reflected physical value
    #
    # u = [u0, u1, u2, ...]
    #
    # ghost = [
    #   2g-u2,
    #   2g-u1,
    #   2g-u0
    # ]
    # --------------------------------------------------------

    expected_left = (
        2.0 * left_value
        - jnp.flip(u[:ng])
    )

    expected_right = (
        2.0 * right_value
        - u[-ng:]
    )

    assert_close(
        result[:ng],
        expected_left,
        "left Dirichlet",
    )

    assert_close(
        result[-ng:],
        expected_right,
        "right Dirichlet",
    )

    assert_close(
        result[ng:-ng],
        u,
        "physical region",
    )


# ============================================================
# TEST 2
# Outflow / Outflow
# ============================================================

def test_outflow():

    print("\n" + "=" * 70)
    print("TEST 2: OUTFLOW / OUTFLOW")
    print("=" * 70)

    ng = 3
    n = 8

    u = jnp.linspace(0.1, 0.8, n)

    result = apply_boundary(
        u,
        axis=0,
        left_type=OUTFLOW,
        right_type=OUTFLOW,
        ng=ng,
    )

    show("Outflow", result, ng)

    expected_left = jnp.full(
        ng,
        u[0],
    )

    expected_right = jnp.full(
        ng,
        u[-1],
    )

    assert_close(
        result[:ng],
        expected_left,
        "left outflow",
    )

    assert_close(
        result[-ng:],
        expected_right,
        "right outflow",
    )

    assert_close(
        result[ng:-ng],
        u,
        "physical region",
    )


# ============================================================
# TEST 3
# Dirichlet / Outflow
# ============================================================

def test_dirichlet_outflow():

    print("\n" + "=" * 70)
    print("TEST 3: DIRICHLET / OUTFLOW")
    print("=" * 70)

    ng = 3
    n = 8

    u = jnp.linspace(0.1, 0.8, n)

    left_value = 5.0

    result = apply_boundary(
        u,
        axis=0,
        left_type=DIRICHLET,
        right_type=OUTFLOW,
        left_value=left_value,
        ng=ng,
    )

    show("Dirichlet / Outflow", result, ng)

    expected_left = (
        2.0 * left_value
        - jnp.flip(u[:ng])
    )

    expected_right = jnp.full(
        ng,
        u[-1],
    )

    assert_close(
        result[:ng],
        expected_left,
        "left Dirichlet",
    )

    assert_close(
        result[-ng:],
        expected_right,
        "right outflow",
    )

    assert_close(
        result[ng:-ng],
        u,
        "physical region",
    )


# ============================================================
# TEST 4
# Outflow / Dirichlet
# ============================================================

def test_outflow_dirichlet():

    print("\n" + "=" * 70)
    print("TEST 4: OUTFLOW / DIRICHLET")
    print("=" * 70)

    ng = 3
    n = 8

    u = jnp.linspace(0.1, 0.8, n)

    right_value = -4.0

    result = apply_boundary(
        u,
        axis=0,
        left_type=OUTFLOW,
        right_type=DIRICHLET,
        right_value=right_value,
        ng=ng,
    )

    show("Outflow / Dirichlet", result, ng)

    expected_left = jnp.full(
        ng,
        u[0],
    )

    expected_right = (
        2.0 * right_value
        - u[-ng:]
    )

    assert_close(
        result[:ng],
        expected_left,
        "left outflow",
    )

    assert_close(
        result[-ng:],
        expected_right,
        "right Dirichlet",
    )

    assert_close(
        result[ng:-ng],
        u,
        "physical region",
    )


# ============================================================
# TEST 5
# Neumann / Neumann
# ============================================================

def test_neumann():

    print("\n" + "=" * 70)
    print("TEST 5: NEUMANN / NEUMANN")
    print("=" * 70)

    ng = 3
    n = 8
    dx = 0.1

    u = jnp.linspace(0.1, 0.8, n)

    left_derivative = 2.0
    right_derivative = -1.5

    result = apply_boundary(
        u,
        axis=0,
        left_type=NEUMANN,
        right_type=NEUMANN,
        left_value=left_derivative,
        right_value=right_derivative,
        ng=ng,
        dx=dx,
    )

    show("Neumann", result, ng)

    # left:
    #
    # u_ghost(k) = u_0 - k*dx*g
    #
    k = jnp.arange(
        1,
        ng + 1,
        dtype=u.dtype,
    )

    expected_left = (
        u[0]
        - k * dx * left_derivative
    )

    # right:
    #
    # u_ghost(k) = u_N + k*dx*g
    #

    expected_right = (
        u[-1]
        + k * dx * right_derivative
    )

    assert_close(
        result[:ng],
        expected_left,
        "left Neumann",
    )

    assert_close(
        result[-ng:],
        expected_right,
        "right Neumann",
    )

    assert_close(
        result[ng:-ng],
        u,
        "physical region",
    )


# ============================================================
# TEST 6
# Periodic
# ============================================================

def test_periodic():

    print("\n" + "=" * 70)
    print("TEST 6: PERIODIC")
    print("=" * 70)

    ng = 3
    n = 8

    u = jnp.linspace(0.1, 0.8, n)

    result = apply_boundary(
        u,
        axis=0,
        left_type=PERIODIC,
        right_type=PERIODIC,
        ng=ng,
    )

    show("Periodic", result, ng)

    expected_left = u[-ng:]
    expected_right = u[:ng]

    assert_close(
        result[:ng],
        expected_left,
        "left periodic",
    )

    assert_close(
        result[-ng:],
        expected_right,
        "right periodic",
    )

    assert_close(
        result[ng:-ng],
        u,
        "physical region",
    )


# ============================================================
# TEST 7
# Different ng
# ============================================================

def test_different_ng():

    print("\n" + "=" * 70)
    print("TEST 7: DIFFERENT NG")
    print("=" * 70)

    n = 10

    u = jnp.linspace(0.1, 1.0, n)

    for ng in (1, 2, 3, 4):

        print(f"\nng = {ng}")

        result = apply_boundary(
            u,
            axis=0,
            left_type=OUTFLOW,
            right_type=OUTFLOW,
            ng=ng,
        )

        assert_close(
            result[:ng],
            jnp.full(ng, u[0]),
            f"left outflow ng={ng}",
        )

        assert_close(
            result[-ng:],
            jnp.full(ng, u[-1]),
            f"right outflow ng={ng}",
        )

        assert_close(
            result[ng:-ng],
            u,
            f"physical region ng={ng}",
        )


# ============================================================
# TEST 8
# Periodic mismatch must fail
# ============================================================

def test_periodic_mismatch():

    print("\n" + "=" * 70)
    print("TEST 8: PERIODIC MISMATCH")
    print("=" * 70)

    u = jnp.linspace(0.1, 0.8, 8)

    try:

        apply_boundary(
            u,
            axis=0,
            left_type=PERIODIC,
            right_type=OUTFLOW,
            ng=3,
        )

    except ValueError as e:

        print("Expected ValueError:")
        print(e)

        return

    raise AssertionError(
        "Periodic mismatch did not raise ValueError"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    tests = [
        test_dirichlet,
        test_outflow,
        test_dirichlet_outflow,
        test_outflow_dirichlet,
        test_neumann,
        test_periodic,
        test_different_ng,
        test_periodic_mismatch,
    ]

    passed = 0

    for test in tests:

        try:
            test()
            print("PASS")
            passed += 1

        except Exception as e:
            print("FAIL")
            print(e)

    print("\n" + "=" * 70)
    print(
        f"RESULT: {passed}/{len(tests)} tests passed"
    )
    print("=" * 70)