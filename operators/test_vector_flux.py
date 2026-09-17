import jax
import jax.numpy as jnp

from operators.flux import burgers_rusanov_flux, vector_burgers_rusanov_flux

jax.config.update(
    "jax_enable_x64",
    True,
)



# ==========================================================
# Helpers
# ==========================================================

def check(name, actual, expected, tol=1e-14):

    error = jnp.max(
        jnp.abs(actual - expected)
    )

    passed = bool(error < tol)

    status = "PASS" if passed else "FAIL"

    print(
        f"{status:<6}"
        f"{name:<40}"
        f"error = {float(error):.3e}"
    )

    return passed


# ==========================================================
# TEST 1
# Physical vector flux
#
# U = (u, v)
#
# F_x = (u^2, uv)
# F_y = (uv, v^2)
# ==========================================================

def test_physical_flux():

    U = jnp.array(
        [2.0, 3.0]
    )

    Fx = vector_burgers_rusanov_flux(
        U,
        U,
        component=0,
    )

    Fy = vector_burgers_rusanov_flux(
        U,
        U,
        component=1,
    )

    expected_Fx = jnp.array(
        [4.0, 6.0]
    )

    expected_Fy = jnp.array(
        [6.0, 9.0]
    )

    ok_x = check(
        "Physical flux F_x",
        Fx,
        expected_Fx,
    )

    ok_y = check(
        "Physical flux F_y",
        Fy,
        expected_Fy,
    )

    return ok_x and ok_y


# ==========================================================
# TEST 2
# Equal states
#
# U_L = U_R
#
# Rusanov must reduce to physical flux.
# ==========================================================

def test_equal_states():

    U = jnp.array(
        [1.5, -2.0]
    )

    Fx = vector_burgers_rusanov_flux(
        U,
        U,
        component=0,
    )

    Fy = vector_burgers_rusanov_flux(
        U,
        U,
        component=1,
    )

    expected_Fx = jnp.array(
        [1.5**2, 1.5 * (-2.0)]
    )

    expected_Fy = jnp.array(
        [1.5 * (-2.0), (-2.0)**2]
    )

    ok_x = check(
        "Equal states F_x",
        Fx,
        expected_Fx,
    )

    ok_y = check(
        "Equal states F_y",
        Fy,
        expected_Fy,
    )

    return ok_x and ok_y


# ==========================================================
# TEST 3
# Scalar reduction
#
# U = (u)
#
# Vector flux must reduce to scalar Burgers flux.
# ==========================================================

def test_scalar_reduction():

    uL = jnp.array([2.0])
    uR = jnp.array([1.0])

    vector_flux = vector_burgers_rusanov_flux(
        uL,
        uR,
        component=0,
    )

    # ------------------------------------------------------
    # Scalar Burgers Rusanov flux
    # ------------------------------------------------------

    alpha = jnp.maximum(
        jnp.abs(uL[0]),
        jnp.abs(uR[0]),
    )

    expected = (
        0.5 * (
            uL[0] ** 2
            + uR[0] ** 2
        )
        - 0.5 * alpha * (
            uR[0] - uL[0]
        )
    )

    ok = check(
        "Scalar reduction",
        vector_flux[0],
        expected,
    )

    return ok

# ==========================================================
# TEST 4
# Rusanov dissipation
#
# U_L != U_R
#
# Check against explicit formula.
# ==========================================================

def test_rusanov_formula():

    uL = jnp.array(
        [2.0, 1.0]
    )

    uR = jnp.array(
        [1.0, 3.0]
    )

    # ------------------------------------------------------
    # X direction
    #
    # velocity_L = 2
    # velocity_R = 1
    # alpha = 2
    # ------------------------------------------------------

    flux_x = vector_burgers_rusanov_flux(
        uL,
        uR,
        component=0,
    )

    physical_L_x = jnp.array(
        [4.0, 2.0]
    )

    physical_R_x = jnp.array(
        [1.0, 3.0]
    )

    alpha_x = 2.0

    expected_x = (
        0.5 * (
            physical_L_x
            + physical_R_x
        )
        - 0.5 * alpha_x * (
            uR - uL
        )
    )

    # ------------------------------------------------------
    # Y direction
    #
    # velocity_L = 1
    # velocity_R = 3
    # alpha = 3
    # ------------------------------------------------------

    flux_y = vector_burgers_rusanov_flux(
        uL,
        uR,
        component=1,
    )

    physical_L_y = jnp.array(
        [2.0, 1.0]
    )

    physical_R_y = jnp.array(
        [3.0, 9.0]
    )

    alpha_y = 3.0

    expected_y = (
        0.5 * (
            physical_L_y
            + physical_R_y
        )
        - 0.5 * alpha_y * (
            uR - uL
        )
    )

    ok_x = check(
        "Explicit Rusanov formula F_x",
        flux_x,
        expected_x,
    )

    ok_y = check(
        "Explicit Rusanov formula F_y",
        flux_y,
        expected_y,
    )

    return ok_x and ok_y


# ==========================================================
# TEST 5
# Vectorized spatial data
#
# U.shape = (2, nx)
# ==========================================================

def test_vectorized_data():

    uL = jnp.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 1.0, 4.0, 3.0],
        ]
    )

    uR = jnp.array(
        [
            [2.0, 1.0, 4.0, 3.0],
            [1.0, 3.0, 2.0, 4.0],
        ]
    )

    Fx = vector_burgers_rusanov_flux(
        uL,
        uR,
        component=0,
    )

    Fy = vector_burgers_rusanov_flux(
        uL,
        uR,
        component=1,
    )

    # Expected shapes
    shape_ok = (
        Fx.shape == uL.shape
        and Fy.shape == uL.shape
    )

    print(
        f"{'PASS' if shape_ok else 'FAIL':<6}"
        f"{'Vectorized shape':<40}"
        f"shape = {Fx.shape}"
    )

    # ------------------------------------------------------
    # Explicit reference
    # ------------------------------------------------------

    velocity_L_x = uL[0]
    velocity_R_x = uR[0]

    alpha_x = jnp.maximum(
        jnp.abs(velocity_L_x),
        jnp.abs(velocity_R_x),
    )

    expected_x = (
        0.5 * (
            velocity_L_x[None, :] * uL
            + velocity_R_x[None, :] * uR
        )
        - 0.5 * alpha_x[None, :]
        * (uR - uL)
    )

    velocity_L_y = uL[1]
    velocity_R_y = uR[1]

    alpha_y = jnp.maximum(
        jnp.abs(velocity_L_y),
        jnp.abs(velocity_R_y),
    )

    expected_y = (
        0.5 * (
            velocity_L_y[None, :] * uL
            + velocity_R_y[None, :] * uR
        )
        - 0.5 * alpha_y[None, :]
        * (uR - uL)
    )

    ok_x = check(
        "Vectorized F_x",
        Fx,
        expected_x,
    )

    ok_y = check(
        "Vectorized F_y",
        Fy,
        expected_y,
    )

    return (
        shape_ok
        and ok_x
        and ok_y
    )


# ==========================================================
# Main
# ==========================================================

def main():

    print()
    print("=" * 72)
    print("VECTOR BURGERS RUSANOV FLUX")
    print("=" * 72)
    print()

    results = [
        test_physical_flux(),
        test_equal_states(),
        test_scalar_reduction(),
        test_rusanov_formula(),
        test_vectorized_data(),
    ]

    print()
    print("-" * 72)

    if all(results):
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")

    print("-" * 72)


if __name__ == "__main__":
    main()