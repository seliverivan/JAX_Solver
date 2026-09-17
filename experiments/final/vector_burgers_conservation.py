import jax
import jax.numpy as jnp

jax.config.update(
    "jax_enable_x64",
    True,
)

from boundary.conditions import PERIODIC
from rhs.vector_burgers import vector_burgers_rhs
from solver.integrator import integrate


# ==========================================================
# Initial condition
# ==========================================================

def make_initial_condition(ndim, n):

    dx = 1.0 / n

    x = (
        jnp.arange(n) + 0.5
    ) * dx

    mesh = jnp.meshgrid(
        *([x] * ndim),
        indexing="ij",
    )

    U = []

    for component in range(ndim):

        field = 1.0

        for axis in range(ndim):

            phase = (
                2.0
                * jnp.pi
                * (component + axis + 1)
                * mesh[axis]
            )

            field = (
                field
                * (
                    0.7
                    + 0.2 * jnp.sin(phase)
                )
            )

        U.append(field)

    return jnp.stack(U, axis=0), dx


# ==========================================================
# Periodic boundaries
# ==========================================================

def make_boundaries(ndim):

    boundary_types = tuple(
        (
            PERIODIC,
            PERIODIC,
        )
        for _ in range(ndim)
    )

    boundary_values = tuple(
        (
            None,
            None,
        )
        for _ in range(ndim)
    )

    return (
        boundary_types,
        boundary_values,
    )


# ==========================================================
# RHS
# ==========================================================

def make_rhs(ndim, dx, mu=0.0):

    boundary_types, boundary_values = (
        make_boundaries(ndim)
    )

    def rhs(U):

        return vector_burgers_rhs(
            U,
            (dx,) * ndim,
            mu,
            boundary_types,
            boundary_values,
            ng=5,
        )

    return rhs


# ==========================================================
# Conservation diagnostics
# ==========================================================

def component_mass(U):

    axes = tuple(
        range(1, U.ndim)
    )

    return jnp.sum(
        U,
        axis=axes,
    )


def conservation_error(
    initial,
    final,
):

    m0 = component_mass(initial)
    m1 = component_mass(final)

    absolute_error = jnp.abs(
        m1 - m0
    )

    relative_error = (
        absolute_error
        / jnp.maximum(
            jnp.abs(m0),
            1e-14,
        )
    )

    return (
        absolute_error,
        relative_error,
    )


# ==========================================================
# One complete solver run
# ==========================================================

def run(ndim, n):

    U0, dx = make_initial_condition(
        ndim,
        n,
    )

    rhs = make_rhs(
        ndim,
        dx,
        mu=0.0,
    )

    # ------------------------------------------------------
    # CFL
    # ------------------------------------------------------

    def compute_dt(U):

        speed = jnp.max(
            jnp.abs(U)
        )

        rate = (
            ndim
            * speed
            / dx
        )

        return (
            0.15
            / jnp.maximum(
                rate,
                1e-14,
            )
        )

    # ------------------------------------------------------
    # Full integration
    #
    # Only initial + final states are stored.
    # The test does not use the history.
    # ------------------------------------------------------

    (
        U,
        history,
        times,
        save_id,
        steps,
    ) = integrate(
        u0=U0,
        t_end=0.01,
        rhs=rhs,
        compute_dt=compute_dt,
        max_steps=10000,
        num_frames=2,
    )

    # ------------------------------------------------------
    # Conservation
    # ------------------------------------------------------

    absolute_error, relative_error = (
        conservation_error(
            U0,
            U,
        )
    )

    # ------------------------------------------------------
    # Finite check
    # ------------------------------------------------------

    finite = bool(
        jnp.all(
            jnp.isfinite(U)
        )
    )

    # ------------------------------------------------------
    # Maximum solution amplitude
    # ------------------------------------------------------

    max_u = float(
        jnp.max(
            jnp.abs(U)
        )
    )

    return {
        "ndim": ndim,
        "n": n,
        "steps": int(steps),
        "mass_error": float(
            jnp.max(
                absolute_error
            )
        ),
        "relative_mass_error": float(
            jnp.max(
                relative_error
            )
        ),
        "max_u": max_u,
        "finite": finite,
        "U0": U0,
        "U": U,
    }

# ==========================================================
# Print detailed result
# ==========================================================

def print_result(result):

    print()
    print("=" * 78)
    print(
        f"VECTOR BURGERS FULL SOLVER — "
        f"{result['ndim']}D"
    )
    print("=" * 78)

    print(
        f"N                    = "
        f"{result['n']}"
    )

    print(
        f"steps                = "
        f"{result['steps']}"
    )

    print(
        f"max |U|              = "
        f"{result['max_u']:.8e}"
    )

    print(
        f"absolute mass error  = "
        f"{result['mass_error']:.8e}"
    )

    print(
        f"relative mass error  = "
        f"{result['relative_mass_error']:.8e}"
    )

    print(
        f"finite               = "
        f"{result['finite']}"
    )

    print()
    print("COMPONENT ERRORS")
    print("-" * 78)

    U0 = result["U0"]
    U = result["U"]

    m0 = component_mass(U0)
    m1 = component_mass(U)

    errors = jnp.abs(
        m1 - m0
    )

    relative = (
        errors
        / jnp.maximum(
            jnp.abs(m0),
            1e-14,
        )
    )

    for i in range(
        result["ndim"]
    ):

        print(
            f"component {i + 1}: "
            f"mass₀ = {float(m0[i]): .12e}   "
            f"mass₁ = {float(m1[i]): .12e}   "
            f"error = {float(errors[i]): .8e}   "
            f"relative = {float(relative[i]): .8e}"
        )


# ==========================================================
# Assertions
# ==========================================================

def validate(result):

    assert result["finite"]

    assert (
        result["mass_error"]
        < 1e-10
    )

    assert (
        result["relative_mass_error"]
        < 1e-10
    )

    print()
    print("PASS")


# ==========================================================
# Dimension test
# ==========================================================

def test_dimension(ndim):

    result = run(
        ndim,
        n=32,
    )

    print_result(
        result
    )

    validate(
        result
    )


# ==========================================================
# Main
# ==========================================================

def main():

    print()
    print("=" * 78)
    print(
        "VECTOR BURGERS — FULL SOLVER TEST"
    )
    print("=" * 78)

    test_dimension(1)
    test_dimension(2)
    test_dimension(3)

    print()
    print("=" * 78)
    print(
        "ALL VECTOR SOLVER TESTS PASSED"
    )
    print("=" * 78)


if __name__ == "__main__":

    main()