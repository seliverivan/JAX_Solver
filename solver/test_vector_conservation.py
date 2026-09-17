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
# Boundary configuration
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
# Conservation error
# ==========================================================

def conservation_error(
    initial,
    final,
):

    initial_mass = jnp.sum(
        initial,
        axis=tuple(
            range(1, initial.ndim)
        ),
    )

    final_mass = jnp.sum(
        final,
        axis=tuple(
            range(1, final.ndim)
        ),
    )

    return jnp.max(
        jnp.abs(
            final_mass - initial_mass
        )
    )


# ==========================================================
# One test
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

        rate = ndim * speed / dx

        return 0.15 / jnp.maximum(
            rate,
            1e-14,
        )

    # ------------------------------------------------------
    # Integrate
    # ------------------------------------------------------

    U, history, times, save_id, steps = (
        integrate(
            u0=U0,
            t_end=0.01,
            rhs=rhs,
            compute_dt=compute_dt,
            max_steps=10000,
            save_every=100,
        )
    )

    error = conservation_error(
        U0,
        U,
    )

    return (
        float(error),
        int(steps),
    )


# ==========================================================
# Dimension test
# ==========================================================

def test_dimension(ndim):

    n = 32

    error, steps = run(
        ndim,
        n,
    )

    print()
    print("=" * 72)
    print(
        f"VECTOR BURGERS CONSERVATION {ndim}D"
    )
    print("=" * 72)

    print(
        f"N              = {n}"
    )

    print(
        f"steps          = {steps}"
    )

    print(
        f"mass error     = {error:.8e}"
    )

    print(
        f"finite         = {jnp.isfinite(error)}"
    )

    # ------------------------------------------------------
    # Expected:
    #
    # finite
    # and conservation error close to machine precision.
    # ------------------------------------------------------

    assert jnp.isfinite(error)

    assert error < 1e-10

    print("PASS")


# ==========================================================
# Main
# ==========================================================

def main():

    test_dimension(1)
    test_dimension(2)
    test_dimension(3)


if __name__ == "__main__":

    main()