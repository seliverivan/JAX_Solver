import jax
import jax.numpy as jnp

jax.config.update(
    "jax_enable_x64",
    True,
)

from boundary.conditions import PERIODIC
from rhs.vector_burgers import vector_burgers_rhs
from solver.vector_weno9_solver import solve_vector_weno9


# ==========================================================
# Smooth initial condition
# ==========================================================

def make_initial_condition(n, ndim):

    dx = 1.0 / n

    x = (
        jnp.arange(n) + 0.5
    ) * dx

    grids = jnp.meshgrid(
        *([x] * ndim),
        indexing="ij",
    )

    U = []

    for component in range(ndim):

        field = 1.0

        for axis in range(ndim):

            X = grids[axis]

            if (component + axis) % 2 == 0:
                field = field * jnp.sin(
                    2.0 * jnp.pi * X
                )
            else:
                field = field * jnp.cos(
                    2.0 * jnp.pi * X
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
# Test 1: RHS sanity
# ==========================================================

def test_rhs_sanity(ndim):

    n = 32
    mu = 0.0

    U0, dx = make_initial_condition(
        n,
        ndim,
    )

    boundary_types, boundary_values = (
        make_boundaries(ndim)
    )

    rhs = vector_burgers_rhs(
        U0,
        (dx,) * ndim,
        mu,
        boundary_types,
        boundary_values,
        ng=5,
    )

    finite = bool(
        jnp.all(
            jnp.isfinite(rhs)
        )
    )

    print()
    print("=" * 72)
    print(
        f"VECTOR RHS SANITY {ndim}D"
    )
    print("=" * 72)

    print(
        f"shape      = {rhs.shape}"
    )

    print(
        f"max |RHS|  = "
        f"{float(jnp.max(jnp.abs(rhs))):.8e}"
    )

    print(
        f"finite     = {finite}"
    )

    assert finite
    assert rhs.shape == U0.shape

    print("PASS")


# ==========================================================
# Test 2: One tiny Euler step
# ==========================================================

def test_euler_consistency(ndim):

    n = 32
    mu = 0.0

    U0, dx = make_initial_condition(
        n,
        ndim,
    )

    boundary_types, boundary_values = (
        make_boundaries(ndim)
    )

    rhs = vector_burgers_rhs(
        U0,
        (dx,) * ndim,
        mu,
        boundary_types,
        boundary_values,
        ng=5,
    )

    dt = 1.0e-7

    expected = (
        U0 + dt * rhs
    )

    result = solve_vector_weno9(
        U0,
        t_end=dt,
        dx=(dx,) * ndim,
        mu=mu,
        boundary_types=boundary_types,
        boundary_values=boundary_values,
        cfl=0.4,
        num_frames=2,
        max_steps=10,
    )

    U1 = result["solution"]

    error = float(
        jnp.max(
            jnp.abs(
                U1 - expected
            )
        )
    )

    print()
    print("=" * 72)
    print(
        f"VECTOR SOLVER EULER CONSISTENCY {ndim}D"
    )
    print("=" * 72)

    print(
        f"dt         = {dt:.3e}"
    )

    print(
        f"error      = {error:.8e}"
    )

    print(
        f"steps      = {result['steps']}"
    )

    assert jnp.all(
        jnp.isfinite(U1)
    )

    # RK5 differs from Euler by O(dt^2),
    # so this should be extremely small.
    assert error < 1.0e-10

    print("PASS")


# ==========================================================
# Test 3: Full short integration
# ==========================================================

def test_full_solver(ndim):

    n = 32
    mu = 0.0
    t_end = 0.01

    U0, dx = make_initial_condition(
        n,
        ndim,
    )

    boundary_types, boundary_values = (
        make_boundaries(ndim)
    )

    result = solve_vector_weno9(
        U0,
        t_end=t_end,
        dx=(dx,) * ndim,
        mu=mu,
        boundary_types=boundary_types,
        boundary_values=boundary_values,
        cfl=0.2,
        num_frames=2,
        max_steps=10000,
    )

    U = result["solution"]

    finite = bool(
        jnp.all(
            jnp.isfinite(U)
        )
    )

    change = float(
        jnp.max(
            jnp.abs(
                U - U0
            )
        )
    )

    print()
    print("=" * 72)
    print(
        f"VECTOR SOLVER FULL TEST {ndim}D"
    )
    print("=" * 72)

    print(
        f"shape      = {U.shape}"
    )

    print(
        f"steps      = {result['steps']}"
    )

    print(
        f"saves      = {len(result['times'])}"
    )

    print(
        f"max change = {change:.8e}"
    )

    print(
        f"finite     = {finite}"
    )

    assert U.shape == U0.shape
    assert finite
    assert result["steps"] > 0
    assert change > 0.0

    print("PASS")


# ==========================================================
# Main
# ==========================================================

def main():

    for ndim in (1, 2, 3):

        test_rhs_sanity(
            ndim
        )

        test_euler_consistency(
            ndim
        )

        test_full_solver(
            ndim
        )


if __name__ == "__main__":
    main()
