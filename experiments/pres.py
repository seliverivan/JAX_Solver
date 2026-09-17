import jax
import jax.numpy as jnp

jax.config.update(
    "jax_enable_x64",
    True,
)

from solver.weno9_solver import solve_weno9

from boundary.conditions import (
    DIRICHLET,
    OUTFLOW,
)

from experiments.common import banner
from experiments.plots import plot_solution


# ==========================================================
# TEST 1
# Burgers with non-periodic Dirichlet boundaries
# ==========================================================

def test_burgers_dirichlet():

    banner(
        "TEST 1: Burgers with Dirichlet boundaries"
    )

    N = 400
    t_end = 0.15
    mu = 0.01
    cfl = 0.2

    dx = 1.0 / N

    x = (
        jnp.arange(N) + 0.5
    ) * dx

    # ------------------------------------------------------
    # Initial condition
    # ------------------------------------------------------н

    u0 = 0.5 + 0.25 * jnp.sin(
        2.0 * jnp.pi * x
    )

    # ------------------------------------------------------
    # Dirichlet boundaries
    #
    # u(0,t) = 1
    # u(1,t) = 0
    # ------------------------------------------------------

    left_bc = 1.0
    right_bc = 0.0

    sol = solve_weno9(
        u0,
        t_end,
        (dx,),
        mu,
        boundary_types=(
            (DIRICHLET, DIRICHLET),
        ),
        boundary_values=(
            (left_bc, right_bc),
        ),
        cfl=cfl,
        save_every=1,
    )

    u = sol["solution"]

    # ------------------------------------------------------
    # Plot
    # ------------------------------------------------------

    plot_solution(
        x,
        u,
        title="Burgers equation: Dirichlet boundaries",
    )

    # ------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------

    print()
    print("BOUNDARY CHECK:")
    print(
        f"left  physical cell = {float(u[0]):.10f}"
    )
    print(
        f"right physical cell = {float(u[-1]):.10f}"
    )

    print()
    print(
        f"min(u) = {float(jnp.min(u)):.10f}"
    )
    print(
        f"max(u) = {float(jnp.max(u)):.10f}"
    )

    print("PASSED")


# ==========================================================
# TEST 2
# Burgers shock with physical outflow boundaries
# ==========================================================

def test_burgers_outflow():

    banner(
        "TEST 2: Burgers shock with Outflow boundaries"
    )

    N = 400

    t_end = 0.2

    mu = 0.0

    cfl = 0.2

    dx = 1.0 / N

    x = (
        jnp.arange(N) + 0.5
    ) * dx

    # ------------------------------------------------------
    # Riemann initial condition
    #
    # u = 1, x < 0.5
    # u = 0, x >= 0.5
    # ------------------------------------------------------

    u0 = jnp.where(
        x < 0.5,
        1.0,
        0.0,
    )

    # ------------------------------------------------------
    # Physical outflow boundaries
    #
    # left  -> copy u[0]
    # right -> copy u[-1]
    # ------------------------------------------------------

    sol = solve_weno9(
        u0,
        t_end,
        (dx,),
        mu,
        boundary_types=(
            (OUTFLOW, OUTFLOW),
        ),
        boundary_values=(
            (0.0, 0.0),
        ),
        cfl=cfl,
        save_every=1,
    )

    u = sol["solution"]

    # ------------------------------------------------------
    # Exact shock position
    #
    # s = 1/2
    #
    # x_s(t) = 0.5 + 0.5*t
    # ------------------------------------------------------

    x_exact = (
        0.5
        + 0.5 * t_end
    )

    exact = jnp.where(
        x < x_exact,
        1.0,
        0.0,
    )

    # ------------------------------------------------------
    # Plot
    # ------------------------------------------------------

    plot_solution(
        x,
        u,
        exact,
        title="Burgers equation: Outflow boundaries",
    )

    # ------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------

    print()
    print("BOUNDARY CHECK:")

    print(
        f"u[0]  = {float(u[0]):.10f}"
    )

    print(
        f"u[-1] = {float(u[-1]):.10f}"
    )

    print()
    print(
        f"min(u) = {float(jnp.min(u)):.10f}"
    )

    print(
        f"max(u) = {float(jnp.max(u)):.10f}"
    )

    # ------------------------------------------------------
    # Shock position
    # ------------------------------------------------------

    idx = jnp.argmin(
        jnp.abs(u - 0.5)
    )

    x_num = float(x[idx])

    print()
    print(
        f"Exact shock position = "
        f"{x_exact:.8f}"
    )

    print(
        f"Numerical position   = "
        f"{x_num:.8f}"
    )

    print(
        f"Position error        = "
        f"{abs(x_num - x_exact):.6e}"
    )

    print("PASSED")


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    test_burgers_dirichlet()

    test_burgers_outflow()