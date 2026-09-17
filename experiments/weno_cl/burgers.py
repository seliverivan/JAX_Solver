import numpy as np
import jax
import jax.numpy as jnp

jax.config.update(
    "jax_enable_x64",
    True,
)

from solver.weno9_solver import solve_weno9

from boundary.conditions import OUTFLOW
from boundary.conditions import DIRICHLET
from experiments.common import banner
from experiments.common import compute_TV

from experiments.plots import plot_solution


# ==========================================================
# TEST 3
# Burgers shock with physical outflow boundaries
# ==========================================================

def test_burgers_shock():

    banner(
        "TEST 3: Burgers shock"
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
    # Physical boundaries
    #
    # Outflow:
    #   left  -> copy u[0]
    #   right -> copy u[-1]
    #
    # For this Riemann problem this means:
    #   left ghost cells  -> 1
    #   right ghost cells -> 0
    # ------------------------------------------------------

    sol = solve_weno9(
        u0,
        t_end,
        (dx,),
        mu,
        boundary_types = (
            (DIRICHLET, OUTFLOW),
        ),
        boundary_values = (
            (1.0, 0.0),
        ),
        cfl=cfl,
        num_frames=2,
    )

    u = sol["solution"]
    
    print()
    print("BC CHECK:")
    print("u[0]    =", float(u[0]))
    print("u[-1]   =", float(u[-1]))
    print("u[:10]  =", u[:10])
    print("u[-10:] =", u[-10:])
    # ------------------------------------------------------
    # Exact shock position
    #
    # Burgers:
    #
    # s = (f(u_L) - f(u_R)) / (u_L - u_R)
    #
    # u_L = 1
    # u_R = 0
    #
    # s = 1/2
    #
    # x_s(t) = 0.5 + 0.5 * t
    # ------------------------------------------------------

    x_exact = 0.5 + 0.5 * t_end

    exact = jnp.where(
        x < x_exact,
        1.0,
        0.0,
    )

    plot_solution(
        x,
        u,
        exact,
        title="Burgers equation: shock wave",
    )

    # ------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------

    print(
        f"min(u) = {float(jnp.min(u)):.10f}"
    )

    print(
        f"max(u) = {float(jnp.max(u)):.10f}"
    )

    tv0 = compute_TV(u0)
    tv = compute_TV(u)

    print(
        f"TV initial = {tv0:.10f}"
    )

    print(
        f"TV final   = {tv:.10f}"
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
        f"Exact shock position = {x_exact:.8f}"
    )

    print(
        f"Numerical position  = {x_num:.8f}"
    )

    print(
        f"Position error      = "
        f"{abs(x_num - x_exact):.6e}"
    )

    # ------------------------------------------------------
    # Sanity check
    # ------------------------------------------------------

    tol = 1e-6

    overshoot = max(
        abs(float(jnp.min(u))),
        abs(float(jnp.max(u)) - 1.0),
    )

    print()

    print(
        f"Overshoot = {overshoot:.6e}"
    )

    if overshoot < tol:
        print("PASSED")
    else:
        print("FAILED")


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    test_burgers_shock()
