import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from operators.flux_divergence import flux_divergence
from rk.rk5 import rk5

from experiments.common import banner, compute_L2
from experiments.plots import plot_convergence

from boundary.apply import apply_boundary
from boundary.conditions import PERIODIC


# ==========================================================
# Linear advection RHS
# ==========================================================

def advection_rhs(u, dx, a):

    ng = 5

    u_ext = apply_boundary(
        u,
        axis=0,
        left_type=PERIODIC,
        right_type=PERIODIC,
        ng=ng,
    )

    return -flux_divergence(
        u_ext,
        dx,
        axis=0,
        flux=lambda uL, uR: (
            0.5 * a * (uL + uR)
            - 0.5 * abs(a) * (uR - uL)
        ),
        ng=ng,
    )


# ==========================================================
# Linear advection solver
# ==========================================================

def solve_advection(
    u0,
    dx,
    t_end,
    a,
    cfl,
):

    u = u0
    t = 0.0

    while t < t_end:

        dt = cfl * dx / abs(a)

        if t + dt > t_end:
            dt = t_end - t

        u = rk5(
            u,
            dt,
            lambda x: advection_rhs(x, dx, a),
        )

        t += dt

    return u


# ==========================================================
# TEST: WENO9 convergence
# ==========================================================

def test_weno9_convergence():

    banner(
        "WENO9 CONVERGENCE: LINEAR ADVECTION"
    )

    # ------------------------------------------------------
    # Problem parameters
    # ------------------------------------------------------

    a = 1.0
    t_end = 0.1
    cfl = 0.1

    Ns = [
        16,
        32,
        64,
        128,
        256,
    ]

    errors = []

    # ------------------------------------------------------
    # Convergence loop
    # ------------------------------------------------------

    for N in Ns:

        dx = 1.0 / N

        x = (
            jnp.arange(N) + 0.5
        ) * dx

        # Smooth initial condition
        u0 = jnp.sin(
            2.0 * jnp.pi * x
        )

        # Numerical solution
        u = solve_advection(
            u0,
            dx,
            t_end,
            a,
            cfl,
        )

        # Exact solution
        exact = jnp.sin(
            2.0 * jnp.pi * (x - a * t_end)
        )

        # L2 error
        error = compute_L2(
            u,
            exact,
            dx,
        )

        errors.append(
            float(error)
        )

    # ------------------------------------------------------
    # Print convergence table
    # ------------------------------------------------------

    print()
    print(
        f"{'N':>8} "
        f"{'dx':>14} "
        f"{'L2 error':>18} "
        f"{'Order':>10}"
    )

    print("-" * 56)

    for i, N in enumerate(Ns):

        dx = 1.0 / N

        if i == 0:

            print(
                f"{N:8d} "
                f"{dx:14.6e} "
                f"{errors[i]:18.8e} "
                f"{'-':>10}"
            )

        else:

            order = np.log2(
                errors[i - 1] / errors[i]
            )

            print(
                f"{N:8d} "
                f"{dx:14.6e} "
                f"{errors[i]:18.8e} "
                f"{order:10.4f}"
            )

    print()

    # ------------------------------------------------------
    # Convergence plot
    # ------------------------------------------------------

    plot_convergence(
        Ns,
        errors,
        order=9,
        title="WENO9: linear advection convergence",
    )


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    test_weno9_convergence()