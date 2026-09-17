import jax
import jax.numpy as jnp

jax.config.update(
    "jax_enable_x64",
    True,
)

from boundary.apply import apply_boundary
from boundary.conditions import PERIODIC

from weno9.point_reconstruction import (
    reconstruct_point,
)


# ==========================================================
# Smooth function
# ==========================================================

def exact_function(x):

    return (
        jnp.sin(2.0 * jnp.pi * x)
        + 0.3 * jnp.cos(4.0 * jnp.pi * x)
    )


# ==========================================================
# Exact cell averages
# ==========================================================

def exact_cell_averages(nx):

    dx = 1.0 / nx

    x = (
        jnp.arange(nx) + 0.5
    ) * dx

    xL = x - 0.5 * dx
    xR = x + 0.5 * dx

    avg_sin = (
        jnp.cos(2.0 * jnp.pi * xL)
        - jnp.cos(2.0 * jnp.pi * xR)
    ) / (
        2.0 * jnp.pi * dx
    )

    avg_cos = (
        jnp.sin(4.0 * jnp.pi * xR)
        - jnp.sin(4.0 * jnp.pi * xL)
    ) / (
        4.0 * jnp.pi * dx
    )

    return (
        avg_sin
        + 0.3 * avg_cos
    )


# ==========================================================
# One resolution
# ==========================================================

def run(nx, xi):

    dx = 1.0 / nx
    ng = 5

    # ------------------------------------------------------
    # Exact FV cell averages
    # ------------------------------------------------------

    u = exact_cell_averages(nx)

    # ------------------------------------------------------
    # Periodic ghost cells
    # ------------------------------------------------------

    u_bc = apply_boundary(
        u,
        axis=0,
        left_type=PERIODIC,
        right_type=PERIODIC,
        ng=ng,
        dx=dx,
    )

    # ------------------------------------------------------
    # Physical cell centers
    # ------------------------------------------------------

    x = (
        jnp.arange(nx) + 0.5
    ) * dx

    # ------------------------------------------------------
    # Indices of physical cells
    # ------------------------------------------------------

    indices = jnp.arange(
        ng,
        ng + nx,
    )

    # ------------------------------------------------------
    # Full 9-cell stencil
    #
    # i-4 ... i ... i+4
    # ------------------------------------------------------

    stencil = jnp.stack(
        [
            jnp.take(
                u_bc,
                indices + offset,
                axis=0,
            )
            for offset in range(-4, 5)
        ],
        axis=-1,
    )

    # ------------------------------------------------------
    # Point reconstruction
    # ------------------------------------------------------

    numerical = reconstruct_point(
        stencil,
        xi=xi,
    )

    # ------------------------------------------------------
    # Exact point value
    #
    # x_i + xi * dx
    # ------------------------------------------------------

    exact = exact_function(
        x + xi * dx
    )

    # ------------------------------------------------------
    # L-infinity error
    # ------------------------------------------------------

    error = jnp.max(
        jnp.abs(
            numerical - exact
        )
    )

    return float(error)


# ==========================================================
# Convergence table for one xi
# ==========================================================

def convergence_table(xi, resolutions):

    previous = None

    print()
    print(
        f"xi = {xi:+.2f}"
    )

    print(
        "-" * 48
    )

    print(
        f"{'N':>6}"
        f"{'error':>20}"
        f"{'order':>12}"
    )

    print(
        "-" * 48
    )

    for nx in resolutions:

        error = run(
            nx,
            xi,
        )

        if previous is None:

            order = 0.0

        else:

            order = float(
                jnp.log2(
                    previous / error
                )
            )

        # --------------------------------------------------
        # Once we hit floating-point precision, the
        # measured order is no longer meaningful.
        # --------------------------------------------------

        if error < 1.0e-14:

            order_text = "---"

        else:

            order_text = f"{order:12.4f}"

        print(
            f"{nx:6d}"
            f"{error:20.8e}"
            f"{order_text}"
        )

        previous = error

    print(
        "-" * 48
    )


# ==========================================================
# Main
# ==========================================================

def main():

    resolutions = [
        8,
        16,
        32,
        64,
        128,
    ]

    xis = [
        -0.50,
        -0.25,
         0.00,
         0.25,
         0.50,
    ]

    print()
    print("=" * 72)
    print("9TH ORDER FV POINT RECONSTRUCTION")
    print("=" * 72)

    print()
    print(
        "Reconstruction from 9 cell averages"
    )

    print(
        "Target: u(x_i + xi * dx)"
    )

    print(
        "Stencil: i-4 ... i+4"
    )

    print(
        "Polynomial degree: 8"
    )

    print(
        "=" * 72
    )

    for xi in xis:

        convergence_table(
            xi,
            resolutions,
        )


# ==========================================================
# Entry point
# ==========================================================

if __name__ == "__main__":

    main()