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

def run(nx):

    dx = 1.0 / nx
    ng = 5

    u = exact_cell_averages(nx)

    u_bc = apply_boundary(
        u,
        axis=0,
        left_type=PERIODIC,
        right_type=PERIODIC,
        ng=ng,
        dx=dx,
    )

    # ------------------------------------------------------
    # Physical cells
    # ------------------------------------------------------

    # Target cell centers
    x = (
        jnp.arange(nx) + 0.5
    ) * dx

    # ------------------------------------------------------
    # Use the 9-cell stencil around each target cell
    #
    # indices:
    #
    #   i-4 ... i ... i+4
    # ------------------------------------------------------

    indices = jnp.arange(
        ng,
        ng + nx,
    )

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
    # Reconstruct at cell center
    # ------------------------------------------------------

    numerical = reconstruct_point(
        stencil,
        xi=0.0,
    )

    exact = exact_function(x)

    error = jnp.max(
        jnp.abs(
            numerical - exact
        )
    )

    return float(error)


# ==========================================================
# Table
# ==========================================================

def main():

    resolutions = [
        8,
        16,
        32,
        64,
        128,
    ]

    print()
    print("=" * 72)
    print("9TH ORDER FV POINT RECONSTRUCTION")
    print("=" * 72)

    print(
        f"{'N':>6}"
        f"{'error':>20}"
        f"{'order':>12}"
    )

    print("-" * 72)

    previous = None

    for nx in resolutions:

        error = run(nx)

        if previous is None:
            order = 0.0
        else:
            order = float(
                jnp.log2(
                    previous / error
                )
            )

        print(
            f"{nx:6d}"
            f"{error:20.8e}"
            f"{order:12.4f}"
        )

        previous = error

    print("-" * 72)


if __name__ == "__main__":
    main()