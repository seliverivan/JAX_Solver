import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from boundary.conditions import PERIODIC
from rhs.burgers import burgers_rhs


def run(nx, mu):

    ng = 5
    L = 1.0
    dx = L / nx

    x = (jnp.arange(nx) + 0.5) * dx

    # --------------------------------------------------------
    # u = sin(2*pi*x)
    # --------------------------------------------------------

    u = jnp.sin(2.0 * jnp.pi * x)

    # --------------------------------------------------------
    # Exact RHS
    #
    # u_t = -(u²/2)_x + mu*u_xx
    #
    # -(u²/2)_x = -u*u_x
    #            = -2*pi*sin(2*pi*x)*cos(2*pi*x)
    #
    # u_xx = -(2*pi)^2*u
    # --------------------------------------------------------

    exact = (
        -2.0 * jnp.pi
        * jnp.sin(2.0 * jnp.pi * x)
        * jnp.cos(2.0 * jnp.pi * x)
        - 4.0 * jnp.pi**2 * mu * u
    )

    boundary_types = (
        (PERIODIC, PERIODIC),
    )

    boundary_values = (
        (0.0, 0.0),
    )

    rhs = burgers_rhs(
        u,
        (dx,),
        mu,
        boundary_types,
        boundary_values,
        ng=ng,
    )

    error = jnp.max(
        jnp.abs(rhs - exact)
    )

    return float(error)


def main():

    mu = 0.01

    resolutions = [
        16,
        32,
        64,
        128,
    ]

    previous = None

    print("=" * 72)
    print("BURGERS RHS PERIODIC CONVERGENCE")
    print("=" * 72)

    print()
    print(
        f"{'N':>6}"
        f"{'error':>20}"
        f"{'order':>12}"
    )

    print("-" * 72)

    for nx in resolutions:

        error = run(
            nx,
            mu,
        )

        if previous is None:
            order = 0.0
        else:
            order = jnp.log2(
                previous / error
            )
            order = float(order)

        print(
            f"{nx:6d}"
            f"{error:20.8e}"
            f"{order:12.4f}"
        )

        previous = error

    print("-" * 72)


if __name__ == "__main__":
    main()