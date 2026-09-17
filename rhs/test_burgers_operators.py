import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from boundary.conditions import PERIODIC
from boundary.apply import apply_boundary
from operators.flux_divergence import flux_divergence
from operators.second_derivative import second_derivative
from operators.flux import burgers_rusanov_flux


def run(nx, mu):

    ng = 5
    dx = 1.0 / nx

    x = (jnp.arange(nx) + 0.5) * dx

    u = jnp.sin(2.0 * jnp.pi * x)

    # --------------------------------------------------------
    # Periodic ghost cells
    # --------------------------------------------------------

    u_ext = apply_boundary(
        u,
        axis=0,
        left_type=PERIODIC,
        right_type=PERIODIC,
        ng=ng,
        dx=dx,
    )

    # ========================================================
    # CONVECTION
    # ========================================================

    numerical_conv = -flux_divergence(
        u_ext,
        dx,
        axis=0,
        flux=burgers_rusanov_flux,
        ng=ng,
    )

    exact_conv = (
        -2.0
        * jnp.pi
        * jnp.sin(2.0 * jnp.pi * x)
        * jnp.cos(2.0 * jnp.pi * x)
    )

    conv_error = jnp.max(
        jnp.abs(
            numerical_conv - exact_conv
        )
    )

    # ========================================================
    # DIFFUSION
    # ========================================================

    numerical_diff = (
        mu
        * second_derivative(
            u_ext,
            dx,
            axis=0,
            ng=ng,
        )
    )

    exact_diff = (
        -4.0
        * jnp.pi**2
        * mu
        * u
    )

    diff_error = jnp.max(
        jnp.abs(
            numerical_diff - exact_diff
        )
    )

    return (
        float(conv_error),
        float(diff_error),
    )


def print_table(name, values):

    print()
    print("=" * 72)
    print(name)
    print("=" * 72)

    print(
        f"{'N':>6}"
        f"{'error':>20}"
        f"{'order':>12}"
    )

    print("-" * 72)

    previous = None

    for nx, error in values:

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


def main():

    mu = 0.01

    resolutions = [
        16,
        32,
        64,
        128,
    ]

    convection = []
    diffusion = []

    for nx in resolutions:

        conv_error, diff_error = run(
            nx,
            mu,
        )

        convection.append(
            (nx, conv_error)
        )

        diffusion.append(
            (nx, diff_error)
        )

    print_table(
        "BURGERS CONVECTION",
        convection,
    )

    print_table(
        "BURGERS DIFFUSION",
        diffusion,
    )


if __name__ == "__main__":
    main()