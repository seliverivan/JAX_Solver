import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from solver.vector_weno9_solver import solve_vector_weno9

jax.config.update(
    "jax_enable_x64",
    True,
)

from boundary.conditions import PERIODIC


# ==========================================================
# PARAMETERS
# ==========================================================

N = 256

NU = 1.0e-2
AMPLITUDE = 0.2
T_END = 0.1


# ==========================================================
# Cole-Hopf exact solution
#
# Equation solved by the 1D vector solver:
#
#     u_t + (u^2)_x = nu u_xx
#
# ==========================================================

def phi(x, t):

    q = (
        AMPLITUDE
        * jnp.exp(
            -4.0
            * jnp.pi**2
            * NU
            * t
        )
    )

    return (
        1.0
        + q
        * jnp.cos(
            2.0 * jnp.pi * x
        )
    )


def exact_point(x, t):

    q = (
        AMPLITUDE
        * jnp.exp(
            -4.0
            * jnp.pi**2
            * NU
            * t
        )
    )

    return (
        2.0
        * jnp.pi
        * NU
        * q
        * jnp.sin(
            2.0 * jnp.pi * x
        )
        / (
            1.0
            + q
            * jnp.cos(
                2.0 * jnp.pi * x
            )
        )
    )


# ==========================================================
# Exact CELL AVERAGE
#
# u = -nu * d/dx log(phi)
#
# Therefore:
#
#     u_bar =
#       -nu/dx *
#       [log(phi_R) - log(phi_L)]
# ==========================================================

def exact_cell_average(
    xL,
    xR,
    t,
):

    return (
        -NU
        / (
            xR - xL
        )
        * (
            jnp.log(
                phi(
                    xR,
                    t,
                )
            )
            -
            jnp.log(
                phi(
                    xL,
                    t,
                )
            )
        )
    )


# ==========================================================
# Initial condition
# ==========================================================

def make_initial_condition():

    dx = 1.0 / N

    x = (
        jnp.arange(N)
        + 0.5
    ) * dx

    xL = x - 0.5 * dx
    xR = x + 0.5 * dx

    U0 = exact_cell_average(
        xL,
        xR,
        0.0,
    )

    # Vector solver expects:
    #
    #     (components, Nx)
    #
    U0 = U0[None, :]

    return U0, dx, x


# ==========================================================
# Boundary conditions
# ==========================================================

def make_boundaries():

    boundary_types = (
        (
            PERIODIC,
            PERIODIC,
        ),
    )

    boundary_values = (
        (
            None,
            None,
        ),
    )

    return (
        boundary_types,
        boundary_values,
    )


# ==========================================================
# Main experiment
# ==========================================================

def main():

    U0, dx, x = (
        make_initial_condition()
    )

    boundary_types, boundary_values = (
        make_boundaries()
    )

    # ------------------------------------------------------
    # FULL PRODUCTION SOLVER
    # ------------------------------------------------------

    result = solve_vector_weno9(
        U0=U0,
        t_end=T_END,
        dx=(dx,),
        mu=NU,
        boundary_types=boundary_types,
        boundary_values=boundary_values,
        cfl=0.4,
        num_frames=2,
        max_steps=100000,
    )

    numerical = result[
        "solution"
    ][0]

    steps = result[
        "steps"
    ]

    # ------------------------------------------------------
    # Exact solution
    #
    # Exact CELL AVERAGES
    # ------------------------------------------------------

    xL = x - 0.5 * dx
    xR = x + 0.5 * dx

    exact_avg = exact_cell_average(
        xL,
        xR,
        T_END,
    )

    # ------------------------------------------------------
    # For visualization:
    #
    # FV cell averages are plotted at cell centers.
    # ------------------------------------------------------

    numerical_plot = numerical
    exact_plot = exact_avg

    # ------------------------------------------------------
    # Error
    # ------------------------------------------------------

    error = (
        numerical
        - exact_avg
    )

    L1 = jnp.mean(
        jnp.abs(error)
    )

    L2 = jnp.sqrt(
        jnp.mean(
            error**2
        )
    )

    Linf = jnp.max(
        jnp.abs(error)
    )

    # ------------------------------------------------------
    # Console
    # ------------------------------------------------------

    print()
    print("=" * 78)
    print(
        "VECTOR BURGERS — "
        "EXACT SOLUTION"
    )
    print("=" * 78)

    print(
        f"N          = {N}"
    )

    print(
        f"nu         = {NU:.6e}"
    )

    print(
        f"t_end      = {T_END:.6e}"
    )

    print(
        f"steps      = {steps}"
    )

    print()

    print(
        f"L1         = "
        f"{float(L1):.10e}"
    )

    print(
        f"L2         = "
        f"{float(L2):.10e}"
    )

    print(
        f"Linf       = "
        f"{float(Linf):.10e}"
    )

    print(
        f"max |U|    = "
        f"{float(jnp.max(jnp.abs(numerical))):.10e}"
    )

    # ------------------------------------------------------
    # Plot
    # ------------------------------------------------------

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        x,
        exact_plot,
        label="Exact",
        linewidth=2.0,
    )

    plt.plot(
        x,
        numerical_plot,
        "--",
        label="WENO9-FV",
        linewidth=1.5,
    )

    plt.xlabel("x")
    plt.ylabel("u")

    plt.title(
        "Vector Burgers: "
        "WENO9-FV vs Exact Solution"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()