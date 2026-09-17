import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

jax.config.update(
    "jax_enable_x64",
    True,
)

from solver.weno9_solver import solve_weno9

from boundary.conditions import PERIODIC

from experiments.common import banner


# ==========================================================
# Exact solution: viscous Burgers via Cole-Hopf
# CELL AVERAGES
# ==========================================================

def burgers_exact_cole_hopf_cell_average(
    x,
    t,
    nu,
):
    """
    Exact CELL AVERAGES of

        u_t + u u_x = nu u_xx

    with

        u(x, 0) = sin(2*pi*x)

    on

        x in [0, 1]

    periodic.

    The FV solution stores

        u_i = 1/dx * integral_cell u(x,t) dx.

    Cole-Hopf:

        u = -2*nu * phi_x / phi

    Therefore

        u = -2*nu * (log phi)_x

    and the exact cell average is

        ubar_i =
            -2*nu/dx *
            [log(phi(x_{i+1/2}))
             - log(phi(x_{i-1/2}))].
    """

    N = x.shape[0]

    dx = 1.0 / N

    # ------------------------------------------------------
    # x are CELL CENTERS:
    #
    # x_i = (i + 1/2) dx
    #
    # Therefore:
    #
    # x_{i-1/2} = i dx
    # x_{i+1/2} = (i+1) dx
    # ------------------------------------------------------

    # Cell-center grid used to construct the Fourier
    # representation of phi.
    x_center = (
        jnp.arange(N) + 0.5
    ) * dx

    # ------------------------------------------------------
    # Initial Cole-Hopf potential
    #
    # phi(x,0) =
    # exp(cos(2*pi*x)/(4*pi*nu))
    # ------------------------------------------------------

    phi0 = jnp.exp(
        jnp.cos(2.0 * jnp.pi * x_center)
        / (4.0 * jnp.pi * nu)
    )

    # ------------------------------------------------------
    # Fourier wave numbers
    # ------------------------------------------------------

    freq = jnp.fft.fftfreq(
        N,
        d=dx,
    )

    k = 2.0 * jnp.pi * freq

    # ------------------------------------------------------
    # Heat equation:
    #
    # phi_t = nu phi_xx
    #
    # phi_hat(t) =
    # phi_hat(0) exp(-nu k^2 t)
    # ------------------------------------------------------

    phi_hat = jnp.fft.fft(phi0)

    phi_hat_t = (
        phi_hat
        * jnp.exp(
            -nu * k**2 * t
        )
    )

    # ------------------------------------------------------
    # IMPORTANT:
    #
    # We need phi on CELL INTERFACES, not centers.
    #
    # The FFT coefficients currently correspond to samples
    # at
    #
    # x_j = (j + 1/2) dx.
    #
    # Shift by -dx/2 to obtain
    #
    # x_j = j dx.
    # ------------------------------------------------------

    phase = jnp.exp(
        -1j * k * (dx / 2.0)
    )

    phi_hat_interface = (
        phi_hat_t * phase
    )

    phi_interface = jnp.fft.ifft(
        phi_hat_interface
    ).real

    # ------------------------------------------------------
    # phi_interface[j] corresponds to
    #
    # phi(j * dx)
    #
    # Therefore:
    #
    # left  = phi(x_{i-1/2})
    # right = phi(x_{i+1/2})
    #
    # with periodic wrapping.
    # ------------------------------------------------------

    phi_left = phi_interface

    phi_right = jnp.roll(
        phi_interface,
        -1,
    )

    # ------------------------------------------------------
    # EXACT CELL AVERAGE
    #
    # ubar_i =
    #
    # -2*nu/dx *
    # [log(phi_right) - log(phi_left)]
    # ------------------------------------------------------

    u_average = (
        -2.0
        * nu
        / dx
        * (
            jnp.log(phi_right)
            - jnp.log(phi_left)
        )
    )

    return u_average


# ==========================================================
# Numerical solution
# ==========================================================

def solve_burgers(
    N,
    t_end,
    nu,
    cfl,
):

    dx = 1.0 / N

    # ------------------------------------------------------
    # CELL CENTERS
    # ------------------------------------------------------

    x = (
        jnp.arange(N) + 0.5
    ) * dx

    # ------------------------------------------------------
    # Initial condition as CELL AVERAGES
    #
    # For
    #
    #     u(x,0) = sin(2*pi*x)
    #
    # the exact cell average is
    #
    #     sin(2*pi*x_i)
    #     * sin(pi*dx)/(pi*dx)
    # ------------------------------------------------------

    u0 = (
        jnp.sin(
            2.0 * jnp.pi * x
        )
        *
        (
            jnp.sin(
                jnp.pi * dx
            )
            /
            (
                jnp.pi * dx
            )
        )
    )

    # ------------------------------------------------------
    # WENO9-FV
    #
    # Only initial + final states are stored.
    # This minimizes GPU memory usage.
    # ------------------------------------------------------

    sol = solve_weno9(
        u0,
        t_end,
        (dx,),
        nu,
        boundary_types=(
            (PERIODIC, PERIODIC),
        ),
        boundary_values=(
            (None, None),
        ),
        cfl=cfl,
        num_frames=2,
    )

    return x, sol["solution"]


# ==========================================================
# Error
# ==========================================================

def compute_errors(
    numerical,
    exact,
    dx,
):
    """
    Absolute errors between numerical and exact
    CELL AVERAGES.
    """

    error = numerical - exact

    L1 = float(
        jnp.sum(
            jnp.abs(error)
        ) * dx
    )

    L2 = float(
        jnp.sqrt(
            jnp.sum(
                error**2
            ) * dx
        )
    )

    Linf = float(
        jnp.max(
            jnp.abs(error)
        )
    )

    return L1, L2, Linf


# ==========================================================
# Plot
# ==========================================================

def plot_solution(
    x,
    numerical,
    exact,
):
    """
    Plot numerical and exact CELL AVERAGES.
    """

    x = jnp.asarray(x)

    error = jnp.abs(
        numerical - exact
    )

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        x,
        exact,
        "--",
        linewidth=2,
        label="Cole-Hopf exact cell average",
    )

    plt.plot(
        x,
        numerical,
        linewidth=1.5,
        label="WENO9-FV",
    )

    plt.xlabel("x")
    plt.ylabel("cell average of u")

    plt.title(
        "Viscous Burgers: WENO9-FV vs Cole-Hopf"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.show()

    # ------------------------------------------------------
    # Absolute error
    # ------------------------------------------------------

    plt.figure(
        figsize=(9, 4)
    )

    plt.plot(
        x,
        error,
        linewidth=1.5,
    )

    plt.xlabel("x")
    plt.ylabel("|u - u_exact|")

    plt.title(
        "Cell-average absolute error"
    )

    plt.yscale("log")

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.show()


# ==========================================================
# Main test
# ==========================================================

def test_burgers_cole_hopf():

    banner(
        "TEST: VISCOUS BURGERS - COLE-HOPF CELL AVERAGES"
    )

    # ------------------------------------------------------
    # Parameters
    # ------------------------------------------------------

    N = 256
    t_end = 0.1
    nu = 0.01
    cfl = 0.1

    dx = 1.0 / N

    # ------------------------------------------------------
    # Numerical solution
    # ------------------------------------------------------

    x, numerical = solve_burgers(
        N=N,
        t_end=t_end,
        nu=nu,
        cfl=cfl,
    )

    # ------------------------------------------------------
    # Exact CELL AVERAGES
    # ------------------------------------------------------

    exact = burgers_exact_cole_hopf_cell_average(
        x,
        t_end,
        nu,
    )

    # ------------------------------------------------------
    # Errors
    # ------------------------------------------------------

    L1, L2, Linf = compute_errors(
        numerical,
        exact,
        dx,
    )

    # ------------------------------------------------------
    # Output
    # ------------------------------------------------------

    print()
    print("=" * 70)

    print(
        f"N       = {N}"
    )

    print(
        f"dx      = {dx:.6e}"
    )

    print(
        f"nu      = {nu:.6e}"
    )

    print(
        f"t_end   = {t_end:.6e}"
    )

    print()
    print(
        "REFERENCE = EXACT CELL AVERAGES"
    )

    print()
    print(
        "ABSOLUTE ERROR"
    )

    print("-" * 70)

    print(
        f"L1      = {L1:.12e}"
    )

    print(
        f"L2      = {L2:.12e}"
    )

    print(
        f"Linf    = {Linf:.12e}"
    )

    print("=" * 70)

    # ------------------------------------------------------
    # Plot
    # ------------------------------------------------------

    plot_solution(
        x,
        numerical,
        exact,
    )


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    test_burgers_cole_hopf()