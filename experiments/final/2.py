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
# EXACT CELL AVERAGES
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

    on x in [0, 1], periodic.

    Cole-Hopf:

        u = -2*nu * phi_x / phi

    Therefore

        u = -2*nu * (log(phi))_x

    and the exact cell average is

        ubar_i =
            -2*nu/dx *
            [log(phi(x_{i+1/2}))
             - log(phi(x_{i-1/2}))]
    """

    N = x.shape[0]

    dx = 1.0 / N

    # ------------------------------------------------------
    # Cell-center grid
    # ------------------------------------------------------

    x_center = (
        jnp.arange(N) + 0.5
    ) * dx

    # ------------------------------------------------------
    # Initial Cole-Hopf potential
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
    # Heat equation
    #
    # phi_t = nu * phi_xx
    # ------------------------------------------------------

    phi_hat = jnp.fft.fft(phi0)

    phi_hat_t = (
        phi_hat
        * jnp.exp(
            -nu * k**2 * t
        )
    )

    # ------------------------------------------------------
    # Move Fourier representation from
    # cell centers to cell interfaces
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
    # Interfaces:
    #
    # phi_left[i]  = phi(x_{i-1/2})
    # phi_right[i] = phi(x_{i+1/2})
    # ------------------------------------------------------

    phi_left = phi_interface

    phi_right = jnp.roll(
        phi_interface,
        -1,
    )

    # ------------------------------------------------------
    # Exact cell average
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
# Numerical Burgers solver
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
# Error norms
# ==========================================================

def compute_errors(
    numerical,
    exact,
    dx,
):
    """
    Compute L1, L2 and Linf errors.
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
# Convergence order
# ==========================================================

def convergence_order(
    error_coarse,
    error_fine,
):
    """
    Since dx_fine = dx_coarse / 2:

        p = log2(E_coarse / E_fine)
    """

    return jnp.log2(
        error_coarse / error_fine
    )


# ==========================================================
# Convergence test
# ==========================================================

def test_burgers_cole_hopf_convergence():

    banner(
        "VISCOUS BURGERS: WENO9-FV / COLE-HOPF CONVERGENCE"
    )

    # ------------------------------------------------------
    # Parameters
    # ------------------------------------------------------

    resolutions = [
        32,
        64,
        128,
        256,
        512,
    ]

    nu = 1.0e-2
    t_end = 0.1
    cfl = 0.1

    # ------------------------------------------------------
    # Storage
    # ------------------------------------------------------

    results = []

    previous_L1 = None
    previous_L2 = None
    previous_Linf = None

    # ------------------------------------------------------
    # Header
    # ------------------------------------------------------

    print()

    print(
        f"Equation : u_t + u u_x = nu u_xx"
    )

    print(
        f"Initial  : u(x,0) = sin(2*pi*x)"
    )

    print(
        f"Reference: Cole-Hopf exact CELL AVERAGES"
    )

    print()

    print(
        f"nu       = {nu:.6e}"
    )

    print(
        f"t_end    = {t_end:.6e}"
    )

    print(
        f"CFL      = {cfl:.6e}"
    )

    print()

    print("=" * 86)

    print(
        f"{'N':>6}"
        f"{'dx':>14}"
        f"{'L1':>18}"
        f"{'order':>10}"
        f"{'L2':>18}"
        f"{'order':>10}"
        f"{'Linf':>18}"
        f"{'order':>10}"
    )

    print("-" * 86)

    # ------------------------------------------------------
    # Resolutions
    # ------------------------------------------------------

    for N in resolutions:

        dx = 1.0 / N

        # Numerical solution
        x, numerical = solve_burgers(
            N=N,
            t_end=t_end,
            nu=nu,
            cfl=cfl,
        )

        # Exact cell averages
        exact = burgers_exact_cole_hopf_cell_average(
            x,
            t_end,
            nu,
        )

        # Errors
        L1, L2, Linf = compute_errors(
            numerical,
            exact,
            dx,
        )

        # Orders
        if previous_L1 is None:

            order_L1 = float("nan")
            order_L2 = float("nan")
            order_Linf = float("nan")

        else:

            order_L1 = float(
                convergence_order(
                    previous_L1,
                    L1,
                )
            )

            order_L2 = float(
                convergence_order(
                    previous_L2,
                    L2,
                )
            )

            order_Linf = float(
                convergence_order(
                    previous_Linf,
                    Linf,
                )
            )

        # Save
        results.append(
            (
                N,
                dx,
                L1,
                order_L1,
                L2,
                order_L2,
                Linf,
                order_Linf,
            )
        )

        # Print
        print(
            f"{N:6d}"
            f"{dx:14.6e}"
            f"{L1:18.10e}"
            f"{order_L1:10.4f}"
            f"{L2:18.10e}"
            f"{order_L2:10.4f}"
            f"{Linf:18.10e}"
            f"{order_Linf:10.4f}"
        )

        # Update
        previous_L1 = L1
        previous_L2 = L2
        previous_Linf = Linf

    print("=" * 86)

    # ------------------------------------------------------
    # Final interpretation
    # ------------------------------------------------------

    print()
    print(
        "Reference solution uses exact FV cell averages."
    )

    print(
        "Initial condition is also initialized as exact cell averages."
    )

    print()


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    test_burgers_cole_hopf_convergence()