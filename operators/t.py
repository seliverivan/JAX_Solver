import jax
import jax.numpy as jnp

jax.config.update(
    "jax_enable_x64",
    True,
)

from boundary.apply import apply_boundary
from boundary.conditions import PERIODIC

from weno9.weno_fv import weno9_fv
from operators.flux import vector_burgers_rusanov_flux


# ==========================================================
# Smooth vector field
# ==========================================================

def make_field(nx):

    dx = 1.0 / nx

    x = (jnp.arange(nx) + 0.5) * dx
    y = (jnp.arange(nx) + 0.5) * dx

    X, Y = jnp.meshgrid(
        x,
        y,
        indexing="ij",
    )

    # ------------------------------------------------------
    # Exact 1D cell averages
    # ------------------------------------------------------

    xL = x - 0.5 * dx
    xR = x + 0.5 * dx

    yL = y - 0.5 * dx
    yR = y + 0.5 * dx

    avg_sin_x = (
        jnp.cos(2.0 * jnp.pi * xL)
        - jnp.cos(2.0 * jnp.pi * xR)
    ) / (2.0 * jnp.pi * dx)

    avg_cos_x = (
        jnp.sin(2.0 * jnp.pi * xR)
        - jnp.sin(2.0 * jnp.pi * xL)
    ) / (2.0 * jnp.pi * dx)

    avg_sin_y = (
        jnp.cos(2.0 * jnp.pi * yL)
        - jnp.cos(2.0 * jnp.pi * yR)
    ) / (2.0 * jnp.pi * dx)

    avg_cos_y = (
        jnp.sin(2.0 * jnp.pi * yR)
        - jnp.sin(2.0 * jnp.pi * yL)
    ) / (2.0 * jnp.pi * dx)

    # ------------------------------------------------------
    # Tensor-product cell averages
    # ------------------------------------------------------

    u = (
        avg_sin_x[:, None]
        * avg_cos_y[None, :]
    )

    v = (
        avg_cos_x[:, None]
        * avg_sin_y[None, :]
    )

    U = jnp.stack(
        (
            u,
            v,
        ),
        axis=0,
    )

    return U, X, Y, dx


# ==========================================================
# Exact physical flux at interfaces
# ==========================================================

def exact_interface_fluxes(nx):

    dx = 1.0 / nx

    x = (
        jnp.arange(nx) + 0.5
    ) * dx

    y = (
        jnp.arange(nx) + 0.5
    ) * dx

    x_face = (
        jnp.arange(nx + 1)
    ) * dx

    y_face = (
        jnp.arange(nx + 1)
    ) * dx

    # ------------------------------------------------------
    # Transverse cell averages
    # ------------------------------------------------------

    yL = y - 0.5 * dx
    yR = y + 0.5 * dx

    avg_sin_y = (
        jnp.cos(2.0 * jnp.pi * yL)
        - jnp.cos(2.0 * jnp.pi * yR)
    ) / (2.0 * jnp.pi * dx)

    avg_cos_y = (
        jnp.sin(2.0 * jnp.pi * yR)
        - jnp.sin(2.0 * jnp.pi * yL)
    ) / (2.0 * jnp.pi * dx)

    xL = x - 0.5 * dx
    xR = x + 0.5 * dx

    avg_sin_x = (
        jnp.cos(2.0 * jnp.pi * xL)
        - jnp.cos(2.0 * jnp.pi * xR)
    ) / (2.0 * jnp.pi * dx)

    avg_cos_x = (
        jnp.sin(2.0 * jnp.pi * xR)
        - jnp.sin(2.0 * jnp.pi * xL)
    ) / (2.0 * jnp.pi * dx)

    # ======================================================
    # X faces
    # ======================================================

    Xf, Yc = jnp.meshgrid(
        x_face,
        y,
        indexing="ij",
    )

    ux = (
        jnp.sin(2.0 * jnp.pi * Xf)
        * avg_cos_y[None, :]
    )

    vx = (
        jnp.cos(2.0 * jnp.pi * Xf)
        * avg_sin_y[None, :]
    )

    Fx = jnp.stack(
        (
            ux * ux,
            ux * vx,
        ),
        axis=0,
    )

    # ======================================================
    # Y faces
    # ======================================================

    Xc, Yf = jnp.meshgrid(
        x,
        y_face,
        indexing="ij",
    )

    uy = (
        avg_sin_x[:, None]
        * jnp.cos(2.0 * jnp.pi * Yf)
    )

    vy = (
        avg_cos_x[:, None]
        * jnp.sin(2.0 * jnp.pi * Yf)
    )

    Fy = jnp.stack(
        (
            uy * vy,
            vy * vy,
        ),
        axis=0,
    )

    return Fx, Fy


# ==========================================================
# Exact divergence
# ==========================================================

def exact_fv_divergence(nx):

    dx = 1.0 / nx

    exact_x, exact_y = exact_interface_fluxes(nx)

    # ------------------------------------------------------
    # X contribution
    # ------------------------------------------------------

    div_x = (
        jnp.take(
            exact_x,
            jnp.arange(
                1,
                exact_x.shape[1],
            ),
            axis=1,
        )
        -
        jnp.take(
            exact_x,
            jnp.arange(
                0,
                exact_x.shape[1] - 1,
            ),
            axis=1,
        )
    ) / dx

    # ------------------------------------------------------
    # Y contribution
    # ------------------------------------------------------

    div_y = (
        jnp.take(
            exact_y,
            jnp.arange(
                1,
                exact_y.shape[2],
            ),
            axis=2,
        )
        -
        jnp.take(
            exact_y,
            jnp.arange(
                0,
                exact_y.shape[2] - 1,
            ),
            axis=2,
        )
    ) / dx

    return div_x + div_y

# ==========================================================
# Central numerical flux
# ==========================================================

def central_flux(uL, uR):

    def physical_flux(U):

        u = U[0]
        v = U[1]

        return jnp.stack(
            (
                u * u,
                u * v,
            ),
            axis=0,
        )

    return 0.5 * (
        physical_flux(uL)
        + physical_flux(uR)
    )


# ==========================================================
# Y-direction central physical flux
# ==========================================================

def central_flux_y(uL, uR):

    def physical_flux(U):

        u = U[0]
        v = U[1]

        return jnp.stack(
            (
                u * v,
                v * v,
            ),
            axis=0,
        )

    return 0.5 * (
        physical_flux(uL)
        + physical_flux(uR)
    )


# ==========================================================
# Flux divergence from interface flux
# ==========================================================

def divergence_from_flux(F, dx, axis):

    right = jnp.take(
        F,
        jnp.arange(
            1,
            F.shape[axis],
        ),
        axis=axis,
    )

    left = jnp.take(
        F,
        jnp.arange(
            0,
            F.shape[axis] - 1,
        ),
        axis=axis,
    )

    return (
        right - left
    ) / dx


# ==========================================================
# Numerical divergence
# ==========================================================

def compute_divergence(nx, flux_type):

    ng = 5

    U, _, _, dx = make_field(nx)

    # ======================================================
    # X direction
    # ======================================================

    Ux = apply_boundary(
        U,
        axis=1,
        left_type=PERIODIC,
        right_type=PERIODIC,
        ng=ng,
        dx=dx,
    )

    Ux_L, Ux_R = weno9_fv(
        Ux,
        axis=1,
        ng=ng,
    )

    if flux_type == "central":

        Fx = central_flux(
            Ux_L,
            Ux_R,
        )

    elif flux_type == "rusanov":

        Fx = vector_burgers_rusanov_flux(
            Ux_L,
            Ux_R,
            component=0,
        )

    else:

        raise ValueError(
            f"Unknown flux type: {flux_type}"
        )

    div_x = divergence_from_flux(
        Fx,
        dx,
        axis=1,
    )

    # ======================================================
    # Y direction
    # ======================================================

    Uy = apply_boundary(
        U,
        axis=2,
        left_type=PERIODIC,
        right_type=PERIODIC,
        ng=ng,
        dx=dx,
    )

    Uy_L, Uy_R = weno9_fv(
        Uy,
        axis=2,
        ng=ng,
    )

    if flux_type == "central":

        Fy = central_flux_y(
            Uy_L,
            Uy_R,
        )

    elif flux_type == "rusanov":

        Fy = vector_burgers_rusanov_flux(
            Uy_L,
            Uy_R,
            component=1,
        )

    else:

        raise ValueError(
            f"Unknown flux type: {flux_type}"
        )

    div_y = divergence_from_flux(
        Fy,
        dx,
        axis=2,
    )

    return div_x + div_y


# ==========================================================
# One resolution
# ==========================================================

def run(nx, flux_type):

    numerical = compute_divergence(
        nx,
        flux_type,
    )

    exact = exact_fv_divergence(
        nx,
    )

    error = jnp.abs(
        numerical - exact
    )

    error_x = jnp.max(
        error[0]
    )

    error_y = jnp.max(
        error[1]
    )

    error_total = jnp.max(
        error
    )

    return (
        float(error_total),
        float(error_x),
        float(error_y),
    )


# ==========================================================
# Table
# ==========================================================

def test_flux(flux_type):

    resolutions = [
        8,
        16,
        32,
        64,
        128,
    ]

    print()
    print("=" * 72)

    if flux_type == "central":
        print("VECTOR BURGERS CENTRAL FLUX DIVERGENCE")

    else:
        print("VECTOR BURGERS RUSANOV FLUX DIVERGENCE")

    print("=" * 72)

    print(
        f"{'N':>6}"
        f"{'error':>20}"
        f"{'order':>12}"
        f"{'error_x':>20}"
        f"{'error_y':>20}"
    )

    print("-" * 80)

    previous = None

    for nx in resolutions:

        error, error_x, error_y = run(
            nx,
            flux_type,
        )

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
            f"{error_x:20.8e}"
            f"{error_y:20.8e}"
        )

        previous = error

    print("-" * 80)


# ==========================================================
# Main
# ==========================================================

def main():

    test_flux("central")
    test_flux("rusanov")


# ==========================================================
# Entry point
# ==========================================================

if __name__ == "__main__":

    main()