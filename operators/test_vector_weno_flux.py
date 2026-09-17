import jax
import jax.numpy as jnp

from operators.flux import vector_burgers_rusanov_flux

jax.config.update(
    "jax_enable_x64",
    True,
)

from boundary.apply import apply_boundary
from boundary.conditions import PERIODIC
from weno9.weno_fv import weno9_fv




# ==========================================================
# Smooth vector field
# ==========================================================

def make_field(nx):

    dx = 1.0 / nx

    x = (
        jnp.arange(nx) + 0.5
    ) * dx

    y = (
        jnp.arange(nx) + 0.5
    ) * dx

    # Cell boundaries
    xL = x - 0.5 * dx
    xR = x + 0.5 * dx

    yL = y - 0.5 * dx
    yR = y + 0.5 * dx

    # ------------------------------------------------------
    # Exact 1D cell averages
    # ------------------------------------------------------

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

    return U, dx


# ==========================================================
# Exact point values
# ==========================================================

def exact_field(x, y):

    u = (
        jnp.sin(2.0 * jnp.pi * x)
        * jnp.cos(2.0 * jnp.pi * y)
    )

    v = (
        jnp.cos(2.0 * jnp.pi * x)
        * jnp.sin(2.0 * jnp.pi * y)
    )

    return jnp.stack(
        (
            u,
            v,
        ),
        axis=0,
    )


# ==========================================================
# Exact physical flux
# ==========================================================

def exact_flux(U, component):

    velocity = U[component]

    return (
        velocity[None, ...]
        * U
    )


# ==========================================================
# One resolution
# ==========================================================

def run(nx):

    ng = 5

    U, dx = make_field(nx)

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

    Fx = vector_burgers_rusanov_flux(
        Ux_L,
        Ux_R,
        component=0,
    )

    # ------------------------------------------------------
    # Exact X-face point values
    #
    # x = interfaces
    # y = cell centers
    # ------------------------------------------------------

    x_face = (
        jnp.arange(nx + 1)
        * dx
    )

    y_center = (
        jnp.arange(nx) + 0.5
    ) * dx

    Xf, Yc = jnp.meshgrid(
        x_face,
        y_center,
        indexing="ij",
    )

    U_exact_x = exact_field(
        Xf,
        Yc,
    )

    Fx_exact = exact_flux(
        U_exact_x,
        component=0,
    )

    error_x = jnp.max(
        jnp.abs(
            Fx - Fx_exact
        )
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

    Fy = vector_burgers_rusanov_flux(
        Uy_L,
        Uy_R,
        component=1,
    )

    # ------------------------------------------------------
    # Exact Y-face point values
    #
    # x = cell centers
    # y = interfaces
    # ------------------------------------------------------

    x_center = (
        jnp.arange(nx) + 0.5
    ) * dx

    y_face = (
        jnp.arange(nx + 1)
        * dx
    )

    Xc, Yf = jnp.meshgrid(
        x_center,
        y_face,
        indexing="ij",
    )

    U_exact_y = exact_field(
        Xc,
        Yf,
    )

    Fy_exact = exact_flux(
        U_exact_y,
        component=1,
    )

    error_y = jnp.max(
        jnp.abs(
            Fy - Fy_exact
        )
    )

    # ======================================================
    # Combined error
    # ======================================================

    error = jnp.maximum(
        error_x,
        error_y,
    )

    return (
        float(error),
        float(error_x),
        float(error_y),
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

    print()
    print("=" * 72)
    print("VECTOR WENO9 + RUSANOV FLUX")
    print("=" * 72)

    print(
        f"{'N':>6}"
        f"{'error':>20}"
        f"{'order':>12}"
        f"{'error_x':>18}"
        f"{'error_y':>18}"
    )

    print("-" * 88)

    previous = None

    for nx in resolutions:

        error, error_x, error_y = run(nx)

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
            f"{error_x:18.8e}"
            f"{error_y:18.8e}"
        )

        previous = error

    print("-" * 88)


if __name__ == "__main__":
    main()