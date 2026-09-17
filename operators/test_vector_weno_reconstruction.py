import jax
import jax.numpy as jnp

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

    x = (jnp.arange(nx) + 0.5) * dx
    y = (jnp.arange(nx) + 0.5) * dx

    X, Y = jnp.meshgrid(
        x,
        y,
        indexing="ij",
    )

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

    u = avg_sin_x[:, None] * avg_cos_y[None, :]
    v = avg_cos_x[:, None] * avg_sin_y[None, :]

    U = jnp.stack(
        (
            u,
            v,
        ),
        axis=0,
    )

    return U, X, Y, dx


# ==========================================================
# Exact values at interfaces
# ==========================================================

def exact_interface_values(nx):

    dx = 1.0 / nx

    x = (
        jnp.arange(nx) + 0.5
    ) * dx

    y = (
        jnp.arange(nx) + 0.5
    ) * dx

    # ------------------------------------------------------
    # 1D cell averages in transverse directions
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
    #
    # x = interface
    # y = cell average
    # ======================================================

    x_face = jnp.arange(nx + 1) * dx

    Xf, Yc = jnp.meshgrid(
        x_face,
        y,
        indexing="ij",
    )

    u_x = (
        jnp.sin(2.0 * jnp.pi * Xf)
        * avg_cos_y[None, :]
    )

    v_x = (
        jnp.cos(2.0 * jnp.pi * Xf)
        * avg_sin_y[None, :]
    )

    exact_x = jnp.stack(
        (
            u_x,
            v_x,
        ),
        axis=0,
    )

    # ======================================================
    # Y faces
    #
    # x = cell average
    # y = interface
    # ======================================================

    y_face = jnp.arange(nx + 1) * dx

    Xc, Yf = jnp.meshgrid(
        x,
        y_face,
        indexing="ij",
    )

    u_y = (
        avg_sin_x[:, None]
        * jnp.cos(2.0 * jnp.pi * Yf)
    )

    v_y = (
        avg_cos_x[:, None]
        * jnp.sin(2.0 * jnp.pi * Yf)
    )

    exact_y = jnp.stack(
        (
            u_y,
            v_y,
        ),
        axis=0,
    )

    return exact_x, exact_y

def test_scalar_vs_vector(nx):

    ng = 5

    U, _, _, dx = make_field(nx)

    # ------------------------------------------------------
    # Scalar
    # ------------------------------------------------------

    u = U[0]

    u_bc = apply_boundary(
        u,
        axis=0,
        left_type=PERIODIC,
        right_type=PERIODIC,
        ng=ng,
        dx=dx,
    )

    u_L, u_R = weno9_fv(
        u_bc,
        axis=0,
        ng=ng,
    )

    # ------------------------------------------------------
    # Vector
    # ------------------------------------------------------

    U_bc = apply_boundary(
        U,
        axis=1,
        left_type=PERIODIC,
        right_type=PERIODIC,
        ng=ng,
        dx=dx,
    )

    U_L, U_R = weno9_fv(
        U_bc,
        axis=1,
        ng=ng,
    )

    # ------------------------------------------------------
    # Compare scalar and vector component
    # ------------------------------------------------------

    scalar_vector_difference = jnp.max(
        jnp.abs(
            u_L - U_L[0]
        )
    )

    print(
        "scalar/vector difference =",
        float(scalar_vector_difference),
    )

# ==========================================================
# One resolution
# ==========================================================

def run(nx):

    ng = 5

    U, X, Y, dx = make_field(nx)

    # ======================================================
    # X reconstruction
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

    # ======================================================
    # Y reconstruction
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

    # ======================================================
    # Exact interface values
    # ======================================================

    exact_x, exact_y = exact_interface_values(nx)

    # ======================================================
    # Reconstruction errors
    # ======================================================

    error_x_L = jnp.max(
        jnp.abs(
            Ux_L - exact_x
        )
    )

    error_x_R = jnp.max(
        jnp.abs(
            Ux_R - exact_x
        )
    )

    error_y_L = jnp.max(
        jnp.abs(
            Uy_L - exact_y
        )
    )

    error_y_R = jnp.max(
        jnp.abs(
            Uy_R - exact_y
        )
    )

    error = jnp.maximum(
        jnp.maximum(
            error_x_L,
            error_x_R,
        ),
        jnp.maximum(
            error_y_L,
            error_y_R,
        ),
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
    print("VECTOR WENO9 RECONSTRUCTION")
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