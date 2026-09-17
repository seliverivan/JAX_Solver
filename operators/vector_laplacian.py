import jax.numpy as jnp

from operators.second_derivative import second_derivative


# ============================================================
# 2D VECTOR LAPLACIAN
#
# u_ext[..., 0] = u
# u_ext[..., 1] = v
#
# u_ext already contains ghost cells.
#
# Δu = u_xx + u_yy
# Δv = v_xx + v_yy
# ============================================================

def vector_laplacian_2d(
    u_ext,
    dx,
    dy,
    ng=4,
):

    result = []

    for component in range(2):

        q = u_ext[..., component]

        # ----------------------------------------------------
        # d²/dx²
        #
        # second_derivative removes ghost cells only along
        # the differentiation axis.
        #
        # Therefore ghost cells along y remain.
        # ----------------------------------------------------

        dxx = second_derivative(
            q,
            dx,
            axis=0,
            ng=ng,
        )

        # dxx: (nx, ny + 2*ng)
        # remove y ghost cells
        dxx = dxx[
            :,
            ng:-ng,
        ]

        # ----------------------------------------------------
        # d²/dy²
        # ----------------------------------------------------

        dyy = second_derivative(
            q,
            dy,
            axis=1,
            ng=ng,
        )

        # dyy: (nx + 2*ng, ny)
        # remove x ghost cells
        dyy = dyy[
            ng:-ng,
            :,
        ]

        # ----------------------------------------------------
        # Now both are (nx, ny)
        # ----------------------------------------------------

        result.append(
            dxx + dyy
        )

    return jnp.stack(
        result,
        axis=-1,
    )


# ============================================================
# 3D VECTOR LAPLACIAN
#
# u_ext[..., 0] = u
# u_ext[..., 1] = v
# u_ext[..., 2] = w
#
# u_ext already contains ghost cells.
#
# Δu = u_xx + u_yy + u_zz
# ============================================================

def vector_laplacian_3d(
    u_ext,
    dx,
    dy,
    dz,
    ng=4,
):

    result = []

    for component in range(3):

        q = u_ext[..., component]

        # ----------------------------------------------------
        # d²/dx²
        # ----------------------------------------------------

        dxx = second_derivative(
            q,
            dx,
            axis=0,
            ng=ng,
        )

        # dxx:
        # (nx, ny + 2*ng, nz + 2*ng)
        #
        # remove y and z ghost cells

        dxx = dxx[
            :,
            ng:-ng,
            ng:-ng,
        ]

        # ----------------------------------------------------
        # d²/dy²
        # ----------------------------------------------------

        dyy = second_derivative(
            q,
            dy,
            axis=1,
            ng=ng,
        )

        # dyy:
        # (nx + 2*ng, ny, nz + 2*ng)
        #
        # remove x and z ghost cells

        dyy = dyy[
            ng:-ng,
            :,
            ng:-ng,
        ]

        # ----------------------------------------------------
        # d²/dz²
        # ----------------------------------------------------

        dzz = second_derivative(
            q,
            dz,
            axis=2,
            ng=ng,
        )

        # dzz:
        # (nx + 2*ng, ny + 2*ng, nz)
        #
        # remove x and y ghost cells

        dzz = dzz[
            ng:-ng,
            ng:-ng,
            :,
        ]

        # ----------------------------------------------------
        # Now all three are:
        #
        # (nx, ny, nz)
        # ----------------------------------------------------

        result.append(
            dxx + dyy + dzz
        )

    return jnp.stack(
        result,
        axis=-1,
    )