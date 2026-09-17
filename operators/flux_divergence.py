import jax.numpy as jnp

from weno9.weno_fv import weno9_fv


# ==========================================================
# Scalar flux divergence
# ==========================================================

def flux_divergence(
    u,
    dx,
    axis,
    flux,
    ng=5,
):
    """
    Finite-volume numerical flux divergence for scalar field.
    """

    uL, uR = weno9_fv(
        u,
        axis=axis,
        ng=ng,
    )

    F = flux(
        uL,
        uR,
    )

    return (
        jnp.take(
            F,
            indices=jnp.arange(
                1,
                F.shape[axis],
            ),
            axis=axis,
        )
        -
        jnp.take(
            F,
            indices=jnp.arange(
                0,
                F.shape[axis] - 1,
            ),
            axis=axis,
        )
    ) / dx


# ==========================================================
# Vector flux divergence
# ==========================================================

def vector_flux_divergence(
    u,
    dx,
    axis,
    component,
    ng=5,
):
    """
    Finite-volume divergence of one vector Burgers flux.

    State convention:

        2D:
            u.shape = (2, Nx, Ny)

        3D:
            u.shape = (3, Nx, Ny, Nz)

    axis:
        spatial direction

            0 -> x
            1 -> y
            2 -> z

    component:
        velocity component defining the flux

            0 -> F_x = u * U
            1 -> F_y = v * U
            2 -> F_z = w * U

    Returns
    -------
    divergence:
        Vector field with the same shape as the physical
        part of u.
    """

    # ------------------------------------------------------
    # WENO reconstruction
    #
    # IMPORTANT:
    #
    # weno9_fv operates along `axis`.
    # Component axis is preserved.
    # ------------------------------------------------------

    uL, uR = weno9_fv(
        u,
        axis=axis,
        ng=ng,
    )

    # ------------------------------------------------------
    # Vector Rusanov flux
    # ------------------------------------------------------

    from operators.flux import (
        vector_burgers_rusanov_flux,
    )

    F = vector_burgers_rusanov_flux(
        uL,
        uR,
        component=component,
    )

    # ------------------------------------------------------
    # Flux difference
    #
    # F has N+1 interfaces along `axis`.
    #
    # Therefore:
    #
    # div[i] = (F[i+1] - F[i]) / dx
    #
    # Component axis is untouched.
    # ------------------------------------------------------

    right_flux = jnp.take(
        F,
        indices=jnp.arange(
            1,
            F.shape[axis],
        ),
        axis=axis,
    )

    left_flux = jnp.take(
        F,
        indices=jnp.arange(
            0,
            F.shape[axis] - 1,
        ),
        axis=axis,
    )

    return (
        right_flux - left_flux
    ) / dx