import jax.numpy as jnp

from weno9.fast_betas import compute_beta
from weno9.reconstruction import (
    get_stencils,
    nonlinear_weights,
    reconstruct_positive,
)


def reconstruct_from_stencils(stencils):
    beta = compute_beta(stencils)
    weights = nonlinear_weights(beta)

    return reconstruct_positive(
        stencils,
        weights,
    )


def weno9_fv_left(u, axis, ng=5):
    """
    LEFT state at all N+1 physical interfaces.

    Interfaces:

        x = 0, dx, ..., N*dx
    """

    n = u.shape[axis] - 2 * ng

    stencil = get_stencils(
        u,
        axis,
        ng=ng,
        start=ng - 1,
        stop=ng + n,
    )

    return reconstruct_from_stencils(stencil)


def weno9_fv_right(u, axis, ng=5):
    """
    RIGHT state at all N+1 physical interfaces.

    Obtained by reversing the spatial direction.
    """

    uf = jnp.flip(
        u,
        axis=axis,
    )

    ur = weno9_fv_left(
        uf,
        axis=axis,
        ng=ng,
    )

    return jnp.flip(
        ur,
        axis=axis,
    )


def weno9_fv(u, axis, ng=5):
    """
    Full WENO9 FV reconstruction.

    Returns:

        uL.shape == (N+1,)
        uR.shape == (N+1,)

    where interface i is:

        x = i * dx
    """

    uL = weno9_fv_left(
        u,
        axis=axis,
        ng=ng,
    )

    uR = weno9_fv_right(
        u,
        axis=axis,
        ng=ng,
    )

    return uL, uR