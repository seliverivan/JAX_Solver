import jax.numpy as jnp

from weno9.constants import C, D, EPS


# ==========================================================
# OLD: periodic roll-based stencils
# ==========================================================

def get_stencils_old(u, axis):

    return jnp.stack([
        jnp.stack([
            jnp.roll(u, 4, axis=axis),
            jnp.roll(u, 3, axis=axis),
            jnp.roll(u, 2, axis=axis),
            jnp.roll(u, 1, axis=axis),
            u,
        ]),

        jnp.stack([
            jnp.roll(u, 3, axis=axis),
            jnp.roll(u, 2, axis=axis),
            jnp.roll(u, 1, axis=axis),
            u,
            jnp.roll(u, -1, axis=axis),
        ]),

        jnp.stack([
            jnp.roll(u, 2, axis=axis),
            jnp.roll(u, 1, axis=axis),
            u,
            jnp.roll(u, -1, axis=axis),
            jnp.roll(u, -2, axis=axis),
        ]),

        jnp.stack([
            jnp.roll(u, 1, axis=axis),
            u,
            jnp.roll(u, -1, axis=axis),
            jnp.roll(u, -2, axis=axis),
            jnp.roll(u, -3, axis=axis),
        ]),

        jnp.stack([
            u,
            jnp.roll(u, -1, axis=axis),
            jnp.roll(u, -2, axis=axis),
            jnp.roll(u, -3, axis=axis),
            jnp.roll(u, -4, axis=axis),
        ]),
    ])


# ==========================================================
# NEW: ghost-cell stencils
# ==========================================================

def get_stencils(
    u,
    axis,
    ng=5,
    start=None,
    stop=None,
):
    """
    Construct the five 5-point FV WENO9 stencils.

    The returned stencils correspond to interfaces whose
    left-adjacent cell indices are given by [start, stop).

    For N physical cells:

        start = ng
        stop  = ng + N

    gives the N internal/right interfaces:

        x = dx, ..., x = N*dx

    To include both physical boundary interfaces:

        start = ng - 1
        stop  = ng + N

    gives N+1 interfaces:

        x = 0, dx, ..., N*dx
    """

    n = u.shape[axis] - 2 * ng

    if start is None:
        start = ng

    if stop is None:
        stop = ng + n

    indices = jnp.arange(start, stop)

    return jnp.stack([
        jnp.stack([
            jnp.take(
                u,
                indices + offset,
                axis=axis,
            )
            for offset in range(-4, 1)
        ]),

        jnp.stack([
            jnp.take(
                u,
                indices + offset,
                axis=axis,
            )
            for offset in range(-3, 2)
        ]),

        jnp.stack([
            jnp.take(
                u,
                indices + offset,
                axis=axis,
            )
            for offset in range(-2, 3)
        ]),

        jnp.stack([
            jnp.take(
                u,
                indices + offset,
                axis=axis,
            )
            for offset in range(-1, 4)
        ]),

        jnp.stack([
            jnp.take(
                u,
                indices + offset,
                axis=axis,
            )
            for offset in range(0, 5)
        ]),
    ])

# ==========================================================
# Reconstruction
# ==========================================================

def nonlinear_weights(beta):

    shape = (5,) + (1,) * (beta.ndim - 1)

    alpha = D.reshape(shape) / (beta + EPS) ** 2

    return alpha / jnp.sum(
        alpha,
        axis=0,
    )


def reconstruct_positive(stencils, weights):

    poly = jnp.einsum(
        'ki...,ki...->k...',
        C,
        stencils,
    )

    return jnp.sum(
        weights * poly,
        axis=0,
    )


def reconstruct_linear(stencils):

    poly = jnp.einsum(
        'ki,kin->kn',
        C,
        stencils,
    )

    return jnp.sum(
        D[:, None] * poly,
        axis=0,
    )

