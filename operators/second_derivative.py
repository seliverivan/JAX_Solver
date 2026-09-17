import jax.numpy as jnp


def second_derivative(
    u,
    dx,
    axis,
    ng=4,
):

    n = u.shape[axis] - 2 * ng

    center = jnp.arange(
        ng,
        ng + n,
    )

    result = (
        -1 / 560 * jnp.take(u, center - 4, axis=axis)
        + 8 / 315 * jnp.take(u, center - 3, axis=axis)
        - 1 / 5 * jnp.take(u, center - 2, axis=axis)
        + 8 / 5 * jnp.take(u, center - 1, axis=axis)
        - 205 / 72 * jnp.take(u, center, axis=axis)
        + 8 / 5 * jnp.take(u, center + 1, axis=axis)
        - 1 / 5 * jnp.take(u, center + 2, axis=axis)
        + 8 / 315 * jnp.take(u, center + 3, axis=axis)
        - 1 / 560 * jnp.take(u, center + 4, axis=axis)
    )

    return result / (dx * dx)