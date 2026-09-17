import jax.numpy as jnp


# ==========================================================
# Scalar Burgers
# ==========================================================

def burgers_flux(u):
    return 0.5 * u**2


def burgers_rusanov_flux(uL, uR):

    alpha = jnp.maximum(
        jnp.abs(uL),
        jnp.abs(uR),
    )

    return (
        0.5 * (
            burgers_flux(uL)
            + burgers_flux(uR)
        )
        - 0.5 * alpha * (uR - uL)
    )


# ==========================================================
# Vector Burgers
# ==========================================================

def vector_burgers_rusanov_flux(
    uL,
    uR,
    component,
):
    """
    Rusanov numerical flux for vector Burgers.

    State convention:

        2D:
            U.shape = (2, ...)

        3D:
            U.shape = (3, ...)

    The first axis stores vector components:

        U[0] = u
        U[1] = v
        U[2] = w

    Parameters
    ----------
    uL, uR :
        Left and right reconstructed vector states.

        Shape:

            (n_components, ...)

    component :
        Direction/component of the flux:

            0 -> F_x = u * U
            1 -> F_y = v * U
            2 -> F_z = w * U

    Returns
    -------
    F :
        Vector numerical flux.

        Same shape as uL and uR.
    """

    # ------------------------------------------------------
    # Velocity in the direction of the flux
    #
    # component = 0:
    #     velocity = u
    #
    # component = 1:
    #     velocity = v
    #
    # component = 2:
    #     velocity = w
    # ------------------------------------------------------

    velocity_L = uL[component]
    velocity_R = uR[component]

    # ------------------------------------------------------
    # Physical flux
    #
    # F = U_component * U
    #
    # velocity has shape (...)
    # U has shape (components, ...)
    #
    # Restore component axis for broadcasting.
    # ------------------------------------------------------

    velocity_L = jnp.expand_dims(
        velocity_L,
        axis=0,
    )

    velocity_R = jnp.expand_dims(
        velocity_R,
        axis=0,
    )

    physical_flux_L = (
        velocity_L * uL
    )

    physical_flux_R = (
        velocity_R * uR
    )

    # ------------------------------------------------------
    # Maximum characteristic speed
    # ------------------------------------------------------

    alpha = jnp.maximum(
        jnp.abs(velocity_L),
        jnp.abs(velocity_R),
    )

    # ------------------------------------------------------
    # Rusanov flux
    # ------------------------------------------------------

    return (
        0.5 * (
            physical_flux_L
            + physical_flux_R
        )
        - 0.5 * alpha * (
            uR - uL
        )
    )