import jax.numpy as jnp


# ==========================================================
# Скалярное уравнение Бюргерса
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
# Векторное уравнение Бюргерса
# ==========================================================

def vector_burgers_rusanov_flux(
    uL,
    uR,
    component,
):
    """
    Численный поток Русанова для векторного уравнения Бюргерса.

    Соглашение о представлении состояния:

        2D:
            U.shape = (2, ...)

        3D:
            U.shape = (3, ...)

    Первая ось содержит компоненты вектора:

        U[0] = u
        U[1] = v
        U[2] = w

    Parameters
    ----------
    uL, uR :
        Реконструированные векторные состояния слева и справа.

        Shape:

            (n_components, ...)

    component :
        Направление и компонента потока:

            0 -> F_x = u * U
            1 -> F_y = v * U
            2 -> F_z = w * U

    Returns
    -------
    F :
        Векторный численный поток.

        Имеет ту же форму, что uL и uR.
    """

    # ------------------------------------------------------
    # Скорость в направлении потока
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
    # Физический поток
    #
    # F = U_component * U
    #
    # velocity имеет форму (...)
    # U имеет форму (components, ...)
    #
    # Восстанавливаем ось компонент для broadcasting.
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
    # Максимальная характеристическая скорость
    # ------------------------------------------------------

    alpha = jnp.maximum(
        jnp.abs(velocity_L),
        jnp.abs(velocity_R),
    )

    # ------------------------------------------------------
    # Поток Русанова
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
