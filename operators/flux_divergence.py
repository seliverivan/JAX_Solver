import jax.numpy as jnp

from weno9.weno_fv import weno9_fv


# ==========================================================
# Дивергенция скалярного потока
# ==========================================================

def flux_divergence(
    u,
    dx,
    axis,
    flux,
    ng=5,
):
    """
    Конечно-объёмная дивергенция численного потока скалярного поля.
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
# Дивергенция векторного потока
# ==========================================================

def vector_flux_divergence(
    u,
    dx,
    axis,
    component,
    ng=5,
):
    """
    Конечно-объёмная дивергенция одного потока векторного уравнения Бюргерса.

    Соглашение о представлении состояния:

        2D:
            u.shape = (2, Nx, Ny)

        3D:
            u.shape = (3, Nx, Ny, Nz)

    axis:
        пространственное направление

            0 -> x
            1 -> y
            2 -> z

    component:
        компонента скорости, определяющая поток

            0 -> F_x = u * U
            1 -> F_y = v * U
            2 -> F_z = w * U

    Returns
    -------
    divergence:
        Векторное поле той же формы, что физическая часть u.
    """

    # ------------------------------------------------------
    # Реконструкция WENO
    #
    # ВАЖНО:
    #
    # weno9_fv работает вдоль `axis`.
    # Ось компонент сохраняется.
    # ------------------------------------------------------

    uL, uR = weno9_fv(
        u,
        axis=axis,
        ng=ng,
    )

    # ------------------------------------------------------
    # Векторный поток Русанова
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
    # Разность потоков
    #
    # F содержит N+1 граней вдоль `axis`.
    #
    # Поэтому:
    #
    # div[i] = (F[i+1] - F[i]) / dx
    #
    # Ось компонент не изменяется.
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
