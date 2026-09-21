import jax.numpy as jnp

from weno9.weno_fv import weno9_fv


def reconstruct_face_quadrature(
    U,
    axis,
    quadrature_points,
    ng,
):
    """
    Reconstruct WENO9 states on face quadrature points.

    Parameters
    ----------
    U :
        Cell averages with shape

            (n_components, ...)

        The first axis stores vector components.

    axis :
        Physical spatial axis.

        Since axis 0 stores vector components,
        spatial axes are:

            axis=1 -> x
            axis=2 -> y
            axis=3 -> z

    quadrature_points :
        Quadrature points in the transverse direction.

        Shape:

            (Nq,)

        Values are in the reference cell:

            [-1/2, 1/2]

    ng :
        Number of ghost cells.

    Returns
    -------
    U_L, U_R :
        Reconstructed states on quadrature points.

        For a 2D input:

            U_L.shape =
                (n_components, Nface, Nq)

            U_R.shape =
                (n_components, Nface, Nq)

        The last axis corresponds to transverse
        quadrature points.
    """

    # ------------------------------------------------------
    # 1. Реконструкция WENO в направлении нормали
    # ------------------------------------------------------

    U_L, U_R = weno9_fv(
        U,
        axis=axis,
        ng=ng,
    )

    # ------------------------------------------------------
    # На этом этапе:
    #
    # 2D:
    #
    #     U_L.shape = (ncomp, Nface, Ntrans)
    #
    # где Ntrans — индекс ячейки в поперечном направлении.
    #
    # Далее выполняем реконструкцию в поперечном направлении.
    # ------------------------------------------------------

    ndim = U.ndim - 1

    if ndim != 2:
        raise NotImplementedError(
            "Current implementation supports 2D only."
        )

    # ------------------------------------------------------
    # Определяем поперечную ось.
    #
    # Пространственные оси:
    #
    #     axis=1 -> x
    #     axis=2 -> y
    # ------------------------------------------------------

    if axis == 1:
        transverse_axis = 2

    elif axis == 2:
        transverse_axis = 1

    else:
        raise ValueError(
            f"Invalid spatial axis: {axis}"
        )

    # ------------------------------------------------------
    # Реконструкция WENO в поперечном направлении
    # ------------------------------------------------------

    # Требуются значения в произвольных точках внутри каждой
    # поперечной ячейки.
    #
    # Пока используем существующую реконструкцию WENO для получения
    # значений на гранях. Реконструкция в произвольных точках Гаусса
    # будет добавлена позднее.
    #
    # Эта функция сохранена как архитектурная точка входа.

    raise NotImplementedError(
        "Quadrature-point reconstruction is the next step."
    )
