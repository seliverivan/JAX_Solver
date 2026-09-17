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
    # 1. WENO reconstruction in the normal direction
    # ------------------------------------------------------

    U_L, U_R = weno9_fv(
        U,
        axis=axis,
        ng=ng,
    )

    # ------------------------------------------------------
    # At this point:
    #
    # 2D:
    #
    #     U_L.shape = (ncomp, Nface, Ntrans)
    #
    # where Ntrans is the transverse cell index.
    #
    # We now reconstruct along the transverse direction.
    # ------------------------------------------------------

    ndim = U.ndim - 1

    if ndim != 2:
        raise NotImplementedError(
            "Current implementation supports 2D only."
        )

    # ------------------------------------------------------
    # Determine transverse axis.
    #
    # Spatial axes:
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
    # WENO reconstruction along transverse direction
    # ------------------------------------------------------

    # We need values at arbitrary points inside each
    # transverse cell.
    #
    # For now use the existing WENO reconstruction
    # to obtain interface values. The actual arbitrary
    # Gauss-point reconstruction will be added next.
    #
    # This function is intentionally kept as the
    # architectural entry point.

    raise NotImplementedError(
        "Quadrature-point reconstruction is the next step."
    )