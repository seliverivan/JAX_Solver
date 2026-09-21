import jax.numpy as jnp


# ==========================================================
# Конечно-объёмный полином на шаблоне из 9 ячеек
# ==========================================================

def _cell_average_moments(
    cell_centers,
    degree,
):
    """
    Moments of monomials over a cell.

    Cell:
        [x - 1/2, x + 1/2]

    Returns matrix A such that

        A[i, k] = average(x^k)

    for the cell centered at cell_centers[i].
    """

    moments = []

    for k in range(degree + 1):

        moment = (
            (cell_centers + 0.5) ** (k + 1)
            - (cell_centers - 0.5) ** (k + 1)
        ) / (k + 1)

        moments.append(moment)

    return jnp.stack(
        moments,
        axis=1,
    )


# ==========================================================
# Коэффициенты для вычисления значения в точке
# ==========================================================

def point_reconstruction_coefficients(
    xi,
):
    """
    Coefficients for evaluating the unique degree-8
    FV reconstruction from 9 cell averages.

    The 9 cells are centered at

        -4, -3, ..., 0, ..., 3, 4

    relative to the target cell.

    Parameters
    ----------
    xi :
        Reference coordinate inside the target cell.

        xi = 0       -> cell center
        xi = -1/2    -> left interface
        xi = +1/2    -> right interface

    Returns
    -------
    coeff :
        Shape (9,)

        Such that

            u(xi) ≈ sum_j coeff[j] * ubar[j]
    """

    centers = jnp.arange(
        -4,
        5,
        dtype=jnp.float64,
    )

    A = _cell_average_moments(
        centers,
        degree=8,
    )

    powers = jnp.arange(
        9,
        dtype=jnp.float64,
    )

    evaluation = xi ** powers

    # A.T @ c = значение в точке
    coeff = jnp.linalg.solve(
        A.T,
        evaluation,
    )

    return coeff


# ==========================================================
# Реконструкция в точке
# ==========================================================

def reconstruct_point(
    stencils,
    xi,
):
    """
    Evaluate the degree-8 FV polynomial at xi.

    Parameters
    ----------
    stencils :
        Last stencil dimension contains 9 cells.

        Expected logical shape:

            (..., 9)

    xi :
        Scalar point inside the reference cell.

    Returns
    -------
    value :
        Reconstructed point value.
    """

    coeff = point_reconstruction_coefficients(
        xi,
    )

    return jnp.einsum(
        "...k,k->...",
        stencils,
        coeff,
    )


# ==========================================================
# Набор квадратурных точек
# ==========================================================

def reconstruct_points(
    stencils,
    points,
):
    """
    Reconstruct at multiple points.

    Parameters
    ----------
    stencils :
        (..., 9)

    points :
        (Nq,)

    Returns
    -------
    values :
        (..., Nq)
    """

    coeffs = jnp.stack(
        [
            point_reconstruction_coefficients(xi)
            for xi in points
        ],
        axis=0,
    )

    return jnp.einsum(
        "...k,qk->...q",
        stencils,
        coeffs,
    )
