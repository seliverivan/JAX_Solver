import jax
import jax.numpy as jnp

jax.config.update(
    "jax_enable_x64",
    True,
)

from boundary.conditions import PERIODIC
from rhs.vector_burgers import vector_burgers_rhs


# ==========================================================
# Smooth separable vector field
#
# U_c(x_0,...,x_{d-1})
#     = product_k f_{c+k}(x_k)
#
# even index -> sin
# odd  index -> cos
# ==========================================================


def basis_value(index, x):

    if index % 2 == 0:
        return jnp.sin(
            2.0 * jnp.pi * x
        )

    return jnp.cos(
        2.0 * jnp.pi * x
    )


def basis_average(index, xL, xR):

    k = 2.0 * jnp.pi
    dx = xR - xL

    if index % 2 == 0:

        return (
            jnp.cos(k * xL)
            - jnp.cos(k * xR)
        ) / (k * dx)

    return (
        jnp.sin(k * xR)
        - jnp.sin(k * xL)
    ) / (k * dx)


def basis_second_average(index, xL, xR):

    k = 2.0 * jnp.pi

    return -k * k * basis_average(
        index,
        xL,
        xR,
    )


# ==========================================================
# Coordinates
# ==========================================================


def coordinates(n, ndim):

    dx = 1.0 / n

    x = (
        jnp.arange(n) + 0.5
    ) * dx

    xL = x - 0.5 * dx
    xR = x + 0.5 * dx

    return x, xL, xR, dx


# ==========================================================
# Cell averages
# ==========================================================


def make_cell_averages(ndim, n):

    x, xL, xR, dx = coordinates(
        n,
        ndim,
    )

    U = []

    for component in range(ndim):

        field = 1.0

        for axis in range(ndim):

            factor = basis_average(
                component + axis,
                xL,
                xR,
            )

            shape = [1] * ndim
            shape[axis] = n

            factor = factor.reshape(shape)

            field = field * factor

        U.append(field)

    return jnp.stack(U, axis=0), dx


# ==========================================================
# Exact smooth field at cell centers
# ==========================================================


def make_point_field(ndim, n):

    x, _, _, _ = coordinates(
        n,
        ndim,
    )

    grids = jnp.meshgrid(
        *([x] * ndim),
        indexing="ij",
    )

    U = []

    for component in range(ndim):

        field = 1.0

        for axis in range(ndim):

            field = (
                field
                * basis_value(
                    component + axis,
                    grids[axis],
                )
            )

        U.append(field)

    return jnp.stack(U, axis=0)


# ==========================================================
# Exact Laplacian cell average
# ==========================================================


def exact_laplacian(ndim, n):

    _, xL, xR, _ = coordinates(
        n,
        ndim,
    )

    result = []

    for component in range(ndim):

        total = 0.0

        for direction in range(ndim):

            field = 1.0

            for axis in range(ndim):

                index = component + axis

                shape = [1] * ndim
                shape[axis] = n

                if axis == direction:

                    factor = basis_second_average(
                        index,
                        xL,
                        xR,
                    )

                else:

                    factor = basis_average(
                        index,
                        xL,
                        xR,
                    )

                factor = factor.reshape(shape)

                field = field * factor

            total = total + field

        result.append(total)

    return jnp.stack(
        result,
        axis=0,
    )


# ==========================================================
# Exact FV flux
#
# IMPORTANT:
#
# This is the flux representation consistent with the
# dimension-by-dimension FV operator:
#
#   F_d = U_d * U
#
# where transverse directions remain represented by
# cell averages.
# ==========================================================

# ==========================================================
# Exact FV flux
# ==========================================================

def exact_flux(
    ndim,
    n,
    direction,
):

    x, xL, xR, dx = coordinates(
        n,
        ndim,
    )

    face = jnp.arange(n + 1) * dx

    flux_components = []

    for component in range(ndim):

        field = 1.0

        for axis in range(ndim):

            index_d = direction + axis
            index_c = component + axis

            if axis == direction:

                # --------------------------------------------------
                # Normal direction:
                # point value at face
                # --------------------------------------------------

                fd = basis_value(
                    index_d,
                    face,
                )

                fc = basis_value(
                    index_c,
                    face,
                )

                factor = fd * fc

                shape = [1] * ndim
                shape[axis] = n + 1

                factor = factor.reshape(shape)

            else:

                # --------------------------------------------------
                # Transverse direction:
                # cell average
                # --------------------------------------------------

                fd = basis_average(
                    index_d,
                    xL,
                    xR,
                )

                fc = basis_average(
                    index_c,
                    xL,
                    xR,
                )

                factor = fd * fc

                shape = [1] * ndim
                shape[axis] = n

                factor = factor.reshape(shape)

            field = field * factor

        flux_components.append(field)

    return jnp.stack(
        flux_components,
        axis=0,
    )


# ==========================================================
# Exact FV convection divergence
# ==========================================================

def exact_convection(
    ndim,
    n,
):

    dx = 1.0 / n

    result = []

    for component in range(ndim):

        total = None

        for direction in range(ndim):

            F = exact_flux(
                ndim,
                n,
                direction,
            )[component]

            # --------------------------------------------------
            # F has ndim spatial axes.
            #
            # direction 0 -> axis 0
            # direction 1 -> axis 1
            # direction 2 -> axis 2
            # --------------------------------------------------

            right = jnp.take(
                F,
                jnp.arange(
                    1,
                    n + 1,
                ),
                axis=direction,
            )

            left = jnp.take(
                F,
                jnp.arange(
                    0,
                    n,
                ),
                axis=direction,
            )

            contribution = (
                right - left
            ) / dx

            if total is None:
                total = contribution
            else:
                total = total + contribution

        result.append(total)

    return jnp.stack(
        result,
        axis=0,
    )

# ==========================================================
# Exact RHS
# ==========================================================


def exact_rhs(
    ndim,
    n,
    mu,
):

    return (
        -exact_convection(
            ndim,
            n,
        )
        + mu * exact_laplacian(
            ndim,
            n,
        )
    )


# ==========================================================
# Numerical RHS
# ==========================================================


def numerical_rhs(
    ndim,
    n,
    mu,
):

    U, dx = make_cell_averages(
        ndim,
        n,
    )

    boundary_types = tuple(
        (
            PERIODIC,
            PERIODIC,
        )
        for _ in range(ndim)
    )

    boundary_values = tuple(
        (
            None,
            None,
        )
        for _ in range(ndim)
    )

    return vector_burgers_rhs(
        U,
        (dx,) * ndim,
        mu,
        boundary_types,
        boundary_values,
        ng=5,
    )


# ==========================================================
# One resolution
# ==========================================================


def run(
    ndim,
    n,
    mu,
):

    numerical = numerical_rhs(
        ndim,
        n,
        mu,
    )

    exact = exact_rhs(
        ndim,
        n,
        mu,
    )

    error = jnp.max(
        jnp.abs(
            numerical - exact
        )
    )

    return float(error)


# ==========================================================
# Test dimension
# ==========================================================


def test_dimension(ndim):

    resolutions = [
        8,
        16,
        32,
        64,
    ]

    mu = 0.1

    print()
    print("=" * 72)
    print(
        f"VECTOR BURGERS RHS {ndim}D"
    )
    print("=" * 72)

    print(
        f"{'N':>6}"
        f"{'error':>20}"
        f"{'order':>12}"
    )

    print("-" * 72)

    previous = None

    for n in resolutions:

        error = run(
            ndim,
            n,
            mu,
        )

        if previous is None:

            order = 0.0

        else:

            order = float(
                jnp.log2(
                    previous / error
                )
            )

        print(
            f"{n:6d}"
            f"{error:20.8e}"
            f"{order:12.4f}"
        )

        previous = error

    print("-" * 72)


# ==========================================================
# Main
# ==========================================================


def main():

    test_dimension(1)
    test_dimension(2)
    test_dimension(3)


if __name__ == "__main__":
    main()