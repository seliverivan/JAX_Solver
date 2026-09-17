import jax.numpy as jnp
from jax import jit

from boundary.apply import apply_boundary

from operators.flux_divergence import (
    vector_flux_divergence,
)

from operators.second_derivative import (
    second_derivative,
)


@jit(
    static_argnames=(
        "boundary_types",
        "ng",
    )
)
def vector_burgers_rhs(
    U,
    dx,
    mu,
    boundary_types,
    boundary_values,
    ng=5,
):

    rhs = jnp.zeros_like(U)

    ndim = U.ndim - 1

    for direction, h in enumerate(dx):

        spatial_axis = direction + 1

        left_type, right_type = (
            boundary_types[direction]
        )

        left_value, right_value = (
            boundary_values[direction]
        )

        # --------------------------------------------------
        # Boundary conditions
        # --------------------------------------------------

        U_ext = apply_boundary(
            U,
            axis=spatial_axis,
            left_type=left_type,
            right_type=right_type,
            left_value=left_value,
            right_value=right_value,
            ng=ng,
            dx=h,
        )

        # --------------------------------------------------
        # Convection
        #
        # direction:
        #
        # 0 -> F_x
        # 1 -> F_y
        # 2 -> F_z
        # --------------------------------------------------

        rhs = rhs - vector_flux_divergence(
            U_ext,
            h,
            axis=spatial_axis,
            component=direction,
            ng=ng,
        )

        # --------------------------------------------------
        # Diffusion
        # --------------------------------------------------

        rhs = rhs + mu * second_derivative(
            U_ext,
            h,
            axis=spatial_axis,
            ng=ng,
        )

    return rhs