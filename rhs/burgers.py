import jax.numpy as jnp
from jax import jit

from boundary.apply import apply_boundary
from operators.flux_divergence import flux_divergence
from operators.second_derivative import second_derivative
from operators.flux import burgers_rusanov_flux


@jit(
    static_argnames=(
        "boundary_types",
        "ng",
    )
)
def burgers_rhs(
    u,
    dx,
    mu,
    boundary_types,
    boundary_values,
    ng=5,
):

    rhs = jnp.zeros_like(u)

    for axis, h in enumerate(dx):

        left_type, right_type = boundary_types[axis]
        left_value, right_value = boundary_values[axis]

        u_ext = apply_boundary(
            u,
            axis=axis,
            left_type=left_type,
            right_type=right_type,
            left_value=left_value,
            right_value=right_value,
            ng=ng,
            dx=h,
        )

        # --------------------------------------------------
        # Convection
        # --------------------------------------------------

        rhs = rhs - flux_divergence(
            u_ext,
            h,
            axis=axis,
            flux=burgers_rusanov_flux,
            ng=ng,
        )

        # --------------------------------------------------
        # Diffusion
        # --------------------------------------------------

        rhs = rhs + mu * second_derivative(
            u_ext,
            h,
            axis=axis,
            ng=ng,
        )

    return rhs