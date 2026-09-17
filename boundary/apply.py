import jax.numpy as jnp

from boundary.conditions import (
    PERIODIC,
    DIRICHLET,
    NEUMANN,
    OUTFLOW,
)


# ==========================================================
# Helpers
# ==========================================================

def _prepare_boundary_value(value, u, axis):
    """
    Convert boundary value to the tangential shape.

    Accepted:
        scalar
        shape compatible with u without `axis`
        same ndim as u with singleton normal axis
    """

    value = jnp.asarray(
        value,
        dtype=u.dtype,
    )

    tangential_shape = (
        u.shape[:axis]
        + u.shape[axis + 1:]
    )

    # Scalar
    if value.ndim == 0:
        return jnp.broadcast_to(
            value,
            tangential_shape,
        )

    # Same ndim as u, singleton normal direction
    if value.ndim == u.ndim:

        if value.shape[axis] != 1:
            raise ValueError(
                "Boundary value with ndim equal to u.ndim "
                "must have size 1 along the normal axis."
            )

        value = jnp.squeeze(
            value,
            axis=axis,
        )

    if value.ndim != u.ndim - 1:
        raise ValueError(
            f"Boundary value has ndim={value.ndim}; "
            f"expected scalar or ndim={u.ndim - 1}."
        )

    return jnp.broadcast_to(
        value,
        tangential_shape,
    )


# ==========================================================
# Periodic
# ==========================================================

def apply_periodic(
    u,
    axis,
    ng,
):
    """
    Periodic ghost cells.

    Output:

        [left ghosts, u, right ghosts]
    """

    left = jnp.take(
        u,
        jnp.arange(-ng, 0),
        axis=axis,
    )

    right = jnp.take(
        u,
        jnp.arange(0, ng),
        axis=axis,
    )

    return jnp.concatenate(
        (
            left,
            u,
            right,
        ),
        axis=axis,
    )


# ==========================================================
# Dirichlet LEFT
# ==========================================================

def _dirichlet_left(
    u,
    axis,
    value,
    ng,
):
    """
    Second-order-compatible FV Dirichlet ghost cells.

    Boundary is located at the face immediately before
    physical cell 0.

    For cell averages:

        ghost[k] = 2*u_boundary - u_mirror

    where the physical cells are mirrored across the
    boundary face.

    Returned ordering:

        far -> near
    """

    value = _prepare_boundary_value(
        value,
        u,
        axis,
    )

    # Physical cells nearest the boundary:
    #
    # u[0], u[1], ..., u[ng-1]
    #
    # Mirror them:
    #
    # ghost nearest = 2*bc - u[0]
    # next           = 2*bc - u[1]
    # ...
    # ------------------------------------------------------

    mirrored = jnp.take(
        u,
        jnp.arange(ng),
        axis=axis,
    )

    bc = jnp.expand_dims(
        value,
        axis=axis,
    )

    ghost_near_to_far = (
        2.0 * bc
        - mirrored
    )

    # We need:
    #
    # [far, ..., near]
    #
    return jnp.flip(
        ghost_near_to_far,
        axis=axis,
    )


# ==========================================================
# Dirichlet RIGHT
# ==========================================================

def _dirichlet_right(
    u,
    axis,
    value,
    ng,
):
    """
    Second-order-compatible FV Dirichlet ghost cells.

    Returned ordering:

        near -> far
    """

    value = _prepare_boundary_value(
        value,
        u,
        axis,
    )

    # Physical cells nearest the right boundary:
    #
    # u[-1], u[-2], ..., u[-ng]
    # ------------------------------------------------------

    mirrored = jnp.take(
        u,
        jnp.arange(-1, -ng - 1, -1),
        axis=axis,
    )

    bc = jnp.expand_dims(
        value,
        axis=axis,
    )

    ghost = (
        2.0 * bc
        - mirrored
    )

    return ghost


# ==========================================================
# Neumann LEFT
# ==========================================================

def _neumann_left(
    u,
    axis,
    value,
    ng,
    dx,
):
    """
    Constant-gradient extrapolation.

        du/dx = value

    ghost cell center is one, two, ... dx away from
    the first physical cell center.
    """

    value = _prepare_boundary_value(
        value,
        u,
        axis,
    )

    base = jnp.take(
        u,
        0,
        axis=axis,
    )

    bc = jnp.expand_dims(
        value,
        axis=axis,
    )

    distance = jnp.arange(
        ng,
        0,
        -1,
        dtype=u.dtype,
    )

    shape = [1] * u.ndim
    shape[axis] = ng

    distance = distance.reshape(shape)

    return (
        jnp.expand_dims(
            base,
            axis=axis,
        )
        - distance * dx * bc
    )


# ==========================================================
# Neumann RIGHT
# ==========================================================

def _neumann_right(
    u,
    axis,
    value,
    ng,
    dx,
):
    """
    Constant-gradient extrapolation.

        du/dx = value
    """

    value = _prepare_boundary_value(
        value,
        u,
        axis,
    )

    base = jnp.take(
        u,
        -1,
        axis=axis,
    )

    bc = jnp.expand_dims(
        value,
        axis=axis,
    )

    distance = jnp.arange(
        1,
        ng + 1,
        dtype=u.dtype,
    )

    shape = [1] * u.ndim
    shape[axis] = ng

    distance = distance.reshape(shape)

    return (
        jnp.expand_dims(
            base,
            axis=axis,
        )
        + distance * dx * bc
    )


# ==========================================================
# Outflow LEFT
# ==========================================================

def _outflow_left(
    u,
    axis,
    ng,
):
    base = jnp.take(
        u,
        0,
        axis=axis,
    )

    return jnp.broadcast_to(
        jnp.expand_dims(
            base,
            axis=axis,
        ),
        list(u.shape[:axis])
        + [ng]
        + list(u.shape[axis + 1:]),
    )


# ==========================================================
# Outflow RIGHT
# ==========================================================

def _outflow_right(
    u,
    axis,
    ng,
):
    base = jnp.take(
        u,
        -1,
        axis=axis,
    )

    return jnp.broadcast_to(
        jnp.expand_dims(
            base,
            axis=axis,
        ),
        list(u.shape[:axis])
        + [ng]
        + list(u.shape[axis + 1:]),
    )


# ==========================================================
# General boundary application
# ==========================================================

def apply_boundary(
    u,
    axis,
    left_type,
    right_type,
    left_value=0.0,
    right_value=0.0,
    ng=4,
    dx=1.0,
):
    """
    Apply boundary conditions.

    Result:

        [left ghosts, physical u, right ghosts]

    Boundary types:

        PERIODIC
        DIRICHLET
        NEUMANN
        OUTFLOW

    Dirichlet convention:

        u_ghost = 2*u_boundary - u_mirror

    Neumann convention:

        du/dx = prescribed value
    """

    # ======================================================
    # Periodic
    # ======================================================

    if (
        left_type == PERIODIC
        or right_type == PERIODIC
    ):

        if (
            left_type != PERIODIC
            or right_type != PERIODIC
        ):
            raise ValueError(
                "PERIODIC boundary must be specified "
                "on both sides."
            )

        return apply_periodic(
            u,
            axis,
            ng,
        )

    # ======================================================
    # LEFT
    # ======================================================

    if left_type == DIRICHLET:

        left = _dirichlet_left(
            u,
            axis,
            left_value,
            ng,
        )

    elif left_type == NEUMANN:

        left = _neumann_left(
            u,
            axis,
            left_value,
            ng,
            dx,
        )

    elif left_type == OUTFLOW:

        left = _outflow_left(
            u,
            axis,
            ng,
        )

    else:

        raise ValueError(
            f"Unsupported left boundary type: "
            f"{left_type}"
        )

    # ======================================================
    # RIGHT
    # ======================================================

    if right_type == DIRICHLET:

        right = _dirichlet_right(
            u,
            axis,
            right_value,
            ng,
        )

    elif right_type == NEUMANN:

        right = _neumann_right(
            u,
            axis,
            right_value,
            ng,
            dx,
        )

    elif right_type == OUTFLOW:

        right = _outflow_right(
            u,
            axis,
            ng,
        )

    else:

        raise ValueError(
            f"Unsupported right boundary type: "
            f"{right_type}"
        )

    # ======================================================
    # Combine
    # ======================================================

    return jnp.concatenate(
        (
            left,
            u,
            right,
        ),
        axis=axis,
    )