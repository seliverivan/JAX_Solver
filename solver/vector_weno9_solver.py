from functools import partial

import jax.numpy as jnp

from rhs.vector_burgers import vector_burgers_rhs

from .integrator import integrate
from .timestep import compute_dt


def vector_burgers_dt(
    U,
    dx,
    mu,
    cfl=0.4,
):
    """
    CFL timestep for vector Burgers.

    U shape:

        1D: (1, Nx)
        2D: (2, Nx, Ny)
        3D: (3, Nx, Ny, Nz)

    The characteristic speed is estimated from
    the maximum absolute velocity over all components.
    """

    speed = jnp.maximum(
        jnp.max(jnp.abs(U)),
        1e-14,
    )

    return compute_dt(
        wave_speeds=(speed,) * len(dx),
        dx=dx,
        mu=mu,
        cfl=cfl,
    )


def solve_vector_weno9(
    U0,
    t_end,
    dx,
    mu,
    boundary_types,
    boundary_values,
    cfl=0.4,
    num_frames=100,
    max_steps=100000,
    rhs_fn=vector_burgers_rhs,
):
    """
    Solve multidimensional vector Burgers equation
    using WENO9-FV with RK5 time integration.

    Parameters
    ----------
    U0 : jax.Array
        Initial cell averages.

        1D:
            (1, Nx)

        2D:
            (2, Nx, Ny)

        3D:
            (3, Nx, Ny, Nz)

    t_end : float
        Final simulation time.

    dx : tuple
        Grid spacing in each spatial direction.

    mu : float
        Diffusion coefficient.

    boundary_types : tuple
        Boundary condition types for each spatial direction.

    boundary_values : tuple
        Boundary values for each spatial direction.

    cfl : float
        CFL coefficient.

    num_frames : int
        Number of solution states stored in history.

        The first frame corresponds to t = 0.
        The final frame corresponds to t = t_end.

        Memory consumption depends on num_frames,
        not on the number of RK timesteps.

    max_steps : int
        Maximum number of timesteps.

    rhs_fn : callable
        RHS function. Defaults to vector_burgers_rhs.

    Returns
    -------
    dict
        Dictionary containing:

        solution :
            Final solution.

        history :
            Saved solution states.

        times :
            Corresponding output times.

        steps :
            Number of RK timesteps performed.
    """

    # ------------------------------------------------------
    # RHS
    # ------------------------------------------------------

    rhs = partial(
        rhs_fn,
        dx=dx,
        mu=mu,
        boundary_types=boundary_types,
        boundary_values=boundary_values,
    )

    # ------------------------------------------------------
    # Adaptive timestep
    # ------------------------------------------------------

    dt_fn = partial(
        vector_burgers_dt,
        dx=dx,
        mu=mu,
        cfl=cfl,
    )

    # ------------------------------------------------------
    # Integration
    #
    # num_frames controls the amount of stored history.
    # The integrator aligns timesteps with output times so
    # every frame corresponds exactly to its reported time.
    # ------------------------------------------------------

    (
        U,
        history,
        times,
        save_id,
        steps,
    ) = integrate(
        u0=U0,
        t_end=t_end,
        rhs=rhs,
        compute_dt=dt_fn,
        max_steps=max_steps,
        num_frames=num_frames,
    )

    # ------------------------------------------------------
    # Convert scalar JAX counters to Python integers
    # ------------------------------------------------------

    save_id = int(save_id)

    steps = int(steps)

    # ------------------------------------------------------
    # Result
    # ------------------------------------------------------

    return {
        "solution": U,
        "history": history[:save_id],
        "times": times[:save_id],
        "steps": steps,
    }
