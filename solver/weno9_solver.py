from functools import partial

import jax.numpy as jnp

from rhs.burgers import burgers_rhs

from .integrator import integrate
from .timestep import compute_dt


def burgers_dt(u, dx, mu, cfl=0.4):

    speed = jnp.maximum(
        jnp.max(jnp.abs(u)),
        1e-14,
    )

    return compute_dt(
        wave_speeds=(speed,),
        dx=dx,
        mu=mu,
        cfl=cfl,
    )


def solve_weno9(
    u0,
    t_end,
    dx,
    mu,
    boundary_types,
    boundary_values,
    cfl=0.4,
    num_frames=100,
    max_steps=100000,
    rhs_fn=burgers_rhs,
):

    rhs = partial(
        rhs_fn,
        dx=dx,
        mu=mu,
        boundary_types=boundary_types,
        boundary_values=boundary_values,
    )

    dt_fn = partial(
        burgers_dt,
        dx=dx,
        mu=mu,
        cfl=cfl,
    )

    (
        u,
        history,
        times,
        save_id,
        steps,
    ) = integrate(
        u0=u0,
        t_end=t_end,
        rhs=rhs,
        compute_dt=dt_fn,
        max_steps=max_steps,
        num_frames=num_frames,
    )

    save_id = int(save_id)

    return {
        "solution": u,
        "history": history[:save_id],
        "times": times[:save_id],
        "steps": int(steps),
    }