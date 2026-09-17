import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


from operators.flux_divergence import flux_divergence
from weno9.weno_fv import weno9_fv
from rk.rk5 import rk5
from solver.weno9_solver import solve_weno9
from experiments.common import banner
from experiments.common import compute_L2
from experiments.common import compute_TV
from experiments.plots import plot_convergence
from experiments.plots import plot_solution
from boundary.apply import apply_boundary
from boundary.conditions import PERIODIC
# ==========================================================
# Linear advection solver
# ==========================================================

def advection_rhs(u, dx, a):

    ng = 5

    u_ext = apply_boundary(
        u,
        axis=0,
        left_type=PERIODIC,
        right_type=PERIODIC,
        ng=ng,
    )

    return -flux_divergence(
        u_ext,
        dx,
        axis=0,
        flux=lambda uL, uR: (
            0.5 * a * (uL + uR)
            - 0.5 * abs(a) * (uR - uL)
        ),
        ng=ng,
    )



def solve_advection(
        u0,
        dx,
        t_end,
        a,
        cfl):


    u=u0
    t=0.0


    while t<t_end:


        dt=cfl*dx/abs(a)


        if t+dt>t_end:
            dt=t_end-t


        rhs=lambda x: advection_rhs(
            x,
            dx,
            a
        )


        u=rk5(
            u,
            dt,
            rhs
        )


        t+=dt


    return u

def test_linear_advection_smooth():

    banner(
        "TEST 1: Linear advection smooth"
    )


    a=1.0
    t_end=0.1
    cfl=0.1


    Ns=[
    8,
    16,
    32
    ]

    errors=[]


    for N in Ns:


        dx=1/N


        x=(
            jnp.arange(N)+0.5
        )*dx


        u0=jnp.sin(
            2*jnp.pi*x
        )


        u=solve_advection(
            u0,
            dx,
            t_end,
            a,
            cfl
        )


        exact=jnp.sin(
            2*jnp.pi*(x-a*t_end)
        )


        err=compute_L2(
            u,
            exact,
            dx
        )


        errors.append(err)



    print(
        f"{'N':>8} {'L2':>15} {'Order':>10}"
    )


    for i,N in enumerate(Ns):

        if i==0:

            print(
                f"{N:8d}"
                f"{errors[i]:15.6e}"
                f"{'-':>10}"
            )

        else:

            order=np.log2(
                errors[i-1]/errors[i]
            )


            print(
                f"{N:8d}"
                f"{errors[i]:15.6e}"
                f"{order:10.2f}"
            )
    plot_convergence(
        Ns,
        errors,
        order=9,
        title="Linear advection: spatial convergence"
    )

# ==========================================================
# TEST 2
# Linear advection discontinuity
# ==========================================================

def test_linear_advection_step():

    banner(
        "TEST 2: Linear advection step"
    )


    a = 1.0
    t_end = 0.2
    cfl = 0.1


    N = 400

    dx = 1/N


    x = (
        jnp.arange(N)+0.5
    )*dx


    # Step initial condition
    u0 = jnp.where(
        x < 0.5,
        1.0,
        0.0
    )

    u_ext = apply_boundary(
        u0,
        axis=0,
        left_type=PERIODIC,
        right_type=PERIODIC,
        ng=5,
    )

    uL, uR = weno9_fv(
        u_ext,
        axis=0,
        ng=5,
    )

    print("shape uL:", uL.shape)
    print("shape uR:", uR.shape)

    print("LEFT:")
    print("uL[0] =", float(uL[0]))
    print("uR[0] =", float(uR[0]))

    print("RIGHT:")
    print("uL[-1] =", float(uL[-1]))
    print("uR[-1] =", float(uR[-1]))

    u = solve_advection(
        u0,
        dx,
        t_end,
        a,
        cfl,
    )


    # Exact solution:
    # shift by a*t
    exact = jnp.where(
        ((x-a*t_end) % 1.0) < 0.5,
        1.0,
        0.0
    )

    plot_solution(
        x,
        u,
        exact,
        title="Linear advection (step)"
    )
    # --------------------------------------------------
    # Diagnostics
    # --------------------------------------------------

    print(
        f"min(u) = {float(jnp.min(u)):.8f}"
    )

    print(
        f"max(u) = {float(jnp.max(u)):.8f}"
    )


    tv0 = compute_TV(u0)
    tv = compute_TV(u)


    print(
        f"TV initial = {tv0:.8f}"
    )

    print(
        f"TV final   = {tv:.8f}"
    )


    error = compute_L2(
        u,
        exact,
        dx
    )


    print(
        f"L2 error = {error:.6e}"
    )

    overshoot = max(
        abs(float(jnp.min(u))),
        abs(float(jnp.max(u))-1)
    )

    print(
        "overshoot =",
        overshoot
    )

    if (
        float(jnp.min(u)) >= -1e-6
        and
        float(jnp.max(u)) <= 1+1e-6
    ):
        print("PASSED: no overshoot")

    else:
        print("FAILED: oscillations detected")

    print("u first:", np.array(u[:15]))
    print("u last: ", np.array(u[-15:]))

    print("exact first:", np.array(exact[:15]))
    print("exact last: ", np.array(exact[-15:]))

    print("min/max:", float(jnp.min(u)), float(jnp.max(u)))

