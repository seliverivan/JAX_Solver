import jax
import jax.numpy as jnp

jax.config.update(
    "jax_enable_x64",
    True,
)

from boundary.conditions import PERIODIC
from solver.vector_weno9_solver import solve_vector_weno9


# ==========================================================
# PARAMETERS
# ==========================================================

N = 256

Lx = 1.0
Ly = 1.0

dx = Lx / N
dy = Ly / N

mu = 1.0e-3
t_end = 0.1

cfl = 0.4

num_frames = 100
max_steps = 100000


# ==========================================================
# GRID
# ==========================================================

x = (
    jnp.arange(N) + 0.5
) * dx

y = (
    jnp.arange(N) + 0.5
) * dy

X, Y = jnp.meshgrid(
    x,
    y,
    indexing="ij",
)


# ==========================================================
# INITIAL CONDITION
#
# Divergence-free field:
#
#     u =  sin(2*pi*x) cos(2*pi*y)
#     v = -cos(2*pi*x) sin(2*pi*y)
#
# ==========================================================

u0 = (
    jnp.sin(
        2.0 * jnp.pi * X
    )
    *
    jnp.cos(
        2.0 * jnp.pi * Y
    )
)

v0 = (
    -jnp.cos(
        2.0 * jnp.pi * X
    )
    *
    jnp.sin(
        2.0 * jnp.pi * Y
    )
)

U0 = jnp.stack(
    [
        u0,
        v0,
    ],
    axis=0,
)


# ==========================================================
# PERIODIC BOUNDARIES
# ==========================================================

boundary_types = (
    (
        PERIODIC,
        PERIODIC,
    ),
    (
        PERIODIC,
        PERIODIC,
    ),
)

boundary_values = (
    (
        None,
        None,
    ),
    (
        None,
        None,
    ),
)


# ==========================================================
# MASS
# ==========================================================

def component_mass(U):
    """
    Integral of each velocity component.
    """

    return (
        jnp.sum(
            U,
            axis=(1, 2),
        )
        * dx
        * dy
    )


# ==========================================================
# KINETIC ENERGY
# ==========================================================

def kinetic_energy(U):
    """
    Kinetic energy:

        E = 1/2 integral |U|^2 dx dy
    """

    speed_squared = (
        U[0] ** 2
        + U[1] ** 2
    )

    return (
        0.5
        * jnp.sum(
            speed_squared
        )
        * dx
        * dy
    )


# ==========================================================
# MAIN TEST
# ==========================================================

def test_vector_physics():

    print()
    print("=" * 78)
    print(
        "VECTOR BURGERS — PHYSICAL DIAGNOSTICS"
    )
    print("=" * 78)

    print(
        f"N              = {N} x {N}"
    )

    print(
        f"dx             = {dx:.6e}"
    )

    print(
        f"dy             = {dy:.6e}"
    )

    print(
        f"mu             = {mu:.6e}"
    )

    print(
        f"t_end          = {t_end:.6e}"
    )

    print(
        f"CFL            = {cfl:.2f}"
    )

    print(
        f"num_frames     = {num_frames}"
    )

    print()

    # ======================================================
    # SOLVE
    # ======================================================

    result = solve_vector_weno9(
        U0=U0,
        t_end=t_end,
        dx=(dx, dy),
        mu=mu,
        boundary_types=boundary_types,
        boundary_values=boundary_values,
        cfl=cfl,
        num_frames=num_frames,
        max_steps=max_steps,
    )

    U = result["solution"]

    history = jnp.asarray(
        result["history"]
    )

    times = jnp.asarray(
        result["times"]
    )

    steps = result["steps"]

    # ======================================================
    # BASIC SOLVER CHECK
    # ======================================================

    print(
        f"steps          = {steps}"
    )

    print(
        f"frames         = {len(times)}"
    )

    print()

    assert len(times) == num_frames, (
        f"Expected {num_frames} frames, "
        f"got {len(times)}."
    )

    assert abs(
        float(times[0])
    ) < 1.0e-14

    assert abs(
        float(times[-1]) - t_end
    ) < 1.0e-12

    # ======================================================
    # MASS CONSERVATION
    # ======================================================

    mass_initial = component_mass(
        U0
    )

    mass_final = component_mass(
        U
    )

    mass_error = jnp.abs(
        mass_final
        - mass_initial
    )

    # ======================================================
    # ENERGY HISTORY
    # ======================================================

    energy_history = jax.vmap(
        kinetic_energy
    )(history)

    energy_initial = (
        energy_history[0]
    )

    energy_final = (
        energy_history[-1]
    )

    energy_change = (
        energy_final
        - energy_initial
    )

    # ------------------------------------------------------
    # Check monotonicity.
    #
    # A tiny positive change is tolerated because of
    # floating-point / nonlinear solver effects.
    # ------------------------------------------------------

    energy_nonincreasing_overall = bool(
        energy_final <= energy_initial
    )

    energy_positive = bool(
        energy_final >= 0.0
    )

    # ======================================================
    # AMPLITUDE
    # ======================================================

    speed_initial = jnp.sqrt(
        U0[0] ** 2
        + U0[1] ** 2
    )

    speed_final = jnp.sqrt(
        U[0] ** 2
        + U[1] ** 2
    )

    max_initial = float(
        jnp.max(
            speed_initial
        )
    )

    max_final = float(
        jnp.max(
            speed_final
        )
    )

    max_history = jax.vmap(
        lambda V: jnp.max(
            jnp.sqrt(
                V[0] ** 2
                + V[1] ** 2
            )
        )
    )(history)

    max_global = float(
        jnp.max(
            max_history
        )
    )

    # ======================================================
    # FINITE CHECK
    # ======================================================

    finite = bool(
        jnp.all(
            jnp.isfinite(U)
        )
    )

    # ======================================================
    # OUTPUT
    # ======================================================

    print(
        "MASS"
    )

    print(
        "-" * 78
    )

    print(
        f"u initial      = "
        f"{float(mass_initial[0]): .12e}"
    )

    print(
        f"u final        = "
        f"{float(mass_final[0]): .12e}"
    )

    print(
        f"u error        = "
        f"{float(mass_error[0]): .12e}"
    )

    print()

    print(
        f"v initial      = "
        f"{float(mass_initial[1]): .12e}"
    )

    print(
        f"v final        = "
        f"{float(mass_final[1]): .12e}"
    )

    print(
        f"v error        = "
        f"{float(mass_error[1]): .12e}"
    )

    # ======================================================

    print()
    print(
        "KINETIC ENERGY"
    )

    print(
        "-" * 78
    )

    print(
        f"E(0)           = "
        f"{float(energy_initial):.12e}"
    )

    print(
        f"E(T)           = "
        f"{float(energy_final):.12e}"
    )

    print(
        f"Delta E        = "
        f"{float(energy_change):.12e}"
    )

    print(
        f"Energy decreased = "
        f"{energy_nonincreasing_overall}"
    )

    print(
        f"Energy positive  = "
        f"{energy_positive}"
    )

    # ======================================================

    print()
    print(
        "AMPLITUDE"
    )

    print(
        "-" * 78
    )

    print(
        f"max |U|(0)      = "
        f"{max_initial:.12e}"
    )

    print(
        f"max |U|(T)      = "
        f"{max_final:.12e}"
    )

    print(
        f"max |U| overall = "
        f"{max_global:.12e}"
    )

    # ======================================================

    print()
    print(
        "NUMERICAL VALIDITY"
    )

    print(
        "-" * 78
    )

    print(
        f"finite          = "
        f"{finite}"
    )

    # ======================================================
    # ASSERTIONS
    # ======================================================

    mass_tol = 1.0e-10

    assert finite, (
        "Solution contains NaN or Inf."
    )

    assert bool(
        jnp.max(
            mass_error
        )
        < mass_tol
    ), (
        "Mass conservation failed."
    )

    assert energy_nonincreasing_overall, (
        "Final kinetic energy is larger "
        "than initial kinetic energy."
    )

    assert energy_positive, (
        "Kinetic energy became negative."
    )

    # ======================================================
    # PASS
    # ======================================================

    print()
    print("=" * 78)
    print(
        "PASS"
    )
    print("=" * 78)


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    test_vector_physics()