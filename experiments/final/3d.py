import os

# ==========================================================
# JAX MEMORY
# ==========================================================

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.80"

import jax
import jax.numpy as jnp

jax.config.update(
    "jax_enable_x64",
    True,
)

import numpy as np
import matplotlib.pyplot as plt

from boundary.conditions import PERIODIC
from solver.weno9_solver import solve_weno9


# ==========================================================
# PARAMETERS
# ==========================================================

N = 64

Lx = 1.0
Ly = 1.0
Lz = 1.0

dx = Lx / N
dy = Ly / N
dz = Lz / N

NU = 1.0e-3

T_END = 0.05

CFL = 0.3

NUM_FRAMES = 6

MAX_STEPS = 100000


# ==========================================================
# GRID
# ==========================================================

x = (
    jnp.arange(N) + 0.5
) * dx

y = (
    jnp.arange(N) + 0.5
) * dy

z = (
    jnp.arange(N) + 0.5
) * dz


X, Y, Z = jnp.meshgrid(
    x,
    y,
    z,
    indexing="ij",
)


# ==========================================================
# INITIAL CONDITION
# ==========================================================
#
# 3D scalar viscous Burgers:
#
# u_t
# + u u_x
# + u u_y
# + u u_z
# =
# nu (u_xx + u_yy + u_zz)
#
# ==========================================================

U0 = (
    jnp.sin(2.0 * jnp.pi * X)
    +
    jnp.sin(2.0 * jnp.pi * Y)
    +
    jnp.sin(2.0 * jnp.pi * Z)
)


# ==========================================================
# PERIODIC BOUNDARY CONDITIONS
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
    (
        None,
        None,
    ),
)


# ==========================================================
# SOLVER
# ==========================================================

def run_solver():

    print()
    print("=" * 78)
    print(
        "3D SCALAR VISCOUS BURGERS — WENO9-FV"
    )
    print("=" * 78)

    print(
        f"N           = {N} x {N} x {N}"
    )

    print(
        f"dx          = {dx:.6e}"
    )

    print(
        f"dy          = {dy:.6e}"
    )

    print(
        f"dz          = {dz:.6e}"
    )

    print(
        f"nu          = {NU:.6e}"
    )

    print(
        f"t_end       = {T_END:.6e}"
    )

    print(
        f"CFL         = {CFL:.2f}"
    )

    print(
        f"num_frames  = {NUM_FRAMES}"
    )

    print()

    result = solve_weno9(
        u0=U0,
        t_end=T_END,
        dx=(dx, dy, dz),
        mu=NU,
        boundary_types=boundary_types,
        boundary_values=boundary_values,
        cfl=CFL,
        num_frames=NUM_FRAMES,
        max_steps=MAX_STEPS,
    )

    U = result["solution"]

    history = result["history"]

    times = result["times"]

    steps = result["steps"]

    return U, history, times, steps


# ==========================================================
# 3D DIAGNOSTICS
# ==========================================================

def diagnostics(
    U,
    history,
    times,
):

    history = jnp.asarray(history)
    times = jnp.asarray(times)

    print()
    print("-" * 78)

    print(
        f"steps       = {len(times)} frames"
    )

    print(
        f"initial max |u| = "
        f"{float(jnp.max(jnp.abs(U0))):.8e}"
    )

    print(
        f"final max |u|   = "
        f"{float(jnp.max(jnp.abs(U))):.8e}"
    )

    print(
        f"global max |u|  = "
        f"{float(jnp.max(jnp.abs(history))):.8e}"
    )

    print(
        f"finite          = "
        f"{bool(jnp.all(jnp.isfinite(U)))}"
    )

    # ------------------------------------------------------
    # Mass conservation
    # ------------------------------------------------------

    cell_volume = (
        dx * dy * dz
    )

    mass_initial = (
        jnp.sum(U0)
        * cell_volume
    )

    mass_final = (
        jnp.sum(U)
        * cell_volume
    )

    mass_error = jnp.abs(
        mass_final - mass_initial
    )

    print()

    print(
        f"mass initial    = "
        f"{float(mass_initial):.8e}"
    )

    print(
        f"mass final      = "
        f"{float(mass_final):.8e}"
    )

    print(
        f"mass error      = "
        f"{float(mass_error):.8e}"
    )

    print("-" * 78)

    return history, times


# ==========================================================
# 3D VISUALIZATION
# ==========================================================
#
# Для презентации не будем пытаться рисовать весь куб.
#
# Показываем центральный срез:
#
#       z = 0.5
#
# ==========================================================

def make_figure(
    history,
    times,
):

    history_np = np.asarray(history)
    times_np = np.asarray(times)

    nframes = len(times_np)

    ids = [
        0,
        nframes // 3,
        2 * nframes // 3,
        nframes - 1,
    ]

    color_limit = float(
        np.max(np.abs(history_np))
    )

    # ------------------------------------------------------
    # Central z slice
    # ------------------------------------------------------

    z_id = N // 2

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4.2),
        constrained_layout=True,
    )

    image = None

    for ax, idx in zip(
        axes,
        ids,
    ):

        field = (
            history_np[idx, :, :, z_id]
        )

        image = ax.imshow(
            field.T,
            origin="lower",
            extent=(
                0.0,
                Lx,
                0.0,
                Ly,
            ),
            aspect="equal",
            vmin=-color_limit,
            vmax=color_limit,
            interpolation="nearest",
        )

        ax.set_title(
            f"$t={times_np[idx]:.4f}$"
        )

        ax.set_xlabel("$x$")

        ax.set_ylabel("$y$")

    cbar = fig.colorbar(
        image,
        ax=axes,
        shrink=0.85,
        pad=0.02,
    )

    cbar.set_label(
        "$u(x,y,z=0.5,t)$"
    )

    fig.suptitle(
        "3D Scalar Viscous Burgers — WENO9-FV",
        fontsize=16,
    )

    fig.savefig(
        "burgers_3d_4frames.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print(
        "Saved: burgers_3d_4frames.png"
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    U, history, times, steps = run_solver()

    history, times = diagnostics(
        U,
        history,
        times,
    )

    make_figure(
        history,
        times,
    )

    print()
    print("=" * 78)
    print(
        "3D BURGERS TEST COMPLETE"
    )
    print("=" * 78)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    main()