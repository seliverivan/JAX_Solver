import os

# ==========================================================
# JAX MEMORY SETTINGS
# ==========================================================

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.80"

import jax

jax.config.update(
    "jax_enable_x64",
    True,
)

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from boundary.conditions import PERIODIC
from solver.weno9_solver import solve_weno9


# ==========================================================
# PARAMETERS
# ==========================================================

N = 256

Lx = 1.0
Ly = 1.0

dx = Lx / N
dy = Ly / N

# Small viscosity:
#
# convection dominates
# sharp gradients develop
#
NU = 1.0e-3

T_END = 0.10

CFL = 0.4

NUM_FRAMES = 100

MAX_STEPS = 100000


# ==========================================================
# OUTPUT FILES
# ==========================================================

GIF_FILE = (
    "scalar_burgers_2d_evolution.gif"
)

PNG_FILE = (
    "scalar_burgers_2d_evolution_4frames.png"
)


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
# ==========================================================
#
# 2D scalar viscous Burgers:
#
#     u_t
#     + u u_x
#     + u u_y
#     =
#     nu (u_xx + u_yy)
#
# Initial condition:
#
#     u(x,y,0)
#       =
#     sin(2 pi x) + sin(2 pi y)
#
# This is deliberately nonlinear and smooth initially.
#
# With small viscosity, nonlinear convection creates
# increasingly steep gradients.
# ==========================================================

U0 = (
    jnp.sin(
        2.0 * jnp.pi * X
    )
    +
    jnp.sin(
        2.0 * jnp.pi * Y
    )
)


# ==========================================================
# BOUNDARY CONDITIONS
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
# CFL
# ==========================================================
#
# For 2D scalar Burgers:
#
#     |u|/dx + |u|/dy
#
# determines the convective rate.
#
# The solver itself has its own timestep machinery, but
# keeping this diagnostic here makes the physical CFL
# interpretation explicit.
# ==========================================================


# ==========================================================
# MAIN
# ==========================================================

def main():

    print()
    print("=" * 78)
    print(
        "2D SCALAR VISCOUS BURGERS — WENO9-FV"
    )
    print("=" * 78)

    print(
        f"N           = {N} x {N}"
    )

    print(
        f"dx          = {dx:.6e}"
    )

    print(
        f"dy          = {dy:.6e}"
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

    # ======================================================
    # SOLVE
    # ======================================================

    result = solve_weno9(
        u0=U0,
        t_end=T_END,
        dx=(dx, dy),
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

    # ======================================================
    # CONVERT HISTORY
    # ======================================================

    history = jnp.asarray(
        history
    )

    times = jnp.asarray(
        times
    )

    # ======================================================
    # DIAGNOSTICS
    # ======================================================

    initial_max = float(
        jnp.max(
            jnp.abs(U0)
        )
    )

    final_max = float(
        jnp.max(
            jnp.abs(U)
        )
    )

    global_max = float(
        jnp.max(
            jnp.abs(history)
        )
    )

    finite = bool(
        jnp.all(
            jnp.isfinite(U)
        )
    )

    print(
        f"steps       = {steps}"
    )

    print(
        f"frames      = {len(times)}"
    )

    print()

    print(
        f"initial max |u| = "
        f"{initial_max:.10e}"
    )

    print(
        f"final max |u|   = "
        f"{final_max:.10e}"
    )

    print(
        f"global max |u|  = "
        f"{global_max:.10e}"
    )

    print(
        f"finite          = "
        f"{finite}"
    )

    print()

    # ======================================================
    # BASIC PHYSICAL DIAGNOSTICS
    # ======================================================

    # Integral / mean value.
    #
    # For periodic boundaries, the spatial mean should
    # remain approximately constant.
    #

    mass_initial = (
        jnp.sum(U0)
        * dx
        * dy
    )

    mass_final = (
        jnp.sum(U)
        * dx
        * dy
    )

    mass_error = jnp.abs(
        mass_final
        - mass_initial
    )

    print(
        f"mass initial   = "
        f"{float(mass_initial):.10e}"
    )

    print(
        f"mass final     = "
        f"{float(mass_final):.10e}"
    )

    print(
        f"mass error     = "
        f"{float(mass_error):.10e}"
    )

    print()

    # ======================================================
    # FRAME SELECTION
    # ======================================================

    nframes = len(times)

    snapshot_ids = [
        0,
        nframes // 3,
        (2 * nframes) // 3,
        nframes - 1,
    ]

    # ======================================================
    # GLOBAL COLOR SCALE
    # ======================================================
    #
    # IMPORTANT:
    #
    # Do NOT independently rescale every frame.
    #
    # Otherwise the viewer can be fooled into thinking
    # that the solution changes more dramatically than
    # it actually does.
    #
    # We use one scale for all frames.
    # ======================================================

    color_limit = float(
        jnp.max(
            jnp.abs(history)
        )
    )

    # ======================================================
    # PRESENTATION FIGURE
    # ======================================================
    #
    # Four temporal snapshots.
    #
    # This is the main figure for the presentation.
    # ======================================================

    print(
        "Creating presentation figure..."
    )

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4.4),
        constrained_layout=True,
    )

    for column, idx in enumerate(
        snapshot_ids
    ):

        t = float(
            times[idx]
        )

        field = history[
            idx
        ]

        ax = axes[column]

        im = ax.imshow(
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
            f"$t = {t:.4f}$"
        )

        ax.set_xlabel(
            "$x$"
        )

        if column == 0:
            ax.set_ylabel(
                "$y$"
            )

        else:
            ax.set_ylabel("")

    # One common colorbar for the entire figure.
    cbar = fig.colorbar(
        im,
        ax=axes,
        shrink=0.82,
        pad=0.02,
    )

    cbar.set_label(
        "$u(x,y,t)$"
    )

    fig.suptitle(
        "2D Scalar Viscous Burgers — WENO9-FV",
        fontsize=16,
    )

    fig.savefig(
        PNG_FILE,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Presentation figure saved: "
        f"{PNG_FILE}"
    )

    # ======================================================
    # ANIMATION
    # ======================================================

    print(
        "Creating animation..."
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    # Initial frame.
    image = ax.imshow(
        history[0].T,
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

    cbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    cbar.set_label(
        "$u(x,y,t)$"
    )

    title = ax.set_title(
        "2D Scalar Viscous Burgers\n"
        f"$t = {float(times[0]):.5f}$"
    )

    ax.set_xlabel(
        "$x$"
    )

    ax.set_ylabel(
        "$y$"
    )

    def update(frame):

        image.set_data(
            history[frame].T
        )

        title.set_text(
            "2D Scalar Viscous Burgers\n"
            f"$t = {float(times[frame]):.5f}$"
        )

        return (
            image,
            title,
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=nframes,
        interval=70,
        blit=True,
    )

    animation.save(
        GIF_FILE,
        writer=PillowWriter(
            fps=15
        ),
    )

    plt.close(fig)

    print(
        f"GIF saved: {GIF_FILE}"
    )

    # ======================================================
    # MAXIMUM AMPLITUDE EVOLUTION
    # ======================================================

    max_history = jnp.max(
        jnp.abs(history),
        axis=(
            1,
            2,
        ),
    )

    plt.figure(
        figsize=(8, 4.5)
    )

    plt.plot(
        times,
        max_history,
        linewidth=2.0,
    )

    plt.xlabel(
        "$t$"
    )

    plt.ylabel(
        "$\\max |u|$"
    )

    plt.title(
        "Maximum amplitude"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        "scalar_burgers_2d_max_u.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "Maximum-amplitude plot saved."
    )

    # ======================================================
    # FINAL
    # ======================================================

    print()
    print("=" * 78)
    print(
        "SCALAR 2D BURGERS VISUALIZATION COMPLETE"
    )
    print("=" * 78)

    print()
    print("Created:")
    print(
        f"  {GIF_FILE}"
    )
    print(
        f"  {PNG_FILE}"
    )
    print(
        "  scalar_burgers_2d_max_u.png"
    )

    print()


if __name__ == "__main__":
    main()