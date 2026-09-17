import os

# ==========================================================
# JAX / CUDA MEMORY
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
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

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
t_end = 0.10

cfl = 0.4

NUM_FRAMES = 100


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
# u = sin(2*pi*x) cos(2*pi*y)
#
# v = -cos(2*pi*x) sin(2*pi*y)
#
# div U = 0
# ==========================================================

u0 = (
    jnp.sin(2.0 * jnp.pi * X)
    *
    jnp.cos(2.0 * jnp.pi * Y)
)

v0 = (
    -jnp.cos(2.0 * jnp.pi * X)
    *
    jnp.sin(2.0 * jnp.pi * Y)
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
# SOLVER
# ==========================================================

def run_solver():

    print("=" * 70)
    print("2D VECTOR BURGERS — PRESENTATION EVOLUTION")
    print("=" * 70)

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
        f"mu          = {mu:.6e}"
    )

    print(
        f"t_end       = {t_end:.6e}"
    )

    print(
        f"CFL         = {cfl:.2f}"
    )

    print(
        f"num_frames  = {NUM_FRAMES}"
    )

    print()

    result = solve_vector_weno9(
        U0=U0,
        t_end=t_end,
        dx=(dx, dy),
        mu=mu,
        boundary_types=boundary_types,
        boundary_values=boundary_values,
        cfl=cfl,
        num_frames=NUM_FRAMES,
        max_steps=100000,
    )

    U = result["solution"]

    history = result["history"]
    times = result["times"]

    steps = result["steps"]

    print(
        f"steps       = {steps}"
    )

    print(
        f"frames      = {len(times)}"
    )

    print()

    return U, history, times


# ==========================================================
# PREPARE MAGNITUDE
# ==========================================================

def prepare_magnitude(history):

    history = jnp.asarray(history)

    magnitude = jnp.sqrt(
        history[:, 0] ** 2
        +
        history[:, 1] ** 2
    )

    return magnitude


# ==========================================================
# PRESENTATION ANIMATION
# ==========================================================

def make_animation(
    magnitude,
    times,
):

    print(
        "Preparing animation..."
    )

    # ------------------------------------------------------
    # Convert once to NumPy.
    #
    # This avoids repeatedly transferring data during
    # animation.
    # ------------------------------------------------------

    magnitude_np = np.asarray(
        magnitude
    )

    times_np = np.asarray(
        times
    )

    # ------------------------------------------------------
    # Global scale.
    #
    # The same color scale is used for EVERY frame.
    #
    # This is important for a presentation:
    # changes in color mean actual changes in the solution,
    # not automatic rescaling.
    # ------------------------------------------------------

    vmin = float(
        np.min(magnitude_np)
    )

    vmax = float(
        np.max(magnitude_np)
    )

    print(
        f"magnitude range = "
        f"[{vmin:.6e}, {vmax:.6e}]"
    )

    # ------------------------------------------------------
    # Figure
    # ------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8.5, 7.5),
    )

    # ------------------------------------------------------
    # First frame
    # ------------------------------------------------------

    image = ax.imshow(
        magnitude_np[0].T,
        origin="lower",
        extent=(
            0.0,
            Lx,
            0.0,
            Ly,
        ),
        aspect="equal",
        interpolation="bilinear",
        vmin=vmin,
        vmax=vmax,
    )

    # ------------------------------------------------------
    # Colorbar
    # ------------------------------------------------------

    cbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    cbar.set_label(
        r"$|\mathbf{U}|$",
        fontsize=13,
    )

    # ------------------------------------------------------
    # Labels
    # ------------------------------------------------------

    ax.set_xlabel(
        "x",
        fontsize=12,
    )

    ax.set_ylabel(
        "y",
        fontsize=12,
    )

    title = ax.set_title(
        "2D Vector Burgers — WENO9-FV\n"
        f"$t = {times_np[0]:.4f}$",
        fontsize=15,
        pad=12,
    )

    # ------------------------------------------------------
    # Remove unnecessary whitespace
    # ------------------------------------------------------

    fig.tight_layout()

    # ------------------------------------------------------
    # Animation update
    # ------------------------------------------------------

    def update(frame):

        image.set_data(
            magnitude_np[frame].T
        )

        title.set_text(
            "2D Vector Burgers — WENO9-FV\n"
            f"$t = {times_np[frame]:.4f}$"
        )

        return (
            image,
            title,
        )

    # ------------------------------------------------------
    # Animation
    #
    # 100 frames.
    #
    # 80 ms/frame -> 12.5 FPS
    # ------------------------------------------------------

    animation = FuncAnimation(
        fig,
        update,
        frames=len(times_np),
        interval=80,
        blit=True,
    )

    # ======================================================
    # GIF
    # ======================================================

    gif_name = (
        "vector_burgers_evolution.gif"
    )

    print()
    print(
        f"Saving GIF: {gif_name}"
    )

    animation.save(
        gif_name,
        writer=PillowWriter(
            fps=12
        ),
        dpi=120,
    )

    print(
        "GIF saved."
    )

    # ======================================================
    # MP4
    # ======================================================

    mp4_name = (
        "vector_burgers_evolution.mp4"
    )

    try:

        from matplotlib.animation import FFMpegWriter

        print()
        print(
            f"Saving MP4: {mp4_name}"
        )

        writer = FFMpegWriter(
            fps=20,
            bitrate=5000,
        )

        animation.save(
            mp4_name,
            writer=writer,
            dpi=150,
        )

        print(
            "MP4 saved."
        )

    except Exception as exc:

        print()
        print(
            "MP4 was not created."
        )

        print(
            f"Reason: {exc}"
        )

    plt.close(fig)


# ==========================================================
# STATIC PRESENTATION FRAME
# ==========================================================

def make_presentation_figure(
    magnitude,
    times,
):

    magnitude_np = np.asarray(
        magnitude
    )

    times_np = np.asarray(
        times
    )

    nframes = len(
        times_np
    )

    # ------------------------------------------------------
    # Four representative frames
    # ------------------------------------------------------

    ids = [
        0,
        nframes // 3,
        2 * nframes // 3,
        nframes - 1,
    ]

    vmin = float(
        np.min(magnitude_np)
    )

    vmax = float(
        np.max(magnitude_np)
    )

    # ------------------------------------------------------
    # Figure
    #
    # Extra space on the right is reserved explicitly
    # for the colorbar.
    # ------------------------------------------------------

    fig = plt.figure(
        figsize=(11, 9),
    )

    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=(
            1.0,
            1.0,
            0.06,
        ),
        wspace=0.25,
        hspace=0.25,
    )

    axes = [
        fig.add_subplot(
            gs[0, 0]
        ),
        fig.add_subplot(
            gs[0, 1]
        ),
        fig.add_subplot(
            gs[1, 0]
        ),
        fig.add_subplot(
            gs[1, 1]
        ),
    ]

    cax = fig.add_subplot(
        gs[:, 2]
    )

    # ------------------------------------------------------
    # Draw frames
    # ------------------------------------------------------

    image = None

    for ax, idx in zip(
        axes,
        ids,
    ):

        image = ax.imshow(
            magnitude_np[idx].T,
            origin="lower",
            extent=(
                0.0,
                Lx,
                0.0,
                Ly,
            ),
            aspect="equal",
            interpolation="bilinear",
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_title(
            f"$t = {times_np[idx]:.4f}$",
            fontsize=14,
            pad=8,
        )

        ax.set_xlabel(
            "x",
            fontsize=11,
        )

        ax.set_ylabel(
            "y",
            fontsize=11,
        )

    # ------------------------------------------------------
    # Dedicated colorbar axis
    # ------------------------------------------------------

    cbar = fig.colorbar(
        image,
        cax=cax,
    )

    cbar.set_label(
        r"$|\mathbf{U}|$",
        fontsize=13,
        labelpad=10,
    )

    # ------------------------------------------------------
    # Main title
    # ------------------------------------------------------

    fig.suptitle(
        "2D Vector Burgers — Evolution of $|\\mathbf{U}|$",
        fontsize=18,
        y=0.97,
    )

    # ------------------------------------------------------
    # Explicit layout
    # ------------------------------------------------------

    fig.subplots_adjust(
        left=0.07,
        right=0.94,
        bottom=0.07,
        top=0.90,
    )

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    filename = (
        "vector_burgers_evolution_4frames.png"
    )

    fig.savefig(
        filename,
        dpi=220,
    )

    plt.close(fig)

    print(
        f"Presentation figure saved: {filename}"
    )

# ==========================================================
# MAIN
# ==========================================================

def main():

    U, history, times = run_solver()

    # ------------------------------------------------------
    # Basic diagnostics
    # ------------------------------------------------------

    speed_final = jnp.sqrt(
        U[0] ** 2
        +
        U[1] ** 2
    )

    print(
        f"final max |u| = "
        f"{float(jnp.max(jnp.abs(U[0]))):.6e}"
    )

    print(
        f"final max |v| = "
        f"{float(jnp.max(jnp.abs(U[1]))):.6e}"
    )

    print(
        f"final max |U| = "
        f"{float(jnp.max(speed_final)):.6e}"
    )

    # ------------------------------------------------------
    # Magnitude history
    # ------------------------------------------------------

    magnitude = prepare_magnitude(
        history
    )

    # ------------------------------------------------------
    # Animation
    # ------------------------------------------------------

    make_animation(
        magnitude,
        times,
    )

    # ------------------------------------------------------
    # Static presentation image
    # ------------------------------------------------------

    make_presentation_figure(
        magnitude,
        times,
    )

    # ------------------------------------------------------
    # Done
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print("PRESENTATION VISUALIZATION COMPLETE")
    print("=" * 70)

    print()
    print(
        "Created:"
    )

    print(
        "  vector_burgers_evolution.gif"
    )

    print(
        "  vector_burgers_evolution.mp4"
    )

    print(
        "  vector_burgers_evolution_4frames.png"
    )


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    main()