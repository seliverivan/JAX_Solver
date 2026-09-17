import sys
from pathlib import Path

import os

# ==========================================================
# PROJECT PATH
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

# ==========================================================
# FORCE CPU
# ==========================================================

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import time

import jax
import jax.numpy as jnp

jax.config.update(
    "jax_enable_x64",
    True,
)

from boundary.conditions import PERIODIC

from solver.weno9_solver import solve_weno9


# ==========================================================
# PARAMETERS
# ==========================================================

T_END = 0.5

MU = 1.0e-4

CFL = 0.4

RUNS = 5

GRID_SIZES = (
    16,
    32,
    48,
    64,
)


# ==========================================================
# INITIAL CONDITION
# ==========================================================

def create_initial_condition(N):

    L = 2.0 * jnp.pi

    x = jnp.linspace(
        0.0,
        L,
        N,
        endpoint=False,
    )

    y = jnp.linspace(
        0.0,
        L,
        N,
        endpoint=False,
    )

    z = jnp.linspace(
        0.0,
        L,
        N,
        endpoint=False,
    )

    X, Y, Z = jnp.meshgrid(
        x,
        y,
        z,
        indexing="ij",
    )

    u0 = (
        0.5
        +
        jnp.sin(X)
        * jnp.sin(Y)
        * jnp.sin(Z)
    )

    dx = (
        L / N,
        L / N,
        L / N,
    )

    return u0, dx


# ==========================================================
# BOUNDARY CONDITIONS
# ==========================================================

def create_boundaries():

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

    return (
        boundary_types,
        boundary_values,
    )


# ==========================================================
# SINGLE SOLVE
# ==========================================================

def run_solver(
    u0,
    dx,
    boundary_types,
    boundary_values,
):

    return solve_weno9(
        u0=u0,

        t_end=T_END,

        dx=dx,

        mu=MU,

        boundary_types=boundary_types,

        boundary_values=boundary_values,

        cfl=CFL,

        num_frames=1,

        max_steps=100000,
    )


# ==========================================================
# BENCHMARK
# ==========================================================

def benchmark(N):

    print()
    print("=" * 70)
    print(
        f"CPU WENO9 BENCHMARK — {N}^3"
    )
    print("=" * 70)

    u0, dx = (
        create_initial_condition(N)
    )

    boundary_types, boundary_values = (
        create_boundaries()
    )

    print(
        f"shape  : {u0.shape}"
    )

    print(
        f"device : {u0.device}"
    )

    # ------------------------------------------------------
    # XLA / JIT WARM-UP
    # ------------------------------------------------------

    print()
    print("XLA/JIT warm-up...")

    start = time.perf_counter()

    warmup = run_solver(
        u0,
        dx,
        boundary_types,
        boundary_values,
    )

    warmup[
        "solution"
    ].block_until_ready()

    compilation_time = (
        time.perf_counter()
        - start
    )

    print(
        f"warm-up time: "
        f"{compilation_time:.6f} s"
    )

    # ------------------------------------------------------
    # MEASURED RUNS
    # ------------------------------------------------------

    print()
    print(
        f"Running {RUNS} measured runs..."
    )

    times = []

    steps = None

    for run_id in range(RUNS):

        start = time.perf_counter()

        result = run_solver(
            u0,
            dx,
            boundary_types,
            boundary_values,
        )

        result[
            "solution"
        ].block_until_ready()

        elapsed = (
            time.perf_counter()
            - start
        )

        times.append(
            elapsed
        )

        steps = result[
            "steps"
        ]

        print(
            f"run {run_id + 1}: "
            f"{elapsed:.6f} s"
        )

    # ------------------------------------------------------
    # STATISTICS
    # ------------------------------------------------------

    average = (
        sum(times)
        / len(times)
    )

    minimum = min(times)

    maximum = max(times)

    print()

    print(
        f"steps       : {steps}"
    )

    print(
        f"average     : {average:.6f} s"
    )

    print(
        f"minimum     : {minimum:.6f} s"
    )

    print(
        f"maximum     : {maximum:.6f} s"
    )

    print(
        f"{N} "
        f"steps={steps} "
        f"time={average:.6f}"
    )

    return {
        "N": N,
        "steps": steps,
        "compilation_time": compilation_time,
        "times": times,
        "average": average,
        "minimum": minimum,
        "maximum": maximum,
    }


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    print()

    print(
        "JAX devices:"
    )

    print(
        jax.devices()
    )

    results = []

    for N in GRID_SIZES:

        results.append(
            benchmark(N)
        )

    print()

    print("=" * 70)
    print(
        "CPU BENCHMARK COMPLETE"
    )
    print("=" * 70)