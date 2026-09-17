from pathlib import Path
import subprocess
import re
import json


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

PYTHON = "/home/ivan/jax-env/bin/python"

GPU_SCRIPT = BASE_DIR / "benchmark_gpu.py"
CPU_SCRIPT = BASE_DIR / "benchmark_cpu.py"


# ==========================================================
# PARSER
# ==========================================================

def parse_output(output):

    results = {}

    pattern = re.compile(
        r"^(\d+)\s+"
        r"steps=(\d+)\s+"
        r"time=([0-9.eE+-]+)"
    )

    for line in output.splitlines():

        match = pattern.match(line.strip())

        if match is None:
            continue

        N = int(match.group(1))
        steps = int(match.group(2))
        time_value = float(match.group(3))

        results[N] = {
            "steps": steps,
            "time": time_value,
        }

    return results


# ==========================================================
# RUN
# ==========================================================

def run(script):

    process = subprocess.run(
        [
            PYTHON,
            str(script),
        ],
        text=True,
        capture_output=True,
    )

    # Показываем stdout
    print(process.stdout)

    # Если произошла ошибка — показываем stderr
    if process.returncode != 0:

        print("=" * 70)
        print("BENCHMARK FAILED")
        print("=" * 70)

        print(
            f"script: {script}"
        )

        print()
        print(process.stderr)

        raise RuntimeError(
            f"Benchmark failed: {script}"
        )

    return parse_output(
        process.stdout
    )


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("WENO9 CPU vs GPU BENCHMARK")
    print("=" * 70)

    print()
    print("Running GPU benchmark...")

    gpu = run(
        GPU_SCRIPT
    )

    print()
    print("Running CPU benchmark...")

    cpu = run(
        CPU_SCRIPT
    )

    # ------------------------------------------------------
    # FINAL TABLE
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        f"{'N':>8}"
        f"{'steps':>10}"
        f"{'CPU [s]':>14}"
        f"{'GPU [s]':>14}"
        f"{'speedup':>14}"
    )

    data = {}

    for N in sorted(gpu):

        if N not in cpu:
            continue

        cpu_time = cpu[N]["time"]
        gpu_time = gpu[N]["time"]

        speedup = (
            cpu_time
            / gpu_time
        )

        data[N] = {
            "steps": gpu[N]["steps"],
            "cpu": cpu_time,
            "gpu": gpu_time,
            "speedup": speedup,
        }

        print(
            f"{N:8d}"
            f"{gpu[N]['steps']:10d}"
            f"{cpu_time:14.6f}"
            f"{gpu_time:14.6f}"
            f"{speedup:14.2f}x"
        )

    # ------------------------------------------------------
    # SAVE
    # ------------------------------------------------------

    output_file = (
        BASE_DIR / "benchmark_results.json"
    )

    with open(
        output_file,
        "w",
    ) as f:

        json.dump(
            data,
            f,
            indent=4,
        )

    print()
    print(
        f"Results saved to:"
    )

    print(
        output_file
    )

    print()
    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)