"""Два опубликованных стационарных профиля вязкого уравнения Бюргерса.

Ниже используются профили и физические параметры из статей. Отрезок [0, 1],
точные значения Дирихле на гранях и t_end=0.1 задают проверку сохранения
стационарности во времени. Они не повторяют все временные и граничные условия
первоисточников. Ошибки сравнивают численные и точные СРЕДНИЕ ПО ЯЧЕЙКАМ.
"""

# Тест 1. Tommaso Tenna, "Projective Integration Schemes for Nonlinear
# Degenerate Parabolic Systems", Journal of Scientific Computing 107, 68 (2026),
# раздел 4.4, формула (97), рисунок 9.
# https://doi.org/10.1007/s10915-026-03286-9
# Перенесены: u_t + (u**2/2)_x = xi*u_xx, xi=1e-3,
# u=-2*xi/delta*tanh((x-0.5)/delta), delta=0.01, dx около 3e-3.
# Здесь N=400 (dx=2.5e-3), точные значения Дирихле и t_end=0.1.
# Статья сравнивает проективные схемы; здесь работает существующий WENO9 FV.

# Тест 2. Rafael A. B. de Queiroz, Fernando A. Kurokawa и
# Valdemir G. Ferreira, "A polynomial upwind scheme for convection
# discretization", труды XXXI CNMAC (2008), раздел 4.2.
# https://arquivo.sbmac.org.br/eventos/cnmac/xxxi_cnmac/PDF/112.pdf
# Перенесены: u_t + (u**2/2)_x = u_xx/Re, Re=100,
# u=tanh(Re*(0.5-x)/2), модуль граничного значения tanh(Re/4)
# и N=50,100,200,400,800. Здесь t_end=0.1 проверяет сохранение
# стационарности; это не расчёт сходимости к стационарному состоянию из статьи.

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

from boundary.conditions import DIRICHLET
from solver.weno9_solver import solve_weno9


RESULTS_DIR = Path("results")
T_END = 0.1


def _log_cosh(z):
    """Устойчивое вычисление log(cosh(z)) вдали от внутреннего слоя."""
    return jnp.logaddexp(z, -z) - jnp.log(2.0)


def exact_cell_averages(n, amplitude, steepness):
    """Среднее A*tanh(B*(0.5-x)) в каждой ячейке отрезка [0, 1]."""
    dx = 1.0 / n
    faces = jnp.arange(n + 1, dtype=jnp.float64) * dx
    z = steepness * (0.5 - faces)
    return amplitude * (_log_cosh(z[:-1]) - _log_cosh(z[1:])) / (steepness * dx)


def _run_case(n, nu, amplitude, steepness):
    dx = 1.0 / n
    x = (jnp.arange(n, dtype=jnp.float64) + 0.5) * dx
    exact = exact_cell_averages(n, amplitude, steepness)
    boundary_magnitude = float(amplitude * jnp.tanh(0.5 * steepness))

    result = solve_weno9(
        u0=exact,
        t_end=T_END,
        dx=(dx,),
        mu=nu,
        boundary_types=((DIRICHLET, DIRICHLET),),
        boundary_values=((boundary_magnitude, -boundary_magnitude),),
        cfl=0.4,
        num_frames=2,
    )
    numerical = result["solution"]
    if not bool(jnp.all(jnp.isfinite(numerical))):
        raise FloatingPointError(f"Численное решение содержит нечисловые значения при N={n}")
    if len(result["times"]) < 2 or result["steps"] >= 100000:
        raise RuntimeError(f"Интегрирование не достигло t_end при N={n}")

    error = jnp.abs(numerical - exact)
    norms = (
        float(jnp.mean(error)),
        float(jnp.sqrt(jnp.mean(error**2))),
        float(jnp.max(error)),
    )
    return x, numerical, exact, norms, result["steps"]


def _save_plots(stem, title, x, numerical, exact, spatial_error=True):
    RESULTS_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, exact, "--", label="Точное среднее по ячейке", linewidth=2)
    ax.plot(x, numerical, label="Численное среднее WENO9 FV", linewidth=1.5)
    ax.set(xlabel="x", ylabel="Среднее значение u в ячейке", title=title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{stem}_profile.png", dpi=160)
    plt.close(fig)

    if not spatial_error:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, jnp.abs(numerical - exact), linewidth=1.5)
    ax.set(xlabel="x", ylabel="Абсолютная ошибка средних", title=f"{title}: ошибка")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{stem}_error.png", dpi=160)
    plt.close(fig)


def _save_convergence_plot(grid_sizes, errors):
    """Показывает изменение трёх норм ошибки при сгущении сетки."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for index, label in enumerate(("L1", "L2", "L∞")):
        ax.loglog(
            grid_sizes,
            [norms[index] for norms in errors],
            "o-",
            label=label,
        )
    ax.set_xticks(grid_sizes, [str(n) for n in grid_sizes])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set(
        xlabel="Число ячеек N",
        ylabel="Ошибка средних по ячейкам",
        title="Тест 2: изменение ошибки при сгущении сетки, Re=100",
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "burgers_re100_convergence.png", dpi=160)
    plt.close(fig)


def benchmark_steady_shock():
    title = "PUBLISHED BENCHMARK 1: VISCOUS BURGERS STEADY SHOCK"
    print("=" * 70, title, "=" * 70, sep="\n")
    n, nu, delta = 400, 1e-3, 1e-2
    x, numerical, exact, (l1, l2, linf), steps = _run_case(
        n, nu, 2 * nu / delta, 1 / delta
    )
    print(f"N             = {n}")
    print(f"dx            = {1/n:.6e}")
    print(f"nu            = {nu:.6e}")
    print(f"delta         = {delta:.6e}")
    print(f"t_end         = {T_END:.6e}")
    print(f"Шагов         = {steps}")
    print("Эталон        = точные средние по ячейкам")
    print(f"L1            = {l1:.12e}")
    print(f"L2            = {l2:.12e}")
    print(f"Linf          = {linf:.12e}")
    print(f"Макс. ошибка  = {linf:.12e}")
    print(f"Численный min = {float(jnp.min(numerical)):.12e}")
    print(f"Численный max = {float(jnp.max(numerical)):.12e}")
    print(f"Точный min    = {float(jnp.min(exact)):.12e}")
    print(f"Точный max    = {float(jnp.max(exact)):.12e}")
    _save_plots("burgers_steady_shock", "Тест 1: стационарный вязкий удар Бюргерса", x, numerical, exact)


def benchmark_reynolds_100():
    title = "PUBLISHED BENCHMARK 2: BURGERS STEADY TANH, Re=100"
    print("\n" + "=" * 70, title, "=" * 70, sep="\n")
    re = 100
    nu = 1 / re
    print(f"Re = {re}")
    print(f"nu = {nu:.6e}")
    print(f"t_end = {T_END:.6e}")
    print("Эталон = точные средние по ячейкам")
    print("N       dx            L1              L2              Linf            шагов")
    print("-" * 83)
    previous = None
    grid_sizes = (50, 100, 200, 400, 800)
    errors = []
    for n in grid_sizes:
        x, numerical, exact, norms, steps = _run_case(n, nu, 1.0, re / 2)
        errors.append(norms)
        print(
            f"{n:<7d} {1/n:.6e}  {norms[0]:.12e}  "
            f"{norms[1]:.12e}  {norms[2]:.12e}  {steps}"
        )
        if previous is not None:
            orders = [float(jnp.log2(old / new)) if new > 0 else float("nan")
                      for old, new in zip(previous, norms)]
            print(f"  Наблюдаемый порядок {n//2}->{n}: "
                  f"L1={orders[0]:.3f}, L2={orders[1]:.3f}, Linf={orders[2]:.3f}")
        previous = norms
        if n == 800:
            _save_plots(
                "burgers_re100_n800",
                "Тест 2: стационарный профиль, Re=100",
                x,
                numerical,
                exact,
                spatial_error=False,
            )
    _save_convergence_plot(grid_sizes, errors)


if __name__ == "__main__":
    benchmark_steady_shock()
    benchmark_reynolds_100()
