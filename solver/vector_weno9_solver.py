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
    Шаг CFL для векторного уравнения Бюргерса.

    Форма массива U:

        1D: (1, Nx)
        2D: (2, Nx, Ny)
        3D: (3, Nx, Ny, Nz)

    Характеристическая скорость оценивается по максимальному
    модулю скорости среди всех компонент.
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
    Решает многомерное векторное уравнение Бюргерса методом
    WENO9-FV с интегрированием RK5 по времени.

    Parameters
    ----------
    U0 : jax.Array
        Начальные средние значения по ячейкам.

        1D:
            (1, Nx)

        2D:
            (2, Nx, Ny)

        3D:
            (3, Nx, Ny, Nz)

    t_end : float
        Конечное время расчёта.

    dx : tuple
        Шаг сетки в каждом пространственном направлении.

    mu : float
        Коэффициент диффузии.

    boundary_types : tuple
        Типы граничных условий для каждого направления.

    boundary_values : tuple
        Граничные значения для каждого направления.

    cfl : float
        Коэффициент CFL.

    num_frames : int
        Число состояний решения, сохраняемых в истории.

        Первый кадр соответствует t = 0, последний — t = t_end.

        Потребление памяти зависит от num_frames, а не от числа
        шагов Рунге--Кутты.

    max_steps : int
        Максимальное число шагов по времени.

    rhs_fn : callable
        Функция правой части. По умолчанию vector_burgers_rhs.

    Returns
    -------
    dict
        Словарь со следующими полями:

        solution :
            Конечное решение.

        history :
            Сохранённые состояния решения.

        times :
            Соответствующие времена вывода.

        steps :
            Число выполненных шагов Рунге--Кутты.
    """

    # ------------------------------------------------------
    # Правая часть
    # ------------------------------------------------------

    rhs = partial(
        rhs_fn,
        dx=dx,
        mu=mu,
        boundary_types=boundary_types,
        boundary_values=boundary_values,
    )

    # ------------------------------------------------------
    # Адаптивный шаг по времени
    # ------------------------------------------------------

    dt_fn = partial(
        vector_burgers_dt,
        dx=dx,
        mu=mu,
        cfl=cfl,
    )

    # ------------------------------------------------------
    # Интегрирование
    #
    # num_frames задаёт объём сохраняемой истории. Интегратор
    # согласует шаги с временами вывода, поэтому каждый кадр
    # точно соответствует указанному времени.
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
    # Преобразование скалярных счётчиков JAX в целые числа Python
    # ------------------------------------------------------

    save_id = int(save_id)

    steps = int(steps)

    # ------------------------------------------------------
    # Результат
    # ------------------------------------------------------

    return {
        "solution": U,
        "history": history[:save_id],
        "times": times[:save_id],
        "steps": steps,
    }
