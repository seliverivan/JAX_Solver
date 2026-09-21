"""Быстрая проверка ключевых частей проекта без зависимости от pytest."""

import sys

from experiments.test_boundary import (
    test_dirichlet,
    test_neumann,
    test_periodic,
    test_periodic_mismatch,
)
from operators.test_vector_flux import (
    test_equal_states,
    test_physical_flux,
    test_rusanov_formula,
    test_scalar_reduction,
    test_vectorized_data,
)
from solver.global_test import test_full_solver, test_rhs_sanity
from solver.test_integrator_api import (
    test_incomplete_run_reports_only_saved_frames,
    test_num_frames_has_a_two_frame_minimum,
    test_output_frames_are_aligned,
)
from solver.test_timestep import (
    test_compute_dt_selects_stricter_limit,
    test_convective_dt,
    test_diffusive_dt_matches_rk5,
    test_rk5_real_stability_boundary,
)
from solver.test_vector_conservation import test_dimension as test_conservation


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _require_true(check):
    """Преобразует старую проверку с bool-результатом в обычный assert."""

    assert check(), f"Проверка {check.__name__} вернула False"


CHECKS = (
    ("CFL: конвективное ограничение", test_convective_dt),
    ("CFL: диффузионное ограничение", test_diffusive_dt_matches_rk5),
    ("CFL: область устойчивости RK5", test_rk5_real_stability_boundary),
    ("CFL: выбор строгого ограничения", test_compute_dt_selects_stricter_limit),
    ("Интегратор: времена кадров", test_output_frames_are_aligned),
    ("Интегратор: минимум кадров", test_num_frames_has_a_two_frame_minimum),
    ("Интегратор: досрочная остановка", test_incomplete_run_reports_only_saved_frames),
    ("Границы: Дирихле", test_dirichlet),
    ("Границы: Нейман", test_neumann),
    ("Границы: периодические", test_periodic),
    ("Границы: несовместимая периодичность", test_periodic_mismatch),
    ("Поток: физический предел", lambda: _require_true(test_physical_flux)),
    ("Поток: одинаковые состояния", lambda: _require_true(test_equal_states)),
    ("Поток: скалярная редукция", lambda: _require_true(test_scalar_reduction)),
    ("Поток: формула Русанова", lambda: _require_true(test_rusanov_formula)),
    ("Поток: векторизованные данные", lambda: _require_true(test_vectorized_data)),
    ("Векторный RHS: базовая проверка 1D", lambda: test_rhs_sanity(1)),
    ("Векторный решатель: короткий запуск 1D", lambda: test_full_solver(1)),
    ("Векторный решатель: сохранение в 1D", lambda: test_conservation(1)),
)


def main():
    print(f"Запуск быстрых проверок: {len(CHECKS)}")

    for index, (name, check) in enumerate(CHECKS, start=1):
        print(f"\n[{index:02d}/{len(CHECKS):02d}] {name}")
        check()

    print("\nВсе быстрые проверки пройдены")


if __name__ == "__main__":
    main()
