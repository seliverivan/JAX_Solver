import sys

import jax
import jax.numpy as jnp

from rk.rk5 import rk5
from solver.timestep import (
    DIFFUSION_SPECTRAL_RADIUS,
    DORMAND_PRINCE_5_REAL_STABILITY,
    compute_dt,
    convective_dt,
    diffusive_dt,
)


jax.config.update("jax_enable_x64", True)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def test_convective_dt():
    dt = convective_dt(
        wave_speeds=(2.0, 3.0),
        dx=(0.1, 0.2),
        cfl=0.4,
    )

    expected = 0.4 / (2.0 / 0.1 + 3.0 / 0.2)
    assert jnp.isclose(dt, expected)


def test_diffusive_dt_matches_rk5():
    dx = (0.1, 0.2)
    mu = 0.05
    cfl = 0.4

    dt = diffusive_dt(dx=dx, mu=mu, cfl=cfl)
    rate = sum(1.0 / h**2 for h in dx)
    expected = (
        cfl
        * DORMAND_PRINCE_5_REAL_STABILITY
        / (mu * DIFFUSION_SPECTRAL_RADIUS * rate)
    )

    assert jnp.isclose(dt, expected)


def test_rk5_real_stability_boundary():
    def amplification(z):
        return rk5(1.0, 1.0, lambda value: z * value)

    radius = DORMAND_PRINCE_5_REAL_STABILITY

    assert abs(amplification(-0.999 * radius)) <= 1.0
    assert abs(amplification(-1.001 * radius)) > 1.0


def test_compute_dt_selects_stricter_limit():
    parameters = {
        "wave_speeds": (1.0,),
        "dx": (0.1,),
        "mu": 1.0,
        "cfl": 0.4,
    }

    actual = compute_dt(**parameters)
    expected = jnp.minimum(
        convective_dt(
            parameters["wave_speeds"],
            parameters["dx"],
            parameters["cfl"],
        ),
        diffusive_dt(
            parameters["dx"],
            parameters["mu"],
            parameters["cfl"],
        ),
    )

    assert jnp.isclose(actual, expected)


def main():
    test_convective_dt()
    test_diffusive_dt_matches_rk5()
    test_rk5_real_stability_boundary()
    test_compute_dt_selects_stricter_limit()
    print("Проверки CFL и шага по времени пройдены")


if __name__ == "__main__":
    main()
