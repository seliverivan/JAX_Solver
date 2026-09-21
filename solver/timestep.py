import jax.numpy as jnp


DIFFUSION_SPECTRAL_RADIUS = 6.501587301587302

# Граница устойчивости используемой схемы Dormand--Prince 5-го порядка
# на отрицательной вещественной полуоси. Она определяется условием
# |R(-r)| = 1 для функции устойчивости метода.
DORMAND_PRINCE_5_REAL_STABILITY = 3.306567892634947


def convective_dt(
    wave_speeds,
    dx,
    cfl=0.4,
):

    rate = sum(
        speed / h
        for speed, h in zip(wave_speeds, dx)
    )

    rate = jnp.maximum(
        rate,
        1e-14,
    )

    return cfl / rate


def diffusive_dt(
    dx,
    mu,
    cfl=0.4,
):

    rate = sum(
        1.0 / (h * h)
        for h in dx
    )

    spectral_radius = (
        DIFFUSION_SPECTRAL_RADIUS * rate
    )

    denominator = jnp.maximum(
        mu * spectral_radius,
        1e-14,
    )

    dt = (
        cfl
        * DORMAND_PRINCE_5_REAL_STABILITY
        / denominator
    )

    return jnp.where(
        mu > 0,
        dt,
        jnp.inf,
    )


def compute_dt(
    wave_speeds,
    dx,
    mu=0.0,
    cfl=0.4,
):

    dt_conv = convective_dt(
        wave_speeds,
        dx,
        cfl,
    )

    dt_diff = diffusive_dt(
        dx,
        mu,
        cfl,
    )

    return jnp.minimum(
        dt_conv,
        dt_diff,
    )
