import jax
import jax.numpy as jnp

from solver.integrator import integrate


jax.config.update("jax_enable_x64", True)


def _constant_rhs(u):
    return jnp.ones_like(u)


def _large_timestep(_):
    return jnp.asarray(0.6, dtype=jnp.float64)


def test_output_frames_are_aligned():
    u, history, times, save_id, steps = integrate(
        u0=jnp.asarray([0.0]),
        t_end=1.0,
        rhs=_constant_rhs,
        compute_dt=_large_timestep,
        max_steps=20,
        num_frames=11,
    )

    expected = jnp.linspace(0.0, 1.0, 11)

    assert int(save_id) == 11
    assert int(steps) == 10
    assert jnp.allclose(times, expected)
    assert jnp.allclose(history[:, 0], expected)
    assert jnp.allclose(u, jnp.asarray([1.0]))


def test_num_frames_has_a_two_frame_minimum():
    _, history, times, save_id, _ = integrate(
        u0=jnp.asarray([0.0]),
        t_end=1.0,
        rhs=_constant_rhs,
        compute_dt=_large_timestep,
        max_steps=20,
        num_frames=1,
    )

    assert history.shape == (2, 1)
    assert times.shape == (2,)
    assert int(save_id) == 2
    assert jnp.allclose(history[:, 0], times)


def test_incomplete_run_reports_only_saved_frames():
    _, history, times, save_id, steps = integrate(
        u0=jnp.asarray([0.0]),
        t_end=1.0,
        rhs=_constant_rhs,
        compute_dt=lambda _: jnp.asarray(0.1),
        max_steps=1,
        num_frames=2,
    )

    assert int(steps) == 1
    assert int(save_id) == 1
    assert history.shape == (2, 1)
    assert times.shape == (2,)


def main():
    test_output_frames_are_aligned()
    test_num_frames_has_a_two_frame_minimum()
    test_incomplete_run_reports_only_saved_frames()
    print("ALL INTEGRATOR API TESTS PASSED")


if __name__ == "__main__":
    main()
