"""NumPy reference for smooth, periodic, inviscid 2D scalar Burgers flow.

The equation is u_t + (u**2/2)_x + (u**2/2)_y = 0 on [0, 1]**2,
with u(x, y, 0) = sin(2*pi*x)*sin(2*pi*y)/(4*pi). Before the first shock
at t=2, characteristics give u = u0(x-t*u, y-t*u).

This module evaluates that implicit solution and its finite-volume cell
averages. Reference generation time is separate from solver benchmarks.
"""

from __future__ import annotations

import argparse
import operator
from pathlib import Path
from time import perf_counter

import numpy as np


_TWO_PI = 2.0 * np.pi
_AMPLITUDE = 1.0 / (4.0 * np.pi)


def initial_condition(x, y):
    """Return u0(x, y), broadcasting scalar or array coordinates in float64."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return _AMPLITUDE * np.sin(_TWO_PI * x) * np.sin(_TWO_PI * y)


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if result < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return result


def _solver_parameters(t, tol, max_iter):
    t = float(t)
    tol = float(tol)
    max_iter = _positive_integer(max_iter, "max_iter")
    if not np.isfinite(t) or not 0.0 <= t < 2.0:
        raise ValueError(f"t must satisfy 0 <= t < 2, got {t!r}")
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError(f"tol must be finite and positive, got {tol!r}")
    return t, tol, max_iter


def _fixed_point_map(x, y, t, u, out, scratch):
    """Evaluate u0(x-t*u, y-t*u) using two reusable full-size buffers."""
    out[...] = u
    out *= t
    np.subtract(x, out, out=out)
    np.multiply(out, _TWO_PI, out=out)
    np.sin(out, out=out)

    scratch[...] = u
    scratch *= t
    np.subtract(y, scratch, out=scratch)
    np.multiply(scratch, _TWO_PI, out=scratch)
    np.sin(scratch, out=scratch)
    np.multiply(out, scratch, out=out)
    out *= _AMPLITUDE


def _solve_fixed_point(x, y, t, tol, max_iter):
    """Return solution, iteration count, and maximum implicit residual."""
    x, y = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    )
    u = np.array(initial_condition(x, y), dtype=np.float64, copy=True)
    if not np.all(np.isfinite(u)):
        raise ValueError("x and y must give finite initial values")
    if t == 0.0:
        return u, 0, 0.0

    candidate = np.empty_like(u)
    scratch = np.empty_like(u)
    for iteration in range(1, max_iter + 1):
        _fixed_point_map(x, y, t, u, candidate, scratch)
        difference = float(np.max(np.abs(candidate - u)))
        u, candidate = candidate, u
        if difference < tol:
            # The iterate difference alone does not certify the implicit
            # equation; evaluate its residual at the returned iterate.
            _fixed_point_map(x, y, t, u, candidate, scratch)
            residual = float(np.max(np.abs(u - candidate)))
            if np.isfinite(residual) and residual < tol:
                return u, iteration, residual
    _fixed_point_map(x, y, t, u, candidate, scratch)
    residual = float(np.max(np.abs(u - candidate)))
    raise RuntimeError(
        f"fixed-point iteration did not converge after {max_iter} iterations "
        f"at t={t:g}: residual={residual:.3e}, tol={tol:.3e}"
    )


def exact_solution(x, y, t, tol=1e-13, max_iter=500):
    """Evaluate the smooth exact solution for 0 <= t < 2.

    Coordinates can be scalars or broadcastable NumPy arrays. The result is
    float64, with a scalar result for scalar coordinates. A RuntimeError is
    raised if the fixed-point iteration or its residual misses ``tol``.
    """
    t, tol, max_iter = _solver_parameters(t, tol, max_iter)
    u, _, _ = _solve_fixed_point(x, y, t, tol, max_iter)
    return u[()] if u.ndim == 0 else u


def _cell_averages_with_diagnostics(N, t, quadrature_order, chunk_size, tol, max_iter):
    N = _positive_integer(N, "N")
    quadrature_order = _positive_integer(quadrature_order, "quadrature_order")
    chunk_size = _positive_integer(chunk_size, "chunk_size")
    t, tol, max_iter = _solver_parameters(t, tol, max_iter)

    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    # Cell averages use weights / 2 along each axis, since dx=dy=1/N.
    scaled_weights = weights * 0.5
    offsets = (nodes + 1.0) / (2.0 * N)
    centers = (np.arange(N, dtype=np.float64) + 0.5) / N
    y_nodes = np.arange(N, dtype=np.float64)[None, :, None, None] / N
    y_nodes = y_nodes + offsets[None, None, None, :]
    tensor_weights = scaled_weights[:, None] * scaled_weights[None, :]
    u_avg = np.empty((N, N), dtype=np.float64)
    max_iterations = 0
    max_residual = 0.0

    for start in range(0, N, chunk_size):
        stop = min(start + chunk_size, N)
        x_nodes = np.arange(start, stop, dtype=np.float64)[:, None, None, None] / N
        x_nodes = x_nodes + offsets[None, None, :, None]
        values, iterations, residual = _solve_fixed_point(
            x_nodes, y_nodes, t, tol, max_iter
        )
        # values[a, j, p, q] is the solution at quadrature point (p, q)
        # in cell (start+a, j). Contract only the two quadrature axes.
        u_avg[start:stop, :] = np.einsum(
            "ijpq,pq->ij", values, tensor_weights, optimize=True
        )
        max_iterations = max(max_iterations, iterations)
        max_residual = max(max_residual, residual)

    return centers, centers.copy(), u_avg, max_iterations, max_residual


def exact_cell_averages(
    N, t, quadrature_order=6, chunk_size=16, *, tol=1e-13, max_iter=500
):
    """Return float64 cell averages on a uniform N by N grid over [0, 1]².

    ``u_avg[i, j]`` corresponds to the cell centered at
    ``x[i] = (i+1/2)/N``, ``y[j] = (j+1/2)/N``. The first array axis is x
    and the second is y, matching ``meshgrid(..., indexing="ij")`` in the
    existing 2D experiment. ``chunk_size`` limits the number of x rows
    whose tensor-product Gauss-Legendre points are held at once.
    """
    return _cell_averages_with_diagnostics(
        N, t, quadrature_order, chunk_size, tol, max_iter
    )[2]


def _run_tests():
    """Check the characteristic solution and true finite-volume averages."""
    x, y = np.meshgrid(
        np.linspace(0.03, 0.97, 17), np.linspace(0.07, 0.93, 19), indexing="ij"
    )

    u0 = exact_solution(x, y, 0.0)
    initial_error = float(np.max(np.abs(u0 - initial_condition(x, y))))
    assert initial_error <= 5e-16
    print(f"1. Initial condition: max error {initial_error:.3e}")

    u = exact_solution(x, y, 1.6)
    residual = float(np.max(np.abs(u - initial_condition(x - 1.6*u, y - 1.6*u))))
    assert residual <= 1e-12
    print(f"2. Implicit residual at t=1.6: {residual:.3e}")

    periodic_x = float(np.max(np.abs(exact_solution(x + 1, y, 1.6) - u)))
    periodic_y = float(np.max(np.abs(exact_solution(x, y + 1, 1.6) - u)))
    assert max(periodic_x, periodic_y) <= 1e-12
    print(f"3. Periodicity: x={periodic_x:.3e}, y={periodic_y:.3e}")

    N = 12
    faces = np.arange(N + 1, dtype=np.float64) / N
    sine_averages = (
        np.cos(_TWO_PI * faces[:-1]) - np.cos(_TWO_PI * faces[1:])
    ) * N / _TWO_PI
    analytic = _AMPLITUDE * sine_averages[:, None] * sine_averages[None, :]
    numeric = exact_cell_averages(N, 0.0, quadrature_order=6, chunk_size=3)
    average_error = float(np.max(np.abs(numeric - analytic)))
    assert average_error <= 1e-13
    print(f"4. Initial cell averages: max error {average_error:.3e}")

    averages = [
        exact_cell_averages(N, 1.6, quadrature_order=order, chunk_size=3)
        for order in (4, 6, 8)
    ]
    difference_46 = float(np.max(np.abs(averages[0] - averages[1])))
    difference_68 = float(np.max(np.abs(averages[1] - averages[2])))
    assert difference_68 < difference_46
    print(f"5. Quadrature at t=1.6: |4-6|={difference_46:.3e}, |6-8|={difference_68:.3e}")

    bound = _AMPLITUDE + 1e-14
    assert np.all(np.isfinite(u)) and np.all(np.isfinite(averages[2]))
    assert np.max(np.abs(u)) <= bound and np.max(np.abs(averages[2])) <= bound
    print(f"6. Finite and bounded: max |u|={np.max(np.abs(u)):.6e}, bound={bound:.6e}")

    for invalid_t in (-0.1, 2.0, np.nan):
        try:
            exact_solution(0.2, 0.3, invalid_t)
        except ValueError:
            pass
        else:
            raise AssertionError(f"time {invalid_t!r} was accepted")
    try:
        exact_solution(0.2, 0.3, 1.6, max_iter=1)
    except RuntimeError:
        pass
    else:
        raise AssertionError("nonconverged iteration was accepted")
    print("7. Time range and nonconvergence checks passed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=128, help="cells on each axis")
    parser.add_argument("--t", type=float, default=1.6, help="time, 0 <= t < 2")
    parser.add_argument("--tol", type=float, default=1e-13, help="fixed-point tolerance")
    parser.add_argument("--max-iter", type=int, default=500, help="iteration limit")
    parser.add_argument("--quadrature-order", type=int, default=6)
    parser.add_argument("--chunk-size", type=int, default=16, help="x rows per block")
    parser.add_argument("--output", type=Path, help="destination .npz path")
    parser.add_argument("--test", action="store_true", help="run built-in checks")
    args = parser.parse_args(argv)

    if args.test:
        _run_tests()
        return

    output = args.output or Path(__file__).parent / "results" / f"exact_{args.N}.npz"
    if output.suffix != ".npz":
        parser.error("--output must end in .npz")

    started = perf_counter()
    x, y, u_avg, iterations, residual = _cell_averages_with_diagnostics(
        args.N, args.t, args.quadrature_order, args.chunk_size, args.tol, args.max_iter
    )
    elapsed = perf_counter() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        x=x,
        y=y,
        u_avg=u_avg,
        N=np.int64(args.N),
        t=np.float64(args.t),
        dx=np.float64(1.0 / args.N),
        dy=np.float64(1.0 / args.N),
        nu=np.float64(0.0),
        quadrature_order=np.int64(args.quadrature_order),
    )
    print(f"Grid: {args.N} x {args.N}; t={args.t:g}; quadrature order={args.quadrature_order}")
    print(f"Cell-average range: [{u_avg.min():.16e}, {u_avg.max():.16e}]")
    print(f"Fixed-point convergence: <= {iterations} iterations per block; max residual={residual:.3e}")
    print(f"Reference computation time: {elapsed:.3f} s (excluded from solver benchmarks)")
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
