import jax
import jax.numpy as jnp

jax.config.update(
    "jax_enable_x64",
    True,
)

from boundary.apply import apply_boundary
from boundary.conditions import PERIODIC

from operators.second_derivative import second_derivative


# ==========================================================
# Exact scalar function
# ==========================================================

def scalar_function(coords):

    result = 0.0

    for x in coords:
        result = (
            result
            + jnp.sin(2.0 * jnp.pi * x)
            + 0.3 * jnp.cos(4.0 * jnp.pi * x)
        )

    return result


# ==========================================================
# Exact second derivative
# ==========================================================

def exact_second(coords, axis):

    x = coords[axis]

    return (
        -(2.0 * jnp.pi) ** 2
        * jnp.sin(2.0 * jnp.pi * x)
        - 0.3
        * (4.0 * jnp.pi) ** 2
        * jnp.cos(4.0 * jnp.pi * x)
    )


# ==========================================================
# Coordinates
# ==========================================================

def make_coordinates(shape):

    ndim = len(shape)

    coordinates = []

    for n in shape:

        dx = 1.0 / n

        x = (
            jnp.arange(n) + 0.5
        ) * dx

        coordinates.append(x)

    mesh = jnp.meshgrid(
        *coordinates,
        indexing="ij",
    )

    return mesh


# ==========================================================
# Scalar test
# ==========================================================

def run_scalar(shape, axis):

    ndim = len(shape)
    ng = 5

    coords = make_coordinates(shape)

    u = scalar_function(coords)

    dx = 1.0 / shape[axis]

    u_ext = apply_boundary(
        u,
        axis=axis,
        left_type=PERIODIC,
        right_type=PERIODIC,
        ng=ng,
        dx=dx,
    )

    numerical = second_derivative(
        u_ext,
        dx,
        axis=axis,
        ng=ng,
    )

    exact = exact_second(
        coords,
        axis,
    )

    error = jnp.max(
        jnp.abs(
            numerical - exact
        )
    )

    return float(error)


# ==========================================================
# Vector test
# ==========================================================

def run_vector(shape, axis):

    ndim = len(shape)
    ng = 5

    coords = make_coordinates(shape)

    # ------------------------------------------------------
    # Two / three vector components
    # ------------------------------------------------------

    components = ndim

    fields = []

    for component in range(components):

        field = (
            (component + 1.0)
            * scalar_function(coords)
        )

        fields.append(field)

    U = jnp.stack(
        fields,
        axis=0,
    )

    dx = 1.0 / shape[axis]

    U_ext = apply_boundary(
        U,
        axis=axis + 1,
        left_type=PERIODIC,
        right_type=PERIODIC,
        ng=ng,
        dx=dx,
    )

    numerical = second_derivative(
        U_ext,
        dx,
        axis=axis + 1,
        ng=ng,
    )

    exact = jnp.stack(
        [
            (component + 1.0)
            * exact_second(
                coords,
                axis,
            )
            for component in range(components)
        ],
        axis=0,
    )

    error = jnp.max(
        jnp.abs(
            numerical - exact
        )
    )

    return float(error)


# ==========================================================
# Convergence table
# ==========================================================

def test_dimension(ndim):

    resolutions = [
        8,
        16,
        32,
        64,
        128,
    ]

    shape_name = f"{ndim}D"

    print()
    print("=" * 72)
    print(
        f"SECOND DERIVATIVE {shape_name}"
    )
    print("=" * 72)

    # ------------------------------------------------------
    # Scalar
    # ------------------------------------------------------

    print()
    print("SCALAR")
    print("-" * 72)

    print(
        f"{'axis':>8}"
        f"{'N':>8}"
        f"{'error':>20}"
        f"{'order':>12}"
    )

    for axis in range(ndim):

        previous = None

        for n in resolutions:

            shape = (
                n,
            ) * ndim

            error = run_scalar(
                shape,
                axis,
            )

            if previous is None:
                order = 0.0
            else:
                order = float(
                    jnp.log2(
                        previous / error
                    )
                )

            print(
                f"{axis:8d}"
                f"{n:8d}"
                f"{error:20.8e}"
                f"{order:12.4f}"
            )

            previous = error

        print("-" * 72)

    # ------------------------------------------------------
    # Vector
    # ------------------------------------------------------

    print()
    print("VECTOR")
    print("-" * 72)

    print(
        f"{'axis':>8}"
        f"{'N':>8}"
        f"{'error':>20}"
        f"{'order':>12}"
    )

    for axis in range(ndim):

        previous = None

        for n in resolutions:

            shape = (
                n,
            ) * ndim

            error = run_vector(
                shape,
                axis,
            )

            if previous is None:
                order = 0.0
            else:
                order = float(
                    jnp.log2(
                        previous / error
                    )
                )

            print(
                f"{axis:8d}"
                f"{n:8d}"
                f"{error:20.8e}"
                f"{order:12.4f}"
            )

            previous = error

        print("-" * 72)


# ==========================================================
# Main
# ==========================================================

def main():

    test_dimension(1)
    test_dimension(2)
    test_dimension(3)


if __name__ == "__main__":
    main()